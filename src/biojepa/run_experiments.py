"""Full experiment suite:

  - k-fold subject-level cross-validation x seeds (default 5 x 3)
  - 8 model configs incl. gradient boosting and two JEPA generations
  - patient-level paired inference: each patient's errors are first averaged
    over seeds, then patients are the resampling / sign-flip unit
    (repeated seeds on the same patients are NOT independent samples)
  - prospective conversion discrimination (baseline index, fixed window,
    time-aware labels, baseline dementia excluded)
  - missingness stress test and multi-step rollouts scored identically for
    every model

Usage:  python -m biojepa.run_experiments --study adni|oasis|both|synthetic
        [--seeds 0 1 2] [--folds 5] [--device cpu|auto] [--max-epochs N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from collections import defaultdict

import numpy as np
import torch

from .data.dataset import check_event_timing, subject_kfold
from .evaluate import (CONVERSION_WINDOW_TOLERANCE, CONVERSION_WINDOW_YEARS,
                       cluster_paired_bootstrap, conversion_analysis, degraded_dyn_lookup,
                       evaluate_predictions, forecast, gru_rollout, jepa_rollout,
                       paired_auc_bootstrap)
from .model.baselines import HGBBaseline, RidgeBaseline, ridge_design_matrix, train_gru
from .pipeline import prepare
from .train import train_jepa

MODEL_CONFIGS = {
    "carry_forward": dict(kind="carry"),
    "ridge": dict(kind="ridge"),
    "hgb": dict(kind="hgb"),
    "gru": dict(kind="gru"),
    "jepa_v1": dict(kind="jepa", version="v1", w_bio=0.0),
    "jepa_v2": dict(kind="jepa", version="v2", w_bio=0.0),
    "jepa_v2_bio": dict(kind="jepa", version="v2", w_bio=0.5),
    "jepa_v2_bio_strong": dict(kind="jepa", version="v2", w_bio=2.0),
}
MODEL_NAMES = list(MODEL_CONFIGS)
FIG_LABELS = ["Carry\nfwd", "Ridge", "HGB", "GRU", "JEPA\nv1", "JEPA\nv2",
              "JEPA v2\n+Bio", "JEPA v2\n+Bio(λ2)"]
# Pre-specified model of interest for every confirmatory comparison. All
# analyses (main table, clinical tests, demo script) use this one model.
PRIMARY_MODEL = "jepa_v2_bio"
COMPARATORS = ["jepa_v2", "gru", "ridge", "hgb"]
PRIMARY_FEATURES = {"adni": ["MMSE", "CDRSB"], "oasis": ["MMSE", "CDR"],
                    "synthetic": ["MMSE_like", "CDR_like"]}
DATA_FILES = {"adni": os.path.join("data", "raw", "adnimerge_sample.rda"),
              "oasis": os.path.join("data", "raw", "oasis_longitudinal.csv")}


# ------------------------------------------------------------------ helpers
def load_study(name: str):
    if name == "adni":
        from .data.adni import load_adni
        return load_adni()
    if name == "oasis":
        from .data.oasis import load_oasis
        return load_oasis()
    if name == "synthetic":
        from .data.synthetic import simulate_cascade
        return simulate_cascade(n_subjects=300, seed=0)
    raise ValueError(name)


def to_jsonable(obj):
    """Strict-JSON conversion: numpy scalars/arrays -> Python, NaN/inf -> None."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    return obj


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2, allow_nan=False)


def _sha256(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def environment_manifest() -> dict:
    """Code version and software environment of a run. The git state is read
    from the repository that contains this file, whatever the working dir."""
    import sklearn
    try:
        git = lambda *a: subprocess.run(["git", "-C", REPO_ROOT, *a], capture_output=True,
                                        text=True, check=True).stdout.strip()
        commit, dirty = git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))
    except Exception:
        commit, dirty = None, None
    return {"git_commit": commit, "git_dirty": dirty, "python": platform.python_version(),
            "torch": torch.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__,
            "platform": platform.platform(),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def run_manifest(study_name: str, study, **extra) -> dict:
    """Provenance block written into EVERY results file: data hash, study
    size, event-timing check, code commit and software versions."""
    return {"study": study_name, "n_subjects": len(study.subjects),
            "features": [s.name for s in study.dyn_specs],
            # the loaders read data/raw/... relative to the working directory
            "data_sha256": _sha256(DATA_FILES[study_name]) if study_name in DATA_FILES else None,
            "event_timing_check": check_event_timing(study),
            "environment": environment_manifest(), **extra}


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        from .utils import get_device
        return get_device()
    return torch.device(name)


# ---------------------------------------------------------- fit / predict
def fit_model(config: dict, prepared, seed: int, device, max_epochs: int | None = None):
    """Train a model on the prepared split (None for carry-forward)."""
    kind = config["kind"]
    if kind == "carry":
        return None
    if kind in ("ridge", "hgb"):
        Xtr = ridge_design_matrix(prepared, prepared.pairs["train"])
        Ytr = np.stack([prepared.dyn_by_subject[p["subject"]][p["j"]] for p in prepared.pairs["train"]])
        model = RidgeBaseline() if kind == "ridge" else HGBBaseline(random_state=seed)
        model.fit(Xtr, np.nan_to_num(Ytr), np.isfinite(Ytr))
        return model
    epoch_kw = {} if max_epochs is None else {"epochs": max_epochs}
    if kind == "gru":
        return train_gru(prepared.study, prepared.pairs["train"], prepared.pairs["val"],
                         lambda s: prepared.dyn_by_subject[s], prepared.x_context,
                         device=device, seed=seed, **epoch_kw)
    if kind == "jepa":
        weights = {k: config[k] for k in ("w_lat", "w_std", "w_cov") if k in config}
        model, _, _ = train_jepa(prepared, device=device, seed=seed, version=config["version"],
                                 w_bio=config["w_bio"], **weights, **epoch_kw)
        return model
    raise ValueError(kind)


def fit_predict(config, prepared, seed: int, device, max_epochs: int | None = None):
    """Train on the prepared split; return (standardized test predictions, model)."""
    model = fit_model(config, prepared, seed, device, max_epochs)
    return forecast(config, model, prepared, prepared.pairs["test"], device), model


# ------------------------------------------------------- cross-validation
def run_cv(study, configs: dict, seeds=(0, 1, 2), k_folds: int = 5,
           device=torch.device("cpu"), max_epochs: int | None = None, log=print,
           conversion: bool = True) -> dict:
    """Train/evaluate every config on every (seed, fold). With
    conversion=False (event timing failed its sanity check) no conversion
    analysis is run.

    Returns per-fold metrics, per-pair absolute errors keyed by
    (patient, input visit, target visit, nominal horizon) with one value per
    seed, and per-patient conversion scores with one value per seed."""
    names = list(configs)
    per_fold = {m: [] for m in names}
    errors = {m: defaultdict(lambda: defaultdict(list)) for m in names}  # m -> feat -> key -> [seed errs]
    conv_scores = {m: defaultdict(list) for m in names}                   # m -> sid -> [seed scores]
    conv_labels: dict[str, int] = {}
    conv_exclusions: dict[str, int] = defaultdict(int)
    runtimes = {m: [] for m in names}
    split_counts = []
    for seed in seeds:
        folds = subject_kfold(study, k=k_folds, seed=seed)
        for fold, (tr, va, te) in enumerate(folds):
            prepared = prepare(study, splits=(tr, va, te))
            split_counts.append({"seed": seed, "fold": fold,
                                 **{f"n_subjects_{k}": len(v) for k, v in
                                    zip(("train", "val", "test"), (tr, va, te))},
                                 **{f"n_pairs_{k}": len(v) for k, v in prepared.pairs.items()}})
            truth, _ = prepared.truth_std("test")
            truth_raw = prepared.dyn_std.inverse(truth)
            keys = [(study.subjects[p["subject"]].sid, p["i"], p["j"], float(p["h"]))
                    for p in prepared.pairs["test"]]
            for name, cfg in configs.items():
                t0 = time.time()
                model = fit_model(cfg, prepared, seed, device, max_epochs)
                pred_std = forecast(cfg, model, prepared, prepared.pairs["test"], device)
                runtimes[name].append(time.time() - t0)
                metrics = evaluate_predictions(prepared, "test", pred_std)
                conv = conversion_analysis(cfg, model, prepared, te, device) if conversion \
                    else {"sids": [], "labels": [], "scores": [], "n": 0, "exclusions": {}}
                if "auc" in conv:
                    metrics["conversion_auc"] = conv["auc"]
                metrics["conversion_n"] = conv["n"]
                if name == names[0] and seed == seeds[0]:
                    for reason, cnt in conv["exclusions"].items():
                        conv_exclusions[reason] += cnt
                per_fold[name].append({"seed": seed, "fold": fold, **metrics})
                pred_raw = prepared.dyn_std.inverse(pred_std)
                for f_idx, spec in enumerate(study.dyn_specs):
                    for k_i, key in enumerate(keys):
                        if np.isfinite(truth_raw[k_i, f_idx]):
                            errors[name][spec.name][key].append(
                                float(abs(pred_raw[k_i, f_idx] - truth_raw[k_i, f_idx])))
                for sid, y, sc in zip(conv["sids"], conv["labels"], conv["scores"]):
                    conv_scores[name][sid].append(float(sc))
                    conv_labels[sid] = int(y)
            log(f"[{study.name} seed{seed} fold{fold + 1}/{k_folds}] done")
    return {"per_fold": per_fold, "errors": errors, "conv_scores": conv_scores,
            "conv_labels": conv_labels, "conv_exclusions": dict(conv_exclusions),
            "conversion_enabled": conversion, "runtimes": runtimes,
            "split_counts": split_counts, "n_seeds": len(seeds)}


def seed_averaged(records: dict, n_seeds: int) -> dict:
    """key -> mean over seeds; every key must have exactly one value per seed."""
    out = {}
    for key, vals in records.items():
        if len(vals) != n_seeds:
            raise ValueError(f"{key}: {len(vals)} values for {n_seeds} seeds")
        out[key] = float(np.mean(vals))
    return out


def patient_level_mae_comparison(cv: dict, a: str, b: str, feature: str,
                                 horizon: float | None = None, n_boot: int = 10000) -> dict:
    """Paired MAE comparison of models a and b on one feature (optionally one
    nominal horizon): errors averaged over seeds per prediction pair, then
    patients as the unit (bootstrap CI + sign-flip p). Negative = a better."""
    ea = seed_averaged(cv["errors"][a][feature], cv["n_seeds"])
    eb = seed_averaged(cv["errors"][b][feature], cv["n_seeds"])
    if set(ea) != set(eb):
        raise ValueError(f"{a} vs {b}: prediction pairs differ for {feature}")
    keys = sorted(k for k in ea if horizon is None or k[3] == float(horizon))
    r = cluster_paired_bootstrap([ea[k] for k in keys], [eb[k] for k in keys],
                                 [k[0] for k in keys], n_boot=n_boot, seed=0)

    def patient_weighted(e):  # same estimand as mean_diff: every patient counts once
        per = defaultdict(list)
        for k in keys:
            per[k[0]].append(e[k])
        return float(np.mean([np.mean(v) for v in per.values()])) if per else None

    return {**r, "n_pairs": len(keys),
            "mae_a_patient_weighted": patient_weighted(ea),
            "mae_b_patient_weighted": patient_weighted(eb),
            "mae_a_pair_weighted": float(np.mean([ea[k] for k in keys])) if keys else None,
            "mae_b_pair_weighted": float(np.mean([eb[k] for k in keys])) if keys else None}


def compare_models(cv: dict, ours: str, opponents: list[str], features: list[str],
                   n_boot: int = 10000) -> dict:
    """Patient-level paired comparisons of `ours` against each opponent.

    MAE: per (patient, input visit, target visit) the absolute error is
    averaged over seeds; patients are then the unit of the bootstrap CI and
    of the sign-flip permutation p-value (each patient counted once).
    Conversion AUC: per-patient scores averaged over seeds; paired bootstrap
    over patients. Bonferroni over the whole reported family."""
    n_seeds = cv["n_seeds"]
    family = [(f, opp) for f in features for opp in opponents] + [(None, opp) for opp in opponents]
    alpha = 0.05 / len(family)
    out: dict = {"_family": {"n_tests": len(family), "bonferroni_alpha": alpha,
                             "ours": ours, "unit": "patient (errors averaged over seeds first)"}}
    for f in features:
        for opp in opponents:
            r = patient_level_mae_comparison(cv, ours, opp, f, n_boot=n_boot)
            out[f"{ours}_vs_{opp}_{f}_MAE"] = {
                **r, "direction": "negative mean_diff favors ours",
                "significant_bonferroni": bool(np.isfinite(r["p"]) and r["p"] < alpha)}
    labels = cv["conv_labels"]
    for opp in opponents:
        key = f"{ours}_vs_{opp}_conversion_AUC"
        if not cv.get("conversion_enabled", True):
            out[key] = {"note": "conversion analysis disabled: event timing failed its check"}
            continue
        sa = seed_averaged(cv["conv_scores"][ours], n_seeds)
        sb = seed_averaged(cv["conv_scores"][opp], n_seeds)
        if set(sa) != set(sb):
            raise ValueError(f"{ours} vs {opp}: conversion subjects differ")
        sids = sorted(sa)
        y = np.array([labels[s] for s in sids])
        if len(sids) < 10 or not (0 < y.sum() < len(y)):
            out[key] = {"n": len(sids), "note": "too few subjects or a single class"}
            continue
        r = paired_auc_bootstrap([sa[s] for s in sids], y, [sb[s] for s in sids],
                                 n_boot=n_boot, seed=0)
        out[key] = {**r, "n_cases": int(y.sum()),
                    "significant_bonferroni": bool(np.isfinite(r["p"]) and r["p"] < alpha)}
    return out


def aggregate(per_fold: list[dict]) -> dict:
    """Descriptive mean / std over all (seed, fold) runs of each metric."""
    keys = set()
    for r in per_fold:
        keys |= {k for k in r if k not in ("seed", "fold")}
    out = {}
    for k in sorted(keys):
        vals = [r[k] for r in per_fold if k in r and r[k] is not None and np.isfinite(r[k])]
        if vals:
            out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n_runs": len(vals)}
    return out


def pooled_conversion_auc(cv: dict, name: str) -> dict:
    """AUC of seed-averaged patient scores pooled over folds (one score per
    patient), with the patient count."""
    from sklearn.metrics import roc_auc_score
    s = seed_averaged(cv["conv_scores"][name], cv["n_seeds"])
    sids = sorted(s)
    y = np.array([cv["conv_labels"][k] for k in sids])
    if len(sids) == 0 or not (0 < y.sum() < len(y)):
        return {"n": len(sids)}
    return {"auc": float(roc_auc_score(y, [s[k] for k in sids])), "n": len(sids),
            "n_cases": int(y.sum())}


def run_study(study_name: str, seeds=(0, 1, 2), k_folds: int = 5,
              device=torch.device("cpu"), out_root="experiments",
              max_epochs: int | None = None, n_boot: int = 10000):
    study = load_study(study_name)
    out_dir = os.path.join(out_root, study_name)
    manifest = run_manifest(study_name, study, seeds=list(seeds), k_folds=k_folds,
                            protocol=f"{k_folds}-fold subject-level CV x {len(seeds)} seeds",
                            primary_model=PRIMARY_MODEL, comparators=COMPARATORS,
                            max_epochs_override=max_epochs, device=str(device))
    timing_ok = manifest["event_timing_check"]["valid"]
    if not timing_ok:
        print("[warn] event timing failed its sanity check — conversion analysis disabled; "
              f"details: {manifest['event_timing_check']}")
    cv = run_cv(study, MODEL_CONFIGS, seeds, k_folds, device, max_epochs, conversion=timing_ok)
    features = [s.name for s in study.dyn_specs]
    result = {
        "manifest": manifest,
        "split_counts": cv["split_counts"],
        "conversion_design": {"enabled": timing_ok, "window_years": CONVERSION_WINDOW_YEARS,
                              "tolerance_years": CONVERSION_WINDOW_TOLERANCE,
                              "exclusions": cv["conv_exclusions"]},
        "aggregate": {m: aggregate(cv["per_fold"][m]) for m in MODEL_NAMES},
        "conversion_pooled": {m: pooled_conversion_auc(cv, m) for m in MODEL_NAMES},
        "per_fold": cv["per_fold"],
        "significance": compare_models(cv, PRIMARY_MODEL, COMPARATORS, features, n_boot=n_boot),
        "n_patients_with_test_pairs": {f: len({k[0] for k in cv["errors"][PRIMARY_MODEL][f]})
                                       for f in features},
        "runtimes_s": {m: {"mean": float(np.mean(v))} for m, v in cv["runtimes"].items()},
    }
    write_json(os.path.join(out_dir, "results.json"), result)
    return result, study


# ------------------------------------------------------- stress & rollouts
def run_stress_and_rollouts(study_name: str, seed: int = 0, device=torch.device("cpu"),
                            out_root: str = "experiments", max_epochs: int | None = None):
    """Hard tests on one held-out fold (descriptive): missingness stress
    curves and multi-step rollouts, scored identically for every model."""
    study = load_study(study_name)
    tr, va, te = subject_kfold(study, k=5, seed=seed)[0]
    prepared = prepare(study, splits=(tr, va, te))

    stress_models = [PRIMARY_MODEL, "jepa_v2", "gru", "ridge", "hgb", "carry_forward"]
    c_keys = [f"mae_{spec.name}_h{h}" for spec in study.dyn_specs
              if spec.group == "C" for h in (1, 2, 3)]
    if not c_keys:
        c_keys = [f"mae_{spec.name}_h{h}" for spec in study.dyn_specs for h in (1, 2, 3)]
    stress: dict = {}
    trained = {}
    for name in stress_models:
        cfg = MODEL_CONFIGS[name]
        model = fit_model(cfg, prepared, seed, device, max_epochs)
        trained[name] = model
        curve = {}
        for frac in (0.0, 0.25, 0.50, 0.75):
            degraded = None if frac == 0.0 else degraded_dyn_lookup(prepared, frac, seed=seed)
            p = forecast(cfg, model, prepared, prepared.pairs["test"], device, dyn_by_subject=degraded)
            m = evaluate_predictions(prepared, "test", p)
            curve[f"{frac:.2f}"] = float(np.mean([m[k] for k in c_keys if k in m]))
        stress[name] = curve

    rollouts = {name: (gru_rollout if name == "gru" else jepa_rollout)(
        trained[name], prepared, "test", device) for name in (PRIMARY_MODEL, "jepa_v2", "gru")}

    out = {"_meta": {"seed": seed, "fold": 0, "n_test_subjects": len(te),
                     "n_test_pairs": len(prepared.pairs["test"]),
                     "note": "single held-out fold; descriptive, no inference"},
           "manifest": run_manifest(study_name, study, max_epochs_override=max_epochs,
                                    device=str(device)),
           "stress_mae": stress, "rollout": rollouts}
    write_json(os.path.join(out_root, study_name, "hard_tests.json"), out)
    return out


# ------------------------------------------------------------- reporting
def _fmt(a: dict, k: str) -> str:
    return f"{a[k]['mean']:.3f}±{a[k]['std']:.3f}" if k in a else "—"


def print_table(result, study_name):
    agg = result["aggregate"]
    feats = PRIMARY_FEATURES[study_name]
    print(f"\n=== {study_name.upper()} — test MAE (mean ± std over runs; descriptive) ===")
    header = ["model"] + [f"{f}@{12 * h}m" for f in feats for h in (1, 2, 3)] + \
             ["convAUC", "capViol", "monoViol"]
    print(" | ".join(f"{h:>16}" for h in header))
    for m in MODEL_NAMES:
        a = agg[m]
        row = [m] + [_fmt(a, f"mae_{f}_h{h}") for f in feats for h in (1, 2, 3)]
        conv = result["conversion_pooled"][m]
        row.append(f"{conv['auc']:.3f}" if "auc" in conv else "—")
        row += [_fmt(a, "capacity_violation_rate"), _fmt(a, "monotonic_violation_rate")]
        print(" | ".join(f"{c:>16}" for c in row))
    sig = result["significance"]
    fam = sig["_family"]
    print(f"\n--- patient-level paired inference (ours = {fam['ours']}; "
          f"Bonferroni alpha = {fam['bonferroni_alpha']:.5f} over {fam['n_tests']} tests) ---")
    for k, v in sig.items():
        if k.startswith("_"):
            continue
        if "auc_a" in v:
            extra = f"AUC {v['auc_a']:.3f} vs {v['auc_b']:.3f} [Δ CI {v['ci_low']:+.3f}, {v['ci_high']:+.3f}]"
        elif "mean_diff" in v and v["mean_diff"] is not None and np.isfinite(v["mean_diff"]):
            extra = f"ΔMAE {v['mean_diff']:+.4f} [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]"
        else:
            print(f"  {k:48s} {v.get('note', 'not estimable')}")
            continue
        flag = "significant" if v.get("significant_bonferroni") else "n.s."
        print(f"  {k:48s} {extra}  p = {v['p']:.4f} ({flag})")


def make_figures(result, hard, study_name, out_root="experiments"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = result["aggregate"]
    fig_dir = os.path.join(out_root, study_name, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    colors = ["#9aa5b1", "#9aa5b1", "#9aa5b1", "#9aa5b1", "#8fb8de", "#4c86c6", "#1f5fa8", "#123c66"]
    n_runs = len(result["per_fold"][MODEL_NAMES[0]])

    feats = PRIMARY_FEATURES[study_name]
    fig, axes = plt.subplots(1, len(feats), figsize=(5 * len(feats), 4))
    axes = np.atleast_1d(axes)
    for ax, f in zip(axes, feats):
        for i, m in enumerate(MODEL_NAMES):
            k = f"mae_{f}_h2"
            if k in agg[m]:
                ax.bar(i, agg[m][k]["mean"], yerr=agg[m][k]["std"], color=colors[i], capsize=3)
        ax.set_xticks(range(len(MODEL_NAMES)))
        ax.set_xticklabels(FIG_LABELS, fontsize=7)
        ax.set_title(f"{f} — MAE at 24 months (lower=better)")
    fig.suptitle(f"Prediction accuracy — {study_name.upper()} (mean ± SD over {n_runs} runs)")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig1_mae.png"), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    # predictions and observed data on the SAME cells (future value observed)
    kinds = [("capacity_violation_rate", "R1: decline beyond capacity", "#c0504d"),
             ("monotonic_violation_rate", "R2: pathology reversal", "#e8a33d")]
    present = [kv for kv in kinds if any(f"{kv[0]}_matched" in agg[m] for m in MODEL_NAMES)]
    for j, (k, lbl, col) in enumerate(present):
        vals = [agg[m].get(f"{k}_matched", {}).get("mean", 0.0) for m in MODEL_NAMES]
        errs = [agg[m].get(f"{k}_matched", {}).get("std", 0.0) for m in MODEL_NAMES]
        off = (j - (len(present) - 1) / 2) * 0.38
        ax.bar(np.arange(len(MODEL_NAMES)) + off, vals, 0.38, yerr=errs, capsize=3, label=lbl, color=col)
        ref = agg[MODEL_NAMES[0]].get(f"observed_{k}")
        if ref:
            ax.axhline(ref["mean"], color=col, ls="--", lw=1,
                       label=f"{lbl.split(':')[0]} rate of the OBSERVED data")
    ax.set_xticks(range(len(MODEL_NAMES)))
    ax.set_xticklabels(FIG_LABELS, fontsize=7)
    ax.set_ylabel("violation rate on test predictions")
    ax.set_title(f"Rule violations vs observed-data reference — {study_name.upper()}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig2_violations.png"), dpi=160)
    plt.close(fig)

    if study_name == "adni":
        fig, ax = plt.subplots(figsize=(9, 4.2))
        for i, m in enumerate(MODEL_NAMES):
            a = agg[m]
            if "decline_rate_pred_Apos" in a and "decline_rate_pred_Aneg" in a:
                ax.bar(i - 0.19, a["decline_rate_pred_Apos"]["mean"], 0.34,
                       yerr=a["decline_rate_pred_Apos"]["std"], color="#b23a3a", capsize=3)
                ax.bar(i + 0.19, a["decline_rate_pred_Aneg"]["mean"], 0.34,
                       yerr=a["decline_rate_pred_Aneg"]["std"], color="#3a7ab2", capsize=3)
        ax.set_xticks(range(len(MODEL_NAMES)))
        ax.set_xticklabels(FIG_LABELS, fontsize=7)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel("mean decline rate (std units / year)")
        ax.set_title("Predicted decline by amyloid status — red A+ vs blue A−")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "fig3_subgroup.png"), dpi=160)
        plt.close(fig)

    if hard:
        fig, ax = plt.subplots(figsize=(7, 4))
        fracs = ["0.00", "0.25", "0.50", "0.75"]
        for name, style in [(PRIMARY_MODEL, "o-"), ("jepa_v2", "s--"), ("gru", "^--"),
                            ("ridge", "v--"), ("hgb", "d--"), ("carry_forward", "x:")]:
            if name in hard["stress_mae"]:
                ys = [hard["stress_mae"][name].get(f, np.nan) for f in fracs]
                ax.plot([float(f) * 100 for f in fracs], ys, style, label=name)
        ax.set_xlabel("input measurements additionally removed (%)")
        ax.set_ylabel("mean MAE over clinical features")
        ax.set_title(f"Missingness stress test — {study_name.upper()} (one held-out fold)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "fig4_stress.png"), dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        for name, style in [(PRIMARY_MODEL, "o-"), ("jepa_v2", "s--"), ("gru", "^--")]:
            if name in hard["rollout"]:
                ys = [hard["rollout"][name].get(f"rollout_mae_{feats[1]}_step{s}", np.nan)
                      for s in (1, 2, 3)]
                ax.plot([12, 24, 36], ys, style, label=name)
        ax.set_xlabel("months ahead (multi-step rollout, predictions fed back)")
        ax.set_ylabel(f"{feats[1]} MAE")
        ax.set_title("Rollout: error accumulation over the trajectory (one fold)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "fig5_rollout.png"), dpi=160)
        plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", choices=["adni", "oasis", "both", "synthetic"], default="both")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--device", default="cpu", help="cpu (default, deterministic), cuda, mps or auto")
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="cap training epochs (pipeline smoke runs only; never for reported results)")
    ap.add_argument("--out-root", default="experiments")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args(argv)
    device = resolve_device(args.device)
    studies = ["adni", "oasis"] if args.study == "both" else [args.study]
    for s in studies:
        res, _ = run_study(s, seeds=args.seeds, k_folds=args.folds, device=device,
                           out_root=args.out_root, max_epochs=args.max_epochs, n_boot=args.n_boot)
        hard = run_stress_and_rollouts(s, seed=args.seeds[-1], device=device,
                                       out_root=args.out_root, max_epochs=args.max_epochs)
        print_table(res, s)
        make_figures(res, hard, s, out_root=args.out_root)
        print(f"figures + results.json + hard_tests.json written to {args.out_root}/{s}/")


if __name__ == "__main__":
    main()
