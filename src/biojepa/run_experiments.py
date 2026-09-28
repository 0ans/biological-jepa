"""Full experiment suite v2 — harder evaluation throughout:

  - 5-fold subject-level cross-validation x 3 seeds (15 runs per model)
  - 8 model configs incl. gradient boosting and two JEPA generations
  - paired bootstrap significance on primary-feature errors and conversion AUC
  - missingness stress test (input measurements dropped 25/50/75%)
  - multi-step rollout evaluation (trajectory error accumulation + violations)

Usage:  python -m biojepa.run_experiments [--study adni|oasis|both] [--seeds 0 1 2]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from .data.adni import load_adni
from .data.dataset import subject_kfold
from .data.oasis import load_oasis
from .evaluate import (degraded_dyn_lookup, evaluate_predictions, gru_predict,
                       gru_predict_degraded, gru_rollout, jepa_predict,
                       jepa_predict_degraded, jepa_rollout, cluster_paired_bootstrap,
                       paired_auc_bootstrap, ridge_predict)
from .model.baselines import (HGBBaseline, RidgeBaseline, carry_forward_predict,
                              train_gru)
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
PRIMARY_FEATURES = {"adni": ["MMSE", "CDRSB"], "oasis": ["MMSE", "CDR"]}
BOOTSTRAP_OPPONENTS = ["jepa_v2", "gru", "ridge", "hgb"]


def fit_predict(config, prepared, seed: int, device):
    """Train (if needed) on the prepared split and return standardized test
    predictions plus the fitted model object (or None)."""
    kind = config["kind"]
    if kind == "carry":
        train_ids = sorted({p["subject"] for p in prepared.pairs["train"]})
        pred_raw, _, _, _ = carry_forward_predict(prepared.study, prepared.pairs["test"], train_ids)
        return prepared.dyn_std.transform(pred_raw), None
    if kind in ("ridge", "hgb"):
        from .model.baselines import ridge_design_matrix
        Xtr = ridge_design_matrix(prepared, prepared.pairs["train"])
        Ytr = np.stack([prepared.dyn_by_subject[p["subject"]][p["j"]] for p in prepared.pairs["train"]])
        Mtr = np.isfinite(Ytr)
        model = RidgeBaseline() if kind == "ridge" else HGBBaseline(random_state=seed)
        model.fit(Xtr, np.nan_to_num(Ytr), Mtr)
        return ridge_predict(prepared, "test", model), model
    if kind == "gru":
        model = train_gru(prepared.study, prepared.pairs["train"], prepared.pairs["val"],
                          lambda s: prepared.dyn_by_subject[s], prepared.x_context,
                          device=device, seed=seed)
        return gru_predict(model, prepared, "test", device), model
    if kind == "jepa":
        model, best_val, _ = train_jepa(prepared, device=device, seed=seed,
                                        version=config["version"], w_bio=config["w_bio"])
        return jepa_predict(model, prepared, "test", device, version=config["version"]), model
    raise ValueError(kind)


def run_study(study_name: str, seeds=(0, 1, 2), k_folds: int = 5,
              device=torch.device("cpu"), out_root="experiments"):
    study = load_adni() if study_name == "adni" else load_oasis()
    out_dir = os.path.join(out_root, study_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)

    feat_idx = {spec.name: d for d, spec in enumerate(study.dyn_specs)}
    per_model: dict[str, list[dict]] = {m: [] for m in MODEL_NAMES}
    runtimes: dict[str, list[float]] = {m: [] for m in MODEL_NAMES}
    # aligned per-pair absolute errors + subject keys per seed (folds partition
    # the pairs; subjects are the resampling clusters for the bootstrap)
    err_seed: dict[str, dict[int, dict[str, list]]] = {m: {} for m in MODEL_NAMES}
    subj_seed: dict[int, dict[str, list]] = {}
    conv_seed: dict[str, dict[int, tuple]] = {m: {} for m in MODEL_NAMES}
    manifest = {"study": study_name, "n_subjects": len(study.subjects),
                "seeds": list(seeds), "k_folds": k_folds, "protocol": "5-fold subject-level CV"}

    for seed in seeds:
        folds = subject_kfold(study, k=k_folds, seed=seed)
        err_seed_m: dict[str, dict[str, list]] = {m: {f: [] for f in feat_idx} for m in MODEL_NAMES}
        subj_seed_m: dict[str, list] = {f: [] for f in feat_idx}
        for fold, (tr, va, te) in enumerate(folds):
            prepared = prepare(study, splits=(tr, va, te))
            fold_subj = np.array([study.subjects[pp["subject"]].sid
                                  for pp in prepared.pairs["test"]])
            for name, cfg in MODEL_CONFIGS.items():
                t0 = time.time()
                pred_std, _ = fit_predict(cfg, prepared, seed, device)
                runtimes[name].append(time.time() - t0)
                metrics, conv = evaluate_predictions(prepared, "test", pred_std, return_conv=True)
                per_model[name].append(metrics)
                truth, cur = prepared.truth_std("test")
                for f_name, f_idx in feat_idx.items():
                    m_ = np.isfinite(truth[:, f_idx])
                    e = np.abs(prepared.dyn_std.inverse(pred_std)[m_, f_idx]
                               - prepared.dyn_std.inverse(truth)[m_, f_idx])
                    err_seed_m[name][f_name].append(e)
                    if name == MODEL_NAMES[0]:
                        subj_seed_m[f_name].append(fold_subj[m_])
                if conv is not None:
                    existing = conv_seed[name].get(seed)
                    conv_seed[name][seed] = conv if existing is None else \
                        (np.concatenate([existing[0], conv[0]]),
                         np.concatenate([existing[1], conv[1]]))
            print(f"[{study_name} seed{seed} fold{fold + 1}/{k_folds}] done", flush=True)
        for name in MODEL_NAMES:
            err_seed[name][seed] = {f: np.concatenate(v) for f, v in err_seed_m[name].items()}
        subj_seed[seed] = {f: np.concatenate(v) for f, v in subj_seed_m.items()}

    # ------- paired bootstrap resampling PATIENTS (cluster level) ------------
    # Visit-pairs of one patient are correlated; resampling patients preserves
    # that correlation. Bonferroni correction over the reported family.
    significance: dict = {}
    ours = "jepa_v2_bio"
    n_family = 0
    prelim = []
    for f_name in feat_idx:
        for opp in BOOTSTRAP_OPPONENTS:
            if opp == ours:
                continue
            prelim.append((f"{ours}_vs_{opp}_{f_name}_MAE", f_name, opp))
    for opp in BOOTSTRAP_OPPONENTS:
        if opp != ours:
            prelim.append((f"{ours}_vs_{opp}_conversion_AUC", None, opp))
    n_family = len(prelim)

    for f_name in feat_idx:
        for opp in BOOTSTRAP_OPPONENTS:
            if opp == ours:
                continue
            # الدلالة الأساسية: تجميع البذور مع بقاء المريض عنقوداً (مريض×بذرة)
            x = np.concatenate([err_seed[ours][s][f_name] for s in seeds])
            y = np.concatenate([err_seed[opp][s][f_name] for s in seeds])
            cl = np.concatenate([subj_seed[s][f_name] + f"__s{s}" for s in seeds])
            pooled = cluster_paired_bootstrap(x, y, cl, seed=0)
            per_seed = [cluster_paired_bootstrap(err_seed[ours][s][f_name],
                                                 err_seed[opp][s][f_name],
                                                 subj_seed[s][f_name], seed=s)
                        for s in seeds]
            significance[f"{ours}_vs_{opp}_{f_name}_MAE"] = {
                "mean_diff": pooled["mean_diff"],
                "ci_low": pooled["ci_low"], "ci_high": pooled["ci_high"],
                "p_cluster_pooled": pooled["p"],
                "n_clusters": pooled.get("n_clusters"),
                "p_per_seed": [round(p["p"], 5) for p in per_seed],
                "bonferroni_alpha": round(0.05 / n_family, 5),
                "direction": "ours_lower_is_better",
            }
    for opp in BOOTSTRAP_OPPONENTS:
        if opp == ours or ours not in conv_seed or opp not in conv_seed:
            continue
        per_seed = [paired_auc_bootstrap(conv_seed[ours][s][0], conv_seed[ours][s][1],
                                         conv_seed[opp][s][0], seed=s) for s in seeds
                    if s in conv_seed[ours] and s in conv_seed[opp]]
        if per_seed:
            significance[f"{ours}_vs_{opp}_conversion_AUC"] = {
                "auc_ours": float(np.mean([r["auc_a"] for r in per_seed])),
                "auc_opp": float(np.mean([r["auc_b"] for r in per_seed])),
                "p_per_seed": [round(r["p"], 5) for r in per_seed],
                "p_family_note": "AUC bootstrap already resamples subjects",
                "bonferroni_alpha": round(0.05 / n_family, 5),
            }

    agg = {m: aggregate(per_model[m]) for m in MODEL_NAMES}
    result = {
        "manifest": manifest,
        "aggregate": agg,
        "per_seed": {m: per_model[m] for m in MODEL_NAMES},
        "significance": significance,
        "runtimes_s": {m: {"mean": float(np.mean(v))} for m, v in runtimes.items()},
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result, study


def aggregate(per_seed: list[dict]) -> dict:
    keys = set()
    for r in per_seed:
        keys |= set(r)
    out = {}
    for k in sorted(keys):
        vals = [r[k] for r in per_seed if k in r and np.isfinite(r[k])]
        if vals:
            out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n_seeds": len(vals)}
    return out


# ------------------------------------------------------- stress & rollouts
def run_stress_and_rollouts(study_name: str, seed: int = 0, device=torch.device("cpu")):
    """Hard tests on one held-out fold: missingness stress curves and
    multi-step rollout evaluation."""
    study = load_adni() if study_name == "adni" else load_oasis()
    folds = subject_kfold(study, k=5, seed=seed)
    tr, va, te = folds[0]
    prepared = prepare(study, splits=(tr, va, te))

    stress_models = ["jepa_v2_bio", "jepa_v2", "gru", "ridge", "hgb"]
    c_keys = [f"mae_{spec.name}_h{h}" for spec in study.dyn_specs
              if spec.group == "C" for h in (1, 2, 3)]
    if not c_keys:  # study without explicit C group: fall back to all features
        c_keys = [f"mae_{spec.name}_h{h}" for spec in study.dyn_specs for h in (1, 2, 3)]
    stress: dict = {}
    trained = {}
    for name in stress_models:
        cfg = MODEL_CONFIGS[name]
        pred_std, model = fit_predict(cfg, prepared, seed, device)
        trained[name] = model
        base, _ = evaluate_predictions(prepared, "test", pred_std, return_conv=True)
        curve = {"0.00": float(np.mean([base[k] for k in c_keys if k in base]))}
        for frac in (0.25, 0.50, 0.75):
            degraded = degraded_dyn_lookup(prepared, frac, seed=seed)
            if name.startswith("jepa"):
                p = jepa_predict_degraded(trained[name], prepared, "test", degraded,
                                          device, version="v2")
            elif name == "gru":
                p = gru_predict_degraded(trained[name], prepared, "test", degraded, device)
            else:
                from .model.baselines import ridge_design_matrix
                X = ridge_design_matrix(prepared, prepared.pairs["test"],
                                        dyn_lookup=lambda s: degraded[s])
                p = trained[name].predict(X)
            m, _ = evaluate_predictions(prepared, "test", p, return_conv=True)
            curve[f"{frac:.2f}"] = float(np.mean([m[k] for k in c_keys if k in m]))
        stress[name] = curve

    rollouts: dict = {}
    for name in ("jepa_v2_bio", "jepa_v2", "gru"):
        if name == "gru":
            rollouts[name] = gru_rollout(trained[name], prepared, "test", device)
        else:
            rollouts[name] = jepa_rollout(trained[name], prepared, "test", device)

    out = {"stress_mae": stress, "rollout": rollouts}
    path = os.path.join("experiments", study_name, "hard_tests.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return out


# ------------------------------------------------------------- reporting
def print_table(result, study_name):
    agg = result["aggregate"]
    feats = PRIMARY_FEATURES[study_name]
    print(f"\n=== {study_name.upper()} — test MAE (mean ± std over 15 runs) ===")
    header = ["model"] + [f"{f}@{h}m" for f in feats for h in (12, 24, 36)] + \
             ["convAUC", "capViol", "monoViol"]
    print(" | ".join(f"{h:>16}" for h in header))
    for m in MODEL_NAMES:
        a = agg[m]
        row = [m]
        for f in feats:
            for h in (1, 2, 3):
                k = f"mae_{f}_h{h}"
                row.append(f"{a[k]['mean']:.3f}±{a[k]['std']:.3f}" if k in a else "—")
        for k in ("conversion_auc", "capacity_violation_rate", "monotonic_violation_rate"):
            row.append(f"{a[k]['mean']:.3f}±{a[k]['std']:.3f}" if k in a else "—")
        print(" | ".join(f"{c:>16}" for c in row))
    print("\n--- significance (cluster bootstrap over patients, ours = jepa_v2_bio) ---")
    for k, v in result["significance"].items():
        extra = f"auc {v.get('auc_ours', 0):.3f} vs {v.get('auc_opp', 0):.3f}" if "AUC" in k \
            else f"ΔMAE {v['mean_diff']:+.4f} [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]"
        pp = v.get("p_cluster_pooled")
        print(f"  {k:48s} {extra}  p_pooled = {pp:.4f}" if pp is not None else
              f"  {k:48s} {extra}  p_per_seed = {v.get('p_per_seed')}")


def make_figures(result, hard, study_name, out_root="experiments"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = result["aggregate"]
    fig_dir = os.path.join(out_root, study_name, "figures")
    colors = ["#9aa5b1", "#9aa5b1", "#9aa5b1", "#9aa5b1", "#8fb8de", "#4c86c6", "#1f5fa8", "#123c66"]

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
    fig.suptitle(f"Prediction accuracy — {study_name.upper()} (5-fold CV × 3 seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig1_mae.png"), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    kinds = [("capacity_violation_rate", "R1: decline w/o biological cause"),
             ("monotonic_violation_rate", "R2: pathology reversal")]
    for j, (k, lbl) in enumerate([kv for kv in kinds if any(kv[0] in agg[m] for m in MODEL_NAMES)]):
        vals = [agg[m].get(k, {}).get("mean", 0.0) for m in MODEL_NAMES]
        errs = [agg[m].get(k, {}).get("std", 0.0) for m in MODEL_NAMES]
        ax.bar(np.arange(len(MODEL_NAMES)) + (j - 0.5) * 0.38, vals, 0.38,
               yerr=errs, capsize=3, label=lbl, color=["#c0504d", "#e8a33d"][j])
    ax.set_xticks(range(len(MODEL_NAMES)))
    ax.set_xticklabels(FIG_LABELS, fontsize=7)
    ax.set_ylabel("violation rate on test predictions")
    ax.set_title(f"Biological plausibility — {study_name.upper()}")
    ax.legend(fontsize=8)
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
        for name, style in [("jepa_v2_bio", "o-"), ("jepa_v2", "s--"), ("gru", "^--"),
                            ("ridge", "v--"), ("hgb", "d--")]:
            if name in hard["stress_mae"]:
                ys = [hard["stress_mae"][name].get(f, np.nan) for f in fracs]
                ax.plot([f * 100 for f in map(float, fracs)], ys, style, label=name)
        ax.set_xlabel("input measurements additionally removed (%)")
        ax.set_ylabel("mean MAE over clinical features")
        ax.set_title(f"Missingness stress test — {study_name.upper()} (held-out fold)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "fig4_stress.png"), dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        for name, style in [("jepa_v2_bio", "o-"), ("jepa_v2", "s--"), ("gru", "^--")]:
            if name in hard["rollout"]:
                ys = [hard["rollout"][name].get(f"rollout_mae_{feats[1]}_step{s}", np.nan)
                      for s in (1, 2, 3)]
                ax.plot([12, 24, 36], ys, style, label=name)
        ax.set_xlabel("months ahead (multi-step rollout, predictions fed back)")
        ax.set_ylabel(f"{feats[1]} MAE")
        ax.set_title("Rollout: error accumulation over the trajectory")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "fig5_rollout.png"), dpi=160)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", choices=["adni", "oasis", "both"], default="both")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    studies = ["adni", "oasis"] if args.study == "both" else [args.study]
    for s in studies:
        res, study = run_study(s, seeds=args.seeds, k_folds=args.folds)
        try:
            hard = run_stress_and_rollouts(s, seed=args.seeds[-1])
        except Exception as e:  # hard tests must never void the main results
            print(f"[warn] hard tests failed for {s}: {e}")
            hard = None
        print_table(res, s)
        if hard:
            print(f"\n--- hard tests ({s}) ---")
            print("stress MAE:", json.dumps(hard["stress_mae"], indent=1))
            print("rollout:", json.dumps(hard["rollout"], indent=1))
        make_figures(res, hard, s)
        print(f"figures + results.json + hard_tests.json written to experiments/{s}/")


if __name__ == "__main__":
    main()
