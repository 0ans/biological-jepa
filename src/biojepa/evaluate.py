"""Evaluation: forecasting for every model kind, prediction accuracy,
prospective conversion discrimination, biological-violation rates (with the
observed-data reference), amyloid-subgroup calibration, patient-level paired
inference, missingness stress tests and multi-step rollouts."""
from __future__ import annotations

import numpy as np
import torch

from .model.baselines import gru_collate, ridge_design_matrix
from .utils import fourier_time

# Pre-specified conversion design (see conversion_queries)
CONVERSION_FEATURES = ("CDRSB", "CDR", "CDR_like")
CONVERSION_WINDOW_YEARS = 3.0


# --------------------------------------------------------------- forecasting
def _jepa_forward(model, t: dict, version: str, device, batch: int = 1024) -> np.ndarray:
    """Standardized forecasts from prepared input tensors (CPU tensors are
    moved to `device` batch by batch)."""
    model.eval()
    preds = []
    with torch.no_grad():
        for lo in range(0, t["x_i"].shape[0], batch):
            sl = slice(lo, lo + batch)
            if version in ("v2", "v3"):
                p = model.predict_dyn(t["seq_i"][sl].to(device), t["len_i"][sl].to(device),
                                      t["ctx"][sl].to(device), t["dt_feat"][sl].to(device),
                                      dyn_i=t["dyn_i"][sl].to(device))
            elif version == "v1":
                p = model.predict_dyn(t["x_i"][sl].to(device), t["dt_feat"][sl].to(device))
            else:
                raise ValueError(f"unknown JEPA version {version!r}")
            preds.append(p.cpu().numpy())
    return np.concatenate(preds)


def jepa_predict(model, prepared, split: str, device=torch.device("cpu"),
                 version: str = "v2") -> np.ndarray:
    return _jepa_forward(model, prepared.pair_tensors(split), version, device)


def carry_forward_std(dyn_by_subject, queries) -> np.ndarray:
    """Carry-forward in standardized units: last observed value at or before
    visit i; a never-observed feature gets 0, i.e. the training mean."""
    D = dyn_by_subject[0].shape[1]
    out = np.zeros((len(queries), D), dtype=np.float32)
    for q, pair in enumerate(queries):
        hist = dyn_by_subject[pair["subject"]][: pair["i"] + 1]
        for d in range(D):
            obs = hist[np.isfinite(hist[:, d]), d]
            out[q, d] = obs[-1] if len(obs) else 0.0
    return out


def forecast(config: dict, model, prepared, queries: list[dict],
             device=torch.device("cpu"), dyn_by_subject=None) -> np.ndarray:
    """Standardized forecasts for (subject, i, dt) queries from any model kind.

    Only the input history up to visit i is read — never a target visit — so
    the same function serves test pairs, forecasts at horizons with no
    observed follow-up (conversion analysis, 48-month extrapolation) and
    degraded input histories (missingness stress test)."""
    kind = config["kind"]
    lookup = dyn_by_subject if dyn_by_subject is not None else prepared.dyn_by_subject
    if kind == "carry":
        return carry_forward_std(lookup, queries)
    if kind in ("ridge", "hgb"):
        X = ridge_design_matrix(prepared, queries, dyn_lookup=lambda s: lookup[s])
        return model.predict(X)
    if kind == "gru":
        seq, lengths = gru_collate(prepared.study, queries, lambda s: lookup[s], prepared.x_context)
        model.eval()
        with torch.no_grad():
            return model(torch.as_tensor(seq, device=device),
                         torch.as_tensor(lengths, device=device)).cpu().numpy()
    if kind == "jepa":
        t = prepared.input_tensors(queries, dyn_by_subject=dyn_by_subject)
        return _jepa_forward(model, t, config["version"], device)
    raise ValueError(f"unknown model kind {kind!r}")


# ------------------------------------------------------------------ metrics
def evaluate_predictions(prepared, split: str, pred_std: np.ndarray) -> dict:
    """Accuracy, rule-violation and subgroup metrics for one model's
    standardized predictions on one split's prediction pairs.

    Violation rates are reported twice: for the predictions, and for the
    OBSERVED future values of the same pairs (`observed_*`). The rule
    envelopes are training-data quantiles, so a prediction rate far below the
    observed rate indicates over-smoothed trajectories, not better biology."""
    study = prepared.study
    pairs = prepared.pairs[split]

    truth_std, cur_std = prepared.truth_std(split)
    dt = np.array([p["dt"] for p in pairs])
    hs = np.array([p["h"] for p in pairs])
    sev = prepared.severity[split]
    obs_mask = np.isfinite(truth_std)

    metrics: dict = {}

    # ---- MAE in raw units per feature per nominal horizon ----
    pred_raw = prepared.dyn_std.inverse(pred_std)
    truth_raw = prepared.dyn_std.inverse(truth_std)
    for f_idx, spec in enumerate(study.dyn_specs):
        for h in sorted(set(hs.tolist())):
            m = obs_mask[:, f_idx] & (hs == h)
            if m.sum() == 0:
                continue
            metrics[f"mae_{spec.name}_h{h:g}"] = float(np.abs(pred_raw[m, f_idx] - truth_raw[m, f_idx]).mean())
            metrics[f"n_pairs_{spec.name}_h{h:g}"] = int(m.sum())

    # ---- biological violation rates: predictions and observed reference ----
    metrics.update(prepared.engine.violation_rates(cur_std, pred_std, dt, sev))
    metrics.update({f"observed_{k}": v for k, v in
                    prepared.engine.violation_rates(cur_std, truth_std, dt, sev).items()})

    # ---- amyloid-subgroup calibration (ADNI) ----
    if study.name == "adni" and study.dyn_group_idx.get("C"):
        pet_status = np.array([s.patho_status[0] if len(s.patho_status) else -1
                               for s in study.subjects])
        pred_rate = prepared.engine.observed_decline_rates(cur_std, pred_std, dt)
        true_rate = prepared.engine.observed_decline_rates(cur_std, truth_std, dt)
        for name, arr in [("pred", pred_rate), ("obs", true_rate)]:
            plus_m, minus_m = [], []
            for p_i, pair in enumerate(pairs):
                st = pet_status[pair["subject"]]
                if st == 1:
                    plus_m.append(arr[p_i])
                elif st == 0:
                    minus_m.append(arr[p_i])
            plus_m = [v for v in plus_m if np.isfinite(v)]
            minus_m = [v for v in minus_m if np.isfinite(v)]
            if plus_m:
                metrics[f"decline_rate_{name}_Apos"] = float(np.mean(plus_m))
            if minus_m:
                metrics[f"decline_rate_{name}_Aneg"] = float(np.mean(minus_m))
        for name in ("pred", "obs"):
            pos, neg = f"decline_rate_{name}_Apos", f"decline_rate_{name}_Aneg"
            if pos in metrics and neg in metrics:
                metrics[f"subgroup_gap_{name}"] = metrics[pos] - metrics[neg]
    return metrics


# ------------------------------------------------- prospective conversion
def conversion_feature(study) -> int | None:
    return next((d for d, s in enumerate(study.dyn_specs) if s.name in CONVERSION_FEATURES), None)


def conversion_queries(prepared, subject_ids, window: float = CONVERSION_WINDOW_YEARS):
    """Pre-specified prospective conversion design.

    Index = each subject's baseline visit; forecast horizon = `window` years.
    Eligible: not demented at baseline, known event timing, conversion
    feature observed at baseline. Label 1 = converted within (0, window];
    label 0 = event-free through the window (never converted with follow-up
    >= window, or converted only after it); subjects censored before the
    window closes are excluded. The forecast needs no follow-up visit, so
    eligibility never depends on a visit near the horizon being observed.
    Returns (queries, labels)."""
    study = prepared.study
    conv_f = conversion_feature(study)
    queries, labels = [], []
    if conv_f is None:
        return queries, np.zeros(0, dtype=int)
    for s_idx in sorted(subject_ids):
        s = study.subjects[s_idx]
        tur = s.time_under_risk
        if s.dementia_at_baseline or tur is None or not np.isfinite(tur):
            continue
        if not np.isfinite(prepared.dyn_by_subject[s_idx][0, conv_f]):
            continue
        if s.converts and tur <= 0:
            continue  # event at the index visit: not a prospective case
        if s.converts and tur <= window:
            y = 1
        elif tur >= window:
            y = 0
        else:
            continue  # censored before the window closes
        queries.append({"subject": s_idx, "i": 0, "dt": float(window), "h": float(window)})
        labels.append(y)
    return queries, np.array(labels, dtype=int)


def conversion_scores(prepared, queries, pred_std: np.ndarray) -> np.ndarray:
    """Predicted worsening of the conversion feature from the observed
    baseline value (standardized units; higher = more risk)."""
    study = prepared.study
    conv_f = conversion_feature(study)
    cur = np.array([prepared.dyn_by_subject[q["subject"]][q["i"], conv_f] for q in queries])
    delta = pred_std[:, conv_f] - cur
    return delta if study.dyn_specs[conv_f].higher_is_worse else -delta


def conversion_analysis(config, model, prepared, subject_ids, device=torch.device("cpu"),
                        window: float = CONVERSION_WINDOW_YEARS) -> dict:
    """Scores, labels and subject ids for the prospective conversion design,
    plus the AUC when both classes are present."""
    from sklearn.metrics import roc_auc_score
    queries, labels = conversion_queries(prepared, subject_ids, window)
    out = {"sids": [prepared.study.subjects[q["subject"]].sid for q in queries],
           "labels": labels, "scores": np.zeros(0), "window_years": window,
           "n": len(queries), "n_cases": int(labels.sum()) if len(labels) else 0}
    if not queries:
        return out
    scores = conversion_scores(prepared, queries,
                               forecast(config, model, prepared, queries, device))
    out["scores"] = scores
    if 0 < labels.sum() < len(labels):
        out["auc"] = float(roc_auc_score(labels, scores))
    return out


# -------------------------------------------------- statistical inference
def paired_bootstrap(x, y, n_boot: int = 10000, seed: int = 0) -> dict:
    """Item-level paired bootstrap for mean(x) - mean(y). Treats items as
    independent: do NOT use it when one patient contributes several items —
    use cluster_paired_bootstrap instead."""
    rng = np.random.RandomState(seed)
    d = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return {"mean_diff": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan}
    boots = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(n_boot)])
    ci = np.percentile(boots, [2.5, 97.5])
    p = 2.0 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"mean_diff": float(d.mean()), "ci_low": float(ci[0]), "ci_high": float(ci[1]),
            "p": float(max(p, 1.0 / n_boot))}


def cluster_paired_bootstrap(x, y, clusters, n_boot: int = 10000, seed: int = 0) -> dict:
    """Paired comparison with PATIENTS as the resampling unit.

    `x`, `y` are aligned per-item losses and `clusters` the per-item patient
    key. Each patient is first reduced to its mean paired difference, so the
    estimand is the patient-weighted mean difference (every patient counts
    once). CI: percentile bootstrap over patients. p: patient-level sign-flip
    permutation test (H0: each patient's difference is symmetric about 0).
    Each patient must appear under exactly ONE cluster key — never one key
    per seed, which would count the same person several times."""
    rng = np.random.RandomState(seed)
    d = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    cl = np.asarray(clusters)
    ok = np.isfinite(d)
    d, cl = d[ok], cl[ok]
    uniq, inv = np.unique(cl, return_inverse=True)
    if len(uniq) < 5:
        return {"mean_diff": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan,
                "n_clusters": int(len(uniq))}
    per_patient = np.bincount(inv, weights=d) / np.bincount(inv)
    boots = np.array([per_patient[rng.randint(0, len(per_patient), len(per_patient))].mean()
                      for _ in range(n_boot)])
    ci = np.percentile(boots, [2.5, 97.5])
    flips = rng.choice([-1.0, 1.0], size=(n_boot, len(per_patient)))
    nulls = (per_patient[None, :] * flips).mean(axis=1)
    p = float(np.mean(np.abs(nulls) >= abs(per_patient.mean())))
    return {"mean_diff": float(per_patient.mean()), "ci_low": float(ci[0]), "ci_high": float(ci[1]),
            "p": float(max(p, 1.0 / n_boot)), "n_clusters": int(len(uniq))}


def paired_auc_bootstrap(scores_a, labels, scores_b, n_boot: int = 10000, seed: int = 0) -> dict:
    """Paired bootstrap for AUC(a) - AUC(b); items must be one per subject,
    aligned across a and b."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    a = np.asarray(scores_a)
    b = np.asarray(scores_b)
    auc_a, auc_b = roc_auc_score(labels, a), roc_auc_score(labels, b)
    n = len(labels)
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if labels[idx].sum() in (0, idx.size):
            continue
        diffs.append(roc_auc_score(labels[idx], a[idx]) - roc_auc_score(labels[idx], b[idx]))
    diffs = np.array(diffs)
    if len(diffs) == 0:  # every resample degenerate (all-one-class labels)
        return {"auc_a": float(auc_a), "auc_b": float(auc_b), "n": int(n),
                "mean_diff": float(auc_a - auc_b), "ci_low": np.nan, "ci_high": np.nan,
                "p": np.nan}
    ci = np.percentile(diffs, [2.5, 97.5])
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"auc_a": float(auc_a), "auc_b": float(auc_b), "n": int(n),
            "mean_diff": float(auc_a - auc_b),
            "ci_low": float(ci[0]), "ci_high": float(ci[1]), "p": float(max(p, 1.0 / n_boot))}


# ------------------------------------------------------ missingness stress
def degraded_dyn_lookup(prepared, frac: float, seed: int = 0) -> list[np.ndarray]:
    """Additional missingness applied to INPUT observations only (measurement
    dropped entirely: value -> NaN so mask bits flip too). Truth is untouched.
    With a fixed seed the dropped sets are nested across `frac` values."""
    rng = np.random.RandomState(seed)
    out = []
    for dyn in prepared.dyn_by_subject:
        d = dyn.copy()
        drop = np.isfinite(d) & (rng.rand(*d.shape) < frac)
        d[drop] = np.nan
        out.append(d)
    return out


# ---------------------------------------------------------------- rollout
def rollout_metrics(prepared, split: str, step_preds: list[np.ndarray]) -> dict:
    """Identical scoring for every model's multi-step rollout.

    MAE at step k uses the split's pairs whose target lies ~k years ahead.
    Violations are computed on the cells observed at the input visit, in two
    ways for every model: 'stepwise' (step k vs step k-1, the observed input
    being step 0; dt = 1 y) and 'cumulative' (step k vs the observed input;
    dt = k years)."""
    study = prepared.study
    pairs = prepared.pairs[split]
    truth_std, cur_std = prepared.truth_std(split)
    dt = np.array([p["dt"] for p in pairs])
    sev = prepared.severity[split]
    n = len(pairs)
    observed_in = np.isfinite(cur_std)
    truth_raw = prepared.dyn_std.inverse(truth_std)
    res: dict = {}
    prev = cur_std.copy()
    for s, pred in enumerate(step_preds):
        obs = np.isfinite(truth_std) & (np.abs(dt - (s + 1)) < 0.26)[:, None]
        pred_raw = prepared.dyn_std.inverse(pred)
        for f_idx, spec in enumerate(study.dyn_specs):
            m = obs[:, f_idx]
            if m.any():
                res[f"rollout_mae_{spec.name}_step{s + 1}"] = float(
                    np.abs(pred_raw[m, f_idx] - truth_raw[m, f_idx]).mean())
        pred_m = np.where(observed_in, pred, np.nan)
        stepwise = prepared.engine.violation_rates(prev, pred_m, np.full(n, 1.0), sev)
        cumulative = prepared.engine.violation_rates(cur_std, pred_m, np.full(n, s + 1.0), sev)
        for k, v in stepwise.items():
            res.setdefault(f"rollout_{k}_stepwise", {})[f"step{s + 1}"] = v
        for k, v in cumulative.items():
            res.setdefault(f"rollout_{k}_cumulative", {})[f"step{s + 1}"] = v
        prev = pred_m
    return res


def jepa_rollout(model, prepared, split: str, device=torch.device("cpu"),
                 steps: int = 3) -> dict:
    """Latent AUTOREGRESSIVE rollout (JEPA v2/v3): start from each pair's
    input visit, predict +1 y, feed the predicted latent back, repeat. Tests
    the latent dynamics — NOT a re-encoded patient-history rollout."""
    if not hasattr(model, "rollout"):
        raise ValueError("latent rollout needs a history-aware (v2/v3) JEPA model")
    t = prepared.pair_tensors(split)
    dt_feats = torch.as_tensor(fourier_time(np.ones(steps)), dtype=torch.float32, device=device)
    model.eval()
    chunks = []
    with torch.no_grad():
        for lo in range(0, t["x_i"].shape[0], 1024):
            sl = slice(lo, lo + 1024)
            rolled = model.rollout(t["seq_i"][sl].to(device), t["len_i"][sl].to(device),
                                   t["ctx"][sl].to(device), dt_feats,
                                   dyn_i=t["dyn_i"][sl].to(device))
            chunks.append([r.cpu().numpy() for r in rolled])
    step_preds = [np.concatenate([chunk[s] for chunk in chunks]) for s in range(steps)]
    return rollout_metrics(prepared, split, step_preds)


def gru_rollout(model, prepared, split: str, device=torch.device("cpu"),
                steps: int = 3) -> dict:
    """Multi-step VALUE rollout for the supervised GRU: append the predicted
    visit (mask=1, gap 1 y, horizon 1 y) to the history and predict again."""
    study = prepared.study
    pairs = prepared.pairs[split]
    D = len(study.dyn_specs)
    # One-year horizon for each rollout step, independently of test pair dt.
    rollout_pairs = [{**p, "dt": 1.0} for p in pairs]
    seqs, lengths = gru_collate(study, rollout_pairs, lambda s: prepared.dyn_by_subject[s],
                                prepared.x_context)
    seqs = np.concatenate([seqs, np.zeros((len(pairs), steps, seqs.shape[2]), dtype=np.float32)], axis=1)
    lens = lengths.copy()
    ctx_part = np.stack([prepared.x_context[p["subject"]] for p in pairs]).astype(np.float32)
    year_features = fourier_time(np.ones(len(pairs)))
    model.eval()
    step_preds = []
    with torch.no_grad():
        for _ in range(steps):
            pred = model(torch.as_tensor(seqs, device=device),
                         torch.as_tensor(lens, device=device)).cpu().numpy()  # [B, D] standardized
            step_preds.append(pred)
            step_vec = np.concatenate([pred, np.ones((len(pairs), D), dtype=np.float32),
                                       year_features, year_features, ctx_part], axis=1)
            seqs[np.arange(len(pairs)), lens, :] = step_vec
            lens = lens + 1
    return rollout_metrics(prepared, split, step_preds)
