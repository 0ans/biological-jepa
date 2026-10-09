"""Evaluation: prediction accuracy, conversion AUC, biological-violation rates,
amyloid-subgroup calibration, paired-bootstrap significance, missingness
stress tests, and multi-step rollout evaluation."""
from __future__ import annotations

import numpy as np
import torch

from .model.baselines import gru_collate
from .utils import fourier_time


# ------------------------------------------------------------------ predict
def jepa_predict(model, prepared, split: str, device=torch.device("cpu"),
                 version: str = "v2") -> np.ndarray:
    t = prepared.pair_tensors(split, device)
    model.eval()
    preds = []
    with torch.no_grad():
        for lo in range(0, t["x_i"].shape[0], 1024):
            if version in ("v2", "v3"):
                p = model.predict_dyn(t["seq_i"][lo:lo + 1024], t["len_i"][lo:lo + 1024],
                                      t["ctx"][lo:lo + 1024], t["dt_feat"][lo:lo + 1024],
                                      dyn_i=t["dyn_i"][lo:lo + 1024])
            else:
                p = model.predict_dyn(t["x_i"][lo:lo + 1024], t["dt_feat"][lo:lo + 1024])
            preds.append(p.cpu().numpy())
    return np.concatenate(preds)


def gru_predict(model, prepared, split: str, device=torch.device("cpu")) -> np.ndarray:
    pairs = prepared.pairs[split]
    seq, lengths = gru_collate(prepared.study, pairs, lambda s: prepared.dyn_by_subject[s],
                               prepared.x_context)
    model.eval()
    with torch.no_grad():
        out = model(torch.as_tensor(seq, device=device),
                    torch.as_tensor(lengths, device=device)).cpu().numpy()
    return out


def ridge_predict(prepared, split: str, model) -> np.ndarray:
    from .model.baselines import ridge_design_matrix
    return model.predict(ridge_design_matrix(prepared, prepared.pairs[split]))


# ----------------------------------------------------------------- metrics
def evaluate_predictions(prepared, split: str, pred_std: np.ndarray,
                         return_conv: bool = False):
    """Metrics for one model's standardized predictions on one split."""
    study = prepared.study
    pairs = prepared.pairs[split]

    truth_std = np.stack([prepared.dyn_by_subject[p["subject"]][p["j"]] for p in pairs])
    cur_std = np.stack([prepared.dyn_by_subject[p["subject"]][p["i"]] for p in pairs])
    dt = np.array([p["dt"] for p in pairs])
    sev = prepared.severity[split]
    obs_mask = np.isfinite(truth_std)

    metrics: dict = {}

    # ---- MAE in raw units per feature per horizon ----
    pred_raw = prepared.dyn_std.inverse(pred_std)
    truth_raw = prepared.dyn_std.inverse(truth_std)
    for f_idx, spec in enumerate(study.dyn_specs):
        for h in (1.0, 2.0, 3.0):
            m = obs_mask[:, f_idx] & (np.abs(dt - h) < 0.26)
            if m.sum() == 0:
                continue
            metrics[f"mae_{spec.name}_h{int(h)}"] = float(np.abs(pred_raw[m, f_idx] - truth_raw[m, f_idx]).mean())

    # ---- biological violation rates on predicted trajectories ----
    metrics.update(prepared.engine.violation_rates(cur_std, pred_std, dt, sev))

    # ---- conversion AUC (subject level) ----
    conv_feature = next((d for d, s in enumerate(study.dyn_specs)
                         if s.name in ("CDRSB", "CDR", "CDR_like")), None)
    conv = None
    if conv_feature is not None:
        by_subj: dict[int, list[int]] = {}
        for p_i, pair in enumerate(pairs):
            by_subj.setdefault(pair["subject"], []).append(p_i)
        scores, labels = [], []
        for s_idx, p_is in by_subj.items():
            cand = [p_i for p_i in p_is if 1.5 <= pairs[p_i]["h"] <= 3.0] or \
                   [p_i for p_i in p_is if pairs[p_i]["h"] == max(pp["h"] for pp in [pairs[q] for q in p_is])]
            if not cand:
                continue
            p_i = cand[0]
            if not (np.isfinite(pred_std[p_i, conv_feature]) and np.isfinite(cur_std[p_i, conv_feature])):
                continue
            delta = pred_std[p_i, conv_feature] - cur_std[p_i, conv_feature]
            if not study.dyn_specs[conv_feature].higher_is_worse:
                delta = -delta
            scores.append(float(delta))
            labels.append(int(study.subjects[s_idx].converts))
        if 0 < sum(labels) < len(labels):
            from sklearn.metrics import roc_auc_score
            metrics["conversion_auc"] = float(roc_auc_score(labels, scores))
            metrics["n_auc_subjects"] = len(labels)
            if return_conv:
                conv = (np.array(scores), np.array(labels))

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
        if "decline_rate_pred_Apos" in metrics and "decline_rate_pred_Aneg" in metrics:
            metrics["subgroup_gap_pred"] = metrics["decline_rate_pred_Apos"] - metrics["decline_rate_pred_Aneg"]
            metrics["subgroup_gap_obs"] = metrics.get("decline_rate_obs_Apos", np.nan) - metrics.get("decline_rate_obs_Aneg", np.nan)
    if return_conv:
        return metrics, conv
    return metrics


# -------------------------------------------------- statistical significance
def paired_bootstrap(x, y, n_boot: int = 10000, seed: int = 0) -> dict:
    """Paired bootstrap for mean(x) - mean(y) (two-sided H0: no difference).
    x, y are aligned per-item losses (lower = better)."""
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
    """Paired bootstrap resampling PATIENTS, not visit-pairs.

    One patient contributes several (i, j) pairs; treating them as independent
    understates variance. Here a bootstrap draw samples whole patients and all
    of that patient's items move together. x, y are aligned per-item losses and
    `clusters` the per-item patient key."""
    rng = np.random.RandomState(seed)
    d = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    cl = np.asarray(clusters)
    ok = np.isfinite(d)
    d, cl = d[ok], cl[ok]
    uniq = np.unique(cl)
    if len(uniq) < 5:
        return {"mean_diff": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan, "n_clusters": int(len(uniq))}
    per_patient = np.array([d[cl == u].mean() for u in uniq])
    # CI: cluster bootstrap percentile of the statistic
    boots = np.array([per_patient[rng.randint(0, len(per_patient), len(per_patient))].mean()
                      for _ in range(n_boot)])
    ci = np.percentile(boots, [2.5, 97.5])
    # p-value: patient-level SIGN-FLIP permutation (H0: the paired difference is
    # symmetric around 0 — flipping a patient's sign is exchangeable under H0).
    # This avoids the degenerate centered-bootstrap p (always ~1.0).
    flips = rng.choice([-1.0, 1.0], size=(n_boot, len(per_patient)))
    nulls = (per_patient[None, :] * flips).mean(axis=1)
    p = float(np.mean(np.abs(nulls) >= abs(per_patient.mean())))
    p = max(p, 1.0 / n_boot)
    return {"mean_diff": float(per_patient.mean()), "ci_low": float(ci[0]), "ci_high": float(ci[1]),
            "p": float(max(p, 1.0 / n_boot)), "n_clusters": int(len(uniq))}


def paired_auc_bootstrap(scores_a, labels, scores_b, n_boot: int = 10000, seed: int = 0) -> dict:
    """Paired bootstrap for AUC(a) - AUC(b) over the same subjects."""
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
        return {"auc_a": float(auc_a), "auc_b": float(auc_b),
                "mean_diff": float(auc_a - auc_b), "ci_low": np.nan, "ci_high": np.nan,
                "p": np.nan}
    ci = np.percentile(diffs, [2.5, 97.5])
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"auc_a": float(auc_a), "auc_b": float(auc_b), "mean_diff": float(auc_a - auc_b),
            "ci_low": float(ci[0]), "ci_high": float(ci[1]), "p": float(max(p, 1.0 / n_boot))}


# ------------------------------------------------------ missingness stress
def degraded_dyn_lookup(prepared, frac: float, seed: int = 0) -> list[np.ndarray]:
    """Additional missingness applied to INPUT observations only (measurement
    dropped entirely: value -> NaN so mask bits flip too). Truth is untouched."""
    rng = np.random.RandomState(seed)
    out = []
    for dyn in prepared.dyn_by_subject:
        d = dyn.copy()
        drop = np.isfinite(d) & (rng.rand(*d.shape) < frac)
        d[drop] = np.nan
        out.append(d)
    return out


def jepa_predict_degraded(model, prepared, split: str, degraded: list[np.ndarray],
                          device=torch.device("cpu"), version: str = "v2") -> np.ndarray:
    """v2 JEPA prediction with degraded input histories (truth untouched)."""
    pairs = prepared.pairs[split]
    L = max(p["i"] + 1 for p in pairs)
    seq = np.zeros((len(pairs), L, prepared.step_dim), dtype=np.float32)
    lens = np.zeros(len(pairs), dtype=np.int64)
    dt_feat = fourier_time(np.array([p["dt"] for p in pairs]))
    for p_i, pair in enumerate(pairs):
        s = prepared.study.subjects[pair["subject"]]
        dyn = degraded[pair["subject"]]
        prev = None
        for t, vt in enumerate(s.times[: pair["i"] + 1]):
            gap = 0.0 if prev is None else vt - prev
            mask = np.isfinite(dyn[t]).astype(np.float32)
            vals = np.nan_to_num(dyn[t]).astype(np.float32)
            dtf = fourier_time(np.array([max(gap, 1e-3)]), 4)[0]
            seq[p_i, t] = np.concatenate([vals, mask, dtf])
            prev = vt
        lens[p_i] = pair["i"] + 1
    model.eval()
    with torch.no_grad():
        out = model.predict_dyn(torch.as_tensor(seq, device=device),
                                torch.as_tensor(lens, device=device),
                                torch.as_tensor(prepared.pair_tensors(split)["ctx"].numpy(), device=device),
                                torch.as_tensor(dt_feat, device=device)).cpu().numpy()
    return out


def gru_predict_degraded(model, prepared, split: str, degraded: list[np.ndarray],
                         device=torch.device("cpu")) -> np.ndarray:
    pairs = prepared.pairs[split]
    seq, lengths = gru_collate(prepared.study, pairs, lambda s: degraded[s], prepared.x_context)
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(seq, device=device),
                     torch.as_tensor(lengths, device=device)).cpu().numpy()


# ---------------------------------------------------------------- rollout
def jepa_rollout(model, prepared, split: str, device=torch.device("cpu"),
                 steps: int = 3) -> dict:
    """Latent AUTOREGRESSIVE rollout (JEPA v2): start from each test pair's
    input visit, predict +1 yr, feed the predicted latent back, repeat. This
    tests the latent dynamical rollout — NOT a re-encoded patient-history
    rollout. Biological violations at step k are measured step-to-step
    (prediction k vs prediction k-1), matching the sequential constraint the
    training penalty would see."""
    study = prepared.study
    pairs = prepared.pairs[split]
    t = prepared.pair_tensors(split, device)
    dt_feats = torch.as_tensor(fourier_time(np.ones(steps)), dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        outs = []
        for lo in range(0, t["x_i"].shape[0], 1024):
            rolled = model.rollout(t["seq_i"][lo:lo + 1024], t["len_i"][lo:lo + 1024],
                                   t["ctx"][lo:lo + 1024], dt_feats,
                                   dyn_i=t["dyn_i"][lo:lo + 1024])
            outs.append([r.cpu().numpy() for r in rolled])
    # outs[s] -> chunk -> concat over chunks
    step_preds = [np.concatenate([chunk[s] for chunk in outs]) for s in range(steps)]

    truth_std, cur_std = prepared.truth_std(split)
    dt = np.array([p["dt"] for p in pairs])
    sev = prepared.severity[split]
    res: dict = {}
    prev_std = cur_std.copy()          # step-to-step reference for violations
    for s in range(steps):
        obs = np.isfinite(truth_std) & (np.abs(dt - (s + 1)) < 0.26)[:, None]
        if obs.any():
            pred_raw = prepared.dyn_std.inverse(step_preds[s])
            truth_raw = prepared.dyn_std.inverse(truth_std)
            for f_idx, spec in enumerate(study.dyn_specs):
                m = obs[:, f_idx]
                if m.any():
                    res[f"rollout_mae_{spec.name}_step{s + 1}"] = float(
                        np.abs(pred_raw[m, f_idx] - truth_raw[m, f_idx]).mean())
        # biological violations STEP-TO-STEP: prediction k+1 vs prediction k
        v = prepared.engine.violation_rates(prev_std, step_preds[s],
                                            np.full(len(pairs), 1.0), sev)
        for k, val in v.items():
            res.setdefault(f"rollout_{k}", {})[f"step{s + 1}"] = val
        prev_std = step_preds[s].copy()
    return res


def gru_rollout(model, prepared, split: str, device=torch.device("cpu"),
                steps: int = 3) -> dict:
    """Multi-step VALUE rollout for the supervised GRU: append the predicted
    visit (mask=1) to the history and predict the next step."""
    study = prepared.study
    pairs = prepared.pairs[split]
    D = len(study.dyn_specs)
    truth_std, cur_std = prepared.truth_std(split)
    dt = np.array([p["dt"] for p in pairs])
    sev = prepared.severity[split]

    # One-year horizon for each rollout step, independently of test pair dt.
    rollout_pairs = [{**p, "dt": 1.0} for p in pairs]
    seqs, lengths = gru_collate(study, rollout_pairs, lambda s: prepared.dyn_by_subject[s],
                                prepared.x_context)
    seqs = np.concatenate([seqs, np.zeros((len(pairs), steps, seqs.shape[2]), dtype=np.float32)], axis=1)
    lens = lengths.copy()
    model.eval()
    step_preds = []
    with torch.no_grad():
        for s in range(steps):
            pred = model(torch.as_tensor(seqs, device=device),
                         torch.as_tensor(lens, device=device)).cpu().numpy()  # [B, D] standardized
            step_preds.append(pred)
            # append predicted visit as new step with mask=1, gap=1 year, real context
            ctx_part = np.stack([prepared.x_context[p["subject"]] for p in pairs]).astype(np.float32)
            year_features = fourier_time(np.ones(len(pairs)))
            step_vec = np.concatenate([pred, np.ones((len(pairs), D), dtype=np.float32),
                                       year_features, year_features, ctx_part], axis=1)
            seqs[np.arange(len(pairs)), lens, :] = step_vec
            lens = lens + 1

    res: dict = {}
    for s in range(steps):
        obs = np.isfinite(truth_std) & (np.abs(dt - (s + 1)) < 0.26)[:, None]
        if not obs.any():
            continue
        pred_raw = prepared.dyn_std.inverse(step_preds[s])
        truth_raw = prepared.dyn_std.inverse(truth_std)
        for f_idx, spec in enumerate(study.dyn_specs):
            m = obs[:, f_idx]
            if m.any():
                res[f"rollout_mae_{spec.name}_step{s + 1}"] = float(
                    np.abs(pred_raw[m, f_idx] - truth_raw[m, f_idx]).mean())
        v = prepared.engine.violation_rates(cur_std, step_preds[s], np.full(len(pairs), s + 1.0), sev)
        for k, val in v.items():
            res.setdefault(f"rollout_{k}", {})[f"step{s + 1}"] = val
    return res
