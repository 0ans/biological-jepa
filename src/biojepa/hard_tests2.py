"""Hard clinical tests (v2): beyond MAE — can the model detect WHO will decline?

Tests (run on one held-out 5-fold-CV fold of real ADNI, models retrained live):
  T1 Decliner detection     : does the model correctly flag patients whose
                              MMSE will drop >=1 point / CDRSB rise >=0.5?
                              accuracy + recall + precision per horizon.
  T2 Clinical tolerance     : fraction of predictions within +-1 and +-2 MMSE
                              points, and +-0.5 CDRSB of the truth.
  T3 Early prognosis        : 36-month prediction using ONLY the baseline visit.
  T4 Unseen-horizon         : extrapolation to 48 months (never seen in training;
                              the time-conditioned predictor must extrapolate).
  T5 Hard cohort            : converted patients only (fast progressors).
  T6 Conversion risk        : prospective AUC (baseline input, 36-month window, time-aware labels).

Usage:  PYTHONPATH=src python -m biojepa.hard_tests2 [--seed 0]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .data.adni import load_adni
from .data.dataset import make_pair_sets, subject_kfold
from .evaluate import evaluate_predictions, gru_predict, jepa_predict, ridge_predict
from .model.bio_rules import compute_severity
from .model.baselines import carry_forward_predict
from .run_experiments import MODEL_CONFIGS, fit_predict
from .pipeline import prepare


def predict_split(cfg, model, prepared, split, device):
    kind = cfg["kind"]
    if kind == "carry":
        pred_raw, _, _, _ = carry_forward_predict(prepared.study, prepared.pairs[split],
                                                  sorted({p["subject"] for p in prepared.pairs["train"]}))
        return prepared.dyn_std.transform(pred_raw)
    if kind in ("ridge", "hgb"):
        return ridge_predict(prepared, split, model)
    if kind == "gru":
        return gru_predict(model, prepared, split, device)
    return jepa_predict(model, prepared, split, device, version=cfg["version"])


def per_pair_raw(prepared, split, pred_std):
    pairs = prepared.pairs[split]
    pred_raw = prepared.dyn_std.inverse(pred_std)
    truth_raw = prepared.dyn_std.inverse(np.stack(
        [prepared.dyn_by_subject[p["subject"]][p["j"]] for p in pairs]))
    cur_raw = prepared.dyn_std.inverse(np.stack(
        [prepared.dyn_by_subject[p["subject"]][p["i"]] for p in pairs]))
    dt = np.array([p["dt"] for p in pairs])
    subj = np.array([p["subject"] for p in pairs])
    return pred_raw, truth_raw, cur_raw, dt, subj, pairs


def detect_decliners(pred_raw, truth_raw, cur_raw, dt, f_idx, name, horizon=None,
                     mmse_drop=1.0, cdrsb_rise=0.5):
    """Flag-future-decline accuracy/recall/precision for one clinical feature."""
    obs = np.isfinite(truth_raw[:, f_idx]) & np.isfinite(cur_raw[:, f_idx])
    if horizon is not None:
        obs &= np.abs(dt - horizon) < 0.26
    if obs.sum() == 0:
        return None
    if name == "MMSE":  # higher is better: decline = drop
        true_dec = (cur_raw[obs, f_idx] - truth_raw[obs, f_idx]) >= mmse_drop
        pred_dec = (cur_raw[obs, f_idx] - pred_raw[obs, f_idx]) >= mmse_drop
    else:  # CDRSB: higher is worse: decline = rise
        true_dec = (truth_raw[obs, f_idx] - cur_raw[obs, f_idx]) >= cdrsb_rise
        pred_dec = (pred_raw[obs, f_idx] - cur_raw[obs, f_idx]) >= cdrsb_rise
    tp = float((pred_dec & true_dec).sum())
    return {
        "n": int(obs.sum()),
        "true_decliners_pct": round(100 * float(true_dec.mean()), 1),
        "accuracy_pct": round(100 * float((pred_dec == true_dec).mean()), 1),
        "recall_pct": round(100 * tp / max(true_dec.sum(), 1), 1),
        "precision_pct": round(100 * tp / max(pred_dec.sum(), 1), 1),
    }


def tolerance_acc(pred_raw, truth_raw, dt, f_idx, name, horizons=(1.0, 2.0, 3.0)):
    out = {}
    tol = 1.0 if name == "MMSE" else 0.5
    for h in horizons:
        m = np.isfinite(truth_raw[:, f_idx]) & (np.abs(dt - h) < 0.26)
        if m.sum() == 0:
            continue
        err = np.abs(pred_raw[m, f_idx] - truth_raw[m, f_idx])
        out[f"h{int(h)}_within±{tol:g}"] = round(100 * float((err <= tol).mean()), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/adni/hard_tests2.json")
    args = ap.parse_args()
    device = torch.device("cpu")

    study = load_adni()
    tr, va, te = subject_kfold(study, k=5, seed=args.seed)[0]
    prepared = prepare(study, splits=(tr, va, te))
    print(f"fold: {len(prepared.pairs['train'])} train / {len(prepared.pairs['test'])} test pairs")

    # T4 setup: unseen 48-month horizon (never in training)
    extra = make_pair_sets(study, horizons=(4.0,))
    p48 = [p for p in extra if p["subject"] in te]
    prepared.pairs["test48"] = p48
    def _sev48(p):
        s = study.subjects[p["subject"]]
        ps = prepared.patho_std.transform(s.patho) if prepared.patho_std is not None and len(s.patho) else s.patho
        return compute_severity(study, prepared.dyn_by_subject[p["subject"]][p["i"]], ps, s.patho_status)
    prepared.severity["test48"] = np.stack([_sev48(p) for p in p48]) if p48 else np.zeros((0, 3))

    feat_idx = {s.name: d for d, s in enumerate(study.dyn_specs)}
    models = ["jepa_v2_bio_strong", "jepa_v2", "gru", "hgb", "ridge", "carry_forward"]
    report: dict = {}

    for name in models:
        cfg = MODEL_CONFIGS[name]
        _, model = fit_predict(cfg, prepared, args.seed, device)
        r: dict = {}

        pred_std = predict_split(cfg, model, prepared, "test", device)
        metrics, conv = evaluate_predictions(prepared, "test", pred_std, return_conv=True)
        pred_raw, truth_raw, cur_raw, dt, subj, pairs = per_pair_raw(prepared, "test", pred_std)

        # T1: decliner detection
        for fname in ("MMSE", "CDRSB"):
            r[f"T1_decline_{fname}_all"] = detect_decliners(
                pred_raw, truth_raw, cur_raw, dt, feat_idx[fname], fname)
            r[f"T1_decline_{fname}_36m"] = detect_decliners(
                pred_raw, truth_raw, cur_raw, dt, feat_idx[fname], fname, horizon=3.0)
        # T2: clinical tolerance
        for fname in ("MMSE", "CDRSB"):
            r[f"T2_tolerance_{fname}"] = tolerance_acc(
                pred_raw, truth_raw, dt, feat_idx[fname], fname)
        # T3: early prognosis — baseline visit only, predict 36 months
        base_mask = np.array([p["i"] == 0 and p["h"] == 3.0 for p in pairs])
        for fname in ("MMSE", "CDRSB"):
            f_idx = feat_idx[fname]
            m = base_mask & np.isfinite(truth_raw[:, f_idx]) & np.isfinite(cur_raw[:, f_idx])
            if m.sum() > 10:
                r[f"T3_baseline_only_{fname}_36m_MAE"] = round(
                    float(np.abs(pred_raw[m, f_idx] - truth_raw[m, f_idx]).mean()), 3)
                r[f"T3_baseline_only_{fname}_36m_n"] = int(m.sum())
        # T5: hard cohort — converted patients only (fast progressors)
        conv_mask = np.array([study.subjects[s].converts for s in subj])
        for fname in ("MMSE", "CDRSB"):
            f_idx = feat_idx[fname]
            m = conv_mask & np.isfinite(truth_raw[:, f_idx]) & np.isfinite(cur_raw[:, f_idx])
            if m.sum() > 10:
                r[f"T5_converters_{fname}_MAE"] = round(
                    float(np.abs(pred_raw[m, f_idx] - truth_raw[m, f_idx]).mean()), 3)
                det = detect_decliners(pred_raw[conv_mask], truth_raw[conv_mask],
                                       cur_raw[conv_mask], dt[conv_mask], f_idx, fname)
                if det:
                    r[f"T5_converters_{fname}_decline_recall"] = det["recall_pct"]
        # T6: PROSPECTIVE conversion risk — unified design: baseline visit only,
        # fixed 36-month window, time-aware labels (converted AND time-under-risk
        # <= 3 y = case; not converted AND followed >= 3 y = control; otherwise
        # excluded). Threshold-based accuracy on test scores is NOT reported
        # (optimistically biased); threshold-free AUC is the metric.
        if study.name == "adni":
            from sklearn.metrics import roc_auc_score
            conv_f = next((d for d, s in enumerate(study.dyn_specs) if s.name == "CDRSB"), None)
            rows = []
            for p_i, pair in enumerate(pairs):
                if pair["i"] != 0 or pair["h"] != 3.0:
                    continue
                s = study.subjects[pair["subject"]]
                tur = s.time_under_risk
                if tur is None or not np.isfinite(tur):
                    continue
                if s.converts and tur <= 3.0:
                    y = 1
                elif (not s.converts) and tur >= 3.0:
                    y = 0
                else:
                    continue  # converted after the window, or follow-up < 3 y
                if not (np.isfinite(cur_raw[p_i, conv_f]) and np.isfinite(truth_raw[p_i, conv_f])):
                    continue
                delta = pred_raw[p_i, conv_f] - cur_raw[p_i, conv_f]
                rows.append((float(delta), y))
            if rows and 0 < sum(y for _, y in rows) < len(rows):
                r["T6_conversion_AUC_prospective"] = round(
                    float(roc_auc_score([y for _, y in rows], [x for x, _ in rows])), 3)
                r["T6_conversion_n_prospective"] = len(rows)
                r["T6_design"] = "baseline input, 36-month window, time-under-risk labels"
        elif conv is not None:
            from sklearn.metrics import roc_auc_score
            scores, labels = conv
            r["T6_conversion_AUC"] = round(float(roc_auc_score(labels, scores)), 3)
            r["T6_conversion_n"] = int(len(labels))

        # T4: 48-month extrapolation with the SAME trained model
        if len(p48) > 5:
            pred48_std = predict_split(cfg, model, prepared, "test48", device)
            p48r, t48r, c48r, dt48, _, _ = per_pair_raw(prepared, "test48", pred48_std)
            row = {"n": len(p48)}
            for fname in ("MMSE", "CDRSB"):
                f_idx = feat_idx[fname]
                m = np.isfinite(t48r[:, f_idx]) & np.isfinite(c48r[:, f_idx])
                if m.sum() > 5:
                    row[f"{fname}_MAE_48m"] = round(
                        float(np.abs(p48r[m, f_idx] - t48r[m, f_idx]).mean()), 3)
                    row[f"{fname}_within±1pt_pct"] = round(
                        100 * float((np.abs(p48r[m, f_idx] - t48r[m, f_idx]) <= 1.0).mean()), 1)
            r["T4_unseen_48m"] = row

        report[name] = r
        print(f"[{name}] done", flush=True)

    report["_meta"] = {"fold_seed": args.seed, "n_test48_pairs": len(p48),
                       "note": "single held-out fold; complement to the 15-run aggregate"}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=1))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
