"""Clinical tests beyond MAE (ADNI, one held-out fold — DESCRIPTIVE ONLY).

Models are retrained live on fold 0 of the subject-level 5-fold split.
  T1 Decliner detection : flag pairs whose MMSE drops >= 1 point / CDR-SB
                          rises >= 0.5 (pair level, not one outcome per patient).
  T2 Clinical tolerance : fraction of predictions within ±1 MMSE point and
                          ±0.5 CDR-SB of the truth.
  T3 Early prognosis    : 36-month prediction using ONLY the baseline visit.
  T4 Unseen horizon     : extrapolation to 48 months (never a training horizon).
  T5 Converter cohort   : pairs from patients who convert during follow-up.

Conversion discrimination is NOT repeated here: it is part of the main
cross-validated analysis (run_experiments), with patient-level inference.
Single-fold numbers carry no confidence intervals and support no claim of
superiority.

Usage:  PYTHONPATH=src python -m biojepa.hard_tests2 [--seed 0]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .data.dataset import make_pair_sets, subject_kfold
from .evaluate import forecast
from .pipeline import prepare
from .run_experiments import MODEL_CONFIGS, PRIMARY_MODEL, fit_model, load_study, write_json

MODELS = [PRIMARY_MODEL, "jepa_v2", "gru", "hgb", "ridge", "carry_forward"]


def per_pair_raw(prepared, pairs, pred_std):
    """Raw-unit predictions, truths and current values for `pairs`."""
    pred_raw = prepared.dyn_std.inverse(pred_std)
    truth_raw = prepared.dyn_std.inverse(np.stack(
        [prepared.dyn_by_subject[p["subject"]][p["j"]] for p in pairs]))
    cur_raw = prepared.dyn_std.inverse(np.stack(
        [prepared.dyn_by_subject[p["subject"]][p["i"]] for p in pairs]))
    dt = np.array([p["dt"] for p in pairs])
    subj = np.array([p["subject"] for p in pairs])
    return pred_raw, truth_raw, cur_raw, dt, subj


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
        "n_pairs": int(obs.sum()),
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/adni/hard_tests2.json")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--study", default="adni", choices=["adni", "synthetic"],
                    help="synthetic = pipeline check only")
    args = ap.parse_args(argv)
    device = torch.device("cpu")

    study = load_study(args.study)
    names = {"adni": ("MMSE", "CDRSB"), "synthetic": ("MMSE_like", "CDR_like")}[args.study]
    tr, va, te = subject_kfold(study, k=5, seed=args.seed)[0]
    prepared = prepare(study, splits=(tr, va, te))
    test_pairs = prepared.pairs["test"]
    # T4: unseen 48-month horizon (never a training horizon); inputs only
    p48 = [p for p in make_pair_sets(study, horizons=(4.0,)) if p["subject"] in te]
    feat_idx = {s.name: d for d, s in enumerate(study.dyn_specs)}
    report: dict = {}

    for name in MODELS:
        cfg = MODEL_CONFIGS[name]
        model = fit_model(cfg, prepared, args.seed, device, args.max_epochs)
        r: dict = {}
        pred_std = forecast(cfg, model, prepared, test_pairs, device)
        pred_raw, truth_raw, cur_raw, dt, subj = per_pair_raw(prepared, test_pairs, pred_std)

        for k, fname in enumerate(names):
            label = ("MMSE", "CDRSB")[k]  # decline direction / tolerance per scale type
            f_idx = feat_idx[fname]
            r[f"T1_decline_{fname}_all"] = detect_decliners(pred_raw, truth_raw, cur_raw, dt, f_idx, label)
            r[f"T1_decline_{fname}_36m"] = detect_decliners(pred_raw, truth_raw, cur_raw, dt, f_idx,
                                                            label, horizon=3.0)
            r[f"T2_tolerance_{fname}"] = tolerance_acc(pred_raw, truth_raw, dt, f_idx, label)
            base = np.array([p["i"] == 0 and p["h"] == 3.0 for p in test_pairs])
            m = base & np.isfinite(truth_raw[:, f_idx]) & np.isfinite(cur_raw[:, f_idx])
            if m.sum() > 10:
                r[f"T3_baseline_only_{fname}_36m_MAE"] = round(
                    float(np.abs(pred_raw[m, f_idx] - truth_raw[m, f_idx]).mean()), 3)
                r[f"T3_baseline_only_{fname}_36m_n"] = int(m.sum())
            conv_mask = np.array([study.subjects[s].converts for s in subj])
            m = conv_mask & np.isfinite(truth_raw[:, f_idx]) & np.isfinite(cur_raw[:, f_idx])
            if m.sum() > 10:
                r[f"T5_converters_{fname}_MAE"] = round(
                    float(np.abs(pred_raw[m, f_idx] - truth_raw[m, f_idx]).mean()), 3)
                det = detect_decliners(pred_raw[conv_mask], truth_raw[conv_mask],
                                       cur_raw[conv_mask], dt[conv_mask], f_idx, label)
                if det:
                    r[f"T5_converters_{fname}_decline_recall"] = det["recall_pct"]

        if len(p48) > 5:
            pred48 = forecast(cfg, model, prepared, p48, device)
            p48r, t48r, c48r, _, _ = per_pair_raw(prepared, p48, pred48)
            row = {"n_pairs": len(p48)}
            for k, fname in enumerate(names):
                f_idx = feat_idx[fname]
                tol = (1.0, 0.5)[k]
                m = np.isfinite(t48r[:, f_idx]) & np.isfinite(c48r[:, f_idx])
                if m.sum() > 5:
                    err = np.abs(p48r[m, f_idx] - t48r[m, f_idx])
                    row[f"{fname}_MAE_48m"] = round(float(err.mean()), 3)
                    row[f"{fname}_within±{tol:g}_pct"] = round(100 * float((err <= tol).mean()), 1)
            r["T4_unseen_48m"] = row

        report[name] = r
        print(f"[{name}] done", flush=True)

    report["_meta"] = {"fold_seed": args.seed, "study": args.study,
                       "n_test_subjects": len(te), "n_test_pairs": len(test_pairs),
                       "n_test48_pairs": len(p48), "primary_model": PRIMARY_MODEL,
                       "note": "single held-out fold; descriptive only, no inference"}
    write_json(args.out, report)
    print(json.dumps(report, indent=1, default=str))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
