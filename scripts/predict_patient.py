#!/usr/bin/env python
"""Research demo: forecast a held-out ADNI patient's MMSE / CDR-SB at 12, 24
and 36 months from the baseline visit.

Trains the pre-specified primary model (run_experiments.PRIMARY_MODEL — the
same configuration evaluated in the main analysis) on fold 0 of the
subject-level 5-fold split, then forecasts held-out patients. Forecasts are
point estimates WITHOUT uncertainty intervals. Research software only: not for
diagnosis, prognosis or any clinical decision.

  python scripts/predict_patient.py                 # demo: 3 held-out patients
  python scripts/predict_patient.py --sid ADNI_77   # a specific held-out patient
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from biojepa.data.adni import load_adni  # noqa: E402
from biojepa.data.dataset import subject_kfold  # noqa: E402
from biojepa.evaluate import forecast  # noqa: E402
from biojepa.pipeline import prepare  # noqa: E402
from biojepa.run_experiments import MODEL_CONFIGS, PRIMARY_MODEL, fit_model  # noqa: E402


def _fmt(x, nd):
    return f"{x:.{nd}f}" if np.isfinite(x) else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", default=None, help="e.g. ADNI_77 (a held-out test patient)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    study = load_adni()
    tr, va, te = subject_kfold(study, k=5, seed=args.seed)[0]
    p = prepare(study, splits=(tr, va, te))
    cfg = MODEL_CONFIGS[PRIMARY_MODEL]
    print(f"Training {PRIMARY_MODEL} on fold 0 (CPU)...", flush=True)
    model = fit_model(cfg, p, args.seed, torch.device("cpu"))
    F = {s.name: d for d, s in enumerate(study.dyn_specs)}

    test_sids = sorted(study.subjects[k].sid for k in te)
    if args.sid:
        if args.sid not in test_sids:
            print(f"Not in this test fold. Examples: {', '.join(test_sids[:10])} ...")
            return
        targets = [args.sid]
    else:
        targets = test_sids[:3]

    for sid in targets:
        s_idx = next(k for k, s in enumerate(study.subjects) if s.sid == sid)
        s = study.subjects[s_idx]
        queries = [{"subject": s_idx, "i": 0, "dt": h, "h": h} for h in (1.0, 2.0, 3.0)]
        pred_raw = p.dyn_std.inverse(forecast(cfg, model, p, queries))
        mm0, cd0 = s.dyn[0, F["MMSE"]], s.dyn[0, F["CDRSB"]]
        print(f"\n{'=' * 62}\n{sid} | baseline dx {s.group} | age {_fmt(s.static[0], 0)} | "
              f"converted during follow-up: {'yes' if s.converts else 'no'}")
        print(f"  Observed baseline: MMSE {_fmt(mm0, 0)} | CDR-SB {_fmt(cd0, 1)}")
        print(f"  {'Horizon':>9} | {'MMSE forecast':>13} | {'CDR-SB forecast':>15} | {'observed MMSE':>13}")
        for q, (h, lbl) in enumerate([(1.0, "12 months"), (2.0, "24 months"), (3.0, "36 months")]):
            target = np.abs(s.times - h)
            j = int(np.argmin(target))
            obs = s.dyn[j, F["MMSE"]] if target[j] <= 0.25 and j > 0 else np.nan
            print(f"  {lbl:>9} | {_fmt(pred_raw[q, F['MMSE']], 1):>13} | "
                  f"{_fmt(pred_raw[q, F['CDRSB']], 2):>15} | {_fmt(obs, 0):>13}")
    print("\nPoint forecasts without uncertainty intervals; research use only.")


if __name__ == "__main__":
    main()
