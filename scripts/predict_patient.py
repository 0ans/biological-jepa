#!/usr/bin/env python
"""Simplest entry point: predict a patient's 3-year cognitive trajectory.

Trains the v3 model (deterministic, ~40 s on a laptop) on one real held-out
fold, then predicts future MMSE / ADAS13 / CDR-SB for held-out patients.

  python scripts/predict_patient.py                 # demo: 3 held-out patients
  python scripts/predict_patient.py --sid ADNI_77   # a specific held-out patient
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from biojepa.data.adni import load_adni
from biojepa.data.dataset import subject_kfold
from biojepa.pipeline import prepare
from biojepa.train import train_jepa
from biojepa.evaluate import jepa_predict
from biojepa.hard_tests2 import per_pair_raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", default=None, help="e.g. ADNI_77 (a held-out test patient)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    study = load_adni()
    tr, va, te = subject_kfold(study, k=5, seed=args.seed)[0]
    p = prepare(study, splits=(tr, va, te))
    print("Training v3 (delta-learning, bio-constrained; seconds)...", flush=True)
    model, _, _ = train_jepa(p, seed=args.seed, version="v3", w_bio=2.0,
                             epochs=400, patience=60)
    pred_std = jepa_predict(model, p, "test", version="v3")
    pred_raw, truth_raw, cur_raw, dt, subj, pairs = per_pair_raw(p, "test", pred_std)
    F = {s.name: d for d, s in enumerate(study.dyn_specs)}
    idx_map = {(pp["subject"], pp["i"], pp["h"]): k for k, pp in enumerate(pairs)}

    test_sids = sorted({study.subjects[pp["subject"]].sid for pp in pairs})
    if args.sid:
        if args.sid not in test_sids:
            print(f"Not in this test fold. Examples: {', '.join(test_sids[:10])} ...")
            return
        targets = [args.sid]
    else:
        targets = test_sids[:3]

    for sid in targets:
        sid_idx = next(k for k, s in enumerate(study.subjects) if s.sid == sid)
        s = study.subjects[sid_idx]
        mm0, ad0, cd0 = s.dyn[0]
        print(f"\n{'='*62}\n{sid} | {s.group} | age {s.static[0]:.0f} | actually converted: {'Yes' if s.converts else 'No'}")
        print(f"  Baseline: MMSE {mm0:.0f} | ADAS13 {ad0:.1f} | CDR-SB {cd0:.1f}")
        print(f"  {'Horizon':>8} | {'MMSE pred':>10} | {'CDR-SB pred':>12} | Direction")
        for h, lbl in [(1.0, "12 months"), (2.0, "24 months"), (3.0, "36 months")]:
            k = idx_map.get((sid_idx, 0, h))
            if k is None:
                continue
            pm, pc = pred_raw[k, F["MMSE"]], pred_raw[k, F["CDRSB"]]
            arrow = "v decline" if pm < mm0 - 1.0 else ("^ questionable gain" if pm > mm0 + 1.0 else "= stable")
            print(f"  {lbl:>8} | {pm:>10.1f} | {pc:>12.2f} | {arrow}")
        mm36k = idx_map.get((sid_idx, 0, 3.0))
        if mm36k is not None and np.isfinite(truth_raw[mm36k, F["MMSE"]]):
            print(f"  [Observed at 36m: MMSE {truth_raw[mm36k, F['MMSE']]:.0f}]")
    print("\nNote: predictions are statistical estimates — always read them with the "
          "calibrated uncertainty documented in docs/RESULTS.md.")


if __name__ == "__main__":
    main()
