"""Latent-objective ablation — the project's PRIMARY hypothesis test.

Question: does the latent objective (latent prediction loss + VICReg
anti-collapse) improve forecasting over the IDENTICAL architecture trained
only on the supervised clinical head?

Arms (all: v2 history encoder, no biological penalty, same optimizer,
epochs, early stopping, splits and seeds):
  jepa_full           w_lat=1, w_std=0.5, w_cov=0.25   (the JEPA model)
  supervised_control  w_lat=0, w_std=0,   w_cov=0      (matched control)
  vicreg_only         w_lat=0, w_std=0.5, w_cov=0.25   (diagnostic only)

Design: k-fold subject-level CV x seeds. Primary endpoint = one
(feature, horizon) pair, fixed BEFORE running (default MMSE at 12 months, as
in paper/SUBMISSION_CHECKLIST.md). Inference: per prediction pair the error
is averaged over seeds, then patients are the unit (bootstrap 95% CI +
sign-flip permutation p). Every other comparison is exploratory.

Run:
    PYTHONPATH=src python scripts/latent_ablation.py --study adni
    PYTHONPATH=src python scripts/latent_ablation.py --study synthetic --seeds 0 --folds 3 --max-epochs 20
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from biojepa.run_experiments import (aggregate, load_study, patient_level_mae_comparison,  # noqa: E402
                                     run_cv, write_json, environment_manifest)

ARMS = {
    "jepa_full": dict(kind="jepa", version="v2", w_bio=0.0, w_lat=1.0, w_std=0.5, w_cov=0.25),
    "supervised_control": dict(kind="jepa", version="v2", w_bio=0.0, w_lat=0.0, w_std=0.0, w_cov=0.0),
    "vicreg_only": dict(kind="jepa", version="v2", w_bio=0.0, w_lat=0.0, w_std=0.5, w_cov=0.25),
}
DEFAULT_ENDPOINT = {"adni": ("MMSE", 1.0), "oasis": ("MMSE", 1.0), "synthetic": ("MMSE_like", 1.0)}


def run(study_name="synthetic", seeds=(0, 1, 2), folds=5, endpoint=None, max_epochs=None,
        n_boot=10000, out=None):
    feature, horizon = endpoint or DEFAULT_ENDPOINT[study_name]
    study = load_study(study_name)
    cv = run_cv(study, ARMS, seeds=seeds, k_folds=folds, device=torch.device("cpu"),
                max_epochs=max_epochs)
    primary = patient_level_mae_comparison(cv, "jepa_full", "supervised_control",
                                           feature, horizon, n_boot=n_boot)
    exploratory = {}
    for spec in study.dyn_specs:
        for other in ("supervised_control", "vicreg_only"):
            exploratory[f"jepa_full_vs_{other}_{spec.name}_all_horizons"] = \
                patient_level_mae_comparison(cv, "jepa_full", other, spec.name, n_boot=n_boot)
    output = {
        "status": "CV ablation; primary endpoint pre-specified; all other comparisons exploratory",
        "dataset": study_name, "n_subjects": len(study.subjects),
        "seeds": list(seeds), "k_folds": folds, "max_epochs_override": max_epochs,
        "arms": ARMS,
        "primary_endpoint": {"feature": feature, "horizon_years": horizon,
                             "comparison": "jepa_full minus supervised_control (MAE)",
                             "interpretation": "negative mean_diff favors the latent objective",
                             **primary},
        "exploratory": exploratory,
        "aggregate": {m: aggregate(cv["per_fold"][m]) for m in ARMS},
        "environment": environment_manifest(),
    }
    target = out or os.path.join("experiments", study_name, "latent_ablation.json")
    write_json(target, output)
    p = primary
    print(f"Primary endpoint {feature}@{horizon:g}y: ΔMAE {p['mean_diff']:+.4f} "
          f"[{p['ci_low']:+.4f}, {p['ci_high']:+.4f}], p = {p['p']:.4f}, "
          f"{p['n_clusters']} patients. Wrote {target}.")
    return output


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", choices=("synthetic", "adni", "oasis"), default="synthetic")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--endpoint", nargs=2, metavar=("FEATURE", "HORIZON_YEARS"), default=None,
                    help="override the primary endpoint — only BEFORE seeing any result")
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="cap epochs (pipeline checks only; never for reported results)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ep = (args.endpoint[0], float(args.endpoint[1])) if args.endpoint else None
    run(args.study, tuple(args.seeds), args.folds, ep, args.max_epochs, args.n_boot, args.out)
