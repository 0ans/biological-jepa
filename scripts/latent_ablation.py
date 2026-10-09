"""Controlled latent-loss ablation on a single, subject-disjoint split.

Run:
    PYTHONPATH=src python scripts/latent_ablation.py --study synthetic
    PYTHONPATH=src python scripts/latent_ablation.py --study adni

This is a reproducibility *experiment runner*, NOT evidence of superiority.
Do not conflate one seed / split with the main 5-fold cross-validation study.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from biojepa.data.synthetic import simulate_cascade
from biojepa.data.adni import load_adni
from biojepa.data.oasis import load_oasis
from biojepa.pipeline import prepare
from biojepa.train import train_jepa
from biojepa.evaluate import evaluate_predictions, jepa_predict


def run(study_name="synthetic", seed=0, epochs=100, out="experiments/latent_ablation.json"):
    loaders = {"synthetic": lambda: simulate_cascade(n_subjects=120, seed=seed),
               "adni": load_adni, "oasis": load_oasis}
    study = loaders[study_name]()
    prepared = prepare(study, seed=seed)
    all_metrics = {}
    for name, weight in [("supervised_without_latent", 0.0), ("jepa_with_latent", 1.0)]:
        model, best_val, epoch = train_jepa(
            prepared, device=torch.device("cpu"), seed=seed,
            epochs=epochs, patience=min(25, epochs), w_bio=0.0,
            w_lat=weight, version="v2")
        yhat = jepa_predict(model, prepared, "test", device=torch.device("cpu"), version="v2")
        assert np.isfinite(yhat).all(), f"non-finite predictions: {name}"
        all_metrics[name] = {"latent_weight": weight,
                             "best_val": float(best_val),
                             "epochs_used": int(epoch + 1),
                             "metrics": evaluate_predictions(prepared, "test", yhat)}
    output = {"status": "single-split exploratory ablation; NOT external validation",
              "dataset": study_name, "seed": seed, "epochs_max": epochs,
              "n_subjects": len(study.subjects), "results": all_metrics}
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Wrote {target}. Do not cite as primary confirmatory evidence.")
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--study", choices=("synthetic", "adni", "oasis"), default="synthetic")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--out", default="experiments/latent_ablation.json")
    args = p.parse_args()
    run(args.study, args.seed, args.epochs, args.out)
