"""Repeated, patient-disjoint JEPA latent-objective ablation.

Runs matched v2 architectures with latent weight 0 and 1 on the SAME folds.
Outputs genuine results only when executed; no bundled result is fabricated.

Example:
  python scripts/repeated_ablation.py --study synthetic --seeds 0 1 --folds 3 --epochs 30
  python scripts/repeated_ablation.py --study adni --seeds 0 1 2 --folds 5 --epochs 400
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from biojepa.data.adni import load_adni
from biojepa.data.dataset import subject_kfold
from biojepa.data.oasis import load_oasis
from biojepa.data.synthetic import simulate_cascade
from biojepa.evaluate import jepa_predict
from biojepa.pipeline import prepare
from biojepa.stats import paired_patient_bootstrap
from biojepa.train import train_jepa


def run(study_name="synthetic", seeds=(0, 1), folds=3, epochs=50,
        output="experiments/repeated_ablation.json", target=None, horizon=1.0):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide at least one unique seed")
    if folds < 2 or epochs < 1:
        raise ValueError("Need at least two folds and one epoch")
    if study_name == "synthetic":
        study = simulate_cascade(n_subjects=120, seed=2026)
    elif study_name == "adni":
        study = load_adni()
    elif study_name == "oasis":
        study = load_oasis()
    else:
        raise ValueError("Unknown dataset")
    features = [s.name for s in study.dyn_specs]
    if target is None:
        target = "MMSE_like" if study_name == "synthetic" else "MMSE"
    if target not in features:
        raise ValueError(f"Feature {target} absent; available: {features}")
    feature = features.index(target)
    records = []
    for seed in seeds:
        # Each seed gives each subject precisely one held-out test fold.
        for fold, split in enumerate(subject_kfold(study, k=folds, seed=seed)):
            prepared = prepare(study, splits=split)
            if not prepared.pairs["test"]:
                continue
            predictions = {}
            for name, weight in (("no_latent", 0.0), ("latent", 1.0)):
                model, _, _ = train_jepa(prepared, device=torch.device("cpu"), seed=seed,
                                         epochs=epochs, patience=min(25, epochs),
                                         w_bio=0., w_lat=weight, version="v2")
                result_std = jepa_predict(model, prepared, "test",
                                          device=torch.device("cpu"), version="v2")
                predictions[name] = prepared.dyn_std.inverse(result_std)[:, feature]
            ids = [study.subjects[p["subject"]].sid for p in prepared.pairs["test"]]
            truth = np.array([study.subjects[p["subject"]].dyn[p["j"], feature]
                              for p in prepared.pairs["test"]])
            gaps = np.array([p["dt"] for p in prepared.pairs["test"]])
            valid = (np.abs(gaps - horizon) <= 0.26) & np.isfinite(truth)
            for i, is_valid in enumerate(valid):
                if is_valid and all(np.isfinite(pred[i]) for pred in predictions.values()):
                    records.append({"seed": int(seed), "fold": int(fold),
                                    "patient": str(ids[i]), "truth": float(truth[i]),
                                    "no_latent": float(predictions["no_latent"][i]),
                                    "latent": float(predictions["latent"][i])})
            print(f"seed={seed} fold={fold+1}/{folds} records={len(records)}", flush=True)

    summaries = []
    for seed in seeds:
        rows = [r for r in records if r["seed"] == seed]
        if len({r["patient"] for r in rows}) < 2:
            raise ValueError(f"Insufficient evaluable patients for seed {seed}")
        stats = paired_patient_bootstrap(
            [r["patient"] for r in rows],
            [r["truth"] for r in rows],
            [r["no_latent"] for r in rows],
            [r["latent"] for r in rows], n_boot=2000, seed=seed)
        summaries.append({"seed": int(seed), **stats})

    payload = {
        "status": "exploratory repeated subject-disjoint CV, not external validation",
        "study": study_name, "target": target, "horizon_years": horizon,
        "seeds": list(seeds), "folds": folds, "epochs_max": epochs,
        "comparison": "identical v2 architecture with latent loss 0 vs 1",
        "important": "Each seed has its own patient-bootstrap CI; seeds are not independent cohorts.",
        "per_seed": summaries,
        "n_valid_pairs": len(records),
        "rows": records,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved {path}; verify methodology before manuscript claims")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", choices=("synthetic", "adni", "oasis"), default="synthetic")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--target", default=None)
    parser.add_argument("--horizon", type=float, default=1.)
    parser.add_argument("--output", default="experiments/repeated_ablation.json")
    args = parser.parse_args()
    run(args.study, args.seeds, args.folds, args.epochs, args.output, args.target, args.horizon)


if __name__ == "__main__":
    main()
