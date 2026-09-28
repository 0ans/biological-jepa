"""Smoke test: full training + evaluation path runs on the synthetic study and
the representation does not collapse."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from biojepa.data.synthetic import simulate_cascade
from biojepa.pipeline import prepare
from biojepa.train import train_jepa
from biojepa.evaluate import jepa_predict, evaluate_predictions
from biojepa.model.baselines import carry_forward_predict, RidgeBaseline, ridge_design_matrix
from biojepa.run_experiments import run_study


def test_end_to_end_smoke(tmp_path):
    res = run_study.__wrapped__ if hasattr(run_study, "__wrapped__") else None
    # tiny end-to-end run on synthetic-like data via direct components
    study = simulate_cascade(n_subjects=60, seed=3)
    prepared = prepare(study, seed=3)
    model, best_val, _ = train_jepa(prepared, seed=3, epochs=8, patience=8, w_bio=0.5)
    preds = jepa_predict(model, prepared, "test")
    assert preds.shape[1] == len(study.dyn_specs)
    assert np.isfinite(preds).all()

    # representation must not collapse: encoder output std should be non-trivial
    t = prepared.pair_tensors("test")
    with torch.no_grad():
        z = model.encoder(t["seq_i"], t["len_i"], t["ctx"])
    assert float(z.std()) > 0.2, f"representation collapse suspected: std={float(z.std()):.3f}"

    metrics = evaluate_predictions(prepared, "test", preds)
    assert any(k.startswith("mae_") for k in metrics)

    # carry-forward + ridge paths
    train_ids = sorted({p["subject"] for p in prepared.pairs["train"]})
    pr, tr, mk, hs = carry_forward_predict(study, prepared.pairs["test"], train_ids)
    assert pr.shape == tr.shape == mk.shape
    X = ridge_design_matrix(prepared, prepared.pairs["train"])
    Y = np.stack([prepared.dyn_by_subject[p["subject"]][p["j"]] for p in prepared.pairs["train"]])
    M = np.isfinite(Y)
    rb = RidgeBaseline(); rb.fit(X, np.nan_to_num(Y), M)
    pr_std = rb.predict(ridge_design_matrix(prepared, prepared.pairs["test"]))
    assert pr_std.shape[1] == len(study.dyn_specs)
