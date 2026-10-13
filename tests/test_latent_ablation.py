"""Guardrails for the latent-objective ablation (the primary hypothesis)."""
import inspect
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from biojepa.data.synthetic import simulate_cascade
from biojepa.model.jepa import BioJEPA2
from biojepa.pipeline import prepare
from biojepa.train import train_jepa


def test_latent_objective_weights_are_exposed():
    p = inspect.signature(train_jepa).parameters
    assert p["w_lat"].default == 1.0
    assert p["w_std"].default == 0.5
    assert p["w_cov"].default == 0.25


def test_control_arm_is_purely_supervised():
    """With w_lat = w_std = w_cov = 0 the total loss equals the supervised
    clinical loss exactly, and the target encoder receives no influence."""
    study = simulate_cascade(n_subjects=60, seed=0)
    prepared = prepare(study, seed=0)
    t = prepared.pair_tensors("train")
    torch.manual_seed(0)
    model = BioJEPA2(prepared.step_dim, prepared.x_context.shape[1], len(study.dyn_specs))
    args = (t["seq_i"], t["len_i"], t["seq_j"], t["len_j"], t["ctx"], t["dt_feat"],
            t["dyn_j"], t["mask_j"], t["dyn_i"], t["mask_i"])
    control = model.loss(*args, w_lat=0.0, w_std=0.0, w_cov=0.0)
    sup = control["sup_future"] + control["sup_current"]
    assert torch.allclose(control["total"], sup)
    full = model.loss(*args)
    assert float(full["total"]) > float(control["total"])


def test_vicreg_terms_only_on_trainable_branch():
    """Anti-collapse terms must not depend on the no-gradient EMA target."""
    study = simulate_cascade(n_subjects=60, seed=0)
    prepared = prepare(study, seed=0)
    t = prepared.pair_tensors("train")
    torch.manual_seed(0)
    model = BioJEPA2(prepared.step_dim, prepared.x_context.shape[1], len(study.dyn_specs))
    args = (t["seq_i"], t["len_i"], t["seq_j"], t["len_j"], t["ctx"], t["dt_feat"],
            t["dyn_j"], t["mask_j"], t["dyn_i"], t["mask_i"])
    before = model.loss(*args)
    with torch.no_grad():
        for prm in model.target_encoder.parameters():
            prm.add_(torch.randn_like(prm))
    after = model.loss(*args)
    assert float(before["std_reg"]) == float(after["std_reg"])
    assert float(before["cov_reg"]) == float(after["cov_reg"])
    assert float(before["latent"]) != float(after["latent"])
    assert before["std_reg"].requires_grad
