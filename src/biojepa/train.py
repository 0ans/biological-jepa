"""Training loops (v1 snapshot-encoder and v2/v3 history-encoder JEPA)."""
from __future__ import annotations

import torch

from .model.jepa import build_model
from .utils import TRAINING_BUDGET as TB
from .utils import make_optimizer, seed_everything


def train_jepa(prepared, device=torch.device("cpu"), seed: int = 0, epochs: int = TB["epochs"],
               lr: float = TB["lr"], batch: int = TB["batch"], patience: int = TB["patience"],
               w_bio: float = 0.0, ema_momentum: float = 0.99, version: str = "v2",
               w_lat: float = 1.0, w_std: float = 0.5, w_cov: float = 0.25,
               min_batch: int = TB["min_batch"]):
    """Train a JEPA model; returns (model, best validation sup_future, last epoch).

    Latent-objective weights: w_lat (latent prediction loss), w_std / w_cov
    (VICReg anti-collapse on the predicted embedding). Setting all three to 0
    leaves the identical architecture trained purely on the supervised
    clinical head — the matched control for the latent-objective ablation.
    """
    seed_everything(seed)
    study = prepared.study
    model = build_model(version, in_dim=prepared.in_dim, step_dim=prepared.step_dim,
                        ctx_dim=prepared.x_context.shape[1],
                        n_dyn=len(study.dyn_specs), ema_momentum=ema_momentum).to(device)
    opt, sched = make_optimizer(model.parameters(), lr=lr, epochs=epochs)

    train = prepared.pair_tensors("train", device)
    val = prepared.pair_tensors("val", device)
    n = train["x_i"].shape[0]
    if n < min_batch:
        raise ValueError(f"only {n} training pairs; at least min_batch={min_batch} are needed")
    is_v2 = version in ("v2", "v3")
    weights = dict(w_lat=w_lat, w_std=w_std, w_cov=w_cov)

    def run_loss(t, idx, w_b, pids=None):
        b = {k: v[idx] for k, v in t.items() if isinstance(v, torch.Tensor)}
        if is_v2:
            return model.loss(b["seq_i"], b["len_i"], b["seq_j"], b["len_j"], b["ctx"],
                              b["dt_feat"], b["dyn_j"], b["mask_j"], b["dyn_i"], b["mask_i"],
                              rule_engine=prepared.engine, severity=b["severity"],
                              dt=b["dt"], w_bio=w_b, patient_ids=pids, **weights)
        return model.loss(b["x_i"], b["x_j"], b["dt_feat"], b["dyn_j"], b["mask_j"],
                          b["dyn_i"], b["mask_i"], rule_engine=prepared.engine,
                          severity=b["severity"], dt=b["dt"], w_bio=w_b,
                          patient_ids=pids, **weights)

    best_val, best_state, wait = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for lo in range(0, n, batch):
            idx = perm[lo: lo + batch]
            if len(idx) < min_batch:
                continue  # VICReg / R3 need a reasonable batch
            losses = run_loss(train, idx, w_bio,
                              pids=[train["patient_ids"][i] for i in idx.tolist()])
            opt.zero_grad()
            losses["total"].backward()
            opt.step()
            model._ema_update()
        sched.step()
        model.eval()
        with torch.no_grad():
            val_losses = run_loss(val, torch.arange(val["x_i"].shape[0]), 0.0)
            # early-stop on the predictive decode component: the latent term
            # moves with the EMA target and is not a stable selection signal
            v = float(val_losses["sup_future"])
        if v < best_val - 1e-5:
            best_val, best_state, wait = v, {k: t.detach().clone() for k, t in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, best_val, epoch
