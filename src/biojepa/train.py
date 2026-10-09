"""Training loops (v1 snapshot-encoder and v2 history-encoder JEPA)."""
from __future__ import annotations

import torch

from .model.jepa import build_model
from .utils import seed_everything


def train_jepa(prepared, device=torch.device("cpu"), seed: int = 0, epochs: int = 400,
               lr: float = 3e-3, batch: int = 256, patience: int = 60,
               w_bio: float = 0.0, ema_momentum: float = 0.99, version: str = "v2",
               w_lat: float = 1.0):
    seed_everything(seed)
    study = prepared.study
    model = build_model(version, in_dim=prepared.in_dim, step_dim=prepared.step_dim,
                        ctx_dim=prepared.x_context.shape[1],
                        n_dyn=len(study.dyn_specs), ema_momentum=ema_momentum).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    train = prepared.pair_tensors("train", device)
    val = prepared.pair_tensors("val", device)
    n = train["x_i"].shape[0]
    is_v2 = version in ("v2", "v3")

    def run_loss(t, idx, w_b, pids=None):
        b = {k: v[idx] for k, v in t.items() if isinstance(v, torch.Tensor)}
        if is_v2:
            return model.loss(b["seq_i"], b["len_i"], b["seq_j"], b["len_j"], b["ctx"],
                              b["dt_feat"], b["dyn_j"], b["mask_j"], b["dyn_i"], b["mask_i"],
                              rule_engine=prepared.engine, severity=b["severity"],
                              dt=b["dt"], w_bio=w_b, w_lat=w_lat, patient_ids=pids)
        return model.loss(b["x_i"], b["x_j"], b["dt_feat"], b["dyn_j"], b["mask_j"],
                          b["dyn_i"], b["mask_i"], rule_engine=prepared.engine,
                          severity=b["severity"], dt=b["dt"], w_bio=w_b, w_lat=w_lat)

    best_val, best_state, wait = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for lo in range(0, n, batch):
            idx = perm[lo: lo + batch]
            if len(idx) < 16:
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
