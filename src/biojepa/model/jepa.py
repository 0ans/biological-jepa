"""Biological JEPA: latent-space disease-progression model.

Architecture (per the research proposal):
  - Encoder maps the patient's current state (masked longitudinal features +
    baseline pathology + demographics) to a latent embedding z_t.
  - Predictor takes z_t plus Fourier features of the time gap and predicts the
    FUTURE state's embedding ẑ_{t+k} — prediction happens in representation
    space, never in raw-data space.
  - A target encoder (EMA copy, no gradients) encodes the actually-observed
    future state; the JEPA loss is the distance between prediction and target
    embedding.
  - VICReg-style variance/covariance regularization prevents representation
    collapse (the classic JEPA failure mode; cf. LeJEPA's theoretical view).
  - A light decode head maps latents back to standardized clinical features:
    it supervises the representation and carries the biological-rule penalty,
    which shapes the PREDICTED trajectories (R1 causal capacity, R2
    monotonicity) — active where labels are missing and at rollout time.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_DIM = 64
HIDDEN = 128


def mlp(dims) -> nn.Sequential:
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b)]
        if b != dims[-1]:
            layers += [nn.SiLU(), nn.LayerNorm(b)]
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = mlp([in_dim, HIDDEN, HIDDEN, LATENT_DIM])

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self, dt_dim: int):
        super().__init__()
        self.net = mlp([LATENT_DIM + dt_dim, HIDDEN, HIDDEN, LATENT_DIM])

    def forward(self, z, dt_feat):
        return self.net(torch.cat([z, dt_feat], dim=-1))


class BioJEPA(nn.Module):
    def __init__(self, in_dim: int, n_dyn: int, dt_dim: int = 8, ema_momentum: float = 0.99):
        super().__init__()
        self.encoder = Encoder(in_dim)
        self.predictor = Predictor(dt_dim)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.head = nn.Linear(LATENT_DIM, n_dyn)
        self.ema_momentum = ema_momentum

    @torch.no_grad()
    def _ema_update(self):
        for pt, po in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            pt.mul_(self.ema_momentum).add_(po.detach(), alpha=1 - self.ema_momentum)
        for bt, bo in zip(self.target_encoder.buffers(), self.encoder.buffers()):
            bt.copy_(bo)

    def forward(self, x_i, dt_feat):
        z_i = self.encoder(x_i)
        z_hat = self.predictor(z_i, dt_feat)
        return z_i, z_hat

    def loss(self, x_i, x_j, dt_feat, dyn_std_j, mask_j, dyn_std_i, mask_i,
             rule_engine=None, severity=None, dt=None,
             w_lat=1.0, w_std=0.5, w_cov=0.25, w_sup=1.0, w_bio=0.0):
        z_i, z_hat = self.forward(x_i, dt_feat)
        with torch.no_grad():
            z_t = self.target_encoder(x_j)

        losses: dict[str, torch.Tensor] = {}
        losses["latent"] = F.smooth_l1_loss(z_hat, z_t)

        # VICReg variance hinge + covariance penalties (anti-collapse)
        def var_hinge(z):
            std = torch.sqrt(z.var(dim=0) + 1e-4)
            return F.relu(1.0 - std).mean()

        def cov_pen(z):
            zc = z - z.mean(dim=0)
            C = (zc.T @ zc) / (z.shape[0] - 1)
            d = C.shape[0]
            return (C - torch.diag(torch.diag(C))).pow(2).sum() / d

        losses["std_reg"] = var_hinge(z_hat) + var_hinge(z_t)
        losses["cov_reg"] = cov_pen(z_hat) + cov_pen(z_t)

        # decode-head supervision: current state (encoder) + future state (through predictor)
        pred_dyn = self.head(z_hat)
        cur_dyn = self.head(z_i)
        losses["sup_future"] = masked_mse(pred_dyn, dyn_std_j, mask_j)
        losses["sup_current"] = masked_mse(cur_dyn, dyn_std_i, mask_i)

        total = w_lat * losses["latent"] + w_std * losses["std_reg"] + w_cov * losses["cov_reg"] \
            + w_sup * (losses["sup_future"] + losses["sup_current"])

        if rule_engine is not None and w_bio > 0:
            with torch.no_grad():
                cur_gt = dyn_std_i  # ground-truth current state: rules compare against reality
            bio = rule_engine.penalty(cur_gt, pred_dyn, dt, severity)
            losses["bio_total"] = bio["total"]
            losses["bio_decline_capacity"] = bio.get("decline_capacity", torch.zeros(()))
            losses["bio_monotonic"] = bio.get("monotonic", torch.zeros(()))
            total = total + w_bio * losses["bio_total"]

        losses["total"] = total
        return losses

    @torch.no_grad()
    def predict_dyn(self, x_i, dt_feat):
        """Standardized future clinical features decoded from the predicted latent."""
        _, z_hat = self.forward(x_i, dt_feat)
        return self.head(z_hat)


# --------------------------------------------------------------------------
# v2: history-aware JEPA ("doctors think in steps")
# --------------------------------------------------------------------------
class HistoryEncoder(nn.Module):
    """Encodes the visit HISTORY up to time t (not just a snapshot) into the
    patient-state embedding: GRU over per-visit steps, concatenated with the
    subject-level context (baseline pathology + demographics), then projected
    to the latent space."""

    def __init__(self, step_dim: int, ctx_dim: int):
        super().__init__()
        self.gru = nn.GRU(step_dim, 64, batch_first=True)
        self.proj = mlp([64 + ctx_dim, HIDDEN, HIDDEN, LATENT_DIM])

    def forward(self, seq, lengths, ctx):
        out, _ = self.gru(seq)
        idx = (lengths - 1).clamp(min=0)
        h = out[torch.arange(out.shape[0], device=out.device), idx]
        return self.proj(torch.cat([h, ctx], dim=-1))


class BioJEPA2(nn.Module):
    """History-aware biological JEPA (v2/v3).

    Upgrades over v1, each ablatable:
      1. HistoryEncoder: the embedding summarizes the whole visit history.
      2. Deep decode head (MLP) instead of a single linear probe.
      3. delta_mode (v3): the head predicts the CHANGE from the current state
         (anchored at inference: pred = current + delta). Anchoring removes
         the pessimism bias of absolute-value regression on stable patients
         while keeping full sensitivity to real decliners.
    Training objective identical in structure to v1: latent prediction to the
    EMA target embedding of the future history, VICReg anti-collapse, decode
    supervision, and the biological-rule penalties on predicted trajectories.
    """

    def __init__(self, step_dim: int, ctx_dim: int, n_dyn: int, dt_dim: int = 8,
                 ema_momentum: float = 0.99, delta_mode: bool = False):
        super().__init__()
        self.encoder = HistoryEncoder(step_dim, ctx_dim)
        self.predictor = Predictor(dt_dim)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.head = mlp([LATENT_DIM, HIDDEN, n_dyn])
        self.ema_momentum = ema_momentum
        self.delta_mode = delta_mode

    @torch.no_grad()
    def _ema_update(self):
        for pt, po in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            pt.mul_(self.ema_momentum).add_(po.detach(), alpha=1 - self.ema_momentum)
        for bt, bo in zip(self.target_encoder.buffers(), self.encoder.buffers()):
            bt.copy_(bo)

    def loss(self, seq_i, len_i, seq_j, len_j, ctx, dt_feat, dyn_j, mask_j,
             dyn_i, mask_i, rule_engine=None, severity=None, dt=None,
             w_lat=1.0, w_std=0.5, w_cov=0.25, w_sup=1.0, w_bio=0.0,
             patient_ids=None):
        z_i = self.encoder(seq_i, len_i, ctx)
        z_hat = self.predictor(z_i, dt_feat)
        with torch.no_grad():
            z_t = self.target_encoder(seq_j, len_j, ctx)

        losses: dict[str, torch.Tensor] = {}
        losses["latent"] = F.smooth_l1_loss(z_hat, z_t)

        def var_hinge(z):
            std = torch.sqrt(z.var(dim=0) + 1e-4)
            return F.relu(1.0 - std).mean()

        def cov_pen(z):
            zc = z - z.mean(dim=0)
            C = (zc.T @ zc) / (z.shape[0] - 1)
            d = C.shape[0]
            return (C - torch.diag(torch.diag(C))).pow(2).sum() / d

        losses["std_reg"] = var_hinge(z_hat) + var_hinge(z_t)
        losses["cov_reg"] = cov_pen(z_hat) + cov_pen(z_t)

        pred_dyn = self.head(z_hat)
        if self.delta_mode:
            # delta head predicts the standardized CHANGE only; anchoring at
            # decode (pred_abs = current observed + predicted delta). The same
            # head is NOT also asked to decode the absolute current state —
            # that mixed objective was inconsistent (external review, round 3).
            losses["sup_future"] = masked_mse(pred_dyn, dyn_j - dyn_i, mask_j)
            pred_abs = dyn_i + pred_dyn
        else:
            cur_dyn = self.head(z_i)
            losses["sup_current"] = masked_mse(cur_dyn, dyn_i, mask_i)
            losses["sup_future"] = masked_mse(pred_dyn, dyn_j, mask_j)
            pred_abs = pred_dyn

        total = w_lat * losses["latent"] + w_std * losses["std_reg"] + w_cov * losses["cov_reg"] \
            + w_sup * (losses["sup_future"] + losses.get("sup_current", 0.0))

        if rule_engine is not None and w_bio > 0:
            # the rules constrain the prediction itself — future ground
            # truth is not a requirement
            bio = rule_engine.penalty(dyn_i, pred_abs, dt, severity,
                                      cur_mask=mask_i, patient_ids=patient_ids)
            losses["bio_total"] = bio["total"]
            total = total + w_bio * losses["bio_total"]

        losses["total"] = total
        return losses

    @torch.no_grad()
    def predict_dyn(self, seq_i, len_i, ctx, dt_feat, dyn_i=None):
        """Standardized future clinical features decoded from the predicted
        latent. In delta_mode the prediction is anchored on the current
        observed state (dyn_i)."""
        z_i = self.encoder(seq_i, len_i, ctx)
        z_hat = self.predictor(z_i, dt_feat)
        out = self.head(z_hat)
        if self.delta_mode and dyn_i is not None:
            out = dyn_i + out
        return out

    @torch.no_grad()
    def rollout(self, seq_i, len_i, ctx, dt_feats, dyn_i=None):
        """Latent multi-step rollout: predict ẑ_{t+1}, feed it back into the
        predictor with the next time gap, decode each step. `dt_feats` holds
        one per-step time-gap feature vector [steps, dt_dim]. Returns decoded
        standardized clinical states per step."""
        z = self.encoder(seq_i, len_i, ctx)
        outs = []
        cum = dyn_i.clone() if (self.delta_mode and dyn_i is not None) else None
        for dtf in dt_feats:
            dtf_b = dtf.to(z.device).unsqueeze(0).expand(z.shape[0], -1)
            z = self.predictor(z, dtf_b)
            step = self.head(z)
            if cum is not None:
                cum = cum + step
                outs.append(cum)
            else:
                outs.append(step)
        return outs


def build_model(version: str, in_dim: int, step_dim: int, ctx_dim: int, n_dyn: int,
                dt_dim: int = 8, ema_momentum: float = 0.99):
    if version == "v1":
        return BioJEPA(in_dim=in_dim, n_dyn=n_dyn, dt_dim=dt_dim, ema_momentum=ema_momentum)
    if version == "v2":
        return BioJEPA2(step_dim=step_dim, ctx_dim=ctx_dim, n_dyn=n_dyn,
                        dt_dim=dt_dim, ema_momentum=ema_momentum)
    if version == "v3":
        return BioJEPA2(step_dim=step_dim, ctx_dim=ctx_dim, n_dyn=n_dyn,
                        dt_dim=dt_dim, ema_momentum=ema_momentum, delta_mode=True)
    raise ValueError(version)


def masked_mse(pred, target, mask):
    m = mask.float()
    denom = m.sum().clamp(min=1.0)
    return ((pred - target) ** 2 * m).sum() / denom
