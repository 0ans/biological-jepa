"""Baselines: standard "pattern-matching" models for disease progression.

- carry_forward : clinical standard — predict the last observed value.
- ridge         : regularized linear map from current state + baseline
                  pathology + demographics + time gap -> future values.
- gru           : supervised sequence model (GRU) consuming the visit history
                  up to time t and directly regressing future raw-value levels.

These supervised baselines directly predict future clinical values. The GRU
receives the same requested future horizon as JEPA and tabular baselines;
these baselines do not use biological penalties.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .jepa import masked_mse, mlp
from ..utils import fourier_time


# ----------------------------------------------------------------- GRU model
class GRUProg(nn.Module):
    def __init__(self, step_dim: int, n_dyn: int, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(step_dim, hidden, batch_first=True)
        self.head = mlp([hidden, hidden, n_dyn])

    def forward(self, seq, lengths):
        out, _ = self.gru(seq)
        idx = (lengths - 1).clamp(min=0)
        last = out[torch.arange(out.shape[0]), idx]
        return self.head(last)


# ------------------------------------------------------------- carry-forward
def carry_forward_predict(study, pairs, train_subject_ids):
    """Predict each feature's last observed raw value at or before visit i.
    Never-observed features fall back to the train-cohort mean."""
    train_ids = list(train_subject_ids)
    train_dyn = np.concatenate([study.subjects[k].dyn for k in train_ids], axis=0)
    feat_mean = np.nanmean(train_dyn, axis=0)

    preds, truths, masks, hs = [], [], [], []
    for pair in pairs:
        s = study.subjects[pair["subject"]]
        hist = s.dyn[: pair["i"] + 1]
        last_obs = np.array([
            hist[np.isfinite(hist[:, d]), d][-1] if np.isfinite(hist[:, d]).any() else feat_mean[d]
            for d in range(hist.shape[1])
        ], dtype=np.float64)
        preds.append(last_obs)
        truths.append(s.dyn[pair["j"]])
        masks.append(np.isfinite(s.dyn[pair["j"]]).astype(int))
        hs.append(pair["h"])
    return np.array(preds), np.array(truths), np.array(masks), np.array(hs)


# --------------------------------------------------------------------- ridge
def ridge_design_matrix(prepared, pairs, dyn_lookup=None) -> np.ndarray:
    """[masked dyn at visit i, mask, subject context, Fourier(time gap)] —
    one shared design for fit and predict. `dyn_lookup` allows swapping in a
    degraded input-history view for the missingness stress test."""
    from ..utils import fourier_time as _fourier
    lookup = dyn_lookup or (lambda s: prepared.dyn_by_subject[s])
    X = []
    for pair in pairs:
        s_idx = pair["subject"]
        di = np.nan_to_num(lookup(s_idx)[pair["i"]], nan=0.0)
        mi = np.isfinite(lookup(s_idx)[pair["i"]]).astype(np.float32)
        X.append(np.concatenate([di, mi, prepared.x_context[s_idx],
                                 _fourier(np.array([pair["dt"]]))[0]]))
    return np.array(X, dtype=np.float32)


class RidgeBaseline:
    """Per-feature ridge regression from the current-state design matrix."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.models: list = []

    def fit(self, X, Y, mask):
        from sklearn.linear_model import Ridge
        self.models = []
        for d in range(Y.shape[1]):
            m = mask[:, d].astype(bool)
            r = Ridge(alpha=self.alpha)
            if m.sum() >= 5:
                r.fit(X[m], Y[m, d])
            else:
                r.fit(X, np.full(len(X), Y[:, d].mean() if np.isfinite(Y[:, d]).any() else 0.0))
            self.models.append(r)

    def predict(self, X):
        return np.stack([r.predict(X) for r in self.models], axis=1)


class HGBBaseline:
    """Per-feature histogram gradient boosting (nonlinear tabular baseline) on
    the same design matrix as ridge — the strong pattern-matching competitor."""

    def __init__(self, max_iter: int = 300, learning_rate: float = 0.1, max_depth: int = 3,
                 random_state: int = 0):
        self.params = dict(max_iter=max_iter, learning_rate=learning_rate,
                           max_depth=max_depth, random_state=random_state)
        self.models: list = []

    def fit(self, X, Y, mask):
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.models = []
        for d in range(Y.shape[1]):
            m = mask[:, d].astype(bool)
            r = HistGradientBoostingRegressor(**self.params)
            if m.sum() >= 20:
                r.fit(X[m], Y[m, d])
            else:
                r.fit(X, np.full(len(X), Y[:, d].mean() if np.isfinite(Y[:, d]).any() else 0.0))
            self.models.append(r)

    def predict(self, X):
        return np.stack([r.predict(X) for r in self.models], axis=1)


# --------------------------------------------------------------- GRU wrapper
def gru_collate(study, pairs, dyn_std_lookup, x_static, dt_dim=8, max_len=None):
    """Build observed history, context and requested future horizon.

    pair["dt"] is measured in years and is known at inference time.
    """
    B = len(pairs)
    D = len(study.dyn_specs)
    ctx_dim = x_static.shape[1]
    L = max_len or max(p["i"] + 1 for p in pairs)
    L = max(L, max(p["i"] + 1 for p in pairs))  # never truncate histories
    seq_dyn = np.zeros((B, L, 2 * D + 2 * dt_dim + ctx_dim), dtype=np.float32)
    lengths = np.zeros(B, dtype=np.int64)
    for b, pair in enumerate(pairs):
        s_idx, i = pair["subject"], pair["i"]
        dyn_std = dyn_std_lookup(s_idx)
        s = study.subjects[s_idx]
        horizon = float(pair["dt"])
        if not np.isfinite(horizon) or horizon <= 0:
            raise ValueError(f"Forecast gap must be positive and finite, got {horizon}")
        horizon_f = fourier_time(np.array([horizon]), dt_dim // 2)[0]
        prev_t = None
        for t, visit_t in enumerate(s.times[: i + 1]):
            gap = 0.0 if prev_t is None else visit_t - prev_t
            mask = np.isfinite(dyn_std[t]).astype(np.float32)
            vals = np.nan_to_num(dyn_std[t])
            dt_f = fourier_time(np.array([max(gap, 1e-3)]), dt_dim // 2)[0]
            ctx = x_static[s_idx]
            seq_dyn[b, t] = np.concatenate([vals, mask, dt_f, horizon_f, ctx])
            prev_t = visit_t
        lengths[b] = i + 1
    return seq_dyn, lengths


def train_gru(study, train_pairs, val_pairs, dyn_std_lookup, x_static,
              device, seed=0, epochs=200, lr=3e-3, batch=256, patience=25):
    from ..utils import seed_everything
    seed_everything(seed)
    D = len(study.dyn_specs)
    dt_dim = 8
    model = GRUProg(step_dim=2 * D + 2 * dt_dim + x_static.shape[1], n_dyn=D).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    tr_seq, tr_len = gru_collate(study, train_pairs, dyn_std_lookup, x_static, dt_dim)
    va_seq, va_len = gru_collate(study, val_pairs, dyn_std_lookup, x_static, dt_dim, max_len=tr_seq.shape[1])
    tr_y = np.stack([dyn_std_lookup(p["subject"])[p["j"]] for p in train_pairs])
    tr_m = np.stack([np.isfinite(dyn_std_lookup(p["subject"])[p["j"]]) for p in train_pairs]).astype(np.float32)
    va_y = np.stack([dyn_std_lookup(p["subject"])[p["j"]] for p in val_pairs])
    va_m = np.stack([np.isfinite(dyn_std_lookup(p["subject"])[p["j"]]) for p in val_pairs]).astype(np.float32)
    tr_y = np.nan_to_num(tr_y, nan=0.0)  # masked out by tr_m in the loss
    va_y = np.nan_to_num(va_y, nan=0.0)

    to_t = lambda a: torch.as_tensor(a, device=device)
    tr_seq_t, tr_len_t, tr_y_t, tr_m_t = map(to_t, (tr_seq, tr_len, tr_y, tr_m))
    va_seq_t, va_len_t, va_y_t, va_m_t = map(to_t, (va_seq, va_len, va_y, va_m))

    best_val, best_state, wait = float("inf"), None, 0
    n = len(train_pairs)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for lo in range(0, n, batch):
            idx = perm[lo: lo + batch]
            if len(idx) < 8:
                continue
            pred = model(tr_seq_t[idx], tr_len_t[idx])
            loss = masked_mse(pred, tr_y_t[idx], tr_m_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(va_seq_t, va_len_t)
            val_loss = float(masked_mse(pred, va_y_t, va_m_t))
        if val_loss < best_val - 1e-5:
            best_val, best_state, wait = val_loss, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model
