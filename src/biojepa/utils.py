"""Shared utilities: seeding, device selection, standardization, time features."""
from __future__ import annotations

import random
import warnings

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# One optimization budget for every neural model (JEPA variants and the GRU
# baseline), so no model wins through a larger training budget.
TRAINING_BUDGET = dict(epochs=400, lr=3e-3, batch=256, patience=60, min_batch=16)


def make_optimizer(params, lr: float, epochs: int, weight_decay: float = 1e-4):
    """AdamW + cosine learning-rate schedule shared by every neural model."""
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    return opt, torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Standardizer:
    """Per-feature z-scoring fit on train statistics only.

    A feature with no finite training value gets mean 0 / std 1, so it stays
    missing (NaN) after transform instead of turning every value into NaN
    through a NaN mean or standard deviation."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float32).copy()
        self.std = np.asarray(std, dtype=np.float32).copy()
        self.mean[~np.isfinite(self.mean)] = 0.0
        self.std[~np.isfinite(self.std) | (self.std < 1e-6)] = 1.0

    @classmethod
    def fit(cls, X: np.ndarray) -> "Standardizer":
        X = np.asarray(X, dtype=np.float32)
        with warnings.catch_warnings():
            # all-NaN columns are handled explicitly in __init__
            warnings.simplefilter("ignore", RuntimeWarning)
            return cls(np.nanmean(X, axis=0), np.nanstd(X, axis=0))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float32) - self.mean) / self.std

    def inverse(self, Z: np.ndarray) -> np.ndarray:
        return np.asarray(Z, dtype=np.float32) * self.std + self.mean


def fourier_time(t_years: np.ndarray, num_harmonics: int = 4, period: float = 8.0) -> np.ndarray:
    """Fourier features of a time delta in years (captures multi-scale time).

    The fundamental harmonic has an 8-year period, so the full feature vector
    is unique for gaps in [0, 8) years — including the 48-month extrapolation
    test — even though the higher harmonics repeat inside that range."""
    t = np.asarray(t_years, dtype=np.float32).reshape(-1)
    freqs = 2.0 * np.pi * np.arange(1, num_harmonics + 1) / period
    return np.concatenate([np.sin(np.outer(t, freqs)), np.cos(np.outer(t, freqs))], axis=1).astype(np.float32)
