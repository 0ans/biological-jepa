"""Synthetic Alzheimer-like cascade simulator — ground truth for the rule engine.

Generates subjects whose biology follows the amyloid->tau->neurodegeneration->
cognition cascade (Jack et al. 2010/2013) exactly, so the biological-rule
engine can be verified end-to-end against known ground truth:

  A(t): amyloid burden; A+ subjects start high and creep up, A- subjects stay low.
  T(t): tau burden grows only where amyloid is active (upstream trigger).
  N(t): neurodegeneration grows only where tau is active.
  C(t): cognition (MMSE-like higher-better, CDR-like higher-worse) declines
        only as neurodegeneration accumulates.
"""
from __future__ import annotations

import numpy as np

from .dataset import FeatureSpec, Study, Subject

DYN_SPECS = [
    FeatureSpec("A_burden", higher_is_worse=True, group="A"),
    FeatureSpec("T_burden", higher_is_worse=True, group="T"),
    FeatureSpec("N_atrophy", higher_is_worse=True, group="N"),
    FeatureSpec("MMSE_like", higher_is_worse=False, group="C"),
    FeatureSpec("CDR_like", higher_is_worse=True, group="C"),
]


def simulate_cascade(n_subjects: int = 120, n_visits: int = 4, seed: int = 0,
                     missing_rate: float = 0.15) -> Study:
    rng = np.random.RandomState(seed)
    times = np.arange(n_visits, dtype=np.float32)  # 0..3 years
    subjects: list[Subject] = []
    for k in range(n_subjects):
        amyloid_positive = rng.rand() < 0.5
        if amyloid_positive:
            a0, a_rate = rng.uniform(0.55, 0.9), rng.uniform(0.01, 0.04)
        else:
            a0, a_rate = rng.uniform(0.05, 0.35), 0.0
        age = rng.uniform(65, 85)

        A = np.clip(a0 + a_rate * times, 0, 1)
        drive_T = np.maximum(A - 0.4, 0)
        T = np.clip(0.08 + drive_T * (0.35 + 0.45 * times / max(times[-1], 1)) + rng.normal(0, 0.02, n_visits), 0, 1)
        drive_N = np.maximum(T - 0.3, 0)
        N = np.clip(0.12 + drive_N * (0.5 + 0.5 * times / max(times[-1], 1)) + rng.normal(0, 0.03, n_visits), 0, 1)
        mmse = np.clip(30 - 6.0 * N * (0.4 + 0.6 * times / max(times[-1], 1)) + rng.normal(0, 0.6, n_visits), 0, 30)
        cdr = np.clip(0.1 + 1.6 * N * (0.4 + 0.6 * times / max(times[-1], 1)) + rng.normal(0, 0.08, n_visits), 0, 4)

        dyn = np.stack([A, T, N, mmse, cdr], axis=1).astype(np.float32)
        mask = rng.rand(*dyn.shape) > missing_rate
        mask[0] = True  # baseline observed
        dyn[~mask] = np.nan

        subjects.append(Subject(
            sid=f"SYN_{k:04d}",
            times=times.copy(),
            dyn=dyn,
            static=np.array([age], dtype=np.float32),
            patho=np.zeros(0, dtype=np.float32),
            patho_status=np.zeros(0, dtype=np.float32),
            group="Apos" if amyloid_positive else "Aneg",
            converts=bool(cdr[-1] >= 1.0),
        ))
    return Study(
        name="synthetic",
        dyn_specs=DYN_SPECS,
        patho_names=[],
        status_names=[],
        static_names=["age"],
        subjects=subjects,
    )
