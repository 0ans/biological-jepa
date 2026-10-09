"""Patient-clustered, paired uncertainty intervals for forecasting comparisons.

Inputs are per-prediction arrays aligned by patient and visit target. Patients
are the resampling unit; repeated observations are never independently drawn.
No distributional, causal, or external-validation claim follows from this test.
"""
from __future__ import annotations

import numpy as np


def paired_patient_bootstrap(patient_ids, truth, prediction_a, prediction_b,
                             observed=None, n_boot=2000, seed=0):
    """Estimate paired difference in MAE (A - B), resampling distinct patients.

    Inputs: 1-D arrays with aligned observations; positive difference means B
    has lower MAE. Cluster means are averaged equally over patients, not pairs.
    Raises for insufficient patients, invalid observations, or zero bootstrap.
    """
    patient_ids = np.asarray(patient_ids).astype(str)
    truth = np.asarray(truth, dtype=float)
    a = np.asarray(prediction_a, dtype=float)
    b = np.asarray(prediction_b, dtype=float)
    if not (patient_ids.ndim == truth.ndim == a.ndim == b.ndim == 1):
        raise ValueError("Expected one-dimensional aligned arrays")
    if not (len(patient_ids) == len(truth) == len(a) == len(b)):
        raise ValueError("Input arrays must be aligned and equal length")
    if not isinstance(n_boot, int) or n_boot < 100:
        raise ValueError("Use at least 100 patient-cluster bootstrap resamples")
    valid = np.isfinite(truth) & np.isfinite(a) & np.isfinite(b)
    if observed is not None:
        mask = np.asarray(observed, dtype=bool)
        if mask.shape != valid.shape:
            raise ValueError("Observed mask shape mismatch")
        valid &= mask
    if not valid.any():
        raise ValueError("No valid overlapping observations")

    subjects, inverse = np.unique(patient_ids[valid], return_inverse=True)
    if len(subjects) < 2:
        raise ValueError("At least two distinct patients are required")
    ea = np.abs(a[valid] - truth[valid])
    eb = np.abs(b[valid] - truth[valid])
    counts = np.bincount(inverse)
    subject_a = np.bincount(inverse, weights=ea) / counts
    subject_b = np.bincount(inverse, weights=eb) / counts
    differences = subject_a - subject_b

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(subjects), size=len(subjects))
        draws[i] = differences[idx].mean()
    low, high = np.quantile(draws, (0.025, 0.975))
    return {
        "n_patients": int(len(subjects)),
        "n_observations": int(valid.sum()),
        "mae_a_patient_mean": float(subject_a.mean()),
        "mae_b_patient_mean": float(subject_b.mean()),
        "difference_a_minus_b": float(differences.mean()),
        "ci95": [float(low), float(high)],
        "bootstrap_resamples": int(n_boot),
        "resampling_unit": "patient",
        "interpretation": "positive difference favors prediction_b",
    }
