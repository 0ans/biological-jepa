"""OASIS-2 longitudinal loader (secondary real-data study).

OASIS-2: 150 subjects, 373 MRI visits over 4+ years (Marcus et al., 2010).
Openly redistributed copies exist in public research repositories; provenance
is documented in docs/ADNI_ACCESS.md. Unlike the ADNI sample used here, OASIS-2
has a *longitudinal* neurodegeneration measure (nWBV, normalized whole-brain
volume) but no amyloid/tau biomarkers — so the biological-rule engine runs its
N->C rules (atrophy monotonicity, pathology-supported decline) while A/T rules
remain inactive. This dataset doubles as a pipeline-validation study and as a
demonstration that the framework transfers across cohorts.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .dataset import FeatureSpec, Study, Subject

DATA_PATH = os.path.join("data", "raw", "oasis_longitudinal.csv")

DYN_SPECS = [
    FeatureSpec("MMSE", higher_is_worse=False, group="C"),
    FeatureSpec("CDR", higher_is_worse=True, group="C"),
    FeatureSpec("nWBV", higher_is_worse=False, group="N"),
]
_DYN_COLS = ["MMSE", "CDR", "nWBV"]
_STATIC_COLS = ["Age", "EDUC", "SES", "eTIV"]  # sex handled separately as binary


def _time_under_risk(times: np.ndarray, cdr: np.ndarray, group: str,
                     converts: bool) -> float | None:
    """OASIS-2 has no event-time column. Converters ("nondemented at the
    initial visit, demented at a later visit") get the first follow-up visit
    with CDR >= 0.5; non-converters get their last visit time (censoring).
    Undeterminable timing (no qualifying visit, or CDR >= 0.5 already at
    baseline) returns None, which excludes the subject from conversion
    analyses. Baseline-demented subjects are never at risk."""
    if group == "Demented":
        return None
    if converts:
        hit = np.where(np.isfinite(cdr) & (cdr >= 0.5))[0]
        if len(hit) == 0 or hit[0] == 0:
            return None
        return float(times[hit[0]])
    return float(times[-1])


def load_oasis(path: str = DATA_PATH) -> Study:
    df = pd.read_csv(path)
    df = df.rename(columns={"Subject ID": "sid", "M/F": "sex"})
    df["sex_num"] = (df["sex"] == "M").astype(np.float32)
    df["Group"] = df["Group"].fillna("Nondemented")

    subjects: list[Subject] = []
    for sid, g in df.groupby("sid"):
        g = g.sort_values("Visit")
        times = (g["MR Delay"].to_numpy(dtype=np.float32) / 365.25)  # days -> years
        dyn = g[_DYN_COLS].to_numpy(dtype=np.float32)
        static = np.concatenate([
            g[["Age", "EDUC", "SES", "eTIV"]].iloc[0].to_numpy(dtype=np.float32),
            [g["sex_num"].iloc[0]],
        ])
        group = str(g["Group"].iloc[0])
        converts = bool((g["Group"] == "Converted").any())
        subjects.append(Subject(
            sid=str(sid),
            times=times,
            dyn=dyn,
            static=static.astype(np.float32),
            patho=np.zeros(0, dtype=np.float32),
            patho_status=np.zeros(0, dtype=np.float32),
            group=group,
            converts=converts,
            time_under_risk=_time_under_risk(times, g["CDR"].to_numpy(dtype=np.float32),
                                             group, converts),
            dementia_at_baseline=group == "Demented",
        ))
    return Study(
        name="oasis",
        dyn_specs=DYN_SPECS,
        patho_names=[],
        status_names=[],
        static_names=_STATIC_COLS + ["sex"],
        subjects=subjects,
    )
