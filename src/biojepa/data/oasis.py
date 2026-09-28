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
        subjects.append(Subject(
            sid=str(sid),
            times=times,
            dyn=dyn,
            static=static.astype(np.float32),
            patho=np.zeros(0, dtype=np.float32),
            patho_status=np.zeros(0, dtype=np.float32),
            group=str(g["Group"].iloc[0]),
            converts=bool((g["Group"] == "Converted").any()),
        ))
    return Study(
        name="oasis",
        dyn_specs=DYN_SPECS,
        patho_names=[],
        status_names=[],
        static_names=_STATIC_COLS + ["sex"],
        subjects=subjects,
    )
