"""ADNI longitudinal sample loader.

Primary real-data source for this project. The file is a sample of the ADNI
study in ADNIMERGE-style long format (one row per subject visit), as
redistributed with the `abaR` R package (ncullen93/abaR, processed from
ADNI's ADNIMERGE table). Provenance and access terms: see docs/ADNI_ACCESS.md.

Because biomarkers in this sample are baseline-only, the modeling task is the
canonical ADNI progression-prediction setup: baseline pathology (amyloid A,
tau T, neurodegeneration N) + current clinical state -> future clinical
trajectory, with the biological-rule engine constraining predicted trajectories.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pyreadr

from .dataset import FeatureSpec, Study, Subject

DATA_PATH = os.path.join("data", "raw", "adnimerge_sample.rda")

DYN_SPECS = [
    FeatureSpec("MMSE", higher_is_worse=False, group="C"),
    FeatureSpec("ADAS13", higher_is_worse=True, group="C"),
    FeatureSpec("CDRSB", higher_is_worse=True, group="C"),
]
# columns used verbatim from the source table
_DYN_COLS = ["MMSE", "ADAS13", "CDRSB"]
_PATHO_COLS = ["CSF_ABETA_bl", "PET_ABETA_bl", "CSF_TAU_bl", "CSF_PTAU_bl", "MRI_HIPP_bl"]
_STATUS_COLS = ["PET_ABETA_STATUS_bl", "CSF_ABETA_STATUS_bl"]
_STATIC_COLS = ["AGE", "GENDER", "EDUCATION", "APOE4"]


def load_adni(path: str = DATA_PATH) -> Study:
    table = list(pyreadr.read_r(path).values())[0]
    with warnings.catch_warnings():
        # pyreadr hands several columns over as float/categorical arrays that
        # contain NaNs; pandas emits benign RuntimeWarnings while casting them
        # to str/float32, and every missing value is handled explicitly below.
        warnings.simplefilter("ignore", RuntimeWarning)
        vis = table["VISCODE"].astype(str)
        table = table[vis.str.match(r"^(bl|m0|m\d+)$")].copy()
        table["DX_bl"] = table["DX_bl"].astype(str).replace("", "UNK").replace("nan", "UNK")

        subjects: list[Subject] = []
        for rid, g in table.groupby("RID"):
            g = g.sort_values("YEARS_bl")
            times = g["YEARS_bl"].to_numpy(dtype=np.float32)
            dyn = g[_DYN_COLS].to_numpy(dtype=np.float32)
            # merge duplicate time stamps (m0 vs bl): the rows are the SAME
            # visit — average the available (non-missing) measurements per
            # feature (a deduplication choice, not an unbiased estimator)
            uniq_t, inv = np.unique(np.round(times, 4), return_inverse=True)
            if len(uniq_t) != len(times):
                dyn_u = np.full((len(uniq_t), dyn.shape[1]), np.nan, dtype=np.float32)
                for gi in range(len(uniq_t)):
                    block = dyn[inv == gi]
                    if np.isfinite(block).any():
                        dyn_u[gi] = np.nanmean(block, axis=0)
                times, dyn = uniq_t.astype(np.float32), dyn_u
            static = g[_STATIC_COLS].iloc[0].to_numpy(dtype=np.float32)
            patho = g[_PATHO_COLS].iloc[0].to_numpy(dtype=np.float32)
            status = g[_STATUS_COLS].iloc[0].fillna(-1).to_numpy(dtype=np.float32)
            subjects.append(Subject(
                sid=f"ADNI_{int(rid)}",
                times=times,
                dyn=dyn,
                static=static,
                patho=patho,
                patho_status=status,
                group=str(g["DX_bl"].iloc[0]),
                converts=bool(g["ConvertedToDementia"].iloc[0] > 0),
                time_under_risk=float(g["TimeUnderRiskDementia"].iloc[0])
                if np.isfinite(g["TimeUnderRiskDementia"].iloc[0]) else None,
            ))
    return Study(
        name="adni",
        dyn_specs=DYN_SPECS,
        patho_names=_PATHO_COLS,
        status_names=_STATUS_COLS,
        static_names=_STATIC_COLS,
        subjects=subjects,
    )
