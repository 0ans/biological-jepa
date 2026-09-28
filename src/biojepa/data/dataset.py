"""Core longitudinal data structures shared by all studies (ADNI / OASIS / synthetic).

A Study is a set of subjects; each subject has a time grid of visits, dynamic
features observed sparsely (with masks), static demographics, and baseline
pathology features (A/T/N biomarkers when available).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FeatureSpec:
    name: str
    higher_is_worse: bool
    group: str  # biological group of a dynamic feature: 'C' (clinical) or 'N' (neurodegeneration)


@dataclass
class Subject:
    sid: str
    times: np.ndarray          # [T] years from baseline visit
    dyn: np.ndarray            # [T, D] raw dynamic feature values (NaN = missing)
    static: np.ndarray         # [S] static demographics (NaN allowed)
    patho: np.ndarray          # [P] baseline pathology raw values (NaN = missing)
    patho_status: np.ndarray   # [K] binary pathology status flags (0/1, -1 = unknown)
    group: str                 # diagnostic group string
    converts: bool             # converts to dementia during study (label)
    time_under_risk: float | None = None   # years to conversion or end of follow-up


@dataclass
class Study:
    name: str
    dyn_specs: list[FeatureSpec]
    patho_names: list[str]
    status_names: list[str]
    static_names: list[str]
    subjects: list[Subject]
    # indices into dyn vector of features belonging to each rule-relevant group
    dyn_group_idx: dict[str, list[int]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.dyn_group_idx:
            idx: dict[str, list[int]] = {}
            for d, spec in enumerate(self.dyn_specs):
                idx.setdefault(spec.group, []).append(d)
            self.dyn_group_idx = idx


def subject_split(study: Study, fractions=(0.7, 0.15, 0.15), seed: int = 0, stratify: bool = True):
    """Split subject indices into train/val/test, stratified by group. No subject
    appears in more than one split (prevents longitudinal leakage)."""
    rng = np.random.RandomState(seed)
    groups = {}
    for idx, s in enumerate(study.subjects):
        groups.setdefault(s.group, []).append(idx)
    train, val, test = [], [], []
    for g, members in sorted(groups.items()):
        members = np.array(members)
        rng.shuffle(members)
        n = len(members)
        if stratify:
            n_tr = int(round(fractions[0] * n))
            n_va = int(round(fractions[1] * n))
            train += members[:n_tr].tolist()
            val += members[n_tr:n_tr + n_va].tolist()
            test += members[n_tr + n_va:].tolist()
        else:
            train += members.tolist()
    return set(train), set(val), set(test)


def subject_kfold(study, k: int = 5, seed: int = 0, val_fraction: float = 0.15):
    """Stratified subject-level K-fold: returns k (train, val, test) subject-id
    sets. Each subject appears in exactly one test fold per seed. The non-test
    part of each fold is split into train/val. No subject crosses splits."""
    rng = np.random.RandomState(seed)
    groups: dict[str, list[int]] = {}
    for idx, s in enumerate(study.subjects):
        groups.setdefault(s.group, []).append(idx)
    folds: list[list[int]] = [[] for _ in range(k)]
    for g, members in sorted(groups.items()):
        members = np.array(members)
        rng.shuffle(members)
        for f, chunk in enumerate(np.array_split(members, k)):
            folds[f] += chunk.tolist()
    splits = []
    for f in range(k):
        test = set(folds[f])
        rest = [i for g in range(k) if g != f for i in folds[g]]
        rest = np.array(rest)
        rng.shuffle(rest)
        n_val = int(round(val_fraction * len(rest)))
        val = set(rest[:n_val].tolist())
        train = set(rest[n_val:].tolist())
        splits.append((train, val, test))
    return splits


def make_pair_sets(study: Study, horizons=(1.0, 2.0, 3.0), window=0.25, min_visits=2):
    """Enumerate (subject, i, j, h) evaluation/training pairs: predict state at
    visit j (near time t_i + h) from visit i. Returns list of dicts.

    A pair is valid when both visits have at least one observed dynamic feature.
    """
    pairs = []
    for subj_idx, s in enumerate(study.subjects):
        if len(s.times) < min_visits:
            continue
        for i in range(len(s.times)):
            if not np.isfinite(s.dyn[i]).any():
                continue
            for h in horizons:
                target_t = s.times[i] + h
                d = np.abs(s.times - target_t)
                j = int(np.argmin(d))
                if j <= i or d[j] > window:
                    continue
                if not np.isfinite(s.dyn[j]).any():
                    continue
                pairs.append({
                    "subject": subj_idx,
                    "i": i,
                    "j": j,
                    "h": h,
                    "dt": float(s.times[j] - s.times[i]),
                })
    return pairs
