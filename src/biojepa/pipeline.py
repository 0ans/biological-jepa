"""Prepares a Study for modeling: standardizers (train-fit), masked input
vectors (v1), visit-history sequences (v2), per-pair tensors, severity proxies
and the calibrated rule engine. Supports both a single subject split and
subject-level K-fold splits."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .data.dataset import Study, make_pair_sets, subject_split
from .model.bio_rules import (BiologicalRuleEngine, RuleConfig,
                              compute_severity, fit_capacity_table)
from .utils import Standardizer, fourier_time

HORIZONS = (1.0, 2.0, 3.0)
DT_DIM = 8


@dataclass
class Prepared:
    study: Study
    dyn_std: Standardizer
    patho_std: Standardizer | None
    static_std: Standardizer
    dyn_by_subject: list[np.ndarray]      # standardized dyn per subject (NaN kept)
    x_context: np.ndarray                 # [n_subj, 2P+K+S]
    in_dim: int
    step_dim: int                         # 2D + dt_dim (per-visit step vector, v2)
    pairs: dict[str, list[dict]]          # train/val/test
    severity: dict[str, np.ndarray]       # [n_pairs, 3] per split
    engine: BiologicalRuleEngine
    _cache: dict = field(default_factory=dict, repr=False)

    def visit_steps(self, subject_idx: int, upto: int) -> np.ndarray:
        """[n_visits, 2D+dt_dim] step vectors for visits 0..upto (inclusive)."""
        s = self.study.subjects[subject_idx]
        dyn_std = self.dyn_by_subject[subject_idx]
        steps, prev = [], None
        for t, vt in enumerate(s.times[: upto + 1]):
            gap = 0.0 if prev is None else vt - prev
            mask = np.isfinite(dyn_std[t]).astype(np.float32)
            vals = np.nan_to_num(dyn_std[t]).astype(np.float32)
            dtf = fourier_time(np.array([max(gap, 1e-3)]), DT_DIM // 2)[0]
            steps.append(np.concatenate([vals, mask, dtf]))
            prev = vt
        return np.array(steps, dtype=np.float32)

    def _build_split(self, split: str):
        pairs = self.pairs[split]
        D = len(self.study.dyn_specs)
        n = len(pairs)
        L = max(p["i"] + 1 for p in pairs)
        Lj = max(p["j"] + 1 for p in pairs)
        xs = np.zeros((n, 2 * D + self.x_context.shape[1]), dtype=np.float32)
        xjs = np.zeros_like(xs)
        seq_i = np.zeros((n, L, self.step_dim), dtype=np.float32)
        seq_j = np.zeros((n, Lj, self.step_dim), dtype=np.float32)
        len_i = np.zeros(n, dtype=np.int64)
        len_j = np.zeros(n, dtype=np.int64)
        ctx = np.zeros((n, self.x_context.shape[1]), dtype=np.float32)
        dts = np.zeros(n, dtype=np.float32)
        for p_i, pair in enumerate(pairs):
            s_idx = pair["subject"]
            di = np.nan_to_num(self.dyn_by_subject[s_idx][pair["i"]], nan=0.0)
            mi = np.isfinite(self.dyn_by_subject[s_idx][pair["i"]]).astype(np.float32)
            dj = np.nan_to_num(self.dyn_by_subject[s_idx][pair["j"]], nan=0.0)
            mj = np.isfinite(self.dyn_by_subject[s_idx][pair["j"]]).astype(np.float32)
            xs[p_i] = np.concatenate([di, mi, self.x_context[s_idx]])
            xjs[p_i] = np.concatenate([dj, mj, self.x_context[s_idx]])
            steps_i = self.visit_steps(s_idx, pair["i"])
            steps_j = self.visit_steps(s_idx, pair["j"])
            seq_i[p_i, : len(steps_i)] = steps_i
            seq_j[p_i, : len(steps_j)] = steps_j
            len_i[p_i] = len(steps_i)
            len_j[p_i] = len(steps_j)
            ctx[p_i] = self.x_context[s_idx]
            dts[p_i] = pair["dt"]
        four = fourier_time(dts)
        return {
            "x_i": torch.as_tensor(xs),
            "x_j": torch.as_tensor(xjs),
            "ctx": torch.as_tensor(ctx),
            "seq_i": torch.as_tensor(seq_i),
            "len_i": torch.as_tensor(len_i),
            "seq_j": torch.as_tensor(seq_j),
            "len_j": torch.as_tensor(len_j),
            "dt_feat": torch.as_tensor(four),
            "dt": torch.as_tensor(dts),
            "dyn_j": torch.as_tensor(np.stack([
                np.nan_to_num(self.dyn_by_subject[p["subject"]][p["j"]], nan=0.0) for p in pairs])),
            "mask_j": torch.as_tensor(np.stack([
                np.isfinite(self.dyn_by_subject[p["subject"]][p["j"]]).astype(np.float32) for p in pairs])),
            "dyn_i": torch.as_tensor(np.stack([
                np.nan_to_num(self.dyn_by_subject[p["subject"]][p["i"]], nan=0.0) for p in pairs])),
            "mask_i": torch.as_tensor(np.stack([
                np.isfinite(self.dyn_by_subject[p["subject"]][p["i"]]).astype(np.float32) for p in pairs])),
            "severity": torch.as_tensor(self.severity[split], dtype=torch.float32),
            "patient_ids": [self.study.subjects[pp["subject"]].sid for pp in pairs],
            "pair_index": pairs,
        }

    def pair_tensors(self, split: str, device=torch.device("cpu")) -> dict[str, torch.Tensor]:
        if split not in self._cache:
            self._cache[split] = self._build_split(split)
        out = self._cache[split]
        if device.type != "cpu":
            out = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in out.items()}
        return out

    def truth_std(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        """(truth, current) standardized dyn arrays for a split's pairs."""
        pairs = self.pairs[split]
        truth = np.stack([self.dyn_by_subject[p["subject"]][p["j"]] for p in pairs])
        cur = np.stack([self.dyn_by_subject[p["subject"]][p["i"]] for p in pairs])
        return truth, cur


def prepare(study: Study, seed: int | None = None, splits=None,
            rule_cfg: RuleConfig | None = None) -> Prepared:
    """splits: optional (train_ids, val_ids, test_ids) tuple (e.g. from
    subject_kfold); otherwise a stratified 70/15/15 split by `seed`."""
    if splits is None:
        splits = subject_split(study, seed=seed or 0)
    train_ids, val_ids, test_ids = splits
    split_map = {"train": train_ids, "val": val_ids, "test": test_ids}

    # standardizers fit on TRAIN subjects only
    train_dyn = np.concatenate([study.subjects[k].dyn for k in sorted(train_ids)], axis=0)
    dyn_std = Standardizer.fit(train_dyn)
    static_mat = np.stack([s.static for s in study.subjects])
    static_std = Standardizer.fit(static_mat[sorted(train_ids)])
    if len(study.patho_names):
        patho_mat = np.stack([s.patho for s in study.subjects])
        patho_std = Standardizer.fit(patho_mat[sorted(train_ids)])
        # standardized pathology per subject — compute_severity expects
        # standardized values (raw CSF/volumes saturate the sigmoid transform)
        patho_std_by_subject = patho_std.transform(patho_mat)
    else:
        patho_std = None
        patho_std_by_subject = None

    dyn_by_subject = [dyn_std.transform(s.dyn) for s in study.subjects]

    ctx_cols = []
    if patho_std is not None:
        pv = patho_std.transform(patho_mat)
        ctx_cols += [np.nan_to_num(pv, nan=0.0), np.isfinite(pv).astype(np.float32)]
        status = np.stack([s.patho_status for s in study.subjects])
        ctx_cols.append(np.nan_to_num(status, nan=0.0).astype(np.float32))
    sv = static_std.transform(static_mat)
    ctx_cols.append(np.nan_to_num(sv, nan=0.0).astype(np.float32))
    x_context = np.concatenate(ctx_cols, axis=1).astype(np.float32)

    all_pairs = make_pair_sets(study, horizons=HORIZONS)
    pairs = {name: [p for p in all_pairs if p["subject"] in ids]
             for name, ids in split_map.items()}

    severity = {}
    for name, plist in pairs.items():
        sev = np.zeros((len(plist), 3))
        for p_i, pair in enumerate(plist):
            s = study.subjects[pair["subject"]]
            sev[p_i] = compute_severity(study, dyn_by_subject[pair["subject"]][pair["i"]],
                                        patho_std_by_subject[pair["subject"]]
                                        if patho_std_by_subject is not None else s.patho,
                                        s.patho_status)
        severity[name] = sev

    cfg = rule_cfg or RuleConfig()

    def dyn_lookup(subject_idx):
        return dyn_by_subject[subject_idx]

    def patho_lookup(subject_idx):
        return patho_std_by_subject[subject_idx] if patho_std_by_subject is not None else study.subjects[subject_idx].patho

    capacity = fit_capacity_table(study, pairs["train"], dyn_lookup, patho_lookup, cfg)
    engine = BiologicalRuleEngine(study, capacity, cfg)

    return Prepared(
        study=study, dyn_std=dyn_std, patho_std=patho_std, static_std=static_std,
        dyn_by_subject=dyn_by_subject, x_context=x_context,
        in_dim=x_context.shape[1] + 2 * len(study.dyn_specs),
        step_dim=2 * len(study.dyn_specs) + DT_DIM,
        pairs=pairs, severity=severity, engine=engine,
    )
