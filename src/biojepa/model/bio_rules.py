"""Biological rule engine: differentiable penalties + non-differentiable
violation metrics for predicted disease trajectories.

Rules implemented (grounded in the AD biomarker-cascade literature, Jack et al.
2010/2013 — see docs/BIOLOGICAL_RULES.md):

R1 Causal-capacity rule ("an effect requires a cause"):
   Predicted cognitive decline rate (in standardized units/year, worse
   direction) may not exceed a patient-specific biological capacity
       capacity(f, severity) = calibrated_quantile(f, severity) + r0
   where Severity_* in [0,1] are pathology-burden proxies from measured
   amyloid / tau / neurodegeneration markers. A patient with no active
   pathology has low capacity, so predicting rapid decline for them violates
   biology: the model is penalized for effects without causes.

R2 Monotonicity ("pathology does not reverse"):
   Amyloid burden (A) and neurodegeneration markers (N) cannot improve beyond
   a noise slack on trial timescales. Penalized (and counted as violations)
   when a predicted trajectory reverses them. Clinical (C) features are
   intentionally NOT monotonic — retest/practice effects are real.

R3 Data-driven calibration:
   Capacity curves are fitted from the *training cohort only*: per clinical
   feature, capacity(severity) is the 90th percentile of observed decline
   rates within severity bins (piecewise-linear). The rule is an empirical
   envelope of real biology, not a hand-picked constant.

Applicability is explicit and dataset-dependent: the ADNI sample (baseline
A/T/N biomarkers) runs R1 fully; OASIS-2 has no A/T biomarkers, so severity_A/T
are neutral and R1 reduces to a neurodegeneration-capacity rule, while R2
applies to its longitudinal brain-volume feature. The synthetic full-cascade
study exercises every rule, including A/T/N monotonicity.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

SEVERITY_BINS = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


@dataclass
class RuleConfig:
    # noise slack in standardized units per year: allows test-retest/practice
    # variability (~0.25 sigma is typical for cognitive scales) before a
    # predicted change counts as a rule violation
    slack: float = 0.25
    decline_weight: float = 1.0    # weight of R1 in the total penalty
    monotonic_weight: float = 1.0  # weight of R2 in the total penalty
    rank_weight: float = 0.5       # weight of R3 (causal ordering across patients)
    rank_margin: float = 0.05      # allowed advantage of low-severity patients, sigma/yr
    r0: float = 0.10               # base capacity floor added to calibrated quantiles
    capacity_quantile: float = 0.90


@dataclass
class CapacityTable:
    """Piecewise-linear biological capacity(severity) per clinical feature
    (standardized-decline-rate units per year), noise-corrected, plus the
    per-feature measurement-noise envelope used as the comparison margin, and
    per-feature monotonicity slacks. All fitted from training data only."""
    bins: np.ndarray                                        # [K+1]
    values: dict[str, np.ndarray] = field(default_factory=dict)   # feature -> [K] knots
    noise_env: dict[str, float] = field(default_factory=dict)     # clinical feature -> sigma/yr
    noise_slacks: dict[str, float] = field(default_factory=dict)  # A/N feature -> sigma

    def __call__(self, feature: str, severity: float | np.ndarray) -> np.ndarray:
        knots = self.values[feature]
        xp = self.bins
        fp = np.concatenate([[knots[0]], knots])
        return np.interp(np.clip(severity, 0.0, 1.0), xp, fp)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def compute_severity(study, dyn_std_i: np.ndarray, patho_std: np.ndarray,
                     patho_status: np.ndarray) -> np.ndarray:
    """Severity proxies [Severity_A, Severity_T, Severity_N] in [0,1] for one
    (subject, visit). ADNI: from baseline pathology markers. Synthetic: from
    the current visit's dynamic A/T/N values. OASIS: N from current brain
    volume; A/T neutral (not measured in this cohort)."""
    sev = np.array([0.5, 0.5, 0.5], dtype=np.float64)

    if study.name == "adni":
        # patho columns: [CSF_ABETA_bl, PET_ABETA_bl, CSF_TAU_bl, CSF_PTAU_bl, MRI_HIPP_bl]
        a_parts, t_parts, n_parts = [], [], []
        if np.isfinite(patho_std[0]):
            a_parts.append(_sig(-patho_std[0]))   # low CSF Aβ42 = high amyloid burden
        if np.isfinite(patho_std[1]):
            a_parts.append(_sig(patho_std[1]))    # high PET SUVR = high burden
        if patho_status[0] >= 0:
            a_parts.append(float(patho_status[0]))
        if np.isfinite(patho_std[2]):
            t_parts.append(_sig(patho_std[2]))
        if np.isfinite(patho_std[3]):
            t_parts.append(_sig(patho_std[3]))
        if np.isfinite(patho_std[4]):
            n_parts.append(_sig(-patho_std[4]))   # small hippocampus = high neurodegeneration
        if a_parts: sev[0] = float(np.mean(a_parts))
        if t_parts: sev[1] = float(np.mean(t_parts))
        if n_parts: sev[2] = float(np.mean(n_parts))
    elif study.name == "synthetic":
        for k, gname in enumerate(["A", "T", "N"]):
            idx = study.dyn_group_idx.get(gname, [])
            if idx and np.isfinite(dyn_std_i[idx[0]]):
                sev[k] = float(_sig(dyn_std_i[idx[0]]))
    elif study.name == "oasis":
        idx_n = study.dyn_group_idx.get("N", [])
        if idx_n and np.isfinite(dyn_std_i[idx_n[0]]):
            sev[2] = float(_sig(-dyn_std_i[idx_n[0]]))  # low nWBV = high neurodegeneration
    return np.clip(sev, 0.0, 1.0)


def fit_capacity_table(study, pairs, dyn_std_lookup, patho_std_lookup,
                       cfg: RuleConfig) -> CapacityTable:
    """Calibrate R1 capacity curves and R2 noise slacks from training pairs.

    Calibration is PATIENT-BALANCED: statistics are aggregated per patient
    first (each patient contributes equally, regardless of how many visit-pairs
    they have), then the envelopes are fitted over per-patient values.

    dyn_std_lookup(subject_idx)[visit_idx] -> standardized dyn vector.
    patho_std_lookup(subject_idx) -> standardized baseline pathology vector.

    Stated assumptions (see docs/BIOLOGICAL_RULES.md):
      - In the lowest-severity stratum, true biological decline is ~0, so the
        observed |rate| envelope there estimates measurement noise.
      - Apparent reversals of A/N markers are measurement noise."""
    table = CapacityTable(bins=SEVERITY_BINS)

    by_pat: dict = {}
    for pair in pairs:
        s = study.subjects[pair["subject"]]
        dyn_std = dyn_std_lookup(pair["subject"])
        patho_std = patho_std_lookup(pair["subject"])
        sev_vec = compute_severity(study, dyn_std[pair["i"]], patho_std, s.patho_status)
        e = by_pat.setdefault(pair["subject"], {
            "sev": [], "rates": {f: [] for f in range(len(study.dyn_specs))},
            "reversals": {f: [] for f in range(len(study.dyn_specs))}})
        e["sev"].append(float(sev_vec.mean()))
        for f_idx, spec in enumerate(study.dyn_specs):
            cur, nxt = dyn_std[pair["i"]][f_idx], dyn_std[pair["j"]][f_idx]
            if not (np.isfinite(cur) and np.isfinite(nxt)):
                continue
            rate = ((nxt - cur) if spec.higher_is_worse else (cur - nxt)) / max(pair["dt"], 1e-3)
            e["rates"][f_idx].append(rate)
            if spec.group in ("A", "N"):
                imp = (cur - nxt) if spec.higher_is_worse else (nxt - cur)
                if imp > 0:
                    e["reversals"][f_idx].append(imp)

    patients = sorted(by_pat)
    pat_sev = np.array([np.mean(by_pat[k]["sev"]) for k in patients])
    pat_rates = {f: np.array([np.mean(by_pat[k]["rates"][f]) if by_pat[k]["rates"][f] else np.nan
                              for k in patients]) for f in range(len(study.dyn_specs))}

    for f_idx, spec in enumerate(study.dyn_specs):
        fin = np.isfinite(obs_all := pat_rates[f_idx])
        obs_f = obs_all[fin]
        sev_f = pat_sev[fin]                      # aligned with obs_f
        if len(obs_f) < 5:
            table.values[spec.name] = np.zeros(len(SEVERITY_BINS) - 1)
            table.noise_env[spec.name] = 0.0
            continue
        # Noise envelope: |rate| quantile among the lowest-severity quintile of
        # PATIENTS (stated assumption: near-zero true decline there).
        order = np.argsort(sev_f)
        n_low = max(5, len(order) // 5)
        low_slice = np.abs(obs_f[order[:n_low]])
        noise_q = float(np.quantile(low_slice, cfg.capacity_quantile)) if len(low_slice) >= 5 else 0.0
        table.noise_env[spec.name] = noise_q
        # Biological envelope: per-severity-bin 90th pct of patient-level rates
        # minus the noise component.
        knots = []
        for lo, hi in zip(SEVERITY_BINS[:-1], SEVERITY_BINS[1:]):
            m = (sev_f >= lo) & (sev_f <= hi)
            q = np.quantile(obs_f[m], cfg.capacity_quantile) if m.sum() >= 5 else np.quantile(obs_f, cfg.capacity_quantile)
            knots.append(max(q - noise_q, 0.0))
        table.values[spec.name] = np.array(knots, dtype=np.float64)

        # R2 slack: per-patient apparent-reversal mean -> RMS across patients,
        # doubled as a 2-sigma allowance, capped.
        if spec.group in ("A", "N"):
            revs = []
            for k in patients:
                rv = by_pat[k]["reversals"][f_idx]
                if rv:
                    revs.append(np.mean(rv))
            slack = 2.0 * float(np.sqrt(np.mean(np.square(revs)))) if revs else 0.0
            table.noise_slacks[spec.name] = float(min(max(cfg.slack, slack), 1.5))
    return table


def _interp1d(xp: torch.Tensor, fp: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
    """Differentiable-in-fp linear interpolation on an increasing grid."""
    xq = xq.clamp(xp[0], xp[-1])
    idx = (torch.searchsorted(xp, xq.contiguous()) - 1).clamp(0, len(xp) - 2)
    x0, x1 = xp[idx], xp[idx + 1]
    y0, y1 = fp[idx], fp[idx + 1]
    w = (xq - x0) / (x1 - x0).clamp(min=1e-8)
    return (1 - w) * y0 + w * y1


class BiologicalRuleEngine:
    """Differentiable penalties (training) + violation counters (evaluation)."""

    def __init__(self, study, capacity: CapacityTable, cfg: RuleConfig | None = None):
        self.study = study
        self.capacity = capacity
        self.cfg = cfg or RuleConfig()
        self.c_idx = study.dyn_group_idx.get("C", [])
        self.monotonic_specs = [
            (d, s) for d, s in enumerate(study.dyn_specs) if s.group in ("A", "N")
        ]
        bins = torch.as_tensor(self.capacity.bins, dtype=torch.float32)
        self._bins = bins
        self._fp = {
            name: torch.cat([torch.as_tensor(k[:1], dtype=torch.float32),
                             torch.as_tensor(k, dtype=torch.float32)])
            for name, k in self.capacity.values.items()
        }

    # ---------- training (differentiable, torch) ----------
    def penalty(self, cur_std: torch.Tensor, pred_std: torch.Tensor,
                dt: torch.Tensor, severity: torch.Tensor,
                cur_mask: torch.Tensor | None = None,
                patient_ids: list | None = None) -> dict[str, torch.Tensor]:
        """Rules constrain the PREDICTED trajectory: only the CURRENT state must
        be observed (cur_mask). Future ground truth is NOT required — the
        constraints apply exactly where clinical information is sparse.

        R3 operates on UNIQUE patients (one entry per subject), using observed
        cells only; a patient contributing several visit-pairs counts once."""
        cfg = self.cfg
        out: dict[str, torch.Tensor] = {}
        if self.c_idx:
            rates, valids = [], []
            for d in self.c_idx:
                hw = self.study.dyn_specs[d].higher_is_worse
                dec = (pred_std[:, d] - cur_std[:, d]) if hw else (cur_std[:, d] - pred_std[:, d])
                rates.append(dec / dt.clamp(min=1e-3))
                v = torch.ones_like(cur_std[:, d])
                if cur_mask is not None: v = v * cur_mask[:, d]
                valids.append(v)
            rate = torch.stack(rates, dim=1)                    # [B, nC]
            sev_mean = severity.mean(dim=1)                     # [B]
            caps = []
            for d in self.c_idx:
                name = self.study.dyn_specs[d].name
                cap_f = _interp1d(self._bins.to(pred_std.dtype),
                                  self._fp[name].to(pred_std.dtype), sev_mean)
                noise_f = self.capacity.noise_env.get(name, 0.0)
                caps.append(cap_f + cfg.r0 + noise_f)  # biological envelope + noise margin
            cap = torch.stack(caps, dim=1)                      # [B, nC]
            viol = F.relu(rate - cap)
            V = torch.stack(valids, dim=1).clamp(0.0, 1.0)      # observed cells only
            out["decline_capacity"] = cfg.decline_weight * (viol ** 2 * V).sum() / V.sum().clamp(min=1.0)

            # R3 population-level severity-decline ORDERING constraint (a
            # statistical group-level regularity, NOT individual-level
            # causality): the top pathology-severity quartile of unique
            # patients may not be predicted to decline slower than the bottom
            # quartile beyond a small margin. One entry per patient
            # (visit-pairs of the same patient are averaged), observed cells
            # only.
            if cfg.rank_weight > 0 and patient_ids is not None and rate.shape[0] >= 8:
                sev_mean = severity.mean(dim=1)                     # [B]
                rm = (rate * V).sum(1) / V.sum(1).clamp(min=1e-6)   # per-item, observed only
                # vectorized per-patient aggregation (unique patients, observed
                # cells only) — patients with no observed cells are excluded
                pid_to_i = {}
                p_idx = []
                for b_i, pid in enumerate(list(patient_ids)):
                    if pid not in pid_to_i:
                        pid_to_i[pid] = len(pid_to_i)
                    p_idx.append(pid_to_i[pid])
                p_idx = torch.tensor(p_idx, device=rate.device, dtype=torch.long)
                n_pat = len(pid_to_i)
                has_obs = (V.sum(1) > 0).to(rate.dtype)
                pat_rate = torch.zeros(n_pat, device=rate.device).index_add_(0, p_idx, rm * has_obs)
                pat_sev = torch.zeros(n_pat, device=rate.device).index_add_(0, p_idx, sev_mean * has_obs)
                pat_cnt = torch.zeros(n_pat, device=rate.device).index_add_(0, p_idx, has_obs)
                valid = pat_cnt > 0
                if int(valid.sum()) >= 8:
                    pr = pat_rate[valid] / pat_cnt[valid].clamp(min=1e-6)
                    ps_ = pat_sev[valid] / pat_cnt[valid].clamp(min=1e-6)
                    k = max(2, len(pr) // 4)
                    top = ps_.topk(k).indices
                    bot = (-ps_).topk(k).indices
                    out["rank_order"] = cfg.rank_weight * F.relu(
                        pr[bot].mean() - pr[top].mean() + cfg.rank_margin) ** 2
        mons, mvalids = [], []
        for d, spec in self.monotonic_specs:
            slack_f = self.capacity.noise_slacks.get(spec.name, cfg.slack)
            imp = (cur_std[:, d] - pred_std[:, d]) if spec.higher_is_worse else (pred_std[:, d] - cur_std[:, d])
            mons.append(F.relu(imp - slack_f * dt.clamp(min=1e-3)))
            v = torch.ones_like(cur_std[:, d])
            if cur_mask is not None: v = v * cur_mask[:, d]
            mvalids.append(v)
        if mons:
            M = torch.stack(mons, dim=1)
            V = torch.stack(mvalids, dim=1).clamp(0.0, 1.0)
            out["monotonic"] = cfg.monotonic_weight * (M ** 2 * V).sum() / V.sum().clamp(min=1.0)
        out["total"] = sum(out.values()) if out else pred_std.sum() * 0.0
        return out

    # ---------- evaluation (numpy rates) ----------
    def violation_rates(self, cur_std: np.ndarray, pred_std: np.ndarray,
                        dt: np.ndarray, severity: np.ndarray) -> dict[str, float]:
        res: dict[str, float] = {}
        cfg = self.cfg
        if self.c_idx:
            n, viol = 0, 0
            for b in range(cur_std.shape[0]):
                for d in self.c_idx:
                    if not (np.isfinite(cur_std[b, d]) and np.isfinite(pred_std[b, d])):
                        continue  # unmeasured cells are excluded, not auto-passes
                    spec = self.study.dyn_specs[d]
                    rate = ((pred_std[b, d] - cur_std[b, d]) if spec.higher_is_worse
                            else (cur_std[b, d] - pred_std[b, d])) / max(dt[b], 1e-3)
                    cap = (float(self.capacity(spec.name, severity[b].mean())) + cfg.r0
                           + self.capacity.noise_env.get(spec.name, 0.0))
                    n += 1
                    viol += rate > cap
            res["capacity_violation_rate"] = float(viol / max(n, 1))
        if self.monotonic_specs:
            n, viol = 0, 0
            for b in range(cur_std.shape[0]):
                for d, spec in self.monotonic_specs:
                    if not np.isfinite(cur_std[b, d]):
                        continue
                    slack_f = self.capacity.noise_slacks.get(spec.name, cfg.slack)
                    imp = ((cur_std[b, d] - pred_std[b, d]) if spec.higher_is_worse
                           else (pred_std[b, d] - cur_std[b, d]))
                    n += 1
                    viol += imp > slack_f * max(dt[b], 1e-3)
            res["monotonic_violation_rate"] = float(viol / max(n, 1))
        return res

    # ---------- helper: observed decline rates (subgroup analyses) ----------
    def observed_decline_rates(self, cur_std: np.ndarray, fut_std: np.ndarray,
                               dt: np.ndarray) -> np.ndarray:
        """Mean clinical decline rate per sample (positive = worsening);
        features unobserved at either end are skipped (nan-mean). Samples with
        no observed clinical cells yield NaN (callers filter them)."""
        rates = []
        for d in self.c_idx:
            spec = self.study.dyn_specs[d]
            dec = (fut_std[:, d] - cur_std[:, d]) if spec.higher_is_worse else (cur_std[:, d] - fut_std[:, d])
            rates.append(dec / np.maximum(dt, 1e-3))
        with warnings.catch_warnings():
            # all-NaN rows are a documented outcome (no observed cells), not an
            # anomaly — silence the empty-slice notice
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(np.stack(rates, axis=1), axis=1)
