"""Unit tests for the biological rule engine using the synthetic cascade,
where the ground-truth biology (A->T->N->C) is known exactly."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
import torch

from biojepa.data.synthetic import simulate_cascade
from biojepa.model.bio_rules import BiologicalRuleEngine, RuleConfig, compute_severity
from biojepa.data.dataset import make_pair_sets, subject_split
from biojepa.pipeline import prepare


def _engine_and_lookup(seed=0):
    study = simulate_cascade(n_subjects=120, seed=seed)
    prepared = prepare(study, seed=seed)
    return study, prepared


def test_split_has_no_subject_leakage():
    study = simulate_cascade(n_subjects=60, seed=1)
    for stratify in (True, False):
        tr, va, te = subject_split(study, seed=1, stratify=stratify)
        assert not (tr & va) and not (tr & te) and not (va & te)
        assert len(tr) + len(va) + len(te) == len(study.subjects)
        assert len(va) > 0 and len(te) > 0, f"empty val/test with stratify={stratify}"


def test_pairs_are_temporally_valid():
    study = simulate_cascade(n_subjects=40, seed=2)
    for p in make_pair_sets(study):
        s = study.subjects[p["subject"]]
        assert p["j"] > p["i"]
        assert abs((s.times[p["i"]] + p["h"]) - s.times[p["j"]]) <= 0.25 + 1e-6
        assert 0 < p["dt"] <= 3.25


def test_capacity_increases_with_pathology_severity():
    study, prepared = _engine_and_lookup()
    cap = prepared.engine.capacity
    for spec in study.dyn_specs:
        if spec.group != "C":
            continue
        low = float(cap(spec.name, 0.1))
        high = float(cap(spec.name, 0.9))
        assert high > low, f"capacity must grow with severity for {spec.name}"


def test_compliant_predictions_barely_penalized():
    study, prepared = _engine_and_lookup()
    t = prepared.pair_tensors("val")
    engine = prepared.engine
    noise = torch.randn_like(t["dyn_j"]) * 0.05
    pen_compliant = float(engine.penalty(t["dyn_i"], t["dyn_j"] + noise, t["dt"], t["severity"])["total"])
    # violating: large unsupported decline on low-pathology subjects
    pred = t["dyn_j"].clone()
    low_bio = t["severity"].mean(dim=1) < 0.30
    assert low_bio.any(), "synthetic study must include low-pathology subjects"
    c_idx = study.dyn_group_idx["C"]
    for d in c_idx:
        if study.dyn_specs[d].higher_is_worse:
            pred[low_bio, d] += 5.0 * t["dt"][low_bio]
        else:
            pred[low_bio, d] -= 5.0 * t["dt"][low_bio]
    pen_violating = float(engine.penalty(t["dyn_i"], pred, t["dt"], t["severity"])["total"])
    assert pen_compliant < 0.15, f"compliant predictions penalized: {pen_compliant}"
    assert pen_violating > 10 * max(pen_compliant, 1e-6), \
        f"violating {pen_violating} not distinguished from compliant {pen_compliant}"


def test_effect_without_cause_is_penalized():
    study, prepared = _engine_and_lookup()
    t = prepared.pair_tensors("val")
    engine = prepared.engine
    pred = t["dyn_j"].clone()
    # force a large clinical decline on low-pathology (A-) samples only
    low_bio = t["severity"].mean(dim=1) < 0.30
    assert low_bio.any(), "synthetic study must include low-pathology subjects"
    c_idx = study.dyn_group_idx["C"]
    for d in c_idx:
        if study.dyn_specs[d].higher_is_worse:
            pred[low_bio, d] += 5.0 * t["dt"][low_bio]  # +1.5 sigma/year decline
        else:
            pred[low_bio, d] -= 5.0 * t["dt"][low_bio]
    pen_violating = engine.penalty(t["dyn_i"], pred, t["dt"], t["severity"])
    pen_truth = engine.penalty(t["dyn_i"], t["dyn_j"], t["dt"], t["severity"])
    assert float(pen_violating["decline_capacity"]) > 5 * max(float(pen_truth["decline_capacity"]), 1e-6)
    rates = engine.violation_rates(t["dyn_i"].numpy(), pred.numpy(),
                                   t["dt"].numpy(), t["severity"].numpy())
    assert rates["capacity_violation_rate"] > 0.10


def test_pathology_reversal_is_penalized():
    study, prepared = _engine_and_lookup()
    t = prepared.pair_tensors("val")
    engine = prepared.engine
    pred = t["dyn_j"].clone()
    # amyloid/tau/atrophy cannot meaningfully reverse: push group A and N down
    # (magnitude scaled with horizon so the slack*dt threshold is exceeded)
    for g in ("A", "N"):
        for d in study.dyn_group_idx.get(g, []):
            pred[:, d] -= 0.5 + 1.0 * t["dt"]
    pen_violating = engine.penalty(t["dyn_i"], pred, t["dt"], t["severity"])
    pen_truth = engine.penalty(t["dyn_i"], t["dyn_j"], t["dt"], t["severity"])
    assert float(pen_violating["monotonic"]) > 5 * max(float(pen_truth["monotonic"]), 1e-6)
    rates = engine.violation_rates(t["dyn_i"].numpy(), pred.numpy(),
                                   t["dt"].numpy(), t["severity"].numpy())
    assert rates["monotonic_violation_rate"] > 0.5


def _r3_only_engine(study, prepared):
    """Engine with R1/R2 switched off, so a test isolates R3."""
    return BiologicalRuleEngine(study, prepared.engine.capacity,
                                RuleConfig(decline_weight=0.0, monotonic_weight=0.0, rank_weight=1.0))


def _with_rates(study, t, rates):
    """Predictions declining at `rates` (sigma/yr, worse direction) on every
    clinical feature, starting from the current state."""
    pred = t["dyn_i"].clone()
    for d in study.dyn_group_idx["C"]:
        sign = 1.0 if study.dyn_specs[d].higher_is_worse else -1.0
        pred[:, d] = t["dyn_i"][:, d] + sign * rates * t["dt"]
    return pred


def test_causal_ranking_rule():
    """R3 alone: predicting faster decline for LOW-severity patients than for
    high-severity ones is penalized; the severity-consistent ordering is not."""
    study, prepared = _engine_and_lookup()
    t = prepared.pair_tensors("val")
    engine = _r3_only_engine(study, prepared)
    sev = t["severity"]
    s = sev.mean(dim=1)
    pids = t["patient_ids"]
    consistent = engine.penalty(t["dyn_i"], _with_rates(study, t, 2.0 * s), t["dt"], sev,
                                patient_ids=pids)
    inverted = engine.penalty(t["dyn_i"], _with_rates(study, t, 2.0 * (1.0 - s)), t["dt"], sev,
                              patient_ids=pids)
    assert "rank_order" in consistent and "rank_order" in inverted
    assert float(consistent["rank_order"]) == 0.0
    assert float(inverted["rank_order"]) > 0.0
    # R3 aggregates per patient, so it needs patient ids
    no_ids = engine.penalty(t["dyn_i"], _with_rates(study, t, 2.0 * (1.0 - s)), t["dt"], sev)
    assert "rank_order" not in no_ids


def test_severity_directions():
    study, prepared = _engine_and_lookup()
    # an A+ subject must have higher Severity_A than an A- subject
    sevs = {}
    for k, s in enumerate(study.subjects):
        dyn_std = prepared.dyn_by_subject[k]
        sev = compute_severity(study, dyn_std[0], s.patho, s.patho_status)
        sevs.setdefault(s.group, []).append(sev[0])
    assert np.mean(sevs["Apos"]) > np.mean(sevs["Aneg"]) + 0.1


def test_bio_penalty_ignores_future_truth():
    """Rules constrain the PREDICTION: inside the model loss the biological
    penalty must not change when the future ground truth is scrambled or
    hidden (regression test for the fut_mask-gating bug)."""
    from biojepa.model.jepa import BioJEPA2
    study, prepared = _engine_and_lookup()
    t = prepared.pair_tensors("val")
    torch.manual_seed(0)
    model = BioJEPA2(prepared.step_dim, prepared.x_context.shape[1], len(study.dyn_specs))
    common = (t["seq_i"], t["len_i"], t["seq_j"], t["len_j"], t["ctx"], t["dt_feat"])
    kw = dict(rule_engine=prepared.engine, severity=t["severity"], dt=t["dt"], w_bio=1.0,
              patient_ids=t["patient_ids"])
    real = model.loss(*common, t["dyn_j"], t["mask_j"], t["dyn_i"], t["mask_i"], **kw)
    scrambled = model.loss(*common, 5.0 * torch.randn_like(t["dyn_j"]), torch.zeros_like(t["mask_j"]),
                           t["dyn_i"], t["mask_i"], **kw)
    assert real["sup_future"].item() != scrambled["sup_future"].item()  # the truth did change
    assert real["bio_total"].item() == scrambled["bio_total"].item() > 0


def test_r3_counts_each_patient_once_and_ignores_unobserved_cells():
    study, prepared = _engine_and_lookup()
    t = prepared.pair_tensors("val")
    engine = _r3_only_engine(study, prepared)
    sev = t["severity"]
    pred = _with_rates(study, t, 2.0 * (1.0 - sev.mean(dim=1)))  # inverted ordering
    pids = list(t["patient_ids"])
    base = float(engine.penalty(t["dyn_i"], pred, t["dt"], sev, patient_ids=pids)["rank_order"])
    assert base > 0
    # repeat every item of one patient 5 more times: per-patient means are
    # unchanged, so a unique-patient rule must give the identical penalty
    extra = [k for k, p in enumerate(pids) if p == pids[0]] * 5
    idx = torch.cat([torch.arange(len(pids)), torch.tensor(extra, dtype=torch.long)])
    dup = float(engine.penalty(t["dyn_i"][idx], pred[idx], t["dt"][idx], sev[idx],
                               patient_ids=[pids[k] for k in idx.tolist()])["rank_order"])
    assert dup == pytest.approx(base, rel=1e-5)
    # every current cell unobserved -> no patient enters R3, nothing penalized
    masked = engine.penalty(t["dyn_i"], pred, t["dt"], sev,
                            cur_mask=torch.zeros_like(t["mask_i"]), patient_ids=pids)
    assert "rank_order" not in masked
    assert float(masked["total"]) == 0.0


def test_r2_slack_is_absolute_not_per_year():
    """Measurement noise does not grow with the gap between visits: the same
    reversal is tolerated (or flagged) identically at 6 months and 3 years."""
    study, prepared = _engine_and_lookup()
    engine = prepared.engine
    cur = np.zeros((4, len(study.dyn_specs)))
    sev = np.full((4, 3), 0.5)
    for factor, offset, expected in ((0.5, 0.0, 0.0), (1.0, 0.1, 1.0)):
        pred = cur.copy()
        for d, spec in engine.monotonic_specs:   # reverse every A/N marker
            amount = factor * engine.capacity.noise_slacks[spec.name] + offset
            pred[:, d] = (-1.0 if spec.higher_is_worse else 1.0) * amount
        for dt in (0.5, 3.0):
            r = engine.violation_rates(cur, pred, np.full(4, dt), sev)
            assert r["monotonic_violation_rate"] == expected, (factor, dt)
            pen = engine.penalty(torch.as_tensor(cur, dtype=torch.float32),
                                 torch.as_tensor(pred, dtype=torch.float32),
                                 torch.full((4,), dt), torch.as_tensor(sev, dtype=torch.float32))
            assert (float(pen["monotonic"]) > 0) == bool(expected), (factor, dt)


def test_violation_rates_skip_unmeasured_predictions():
    study, prepared = _engine_and_lookup()
    engine = prepared.engine
    cur = np.zeros((2, len(study.dyn_specs)))
    pred = cur.copy()
    for d, spec in engine.monotonic_specs:                        # clear reversals
        pred[0, d] = (-1.0 if spec.higher_is_worse else 1.0) * 10.0
    pred[1, :] = np.nan                                           # nothing measured
    r = engine.violation_rates(cur, pred, np.ones(2), np.full((2, 3), 0.5))
    assert r["monotonic_violation_rate"] == 1.0   # NaN row excluded, not a pass
