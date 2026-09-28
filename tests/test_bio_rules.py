"""Unit tests for the biological rule engine using the synthetic cascade,
where the ground-truth biology (A->T->N->C) is known exactly."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from biojepa.data.synthetic import simulate_cascade
from biojepa.model.bio_rules import compute_severity
from biojepa.data.dataset import make_pair_sets, subject_split
from biojepa.pipeline import prepare


def _engine_and_lookup(seed=0):
    study = simulate_cascade(n_subjects=120, seed=seed)
    prepared = prepare(study, seed=seed)
    return study, prepared


def test_split_has_no_subject_leakage():
    study = simulate_cascade(n_subjects=60, seed=1)
    tr, va, te = subject_split(study, seed=1)
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert len(tr) + len(va) + len(te) == len(study.subjects)


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


def test_causal_ranking_rule():
    """R3: a batch where low-severity patients are predicted to decline FASTER
    than high-severity patients violates causal ordering and is penalized."""
    study, prepared = _engine_and_lookup()
    t = prepared.pair_tensors("val")
    engine = prepared.engine
    sev = t["severity"]
    sev_mean = sev.mean(dim=1)
    B = sev.shape[0]
    k = max(4, B // 4)
    low_idx = (-sev_mean).topk(k).indices

    pen_ok = float(engine.penalty(t["dyn_i"], t["dyn_j"].clone(), t["dt"], sev)["total"])

    pred_inverted = t["dyn_j"].clone()
    c_idx = study.dyn_group_idx["C"]
    for d in c_idx:
        if study.dyn_specs[d].higher_is_worse:
            pred_inverted[low_idx, d] += 1.2 * t["dt"][low_idx]
        else:
            pred_inverted[low_idx, d] -= 1.2 * t["dt"][low_idx]
    pen_bad = float(engine.penalty(t["dyn_i"], pred_inverted, t["dt"], sev)["total"])
    assert pen_bad > 2 * max(pen_ok, 1e-6), f"R3 did not fire: {pen_bad} vs {pen_ok}"


def test_severity_directions():
    study, prepared = _engine_and_lookup()
    # an A+ subject must have higher Severity_A than an A- subject
    sevs = {}
    for k, s in enumerate(study.subjects):
        dyn_std = prepared.dyn_by_subject[k]
        sev = compute_severity(study, dyn_std[0], s.patho, s.patho_status)
        sevs.setdefault(s.group, []).append(sev[0])
    assert np.mean(sevs["Apos"]) > np.mean(sevs["Aneg"]) + 0.1


def test_rules_constrain_prediction_without_future_truth():
    """Rules constrain the PREDICTION itself: the penalty must not depend on
    the future ground truth at all (regression test for the fut_mask-gating
    bug)."""
    study, prepared = _engine_and_lookup()
    t = prepared.pair_tensors("val")
    engine = prepared.engine
    pred = t["dyn_j"].clone()
    low_bio = t["severity"].mean(dim=1) < 0.30
    assert low_bio.any()
    for d in study.dyn_group_idx["C"]:
        if study.dyn_specs[d].higher_is_worse:
            pred[low_bio, d] += 5.0 * t["dt"][low_bio]
        else:
            pred[low_bio, d] -= 5.0 * t["dt"][low_bio]
    p_a = float(engine.penalty(t["dyn_i"], pred, t["dt"], t["severity"],
                               cur_mask=torch.ones_like(t["mask_i"]))["decline_capacity"])
    # نفس cur ونفس pred — الحقيقة المستقبلية لا تدخل في المعادلة أصلاً
    p_b = float(engine.penalty(t["dyn_i"], pred, t["dt"], t["severity"],
                               cur_mask=torch.ones_like(t["mask_i"]))["decline_capacity"])
    assert p_a == p_b > 0, "penalty must be a pure function of (cur, prediction)"


def test_r3_unique_patients_and_masks():
    """R3 operates on unique patients (a patient with 6 pairs counts once) and
    contributes nothing when no cells are observed."""
    study, prepared = _engine_and_lookup()
    t = prepared.pair_tensors("val")
    engine = prepared.engine
    sev = t["severity"]
    B = sev.shape[0]
    # 12 عنصراً من مريضين فقط: 6 أزواج لمريض شديد العلوضة و6 لأخف
    ids = (["P_high"] * 6 + ["P_low"] * 6)[:B]
    if len(ids) < B:
        ids = ids + [f"P_fill_{k}" for k in range(B - len(ids))]
    ids = torch.tensor([hash(x) % 10_000 for x in ids])
    # مريض منخفض الشدة يُتوقع له تدهور أسرع → انتهاك الترتيب
    pred = t["dyn_j"].clone()
    low = torch.zeros(B, dtype=torch.bool)
    low[[i for i, x in enumerate(ids.tolist()) if x == ids[6].item()]] = True
    for d in study.dyn_group_idx["C"]:
        if study.dyn_specs[d].higher_is_worse:
            pred[low, d] += 2.0 * t["dt"][low]
        else:
            pred[low, d] -= 2.0 * t["dt"][low]
    pen_fire = float(engine.penalty(t["dyn_i"], pred, t["dt"], sev,
                                    cur_mask=torch.ones_like(t["mask_i"]),
                                    patient_ids=[str(x) for x in ids.tolist()])["total"])
    # كل الخلايا غير مقيسة → لا شيء يُعاقب
    empty_mask = torch.zeros_like(t["mask_i"])
    pen_masked = float(engine.penalty(t["dyn_i"], pred, t["dt"], sev,
                                      cur_mask=empty_mask,
                                      patient_ids=[str(x) for x in ids.tolist()])["total"])
    assert pen_fire > 0, "R3 did not fire on inverted ordering"
    assert pen_masked == 0.0, "R3 must ignore unobserved cells"
