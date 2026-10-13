"""Regression tests for the October 2026 scientific audit fixes."""
import inspect
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from biojepa.data.synthetic import simulate_cascade
from biojepa.evaluate import (carry_forward_std, conversion_queries, degraded_dyn_lookup,
                              forecast, jepa_predict, rollout_metrics)
from biojepa.model.baselines import carry_forward_predict, train_gru
from biojepa.model.jepa import mlp
from biojepa.pipeline import prepare
from biojepa.run_experiments import compare_models, to_jsonable
from biojepa.train import train_jepa
from biojepa.utils import Standardizer


@pytest.fixture(scope="module")
def prepared():
    return prepare(simulate_cascade(n_subjects=120, seed=0), seed=0)


def test_mlp_activation_placed_by_position_not_width():
    names = [type(m).__name__ for m in mlp([8, 4, 4])]
    assert names == ["Linear", "SiLU", "LayerNorm", "Linear"]
    names = [type(m).__name__ for m in mlp([8, 16, 16, 3])]
    assert names.count("SiLU") == 2 and names[-1] == "Linear"


def test_standardizer_all_missing_column_stays_missing_only():
    X = np.array([[1.0, np.nan], [3.0, np.nan]], dtype=np.float32)
    s = Standardizer.fit(X)
    Z = s.transform(np.array([[2.0, 5.0], [2.0, np.nan]], dtype=np.float32))
    assert np.isfinite(Z[0]).all()            # observed values stay usable
    assert Z[0, 0] == pytest.approx(0.0)
    assert np.isnan(Z[1, 1])                  # missing stays missing


def test_v3_degraded_path_is_anchored(prepared):
    """0% extra missingness must reproduce the normal predictions exactly
    (the stress path used to drop the delta-mode anchor)."""
    model, _, _ = train_jepa(prepared, seed=0, epochs=3, patience=3, version="v3")
    cfg = {"kind": "jepa", "version": "v3"}
    normal = jepa_predict(model, prepared, "test", version="v3")
    via_forecast = forecast(cfg, model, prepared, prepared.pairs["test"])
    degraded0 = forecast(cfg, model, prepared, prepared.pairs["test"],
                         dyn_by_subject=degraded_dyn_lookup(prepared, 0.0))
    np.testing.assert_allclose(normal, via_forecast, atol=1e-5)
    np.testing.assert_allclose(normal, degraded0, atol=1e-5)


def test_forecast_never_reads_the_target_visit(prepared):
    """Queries without a target visit work for every model kind."""
    queries = [{"subject": p["subject"], "i": p["i"], "dt": p["dt"], "h": p["h"]}
               for p in prepared.pairs["test"][:20]]
    D = len(prepared.study.dyn_specs)
    assert forecast({"kind": "carry"}, None, prepared, queries).shape == (20, D)
    model, _, _ = train_jepa(prepared, seed=0, epochs=2, patience=2, version="v2")
    assert forecast({"kind": "jepa", "version": "v2"}, model, prepared, queries).shape == (20, D)


def test_carry_forward_std_matches_raw_carry_forward(prepared):
    pairs = prepared.pairs["test"]
    train_ids = sorted({p["subject"] for p in prepared.pairs["train"]})
    raw, _, _, _ = carry_forward_predict(prepared.study, pairs, train_ids)
    std = carry_forward_std(prepared.dyn_by_subject, pairs)
    observed = np.array([[np.isfinite(prepared.study.subjects[p["subject"]].dyn[: p["i"] + 1, d]).any()
                          for d in range(raw.shape[1])] for p in pairs])
    np.testing.assert_allclose(prepared.dyn_std.inverse(std)[observed], raw[observed], rtol=1e-4)


def test_gru_gets_the_jepa_training_budget():
    g = inspect.signature(train_gru).parameters
    j = inspect.signature(train_jepa).parameters
    for k in ("epochs", "patience", "lr", "batch", "min_batch"):
        assert g[k].default == j[k].default, k


def test_synthetic_cognition_declines_only_with_neurodegeneration():
    study = simulate_cascade(n_subjects=400, seed=1)
    mmse = [d for d, s in enumerate(study.dyn_specs) if s.name == "MMSE_like"][0]
    drop = {"Aneg": [], "Apos": []}
    for s in study.subjects:
        if np.isfinite(s.dyn[[0, -1], mmse]).all():
            drop[s.group].append(s.dyn[0, mmse] - s.dyn[-1, mmse])
    assert abs(np.mean(drop["Aneg"])) < 0.2           # no true decline without pathology
    assert np.mean(drop["Apos"]) > np.mean(drop["Aneg"]) + 0.2
    assert sum(s.converts for s in study.subjects) > 0


def test_conversion_design(prepared):
    study = prepared.study
    subs = study.subjects
    saved = [(s.converts, s.time_under_risk, s.dementia_at_baseline) for s in subs[:6]]
    try:
        cases = [  # (converts, tur, demented at baseline) -> expected label or None
            (True, 2.0, False, 1),      # converted inside the window
            (True, 5.0, False, 0),      # converted after the window -> event-free control
            (False, 4.0, False, 0),     # followed past the window, no event
            (False, 1.5, False, None),  # censored before the window closes
            (True, 1.0, True, None),    # demented at baseline
            (False, None, False, None),  # unknown timing
        ]
        for s, (c, tur, dem, _) in zip(subs, cases):
            s.converts, s.time_under_risk, s.dementia_at_baseline = c, tur, dem
        queries, labels = conversion_queries(prepared, range(6), window=3.0)
        got = {q["subject"]: y for q, y in zip(queries, labels)}
        conv_f = [d for d, sp in enumerate(study.dyn_specs) if sp.name == "CDR_like"][0]
        for k, case in enumerate(cases):
            expected = case[3] if np.isfinite(prepared.dyn_by_subject[k][0, conv_f]) else None
            assert got.get(k) == expected, (k, case)
        assert all(q["i"] == 0 and q["dt"] == 3.0 and "j" not in q for q in queries)
    finally:
        for s, (c, tur, dem) in zip(subs, saved):
            s.converts, s.time_under_risk, s.dementia_at_baseline = c, tur, dem


def _fake_cv(n_seeds, n_patients=40, pairs_per_patient=3, effect=-0.1, seed=0):
    rng = np.random.RandomState(seed)
    errors = {m: {"F": defaultdict(list)} for m in ("ours", "opp")}
    for p in range(n_patients):
        base = rng.rand()
        for k in range(pairs_per_patient):
            key = (f"P{p}", k, k + 1, 1.0)
            for _ in range(n_seeds):
                noise = rng.normal(0, 0.01)
                errors["opp"]["F"][key].append(base + noise)
                errors["ours"]["F"][key].append(base + effect + rng.normal(0, 0.3) + noise)
    return {"errors": errors, "n_seeds": n_seeds,
            "conv_scores": {"ours": {}, "opp": {}}, "conv_labels": {}}


def test_patients_are_counted_once_across_seeds():
    one = compare_models(_fake_cv(1), "ours", ["opp"], ["F"], n_boot=500)
    three = compare_models(_fake_cv(3), "ours", ["opp"], ["F"], n_boot=500)
    r1, r3 = one["ours_vs_opp_F_MAE"], three["ours_vs_opp_F_MAE"]
    assert r1["n_clusters"] == r3["n_clusters"] == 40   # never 40 x seeds
    assert r3["n_pairs"] == 120


def test_seed_count_mismatch_is_rejected():
    cv = _fake_cv(2)
    cv["errors"]["ours"]["F"][("P0", 0, 1, 1.0)].append(0.0)
    with pytest.raises(ValueError, match="seeds"):
        compare_models(cv, "ours", ["opp"], ["F"], n_boot=200)


def test_rollout_scoring_is_identical_for_every_model(prepared):
    truth, cur = prepared.truth_std("test")
    steps = [np.nan_to_num(cur) + 0.1 * (k + 1) for k in range(3)]
    res = rollout_metrics(prepared, "test", steps)
    for k in [k for k in res if k.endswith("_stepwise")]:
        cum = k.replace("_stepwise", "_cumulative")
        assert res[k]["step1"] == res[cum]["step1"]   # same reference at step 1
        assert set(res[k]) == {"step1", "step2", "step3"}


def test_strict_json_has_no_nan():
    out = to_jsonable({"a": np.nan, "b": [np.float32(1.5), np.inf], "c": np.int64(3)})
    assert out == {"a": None, "b": [1.5, None], "c": 3}


def test_oasis_timing_and_baseline_dementia(tmp_path):
    from biojepa.data.oasis import load_oasis
    rows = []
    def add(sid, group, visits):
        for v, (days, cdr) in enumerate(visits, start=1):
            rows.append({"Subject ID": sid, "MRI ID": f"{sid}_MR{v}", "Group": group, "Visit": v,
                         "MR Delay": days, "M/F": "F", "Hand": "R", "Age": 75, "EDUC": 14,
                         "SES": 2, "MMSE": 28, "CDR": cdr, "eTIV": 1500, "nWBV": 0.72, "ASF": 1.2})
    add("OAS2_0001", "Nondemented", [(0, 0.0), (1200, 0.0)])
    add("OAS2_0002", "Converted", [(0, 0.0), (700, 0.0), (1400, 0.5)])
    add("OAS2_0003", "Demented", [(0, 1.0), (800, 1.0)])
    add("OAS2_0004", "Converted", [(0, 0.5), (900, 0.5)])
    path = tmp_path / "oasis.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    subs = {s.sid: s for s in load_oasis(str(path)).subjects}
    assert subs["OAS2_0001"].time_under_risk == pytest.approx(1200 / 365.25, rel=1e-5)
    assert not subs["OAS2_0001"].converts
    assert subs["OAS2_0002"].converts
    assert subs["OAS2_0002"].time_under_risk == pytest.approx(1400 / 365.25, rel=1e-5)
    assert subs["OAS2_0003"].dementia_at_baseline and subs["OAS2_0003"].time_under_risk is None
    assert subs["OAS2_0004"].time_under_risk is None    # CDR >= 0.5 already at baseline
