"""End-to-end check of every experiment entry point on the synthetic cascade
(tiny budgets — this verifies the pipeline, it produces no evidence)."""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from biojepa import run_experiments as rx

ROOT = os.path.join(os.path.dirname(__file__), "..")


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    out = tmp_path_factory.mktemp("exp")
    res, _ = rx.run_study("synthetic", seeds=(0, 1), k_folds=3, out_root=str(out),
                          max_epochs=2, n_boot=200)
    hard = rx.run_stress_and_rollouts("synthetic", seed=1, out_root=str(out), max_epochs=2)
    rx.make_figures(res, hard, "synthetic", out_root=str(out))
    return out, res, hard


def test_results_file_is_strict_json_with_patient_level_inference(results):
    out, res, _ = results
    with open(out / "synthetic" / "results.json") as f:
        stored = json.load(f, parse_constant=lambda c: pytest.fail(f"non-JSON constant {c}"))
    n_subjects = stored["manifest"]["n_subjects"]
    for k, v in stored["significance"].items():
        if k.endswith("_MAE"):
            assert v["n_clusters"] <= n_subjects     # patients counted once, not per seed
    assert stored["manifest"]["primary_model"] == rx.PRIMARY_MODEL
    assert stored["significance"]["_family"]["n_tests"] == \
        (len(stored["n_patients_with_test_pairs"]) + 1) * len(rx.COMPARATORS)
    assert any("auc" in v for v in stored["conversion_pooled"].values())


def test_every_model_reports_observed_reference_violations(results):
    _, res, _ = results
    for m in rx.MODEL_NAMES:
        assert "observed_capacity_violation_rate" in res["aggregate"][m]


def test_hard_tests_and_figures(results):
    out, _, hard = results
    assert set(hard["stress_mae"]) >= {rx.PRIMARY_MODEL, "gru", "carry_forward"}
    for model_res in hard["rollout"].values():
        assert any(k.endswith("_stepwise") for k in model_res)
        assert any(k.endswith("_cumulative") for k in model_res)
    for fig in ("fig1_mae.png", "fig2_violations.png", "fig4_stress.png", "fig5_rollout.png"):
        assert (out / "synthetic" / "figures" / fig).exists()


def test_results_report_generator(results, tmp_path):
    out, _, _ = results
    target = tmp_path / "RESULTS.md"
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_results_md.py"),
                    "--root", str(out), "--studies", "synthetic", "--out", str(target)],
                   check=True)
    text = target.read_text(encoding="utf-8")
    assert "Generated automatically" in text and "SYNTHETIC" in text


def test_latent_ablation_and_clinical_tests_run(tmp_path):
    env = {**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")}
    ab = tmp_path / "ablation.json"
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "latent_ablation.py"),
                    "--study", "synthetic", "--seeds", "0", "--folds", "2", "--max-epochs", "2",
                    "--n-boot", "200", "--out", str(ab)], check=True, env=env)
    out = json.loads(ab.read_text())
    assert out["primary_endpoint"]["feature"] == "MMSE_like"
    assert out["primary_endpoint"]["n_clusters"] > 5
    h2 = tmp_path / "h2.json"
    subprocess.run([sys.executable, "-m", "biojepa.hard_tests2", "--study", "synthetic",
                    "--max-epochs", "2", "--out", str(h2)], check=True, env=env)
    assert rx.PRIMARY_MODEL in json.loads(h2.read_text())
