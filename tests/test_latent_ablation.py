"""Guardrails for the controlled latent objective ablation."""
import inspect

from biojepa.train import train_jepa


def test_latent_weight_is_exposed_and_defaults_to_one():
    p = inspect.signature(train_jepa).parameters
    assert "w_lat" in p
    assert p["w_lat"].default == 1.0


def test_latent_weight_forwarded_to_both_model_variants():
    source = inspect.getsource(train_jepa)
    assert source.count("w_lat=w_lat") == 2
