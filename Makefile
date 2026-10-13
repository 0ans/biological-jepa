# Python >= 3.10 is required (checked by pip via pyproject.toml).
PYTHON ?= python3

# PYTHONPATH=src lets every target work even in a venv where the editable
# `pip install -e .` from `make setup` was skipped
export PYTHONPATH := src:$(PYTHONPATH)

.PHONY: setup data test experiments adni oasis ablation clinical results predict

setup:
	$(PYTHON) -m venv .venv && ./.venv/bin/pip install -r requirements.txt && ./.venv/bin/pip install -e .

data:
	./.venv/bin/python scripts/download_data.py

test:
	./.venv/bin/python -m pytest tests/ -q

experiments: adni oasis

adni:
	./.venv/bin/python -m biojepa.run_experiments --study adni --seeds 0 1 2

oasis:
	./.venv/bin/python -m biojepa.run_experiments --study oasis --seeds 0 1 2

ablation:
	./.venv/bin/python scripts/latent_ablation.py --study adni
	./.venv/bin/python scripts/latent_ablation.py --study oasis

clinical:
	./.venv/bin/python -m biojepa.hard_tests2

results:
	./.venv/bin/python scripts/make_results_md.py

predict:
	./.venv/bin/python scripts/predict_patient.py
