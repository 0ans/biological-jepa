PYTHON ?= python3.11   # project requires >= 3.10; 3.11 recommended

# PYTHONPATH=src lets every target work even in a venv where the editable
# `pip install -e .` from `make setup` was skipped
export PYTHONPATH := src:$(PYTHONPATH)

.PHONY: setup data test experiments adni oasis predict

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

predict:
	./.venv/bin/python scripts/predict_patient.py
