# biological-jepa

**Disease progression modeling with a biologically-constrained JEPA — from
recognition to reasoning.**

Medical AI today labels a scan; it does not reason about how the disease will
move. This project implements and tests the idea that a JEPA
(Joint-Embedding Predictive Architecture) trained to predict the *latent*
future state of a patient — and penalized whenever its predicted trajectory
violates known biology — produces disease trajectories that are more
biologically plausible than standard pattern-matching models.

Evaluated on **real longitudinal data**: an ADNI sample (2,347 participants,
~9,300 training trajectories, baseline amyloid/tau/hippocampal markers,
longitudinal MMSE/ADAS13/CDR-SB, conversion labels) and OASIS-2
(150 subjects, 373 visits), with a synthetic Alzheimer cascade used as
ground truth for rule verification.

## The idea in one figure

![Architecture](docs/architecture.png)

Why latent prediction: raw-value prediction forces the model to also model
measurement noise, scanner variation, and missing-entry artifacts — the "messy
and often missing" reality of medical data. Predicting in representation
space lets unpredictable detail dissolve and keeps the disease trajectory as
the learning target. The biological penalty then shapes *what trajectories*
the model is allowed to imagine: an effect requires a cause.

## Headline results (ADNI, 5-fold subject-level CV × 3 seeds, patient-level inference)

| | JEPA-v2 | **JEPA-v2+Bio** | GRU | HGB | Ridge | Carry-fwd |
|---|---|---|---|---|---|---|
| MMSE MAE @12 mo ↓ | 1.557 | **1.515** (λ2: **1.511**) | 1.582 | 1.545 | 1.599 | 1.696 |
| CDR-SB MAE @36 mo ↓ | 1.227 | 1.189 (λ2) | 1.202 | **1.201** | 1.299 | 1.408 |
| bio-penalty ablation (patient-level permutation) | — | **ADAS13 p=0.0001 ✓ (survives Bonferroni)** · CDRSB p=0.0054 · MMSE p=0.083 | | | | |
| MAE at 75% input degradation ↓ | 3.99 | 4.08 | **3.98** | 4.49 | 4.13 | — |

**Conversion AUC — two different measurements, never mixed:**
- 15-run CV aggregate (mixed follow-up windows): JEPA+Bio 0.686 — Ridge
  leads this metric (0.724).
- Single held-out fold, prospective design (baseline input, fixed 36-month
  window, time-under-risk labels, n=141): JEPA+Bio λ2 **0.952** vs GRU 0.927
  vs HGB 0.935 — a hard-test result, NOT a CV aggregate.

Three honest statements:

1. **The biological constraints help on 2 of 3 clinical scales** (ADAS13
   survives full Bonferroni correction; MMSE does not at the patient level —
   the earlier pair-level p=0.0002 was inflated and is superseded).
2. **Robustness to missing data is the structural win**: at 25-50% input
   degradation JEPA+Bio is the most accurate of all models, and its relative
   degradation at 75% (+85%) is far below HGB's (+98%); GRU is the closest
   competitor (+75%) with worse absolute error.
3. **Honest losses**: HGB keeps CDR-SB@24/36; Ridge keeps conversion AUC.

Full tables, significance details, and limitations:
**[docs/RESULTS.md](docs/RESULTS.md)**.

The rule engine is verified for **implementation consistency** against a
synthetic Alzheimer cascade with known ground truth (**11/11 tests**,
including two regression tests added after external review: rules constrain
the *prediction* — not gated by future ground truth — and R3 operates on
*unique patients* with observed cells only). These tests validate the code
against its own pre-specified rules; they do not validate the rules as
clinical causal truth.
See **[docs/BIOLOGICAL_RULES.md](docs/BIOLOGICAL_RULES.md)**.

The rule engine itself is verified against a synthetic Alzheimer cascade with
known ground truth (11/11 tests): compliant trajectories ≈ 0 penalty; unsupported
decline, pathology reversal, and anti-causal ordering are flagged with >10×
margin. See **[docs/BIOLOGICAL_RULES.md](docs/BIOLOGICAL_RULES.md)**.

## Quickstart

```bash
git clone https://github.com/0ans/biological-jepa.git && cd biological-jepa
make setup          # venv + torch/pandas/sklearn
make data           # downloads & verifies both real datasets (checksums printed)
make test           # 11 unit tests incl. rule-engine vs ground truth
make experiments    # 8 models × 15 runs × 2 studies + hard tests (~45 min on a laptop)
```

Try it on a patient (trains in ~40 s, then predicts the 3-year trajectory):

```bash
python scripts/predict_patient.py                 # 3 held-out patients
python scripts/predict_patient.py --sid ADNI_77   # a specific held-out patient
```

Results land in `experiments/{adni,oasis}/`: `results.json` (15-run CV +
significance), `hard_tests.json` (missingness stress + rollout), and figures.
See [experiments/README.md](experiments/README.md) for a map of every artifact.

## Repository layout

```
src/biojepa/
├── data/            adni.py · oasis.py · synthetic.py · dataset.py (splits, K-fold, pairs, masks)
├── model/
│   ├── jepa.py      v1 snapshot encoder · v2 history encoder · EMA target · VICReg · latent rollout
│   ├── bio_rules.py differentiable rule engine (R1 capacity, R2 monotonicity, R3 ordering)
│   └── baselines.py carry-forward · ridge · HGB · supervised GRU
├── pipeline.py      train-fit standardizers, per-pair tensors, capacity calibration
├── train.py · evaluate.py (bootstrap · stress · rollout) · run_experiments.py
docs/                RESULTS · RESEARCH_LOG (incl. failures) · BIOLOGICAL_RULES · ADNI_ACCESS
experiments/         committed results.json + hard_tests.json + figures (evidence)
tests/               rule-engine ground-truth tests + end-to-end smoke
```

## Data & ethics

Participant-level data are **not committed**. `scripts/download_data.py`
fetches the ADNI sample (via the `abaR` R package redistribution) and the
OASIS-2 longitudinal CSV from public research mirrors, verifies them, and
prints checksums. For production-grade runs, register at
[adni.loni.usc.edu](https://adni.loni.usc.edu) (free, DUA) — the loader
targets the ADNIMERGE schema; see [docs/ADNI_ACCESS.md](docs/ADNI_ACCESS.md).

## Honest limitations

- The accessible ADNI sample has baseline-only A/T/N biomarkers, so
  trajectory-monotonicity rules are verified on the synthetic cascade and
  applied to OASIS brain volumes; full ADNI activates them on real PET/MRI.
- HGB remains the single-target accuracy leader (CDR-SB@24/36) and ridge the
  conversion-AUC leader — JEPA+Bio's demonstrated edge is plausibility, robustness
  to missing data, and statistically significant accuracy gains over its own
  unconstrained ablation.
- OASIS-2 numbers are small-n pipeline validation only.

See [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md) for the full honest account,
including everything that failed along the way.

## Citation

```bibtex
@software{alharbi2026biologicaljepa,
  author = {Al-Harbi, Anas},
  title  = {biological-jepa: biologically-constrained JEPA for disease progression},
  year   = {2026},
  url    = {https://github.com/0ans/biological-jepa}
}
```

Foundations: JEPA/LeCun et al.; V-JEPA 2 (Assran et al., 2025);
LeJEPA (Balestriero & LeCun, 2025); AD biomarker cascade (Jack et al., 2010,
2013, 2016); ADNI and OASIS-2 datasets.

## License

MIT — see [LICENSE](LICENSE).

---
