# Research protocol and limitations

**Status:** exploratory prototype. Not a clinically validated disease prediction tool.

## Architecture
History encoder (GRU) produces a patient embedding, predictor maps it plus the requested time gap to a *future* embedding, an EMA target encoder supplies a training target, and a supervised clinical head predicts outcome values. VICReg-style variance/covariance terms regularize the predicted embedding. Optional rules penalize trajectories that break explicitly programmed assumptions. This is **JEPA-inspired supervised temporal latent prediction**, not evidence of causal biology.

## Pre-specified analysis plan (implemented in code)
1. **Primary hypothesis — latent objective** (`scripts/latent_ablation.py`), primary dataset **ADNI** (OASIS-2 secondary): identical v2 architecture, no biological penalty, `w_lat = w_std = w_cov = 0` (supervised control) versus the JEPA defaults, same optimizer, epochs, early stopping, splits and seeds. One primary endpoint, fixed before running (default MMSE MAE at 12 months).
2. **Model comparison** (`run_experiments`): one model of interest (`PRIMARY_MODEL = jepa_v2_bio`) against `jepa_v2`, GRU, Ridge and HGB on every dynamic feature of the cohort (MAE pooled over horizons; for OASIS-2 this includes nWBV) and on conversion AUC; Bonferroni over the whole family.
3. **Splits:** 5-fold subject-level cross-validation × 3 seeds; train/validation/test are frozen by patient before any prediction pair is derived; standardizers and rule calibration are fit on training patients only.
4. **Equal access and budget:** every learned model receives the current visit, missingness masks, baseline context and the requested horizon; the sequence models (GRU, JEPA) also receive the full visit history, while Ridge and HGB use the current visit only (a stated design choice). GRU and JEPA share one optimizer factory and one budget (`utils.make_optimizer`, `utils.TRAINING_BUDGET`).
5. **Inference unit = patient:** each prediction pair's error is averaged over seeds first; patients are then the bootstrap and sign-flip unit. Seeds re-use the same people and are never treated as independent samples.
6. **Conversion:** baseline index, fixed 3-year window with ±0.25-year tolerance for visit scatter, time-aware labels (case = converted within the window; control = event-free through the window; censored-before-window excluded), baseline dementia excluded, forecasts that do not depend on a follow-up visit being observed. Exclusions are counted by reason, and the analysis is disabled automatically if the event times fail a units/origin sanity check (`check_event_timing`).
7. **Rule metrics** are always reported next to the same metric computed on the observed data, on the same cells.
8. **Descriptive only:** mean ± SD over runs, single-fold stress/rollout/clinical tests.

## Corrections, October 2026 audit
- The legacy GRU baseline did not receive the forecast horizon (fixed earlier in October) and had a smaller training budget than JEPA (fixed now).
- Significance tests counted every patient once per seed (pseudo-replication); replaced by patient-level inference after averaging over seeds.
- The main conversion AUC kept baseline-demented subjects as negatives and used no event window; the single-fold T6 had a window but could also count baseline-demented subjects as controls, excluded later converters, required an observed 36-month visit and used a different model. Both replaced by the design in item 6.
- Rollout violations were scored differently for JEPA and GRU; both are now scored identically (stepwise and cumulative).
- R2 slack units were inconsistent (absolute calibration, per-year application); now absolute throughout.
- VICReg terms on the no-gradient target were constants and the ablation's control kept VICReg; both corrected.
- **All earlier numbers are superseded** and archived in `experiments/archive/2026-09_pre-audit/` with the reasons.

## Pre-specification status
These cohorts were analysed in earlier exploratory versions of the repository. The plan above was fixed in October 2026 before the corrected pipeline was run, but not before any contact with the data; this must be disclosed in any publication, and confirmatory claims are limited accordingly.

## Claims we cannot yet make
- That the latent loss is essential or improves generalization (pending the primary analysis on licensed data).
- That regularization discovers causality or clinical truth.
- That disease conversion AUC is calibrated patient risk.
- That the tool is safe for medical decision-making.
- Anything about external validity: no independent external cohort is evaluated.

## Reproduce
```bash
make setup test                     # environment + test suite
make data                           # OASIS + verification of your licensed ADNI file
make experiments ablation           # all reported analyses
python scripts/make_results_md.py   # regenerate docs/RESULTS.md from the artifacts
```
Update tables **only** through the generator. External ADNI access and applicable terms must be respected (see `docs/ADNI_ACCESS.md`).
