# Research protocol and limitations

**Status:** exploratory prototype. Not a clinically validated disease prediction tool.

## Architecture
History encoder (GRU) produces a patient embedding, predictor maps it plus the requested time gap to a *future* embedding, an EMA target encoder supplies a training target, and a supervised clinical head predicts outcome values. Optional rules penalize trajectories that break explicitly programmed assumptions. This is **JEPA-inspired supervised temporal latent prediction**, not evidence of causal biology.

## Required experimental comparisons
1. Carry forward, Ridge, HGB, and **time-conditioned GRU** must all receive suitable input histories, missingness masks, baseline variables, and requested future horizon.
2. Compare full JEPA against the **same architecture without latent loss**, keeping everything else fixed.
3. Compare JEPA without rules against JEPA with rules; include simpler rule-constrained baselines.
4. Freeze train/validation/test splits by **patient ID before deriving visits or prediction pairs**. Fit imputers and normalizers on training patients only.
5. Report patient counts and paired patient-bootstrap confidence intervals, not just pair-weighted metrics. Seeds applied to the same patients are not independent samples.
6. Report per-horizon error and pre-specified multiple testing, performance under realistic non-random missingness, group calibration, and external validation.

## Important correction, October 2026
The legacy GRU baseline did not receive the future forecast horizon, while competing predictors did. The code now adds horizon Fourier features to the GRU training/evaluation inputs and fixes autoregressive rollout input shape. **No benchmark numbers were recalculated by this code change.** The reported legacy results are preserved for traceability, but are not valid evidence of superiority over a time-conditioned GRU.

## Claims we cannot yet make
- That the latent loss is essential or improves generalization.
- That regularization discovers causality or clinical truth.
- That disease conversion AUC is calibrated patient risk.
- That historical GRU results reflect the fixed implementation.
- That the tool is safe for medical decision-making.

## Reproduce
Install project requirements, run `python -m pytest tests -q`, then rerun the complete ADNI/OASIS experiments with recorded environment, versions, splits and random seeds. Update tables **only** from those outputs. External ADNI access and applicable terms must be respected.
