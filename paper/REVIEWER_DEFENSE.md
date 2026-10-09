# Reviewer defense sheet — evidence, not immunity to criticism

## Is this truly JEPA?
**Accurate term:** JEPA-inspired supervised temporal latent prediction. A history encoder maps observed visits to latent state, a predictor forecasts a future embedding conditioned on the target horizon, and an EMA target encoder supplies training targets from future observations. A supervised clinical head and rule penalty are additional components. This does not imply the method is identical to image-based I-JEPA.

## Is the architecture novel?
A combination of known building blocks is not by itself proof of algorithmic novelty. Novel contribution must be articulated as a testable modeling or evaluation finding, backed by a fair same-architecture ablation and strong prior-work review. Do not claim a new JEPA paradigm solely based on medical application.

## Do the biological constraints demonstrate cause and effect?
No. The penalties are researcher-defined assumptions. Satisfaction demonstrates consistency with those assumptions, not biological mechanism discovery or interventions.

## Does the latent objective itself help?
Unresolved until the matched comparison with identical architecture and `w_lat=0` versus `w_lat=1` is run on real datasets across subject-disjoint folds. See `scripts/repeated_ablation.py`.

## Is the GRU baseline fair?
The old GRU lacked the target forecast horizon. Code now supplies it; historical ranking against GRU must be completely recomputed. Verify equal feature availability and compatible hyperparameter tuning.

## Are the old reported p-values final?
No. Multiple outcomes, horizons, reused patients and repeated seeds demand careful prespecification and patient-cluster uncertainty. Any pooled resampling that treats patient-seed groups as independent patients can overstate evidence.

## Is conversion AUC enough?
No. Conversion labels depend on follow-up time, censoring and index date. A fixed-horizon risk task or appropriate survival modeling is required for meaningful clinical interpretation.

## What if JEPA loses?
Publish that result honestly. A negative well-controlled analysis can be scientifically useful; it cannot establish superiority.

## What evidence would materially strengthen the paper?
1. Independently reproduce model and baseline runs, with frozen code/data/splits.
2. Show matched loss-weight ablation with effect sizes, confidence intervals and patient counts.
3. Repeat on a geographically/institutionally independent dataset with documented inclusion and temporal alignment.
4. Examine calibration, informative missingness, subgroup performance and clinically meaningful errors.
5. Have a domain expert review rule plausibility and outcome definitions before submission.

**Bottom line:** Defend what the evidence establishes, distinguish hypotheses from findings, and invite reproducible critique. No research project can be made 100% immune to peer review.
