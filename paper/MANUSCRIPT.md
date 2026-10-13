# A Biologically Regularized Joint-Embedding Predictor for Longitudinal Alzheimer's Disease Progression
**Manuscript draft — research prototype; not ready for submission**

Author: Anas Al-Harbi (author/affiliation subject to confirmation)

## Abstract
**Background:** Predicting patient-specific Alzheimer's disease trajectories from irregular, sparsely observed longitudinal visits remains challenging. **Objective:** We investigate whether a history-aware joint-embedding predictive architecture (JEPA-inspired) paired with supervised clinical prediction and soft biomedical constraints is a useful framework for temporal forecasting. **Methods:** A recurrent history encoder generates a 64-dimensional representation from prior visits and patient context; a time-conditioned predictor estimates its future embedding. An exponential-moving-average target encoder provides a latent learning target during training, while a clinical regression head and optional biological penalties provide additional supervision. The repository includes experiments on an accessible ADNI-derived sample and OASIS-2, and subject-level cross-validation. **Results:** Pending. An October 2026 audit corrected the statistical unit (patients, not patient × seed), the conversion design, baseline training budgets and rule scoring; all earlier repository numbers are superseded and are not reported. Results will be filled in only from reruns of the corrected, pre-specified pipeline on licensed data. **Conclusion:** This work establishes an executable framework and falsifiable study design; fair baseline reruns, latent-objective ablations, uncertainty estimates, and external validation remain necessary before any efficacy or clinical claims.

## 1. Introduction
Alzheimer's disease follows heterogeneous and incompletely observed clinical trajectories. Forecasting MMSE, ADAS13, and CDR-SB from prior assessments and baseline context is useful as an ML research task, but cannot alone demonstrate disease mechanism discovery. Explicit time-conditioning is vital because a forecast at 12 months differs from a forecast at 36 months. Latent-state prediction is a plausible inductive bias, not automatically a proven advantage. This manuscript asks: **does latent prediction provide measurable value after fair conditioning, equal data access and careful subject-level testing?**

### Contributions
1. A transparent history-aware latent predictive model with EMA targets and a supervised clinical head.
2. An optional differentiable biomedical penalty module, treated as assumptions rather than causal evidence.
3. A publicly inspectable evaluation pipeline with subject-level partitioning and reproducibility tests.
4. A documented negative-control and ablation agenda to establish whether latent prediction, rather than model capacity or regularization, explains performance.

## 2. Methods
### 2.1 Data
The repository documents an accessible ADNI-derived sample (reported n=2,347 participants) and OASIS-2 (reported n=150). The data are not redistributed within this repository. Longitudinal input comprises observed visits through time t, masks for missing measurements, time gaps and baseline context. Data provenance and usage permissions require independent verification prior to formal submission.

### 2.2 Encoder, target and predictor
A GRU plus projection computes `z_t = Eθ(H≤t, c)`. A predictor uses Fourier encoded horizon Δt to compute `ẑ_{t+Δt}=Pφ(z_t,e(Δt))`. The non-gradient target encoder `Eξ` processes the observed future history only during training; `ξ ← mξ + (1−m)θ` (EMA). The clinical head predicts standardized future measurements `ŷ=Dψ(ẑ)`. Dimensions and weights are documented in source code; hyperparameters are not claimed optimal.

### 2.3 Objectives
The optimization combines masked clinical prediction loss, a SmoothL1 latent target loss, variance/covariance regularization of the predicted embedding (the EMA target carries no gradient) and an optional rule penalty. Biomedical rules R1–R3 encode domain-motivated smooth constraints; a low rule violation score measures agreement with these rules only. It does not demonstrate biological correctness or counterfactual causality.

### 2.4 Baselines and comparability
Baselines are carry-forward, ridge, histogram gradient boosting and a supervised GRU. All nontrivial baselines receive the input history, missingness masks, baseline context and the requested forecast horizon; the GRU and JEPA models share optimizer, learning-rate schedule, epoch limit, early stopping and minimum batch size. The primary hypothesis compares the JEPA model with the identical architecture trained without any latent objective (`w_lat = w_std = w_cov = 0`).

### 2.5 Evaluation design
Five-fold subject-disjoint cross-validation is repeated with three seeds; normalizers and rule calibration are fit on training subjects only. We report per-horizon MMSE, ADAS13 and CDR-SB MAE. For inference, each prediction pair's error is averaged over seeds, and patients are the unit of the paired bootstrap 95% confidence interval and of a sign-flip permutation test; Bonferroni correction covers the full comparison family. Conversion is evaluated prospectively: baseline index, fixed 3-year window, cases converting within the window, controls event-free through it, subjects censored earlier excluded and baseline dementia excluded; forecasts do not require an observed follow-up visit. Rule-violation rates are reported alongside the same rates computed on observed data. The missingness stress test removes inputs completely at random (MCAR) and does not address informative missingness.

## 3. Results — pending rerun
No results are reported. Earlier repository numbers were produced by code with errors that change their interpretation (see `experiments/archive/2026-09_pre-audit/README.md`) and are not used. This section will be written only from `docs/RESULTS.md`, which is generated automatically from the rerun artifacts.

## 4. Discussion
The architecture is reasonably described as *JEPA-inspired temporal latent prediction with supervised outcomes*. A target EMA encoder and latent predictor satisfy important structural properties, but do not prove that the latent objective drives performance. It is necessary to compare the same architecture with latent-loss weight zero, holding all other components fixed. Further tests should determine whether soft biological constraints generalize outside their hand-coded assumptions, and whether improvements are meaningful relative to strong time-aware baselines.

## 5. Limitations
Single-source research samples; potential ADNI-related data reuse; retrospective irregular sampling; restricted repeated biomarkers; limited OASIS-2 sample size; uncertain external generalization; non-causal constraints; and historical comparison errors now fixed in code but not yet re-evaluated. Conversion evaluation requires particular care for censoring and follow-up. The system is not a medical device.

## 6. Reproducibility and ethics
Code: https://github.com/0ans/biological-jepa . Patient-level data are excluded. Every results file records the data checksum, code commit, software versions, seeds and split counts. Data used in this work must be obtained under the ADNI Data Use Agreement and the OASIS data use terms; the manuscript must pass ADNI Data and Publications Committee review and carry the standard ADNI acknowledgement and funding statement (to be inserted verbatim from the current DUA). All failed experiments and negative ablations will be reported.

## 7. Conclusions
A clear technical hypothesis and an executable prototype have been established. **Publication-strength evidence is pending**, especially fair reruns of baseline comparisons, loss ablations, independent cohort evaluation and confidence intervals.

## References to verify before submission
- Assran et al. *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*. 2023.
- Bardes et al. *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning*. 2022.
- Jack et al. *Hypothetical model of dynamic biomarkers of the Alzheimer's pathological cascade*. 2010.
- Alzheimer's Disease Neuroimaging Initiative (ADNI): original cohort and data-use documentation.
- OASIS longitudinal data release and original dataset publication.

**Editorial note:** Replace this starter reference list with complete bibliographic metadata and verify every dataset and empirical claim before submission.
