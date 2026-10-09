# Submission-readiness checklist

This is a research-quality gate, not a checklist to bypass peer review.

## Source, ethics and reproducibility
- [ ] Confirm lawful dataset provenance, consent/DUA terms, cohort version, checksums and permissible redistribution.
- [ ] Lock inclusion/exclusion criteria, diagnosis labels, follow-up horizon, missingness and feature availability before testing.
- [ ] Publish reproducible environments, training config, exact subject-level split manifests (without identifiers), seeds and hardware.
- [ ] Verify training-only standardization, patient-level non-leakage and absence of post-index covariates.

## Primary hypothesis and statistical rigor
- [ ] **PRIMARY:** compare identical v2 architecture with latent weight `w_lat=0` versus `w_lat=1`, no bio penalty and matched compute.
- [ ] Rerun corrected horizon-conditioned GRU and all comparators at the same horizons, with matched accessible information.
- [ ] Lock MMSE@12mo as a pre-specified primary endpoint *or* explicitly choose and justify another before viewing new results.
- [ ] Report 95% patient-cluster bootstrap confidence intervals and unique patient counts alongside MAE.
- [ ] Control multiple tests and distinguish exploratory versus confirmatory findings.
- [ ] Vary random seed/split; repeated overlapping participants must not be treated as independent cohorts.

## Scientific strength
- [ ] External cohort evaluation; OASIS-2 exploratory small sample is not sufficient alone.
- [ ] Missing-not-at-random sensitivity analysis, demographic subgroup errors, and uncertainty calibration.
- [ ] Biological-rule removal, individual-rule ablations and fair regularized baseline.
- [ ] Prospective index dates; conversion labels conditioned on event timing/censoring.
- [ ] Demonstrate whether latent state contributes beyond direct supervised forecasting.
- [ ] Independent code/data audit and expert clinical review.

## Visuals and reporting
- [x] Editable high-resolution architecture schematic (`docs/architecture.svg`).
- [x] Structured publication manuscript draft (`paper/MANUSCRIPT.md`).
- [x] Reproducible single-split latent objective ablation runner.
- [ ] Replace archived results tables and figures only with rerun artifacts.
- [ ] Add dataset flow diagram, patient-counts per split and honest confidence intervals.
- [ ] Verify all manuscript references, citations, disclosure statements and journal formatting.
- [ ] Remove any unsupported clinical, causal, superiority or deployment claims.

## Submission decision
**NOT YET SUBMISSION-READY.** The scholarly output remains a draft until benchmark reruns, confirmatory comparisons, validation and domain review are completed.
