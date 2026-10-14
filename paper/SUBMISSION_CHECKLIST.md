# Submission-readiness checklist

This is a research-quality gate, not a checklist to bypass peer review.

## Source, ethics and reproducibility
- [ ] Confirm lawful dataset provenance, consent/DUA terms, cohort version, checksums and permissible redistribution — rerun on ADNI data downloaded through your own LONI approval.
- [ ] ADNI Data and Publications Committee review; insert the standard ADNI acknowledgement and funding statement verbatim from the current DUA.
- [ ] Confirm that no participant-level ADNI data were shared with third-party services (e.g. the archived LLM comparison).
- [ ] Lock inclusion/exclusion criteria, diagnosis labels, follow-up horizon, missingness and feature availability before testing.
- [x] Every `results.json` records data hash, git commit, software versions, seeds and split counts.
- [x] Training-only standardization and patient-level non-leakage are tested in code.
- [ ] Publish the exact environment (lock file) and hardware used for the reported run.

## Primary hypothesis and statistical rigor
- [x] **PRIMARY** ablation implemented: identical v2 architecture, JEPA objective vs `w_lat=w_std=w_cov=0`, no bio penalty, matched compute, 5 folds × 3 seeds (`make ablation`).
- [ ] Run it on licensed data.
- [x] All comparators receive the same horizons, current-visit inputs and baseline context; GRU and JEPA also get the history and share one training budget (Ridge/HGB use the current visit only — state this).
- [ ] Lock the primary endpoint before running (code default: MMSE @ 12 months) and record the decision with a date.
- [x] Patient-level 95% bootstrap CIs, sign-flip p-values and unique patient counts are produced automatically.
- [x] Bonferroni over the full comparison family; single-fold tests labeled descriptive.
- [x] Seeds are averaged per prediction pair before inference; patients are never counted once per seed.

## Scientific strength
- [ ] External cohort evaluation; OASIS-2 exploratory small sample is not sufficient alone.
- [ ] Missing-not-at-random sensitivity analysis, demographic subgroup errors, and uncertainty calibration.
- [ ] Biological-rule removal, individual-rule ablations and fair regularized baseline.
- [x] Prospective index dates; conversion labels conditioned on event timing/censoring; baseline dementia excluded.
- [ ] Demonstrate whether latent state contributes beyond direct supervised forecasting.
- [ ] Independent code/data audit and expert clinical review.

## Visuals and reporting
- [x] Editable high-resolution architecture schematic (`docs/architecture.svg`).
- [x] Structured publication manuscript draft (`paper/MANUSCRIPT.md`).
- [x] Latent-objective ablation runner: 5-fold × 3-seed CV, pre-specified endpoint, patient-level inference.
- [x] Cohort table generated from the data (`scripts/cohort_summary.py`).
- [ ] Disclose that the cohorts were analysed in exploratory versions before the analysis plan was fixed.
- [x] Pre-audit results archived with reasons; `docs/RESULTS.md` is generated only from rerun artifacts.
- [ ] Add dataset flow diagram, patient-counts per split and honest confidence intervals.
- [ ] Verify all manuscript references, citations, disclosure statements and journal formatting.
- [ ] Remove any unsupported clinical, causal, superiority or deployment claims.

## Submission decision
**NOT YET SUBMISSION-READY.** The scholarly output remains a draft until benchmark reruns, confirmatory comparisons, validation and domain review are completed.
