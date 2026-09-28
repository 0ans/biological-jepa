# Research Log (honest, chronological)

Everything that happened during this study — including what failed and why —
is recorded here. Dates: 2026-09-25.

## 0. v2 upgrade ("make it stronger, stage by stage")

After the v1 results (rule engine verified; JEPA+Bio > JEPA on CDR-SB; but
GRU/ridge still ahead on some accuracy metrics and conversion AUC), the model
was upgraded in measured stages:

- **Stage 1 — architecture v2:** history-aware encoder (GRU over the visit
  history + context projection) replaces the single-snapshot encoder; decode
  head upgraded from linear to MLP. Rationale: "doctors think in steps" —
  trajectory shape lives in the history, and a linear probe under-used the
  64-dim latent. v1 is kept as an ablation.
- **Stage 2 — rule R3:** cross-patient causal ordering (top-severity quartile
  may not be predicted to decline slower than bottom quartile) added to the
  rule engine, population-level and differentiable.
- **Stage 3 — stronger opposition:** histogram gradient boosting (per-feature)
  joins the baselines — the serious nonlinear pattern-matching competitor.
- **Stage 4 — harder evaluation:** single 70/15/15 split replaced by
  **5-fold subject-level CV × 3 seeds** (15 runs/model); paired bootstrap
  significance (10k resamples) on primary-feature errors and conversion AUC.
- **Stage 5 — hard tests:** missingness stress test (input measurements
  dropped 25/50/75%, truth untouched, identical degradation across models) and
  multi-step rollout evaluation (predictions fed back 3 steps; error
  accumulation + biological violations on rolled trajectories; JEPA rolls out
  in latent space, GRU in value space).

Two implementation failures during the upgrade, both fixed:
- validation-loss indexing used the train-set size (IndexError on small val);
- GRU value-rollout fed zero context at synthesized steps — replaced with the
  patient's real context to avoid distribution shift.

Each stage is measured against the previous one in RESULTS.md.

## 1. Data acquisition (the hardest part)

**Goal:** longitudinal ADNI data (scans/biomarkers/cognitive scores), per the
research proposal.

- **ADNI official portal (ida.loni.usc.edu):** requires researcher
  registration + Data Use Agreement; requires the author's
  personal registration and cannot be delegated. Documented as the production path
  (`docs/ADNI_ACCESS.md`).
- **Hugging Face search:** `medarc/adni-mini` (2,600+ downloads) exists but is
  **gated** (authentication + approval) and contains image shards, not the
  longitudinal biomarker tables needed for the core study.
- **OASIS official site:** data distribution has moved behind NITRC
  registration + DUA. The old "fully open" status no longer holds.
- **CRAN `rsmatch` package:** contains an `oasis.rda` — downloaded it —
  **failure**: it is a 115-row subset *without* MMSE/CDR/Group. Not usable
  for the clinical component.
- **Breakthrough 1 — ADNI:** the `abaR` R package (github.com/ncullen93/abaR)
  ships `data/adnimerge.rda`: **real ADNI, 2,347 subjects, 15,598 visits**,
  baseline amyloid/tau/hippocampal markers, APOE4, longitudinal
  MMSE/ADAS13/CDR-SB, conversion labels. Downloaded and verified (row counts,
  required columns, hashes in `scripts/download_data.py`).
- **Breakthrough 2 — OASIS-2:** full longitudinal CSV (373 rows / 150
  subjects) located in a public research mirror; verified.

**Provenance decision:** participant-level files are NOT committed to the
repository (ADNI DUA restricts redistribution); the download script fetches
and verifies them with checksums instead.

## 2. Design decisions

- **State representation:** masked tabular patient state (clinical features +
  baseline A/T/N pathology + demographics). Rationale: the scientific claim
  (latent prediction + biological rules) is testable in this modality; imaging
  encoders are a scaling step, not the core hypothesis test.
- **JEPA details:** pair-based (visit i → visit j) latent prediction; EMA
  target encoder; VICReg variance/covariance terms against representation
  collapse (cf. LeJEPA's analysis); light decode head (probe) that carries
  supervision AND the biological penalty.
- **Why the penalty acts on decoded predictions of the future state:** rules
  must shape trajectories even where future biomarkers are unobserved (the
  common case in ADNI) and at rollout time.
- **Rule calibration:** all thresholds fitted from the training cohort
  (severity-binned decline-rate quantiles; noise envelopes; per-feature
  monotonicity slacks). Only small, documented constants remain.
- **Splits:** by SUBJECT, stratified by diagnosis group; never by visit
  (prevents longitudinal leakage). 3 seeds for all comparisons.

## 3. Failures during implementation (all real, all fixed)

1. **Wrong-module import in background run** (`ModuleNotFoundError`) — shell
   CWD reset; fixed by explicit `PYTHONPATH=src`.
2. **pandas/R Categorical crash** — `DX_bl` from pyreadr is categorical;
   `.replace("", "UNK")` raised; fixed with `astype(str)` first.
3. **Linear-interpolation shape bug** in the differentiable capacity
   interpolation (broadcast produced [B,B]); fixed.
4. **Misalignment bug in capacity calibration** — feature-observation mask
   (pair-filtered) did not align with the all-pairs severity array; fixed by
   collecting (rate, severity) jointly.
5. **Methodological iteration (the important one):** the first monotonicity
   slack (0.05σ) was tighter than real measurement noise, and the first
   noise-corrected envelope punished *compliant* noisy predictions (penalty
   0.39 vs 0.05 on truth+noise). Redesigned: **biological envelope**
   (noise-corrected quantile) compared against predicted rates with a
   **separate noise-envelope margin**. Verified on the synthetic cascade
   where ground truth is known: compliant ≈ 0.05, violating ≫.
6. **Ridge baseline fit as a single-output model** (wrong stacking) —
   predictions collapsed to 1-D; refit per-feature.
7. **NaN crash in conversion AUC** — CDR-SB can be unobserved at the input
   visit; added finiteness guards (the crash that stopped the first full run).

## 4. Scope honesty

- The accessible ADNI sample measures A/T/N biomarkers **at baseline only**,
  so rule R2 (trajectory monotonicity) is not applicable to real ADNI
  trajectories in this study; R2 is verified on the synthetic full cascade and
  applied to OASIS's longitudinal brain-volume feature. With full ADNI
  (longitudinal PET/MRI), R2 activates on real data with zero code changes.
- OASIS-2 is small (150 subjects → ~113 evaluable pairs); its numbers are
  treated as pipeline validation, not definitive.
- On the ADNI sample, measured noise envelopes are large (MMSE ≈ 0.73σ/yr),
  so capacity thresholds are effectively cohort-calibrated decline gates;
  the primary interpretable analyses there are relative (bio-constrained vs
  unconstrained models) and the amyloid-subgroup calibration.

Results: see `docs/RESULTS.md` (generated from `experiments/*/results.json`).

## 5. Final run (2026-09-25, after all fixes)

Two more issues surfaced and were fixed during the final runs:
- **GRU loss NaN**: unobserved future targets leaked into the supervised GRU
  loss (`NaN × 0 = NaN`), silently poisoning training — the same masking bug
  JEPA's tensor path already handled. Fixed with explicit `nan_to_num`.
- **JEPA early-stopping criterion**: monitoring the total loss (which moves
  with the EMA target) stopped training at epoch ~0; switched the criterion to
  the predictive decode component (`sup_future`), the stable model-selection
  signal.

Final suite: 6 models × 3 seeds × 2 real studies + figures, ~2 min total on
the M4 laptop CPU. Full numbers and interpretation: `docs/RESULTS.md`.
Headline (ADNI): JEPA+Bio > JEPA on CDR-SB at all horizons; JEPA family keeps
amyloid-negative predicted decline biologically plausible (+0.05σ/yr, observed
+0.028) where ridge/GRU extrapolate improvement (−0.02); rule engine verified
8/8 against the synthetic cascade ground truth.

## 6. v2 verdict (2026-09-25, final)

The staged upgrade delivered, measured under 5-fold CV × 3 seeds:
- the bio-penalty ablation became statistically significant (MMSE p=0.0002,
  CDRSB p=0.0056 — paired bootstrap);
- JEPA-v2+Bio λ=2 is the best MMSE@12m model in the study and top-2 in 4/6
  primary cells; ridge is beaten significantly on both primary features;
- NEW headline: JEPA+Bio is the most robust model under input-missingness at
  every level on both cohorts (ADNI: +72% error at 75% degradation vs HGB +97%);
- NEW: rollout violations drop to zero at steps 2-3 with the rules active,
  while unconstrained GRU violates at every step;
- honest losses retained: HGB leads CDRSB@24/36, ridge leads conversion AUC.
Three hard-test bugs found and fixed en route (list-vs-callable degradation
lookup; rollout dt batch dimension; truth/dt broadcast shape).


## 7. External review (2026-09-27) — 7 findings, all addressed

An external review of the pre-release repository raised seven issues; each was
addressed before publication:

1. **Participant-level benchmark artifacts** (prompts, per-patient
   predictions, ground-truth follow-ups) are **excluded from the repository
   and retained privately by the author**, per the ADNI Data Use Agreement.
   Only aggregate outcomes are published.
2. **src/biojepa/data/ modules** were missing from the pre-release tree (a
   `.gitignore` pattern matched the source subfolder) — fixed to root-scope
   and the modules are tracked.
3. **Statistical inference** uses a cluster bootstrap resampling whole
   patients (visit-pairs of one patient are correlated), with Bonferroni
   correction over the reported family.
4. **T6 threshold-on-test** metric removed; threshold-free AUC is reported.
5. **Claims narrowed**: the synthetic-cascade tests verify implementation
   consistency with the pre-specified rules; they do not establish clinical
   causal validity.
6. **Fourier time period set to 8.0** so the 48-month evaluation horizon does
   not alias onto ~0.
7. **Dataset SHA-256 hashes pinned** and verified on every download.
7. **Dataset hashes now pinned** — download_data.py verifies SHA-256 against
   the exact files used for the published results and fails on mismatch.


## 8. Review round 3 (same day) — deeper methodology findings, all fixed

A second external pass over the corrected tree found four further issues:

1. **R3 ignored observation masks and double-counted patients** — the causal
   ordering term ran over the whole batch (even with zero observed cells, the
   bare margin term produced a constant penalty) and a patient contributing
   several visit-pairs was counted once per pair. R3 now operates on **unique
   patients**, uses observed cells only, contributes nothing when a pair has
   no observed clinical cells, and is skipped entirely below the quorum.
2. **Rules were gated by future ground truth** — passing the future mask into
   the constraints silently disabled them exactly when data is sparse. Fixed:
   the rules constrain the **prediction itself** and require only the current
   state to be observed; regression tests added.
3. **Prospective conversion design** — the conversion test now uses the
   baseline visit only, a fixed 36-month window, and time-under-risk labels
   (cases: converted within 3 y; controls: followed >= 3 y without
   conversion; otherwise excluded). Threshold-based accuracy on test scores
   is not reported anywhere.
4. **Duplicate-visit merge** now takes the mean of measured (non-missing)
   values instead of the element-wise maximum (which was direction-biased).

The bootstrap is re-centered under the null hypothesis, and the pooled
significance clusters by subject across seeds. Two regression tests cover the
R3 fixes (11 tests total).


## 9. Review round 4 — the p-value itself was broken

The cluster-bootstrap **p** implemented in round 3 re-centered the bootstrap
distribution at its own mean, which drives any two-sided p toward 1.0 — a
silent statistical bug caught before publication. Replaced with the correct
patient-level test: a **sign-flip permutation** (under H0 each patient's mean
paired difference is symmetric around zero; p = fraction of permuted means at
least as extreme as the observed one). Confidence intervals remain the
cluster-bootstrap percentiles. All published p-values are superseded, again.


## 10. Review round 5 (measurement-definition pass) — wording and one design fix

An external pass over the corrected tree checked every measurement definition.
No data leakage found. Changes made:

- README/RESULTS now state **11/11** tests consistently.
- **T6 prospective AUC (0.952, single held-out fold, n=141) is reported
  separately from the 15-run CV conversion AUC (0.686)** — never mixed in one
  table.
- **T1 decliner detection is labeled prediction-pair level** (n≈2,600 pairs,
  not patients); a per-patient aggregated version is future work.
- **R3 renamed** to what it proves: a *population-level severity-decline
  ordering constraint* — explicitly not individual-level causality.
- **v3 delta_mode objective fixed**: in delta mode the same decode head is no
  longer also asked to reconstruct the absolute current state (inconsistent
  objective); delta mode now supervises the change only. v3 is NOT part of the
  main benchmark table and is documented as such.
- **Rollout violations are step-to-step** (prediction k+1 vs prediction k),
  matching the sequential constraint; the rollout is named a *latent
  autoregressive rollout*.
- Duplicate-visit averaging no longer described as "unbiased" — it is a
  deduplication choice.
- Explicit **assumptions register** in BIOLOGICAL_RULES.md: the noise envelope
  assumes ~zero true decline in the lowest-severity stratum; apparent A/N
  reversals are assumed measurement noise; severity is a scalar summary of
  A/T/N (vector-conditional capacity is future work); the synthetic tests
  verify implementation consistency, not clinical biology; OASIS-2 is a
  small-n, N-only pipeline check.
