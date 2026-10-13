# Biological Rules in This Project

This document defines every biological rule used by the rule engine
(`src/biojepa/model/bio_rules.py`), its literature grounding, how it is made
differentiable, and in which dataset it is active.

## Background: the Alzheimer biomarker cascade

Alzheimer's disease follows a well-characterized temporal cascade
(Jack et al., *Lancet Neurology* 2010; Jack et al., *Lancet Neurology* 2013;
synthesized as the A/T/N framework by Jack et al., *Alzheimer's & Dementia* 2016):

    Amyloid (A)  ->  Tau (T)  ->  Neurodegeneration (N)  ->  Clinical decline (C)

Pathology precedes and (partially) causes symptoms. An effect requires a cause;
a disease cannot progress down a route whose upstream biology is not active.

## Rule R1 — Causal capacity ("an effect requires a cause")

**Statement.** The predicted clinical decline rate of a patient (in
standardized units per year, measured in the worse-direction) may not exceed a
patient-specific biological capacity:

    capacity(f, severity) = BioEnvelope_f(severity) + r0 + NoiseEnv_f

where `BioEnvelope_f(severity)` is a piecewise-linear curve over pathology
severity and `NoiseEnv_f` is the feature's measurement-noise envelope.

**Grounding.** Clinical decline in AD is driven by pathology burden: patients
without amyloid/tau/neurodegeneration activity decline slowly if at all
(e.g., amyloid-negative cognitively-unimpaired subjects). A model that
predicts rapid cognitive collapse for a biologically quiet patient is not
reasoning about the disease — it is exploiting statistical shortcuts.

**Data-driven calibration (no hand-picked constants).**
1. Severity proxies in [0,1] per patient: amyloid burden from CSF Aβ42
   (inverted — low Aβ42 = high burden) and amyloid PET; tau burden from CSF
   total/phospho-tau; neurodegeneration from hippocampal volume (inverted).
   Missing markers are skipped; if all are missing, severity is neutral (0.5).
2. Observed decline rates are computed on training pairs only and binned by
   mean severity (5 bins).
3. The **noise envelope** `NoiseEnv_f` is the 90th percentile of |rate| in the
   lowest-severity quintile of patients, where cascade biology implies
   near-zero true
   decline — so that quantile estimates pure measurement noise.
4. The **biological envelope** is the per-bin 90th percentile of observed
   rates minus the noise envelope: an estimate of what biology — not noise —
   actually produces.
5. `r0 = 0.10` (a small floor) is the only constant, chosen conservatively.

**Known limitation (rate units of the noise margin):** R1 compares decline
*rates* (σ per year). Measurement noise in a rate scales with 1/Δt, but the
noise envelope is a single quantile pooled over the 1-, 2- and 3-year
training pairs, so it is relatively loose at 3 years and tight at 1 year.
This is a stated design choice; a horizon-specific envelope is future work.

**Known limitation (scalar severity):** severity is summarized as a scalar
(mean of the A/T/N proxies). Two patients with different biomarker profiles
can share the same scalar severity — biologically distinct pathways are not
distinguished. A vector-conditional capacity (per-marker capacity curves) is
the natural next step and is listed as future work.

**Penalty (differentiable, training only).** `ReLU(rate − capacity)²`,
averaged over the clinical cells that are *observed at the current visit*
(imputed current values are masked out). It compares the decoded *predicted*
future state with the observed current state and never reads the future
ground truth, so it also shapes predictions for pairs whose future values are
missing. At inference no penalty is computed; the rules act only through what
the model learned.

## Rule R2 — Monotonicity ("pathology does not reverse")

**Statement.** Amyloid burden (A) and neurodegeneration markers (N) cannot
improve beyond a per-feature measurement-noise slack:

    penalty = ReLU(improvement − slack_f)²

The slack is an **absolute** amount in standardized units (σ), identical at
every horizon: measurement noise on the difference between two visits does
not grow with the time between them. (Before October 2026 the slack was
calibrated as an absolute amount but applied multiplied by Δt, which allowed
three times more reversal at 3 years than at 1 year; corrected.)

**Grounding.** Fibrillar amyloid deposition and neuronal death are effectively
irreversible within trial/follow-up horizons; observed "improvements" are
measurement variability. `slack_f` is calibrated from training data as
`2 × RMS(per-patient mean apparent reversal)` (σ units, at least
`RuleConfig.slack` = 0.25σ, capped at 1.5σ) — apparent reversals in the
observed cohort are treated as noise by assumption of the cascade.

**Tau (T) is not constrained:** neither real cohort has a longitudinal tau
measure, so no tau-monotonicity rule is encoded.

**Clinical features (C) are intentionally NOT monotonic** — retest/practice
effects are real and documented; forcing monotone cognition would encode a
false rule.

## Rule R3 — Population-level severity–decline ORDERING constraint ("stronger cause, stronger effect")

> **A statistical group-level regularity — NOT a demonstration of
> individual-level causality.** The constraint restricts a population ordering;
> no causal proof is claimed.

**Statement.** Within a training batch, patients in the top
pathology-severity quartile may not be *predicted* to decline slower than
patients in the bottom quartile, beyond a small margin:

    penalty = ReLU(mean_rate(bottom-severity) − mean_rate(top-severity) + margin)²

**Grounding.** The cascade literature consistently shows pathology burden
(amyloid/tau/atrophy) predicts faster clinical decline at the population
level. A model that assigns faster decline to the biologically quieter
patients has learned an anti-causal shortcut. The quartile (population-level)
formulation keeps the rule statistical — individual variation is respected —
while constraining the population ordering.

**Applicability:** all studies with severity variation (ADNI: baseline A/T/N
severity; OASIS: N-severity only; synthetic). Requires patient identifiers
in the batch (per-patient aggregation) and at least 8 unique patients with
observed cells. Also checked diagnostically via the amyloid-subgroup
calibration analysis.

## Dataset applicability (explicit, not hidden)

| Rule | ADNI sample | OASIS-2 | Synthetic cascade |
|---|---|---|---|
| R1 causal capacity (A/T/N severity) | Yes (baseline A/T/N biomarkers) | Partial — N only (no A/T measured; A/T severity neutral) | Yes |
| R2 amyloid monotonicity | N/A (amyloid at baseline only — no trajectory) | N/A | Yes |
| R2 neurodegeneration monotonicity | N/A (no longitudinal N feature) | Yes — longitudinal nWBV | Yes |
| Tau monotonicity | not encoded | not encoded | not encoded |
| R3 population ordering | Yes | Yes (N-severity) | Yes |

The synthetic cascade study exists so that R1, R2 (A and N) and R3 are
verified end-to-end against known ground truth (see `tests/test_bio_rules.py`),
independent of real-data availability limits.

## How to read violation rates

R1 envelopes are 90th-percentile curves of the *training* data plus margins,
so real observed trajectories also exceed them at some rate. Every results
table therefore reports the violation rate of the **observed** future values
next to each model's rate. A model far below the observed rate is producing
smoother trajectories than real patients show — that is not evidence of
better biology, only of agreement with the programmed assumptions.

## What we did NOT encode (and why)

- **Cognitive monotonicity**: false (practice effects).
- **Hard cascade ordering within a patient** (e.g., "tau cannot rise before
  amyloid"): the accessible ADNI sample measures A/T at baseline only, so
  there is no trajectory to constrain. The cross-patient, population-level
  version is encoded as R3 and additionally checked with the amyloid-subgroup
  calibration analysis.
- **Tau monotonicity**: no longitudinal tau measure in either real cohort.
- **SNAP caveat**: ~10-30% of amyloid-negative subjects show
  neurodegeneration-of-unknown-cause and can decline (Jack et al. 2016). R1
  handles this conservatively: neurodegeneration severity raises the capacity
  even when amyloid is low.
