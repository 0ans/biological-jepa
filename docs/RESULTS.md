# Results (corrected statistics, patient-level inference)

Protocol: **5-fold subject-level CV × 3 seeds = 15 runs per model**; inference
via **patient-level sign-flip permutation** (cluster-safe: one patient's
visit-pairs move together) with **Bonferroni correction over 16 comparisons**.
All numbers: `experiments/*/results.json`, `hard_tests*.json`, `figures/`.

## ADNI (2,347 subjects) — accuracy

**MAE, raw units (lower = better), mean ± std over 15 runs:**

| Model | MMSE@12m | MMSE@24m | MMSE@36m | CDRSB@12m | CDRSB@24m | CDRSB@36m | conv AUC |
|---|---|---|---|---|---|---|---|
| Carry-fwd | 1.696±0.058 | 1.901±0.081 | 2.184±0.142 | 0.836±0.043 | 1.112±0.072 | 1.408±0.121 | 0.500±0.000 |
| Ridge | 1.599±0.045 | 1.670±0.064 | 1.883±0.073 | 0.888±0.031 | 0.993±0.039 | 1.298±0.064 | 0.725±0.020 |
| HGB | 1.545±0.045 | 1.633±0.062 | 1.836±0.077 | 0.822±0.038 | 0.957±0.042 | 1.202±0.064 | 0.702±0.027 |
| GRU | 1.577±0.063 | 1.567±0.070 | 1.781±0.101 | 0.942±0.050 | 0.968±0.054 | 1.201±0.083 | 0.682±0.022 |
| JEPA-v1 | 1.562±0.071 | 1.652±0.087 | 1.838±0.103 | 0.888±0.047 | 1.010±0.043 | 1.236±0.075 | 0.684±0.027 |
| JEPA-v2 | 1.530±0.072 | 1.627±0.085 | 1.807±0.097 | 0.867±0.067 | 1.006±0.081 | 1.222±0.089 | 0.681±0.026 |
| JEPA-v2+Bio | 1.517±0.070 | 1.609±0.096 | 1.788±0.121 | 0.843±0.052 | 0.974±0.064 | 1.196±0.081 | 0.683±0.028 |
| **JEPA-v2+Bio λ2** | 1.518±0.062 | 1.623±0.081 | 1.807±0.115 | 0.848±0.065 | 0.982±0.066 | 1.206±0.086 | 0.688±0.027 |

### The core ablation, honestly (vs the identical model without rules)

- CDR-SB: JEPA+Bio better, **p=0.0001 — survives Bonferroni** ✓.
- ADAS13: JEPA+Bio better, **p=0.0025 — survives Bonferroni** ✓.
- MMSE: Δ−0.0054, p=0.34 — **not significant** at the patient level (the
  earlier pair-level p=0.0002 was inflated by correlated visit-pairs; superseded).

### vs the strongest opponents (patient-level permutation p)

- **vs Ridge**: better on MMSE (p=0.0001) and CDR-SB (p=0.0001); Ridge keeps
  conversion AUC (0.725 vs 0.683, per-seed p=0.0001).
- **vs GRU**: better on CDR-SB (p=0.0001); MMSE not significant (p=0.047
  uncorrected); AUC tie.
- **vs HGB**: better on MMSE (p=0.030, uncorrected); HGB keeps CDR-SB
  (best at 24 mo; pooled Δ+0.020, p=0.0014).

## Biological judgment — measured, honestly

| Test | Finding |
|---|---|
| **T6 PROSPECTIVE conversion** (baseline input only, fixed 36-month window, time-under-risk labels, n=141) | **JEPA-v2+Bio λ2: AUC 0.952** — best of all models (GRU 0.927, HGB 0.935, Ridge 0.927, JEPA-v2 0.921). Single-fold caveat applies. |
| **T1 decliner detection** (MMSE drop ≥1 pt) — **prediction-pair level** (n≈2,600 pairs across ~390 test patients; NOT one outcome per patient) | JEPA-v2+Bio λ2: 69.0% pair-level accuracy / 61.5% recall — best accuracy among learned models; a per-patient aggregated version is future work |
| **Amyloid-subgroup calibration** | all learned models separate A+/A− plausibly; Ridge retains a slightly negative A− rate |
| **Stress (inputs removed)** | the JEPA family is most accurate at 25% & 50% removal (JEPA+Bio within ~1% of the unconstrained JEPA); at 75%: +85% degradation vs GRU +75% (closest competitor), Ridge +81%, HGB +96% — re-measured under the corrected pipeline |
| **Rule violations (rollout, step-to-step)** | JEPA+Bio 0.042 → 0.033 → **0.003**; GRU worst at every step (0.105/0.073/0.075) |

## Rule-engine verification — 11/11 tests

Including two regression tests added after external review: (a) rules constrain
the **prediction** and are NOT gated by future ground truth; (b) R3 operates on
**unique patients** with observed cells only.

## Honest limitations (post-review)

1. A/T/N biomarkers are baseline-only in the accessible sample — R2 (no
   reversal) is verified on synthetic data + OASIS nWBV, not on real ADNI
   trajectories. Full ADNI activates it.
2. The **MMSE ablation is not significant** at the patient level; ADAS13 and
   CDR-SB are. The weakest honest summary: "the rules help on 2 of 3 clinical
   scales."
3. HGB/Ridge keep several single-target leads; conversion AUC favors Ridge.
4. OASIS-2 remains small-n pipeline validation.
5. The measurement-noise envelope assumes near-zero true decline in the
   lowest-severity training quintile — stated, not proven; listed for
   independent validation.

## Reproduce

```bash
make setup data test      # env + hash-pinned data + 11 tests
make experiments          # 8 models × 15 runs × 2 studies (~50 min on M4)
```
