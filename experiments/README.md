# experiments/ — evidence artifacts

Every number claimed in `docs/RESULTS.md` lives here. Nothing hand-written:
all files are produced by `make experiments` / `python -m biojepa.hard_tests2`.

## Per study (`adni/`, `oasis/`)

| File | Contents |
|---|---|
| `results.json` | 15-run (5-fold CV × 3 seeds) aggregate + per-seed metrics + paired-bootstrap significance for all 8 models |
| `hard_tests.json` | Missingness stress curves + multi-step rollout trajectories/violations |
| `hard_tests2.json` | Clinical tests T1–T6 (decliner detection, tolerance, early prognosis, unseen 48-month horizon, converter cohort, conversion) |
| `real_vs_prediction.json` | Population trajectory bias, predicted-vs-actual change correlation, quintile calibration |
| `figures/` | fig1 accuracy · fig2 biological violations · fig3 amyloid-subgroup · fig4 stress · fig5 rollout |

## `chatgpt_benchmark/` — head-to-head vs a general LLM

Identical patient inputs were given to both systems; our predictions were
sealed before the LLM's answers arrived.

| File | Contents |
|---|---|
Participant-level artifacts (prompts, per-patient predictions, ground-truth
follow-ups) were **removed from the repository** for ADNI
DUA compliance and are kept privately by the author. Only aggregate outcomes
remain: `comparison_result.json` (per-metric MAE + conversion counts) and
`cell_tally.json` (closer-prediction cell counts). The benchmark regenerates
locally from an ADNI copy via `python -m biojepa.hard_tests2` and the batched
prompts builder.

Participant-level data is **not committed** (see `docs/ADNI_ACCESS.md`).
