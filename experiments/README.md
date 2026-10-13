# experiments/ — evidence artifacts

Every number in `docs/RESULTS.md` is generated from files in this folder by
`scripts/make_results_md.py`. Nothing here is written by hand.

## Per study (`adni/`, `oasis/`) — created by the corrected pipeline

| File | Produced by | Contents |
|---|---|---|
| `results.json` | `python -m biojepa.run_experiments` | manifest (data SHA-256, git commit, software versions, seeds, folds), split counts, per-run metrics, descriptive aggregates, pooled conversion AUC, patient-level paired inference |
| `hard_tests.json` | same command | missingness stress curves and multi-step rollouts (one held-out fold, descriptive) |
| `latent_ablation.json` | `python scripts/latent_ablation.py --study <study>` | primary hypothesis: JEPA vs identical supervised control, pre-specified endpoint |
| `hard_tests2.json` | `python -m biojepa.hard_tests2` | ADNI clinical tests T1–T5 (one held-out fold, descriptive) |
| `figures/` | `run_experiments` | fig1 accuracy · fig2 violations vs observed data · fig3 amyloid subgroup · fig4 stress · fig5 rollout |

The folders do not exist until the experiments are rerun on licensed data.

## `archive/`

Pre-audit artifacts kept for traceability only — see
`archive/2026-09_pre-audit/README.md` for why they must not be cited.

Participant-level data are never committed (see `docs/ADNI_ACCESS.md`).
