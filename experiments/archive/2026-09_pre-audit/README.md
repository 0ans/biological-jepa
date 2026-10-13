# Archived pre-audit artifacts (September 2026 and earlier) — do not cite

These files are kept only for traceability. They were produced by code that
an October 2026 audit found to be flawed in ways that change the conclusions,
so none of their numbers, p-values or figures may be cited as evidence.

| Problem in the code that produced them | Affected artifacts |
|---|---|
| The GRU baseline did not receive the forecast horizon and had a smaller training budget (200 epochs / patience 25 vs 400 / 60 for JEPA, no LR schedule) | every GRU comparison |
| Significance tests treated each patient as a new cluster for every seed (`patient__s0`, `patient__s1`, ...), counting the same people three times (e.g. 5,610 "clusters" for ~1,870 patients) | all p-values and "survives Bonferroni" statements |
| Conversion AUC kept baseline-demented subjects as non-converters, used no event window, and required an observed follow-up visit | `conversion_auc`, T6 |
| Rollout rule violations were scored step-to-step for JEPA but cumulatively for the GRU | rollout violation comparisons |
| The R2 reversal slack was calibrated as an absolute amount but applied per year (multiplied by the horizon) | R2 penalties and violation rates (OASIS, synthetic) |
| VICReg terms on the no-gradient EMA target were constants; the latent ablation kept VICReg in its "no latent objective" arm | latent-objective conclusions |
| Single-fold clinical tests (T6 "best AUC") used a different model than the main analysis | `hard_tests2.json` |
| `real_vs_prediction.json` has no generating code in the repository | `adni/real_vs_prediction.json` |
| `chatgpt_benchmark/` is an n = 6 demonstration, not a statistical benchmark | `adni/chatgpt_benchmark/` |

Current results are produced only by the corrected pipeline
(`python -m biojepa.run_experiments`) and summarized automatically in
`docs/RESULTS.md` by `scripts/make_results_md.py`.
