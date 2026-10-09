# Controlled repeated latent-loss experiment

**Question:** Does the same history-aware model predict MMSE at a prespecified horizon more accurately when the future-embedding SmoothL1 objective is enabled?

**Treatment:** latent coefficient `w_lat=1` vs `w_lat=0`. Both have the same v2 architecture, initialization seed, context, visits, labels and biological penalty `w_bio=0`. The anti-collapse losses and EMA infrastructure remain in both arms; the comparison isolates only the latent loss coefficient, not all forms of self-supervision.

```bash
python scripts/repeated_ablation.py --study synthetic --seeds 0 1 --folds 3 --epochs 50
python scripts/plot_ablation.py --input experiments/repeated_ablation.json --output experiments/repeated_ablation.png
```

For real data, download authorized dataset files and choose `--study adni` or `--study oasis`. Do not publish the automatically generated per-pair `rows` of real patient data without confirming the relevant dataset agreement and privacy disclosure restrictions. The pipeline outputs these rows locally for auditing; **do not commit them to GitHub**. A publication-facing export should contain aggregates only.

### Interpretation
The `difference_a_minus_b` value is patient-average MAE without latent loss **minus** patient-average MAE with latent loss. Positive values favor latent loss. Confidence intervals resample patients, never visits; CIs are calculated separately for each seed because seeds reuse patients and are not independent cohorts. No significance claims should be inferred from the number of seeds alone.

### Requirements for confirmatory publication
Lock a primary outcome and horizon; use predefined patient splits, appropriate multiple-comparison correction, prospectively frozen model configs, a strong horizon-conditioned GRU comparator, and external data. This script supplies a useful controlled experiment, not the entire confirmatory analysis.
