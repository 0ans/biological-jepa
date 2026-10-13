#!/usr/bin/env python
"""Writes docs/RESULTS.md exclusively from experiment artifacts.

Every number in the report is read from experiments/<study>/results.json,
hard_tests.json and latent_ablation.json, so the document can never drift
from the runs. Nothing in docs/RESULTS.md should be edited by hand.

    python scripts/make_results_md.py                      # adni + oasis
    python scripts/make_results_md.py --studies synthetic --out /tmp/R.md
"""
import argparse
import json
import os
import re
import time

H_LABEL = {1: "12m", 2: "24m", 3: "36m"}


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _ms(d, key, nd=3):
    v = d.get(key)
    return f"{v['mean']:.{nd}f} ± {v['std']:.{nd}f}" if v else "—"


def _num(x, nd=3, signed=False):
    if x is None:
        return "—"
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def study_section(name, res, hard, ablation):
    m = res["manifest"]
    env = m.get("environment", {})
    lines = [f"## {name.upper()}", ""]
    lines += [f"- Subjects: {m['n_subjects']}; protocol: {m['protocol']}",
              f"- Data SHA-256: `{m.get('data_sha256')}`",
              f"- Code commit: `{env.get('git_commit')}`"
              + (" (**uncommitted changes present**)" if env.get("git_dirty") else ""),
              f"- Software: Python {env.get('python')}, torch {env.get('torch')}, "
              f"numpy {env.get('numpy')}, scikit-learn {env.get('sklearn')}; device {m.get('device')}"]
    if m.get("max_epochs_override"):
        lines.append(f"- **WARNING: epochs capped at {m['max_epochs_override']} — "
                     "pipeline check only, not reportable evidence.**")
    lines.append("")

    agg = res["aggregate"]
    models = list(agg)
    feats = m.get("features") or sorted({re.match(r"^mae_(.+)_h\d+$", k).group(1)
                                         for a in agg.values() for k in a
                                         if re.match(r"^mae_(.+)_h\d+$", k)})
    lines += ["### Forecast error (MAE, raw units; mean ± SD over runs — descriptive)", ""]
    cols = [f"{f}@{H_LABEL[h]}" for f in feats for h in (1, 2, 3)]
    lines += ["| Model | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for mod in models:
        lines.append(f"| {mod} | " + " | ".join(_ms(agg[mod], f"mae_{f}_h{h}") for f in feats
                                                  for h in (1, 2, 3)) + " |")
    lines.append("")

    lines += ["### Prospective conversion (baseline index, fixed window, time-aware labels; "
              "seed-averaged patient scores pooled over folds)", "",
              "| Model | AUC | patients | converters |", "|---|---|---|---|"]
    for mod, c in res["conversion_pooled"].items():
        lines.append(f"| {mod} | {_num(c.get('auc'))} | {c.get('n', 0)} | {c.get('n_cases', '—')} |")
    lines.append("")

    lines += ["### Rule-violation rates (predictions vs the observed data)", "",
              "The envelopes are training-data quantiles: a prediction rate far below the "
              "observed-data rate means over-smoothed trajectories, not better biology.", "",
              "| Model | R1 capacity | R1 observed data | R2 reversal | R2 observed data |",
              "|---|---|---|---|---|"]
    for mod in models:
        a = agg[mod]
        lines.append(f"| {mod} | {_ms(a, 'capacity_violation_rate')} | "
                     f"{_ms(a, 'observed_capacity_violation_rate')} | "
                     f"{_ms(a, 'monotonic_violation_rate')} | "
                     f"{_ms(a, 'observed_monotonic_violation_rate')} |")
    lines.append("")

    sig = res["significance"]
    fam = sig["_family"]
    lines += [f"### Paired comparisons — {fam['ours']} vs comparators "
              f"(unit: {fam['unit']}; Bonferroni α = {fam['bonferroni_alpha']:.5f} "
              f"over {fam['n_tests']} tests)", "",
              "| Comparison | Difference (ours − comparator) | 95% CI | p | patients | Bonferroni |",
              "|---|---|---|---|---|---|"]
    for k, v in sig.items():
        if k.startswith("_"):
            continue
        if "auc_a" in v:
            diff, n = v.get("mean_diff"), v.get("n")
        elif "mean_diff" in v:
            diff, n = v.get("mean_diff"), v.get("n_clusters")
        else:
            lines.append(f"| {k} | {v.get('note', 'not estimable')} | | | | |")
            continue
        flag = "significant" if v.get("significant_bonferroni") else "not significant"
        lines.append(f"| {k} | {_num(diff, 4, True)} | [{_num(v.get('ci_low'), 4, True)}, "
                     f"{_num(v.get('ci_high'), 4, True)}] | {_num(v.get('p'), 4)} | {n} | {flag} |")
    lines += ["", "MAE rows: negative difference favors ours. AUC rows: positive difference "
              "favors ours.", ""]

    if ablation:
        p = ablation["primary_endpoint"]
        lines += ["### Primary hypothesis — latent-objective ablation", "",
                  f"Endpoint (pre-specified): {p['feature']} at {p['horizon_years']:g} years; "
                  f"{p['comparison']}.", "",
                  f"- MAE jepa_full {_num(p.get('mae_a'))} vs supervised_control {_num(p.get('mae_b'))}",
                  f"- Difference {_num(p.get('mean_diff'), 4, True)}, 95% CI "
                  f"[{_num(p.get('ci_low'), 4, True)}, {_num(p.get('ci_high'), 4, True)}], "
                  f"p = {_num(p.get('p'), 4)}, {p.get('n_clusters')} patients", ""]
        if ablation.get("max_epochs_override"):
            lines += ["**WARNING: ablation epochs were capped — not reportable.**", ""]

    if hard:
        lines += ["### Hard tests (one held-out fold — descriptive, no inference)", "",
                  "Missingness stress (mean clinical MAE by fraction of inputs removed):", "",
                  "| Model | 0% | 25% | 50% | 75% |", "|---|---|---|---|---|"]
        for mod, c in hard["stress_mae"].items():
            lines.append(f"| {mod} | " + " | ".join(_num(c.get(f)) for f in
                                                    ("0.00", "0.25", "0.50", "0.75")) + " |")
        lines.append("")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments")
    ap.add_argument("--studies", nargs="+", default=["adni", "oasis"])
    ap.add_argument("--out", default=os.path.join("docs", "RESULTS.md"))
    args = ap.parse_args()

    lines = ["# Results", "",
             f"_Generated automatically by `scripts/make_results_md.py` on "
             f"{time.strftime('%Y-%m-%d', time.gmtime())} from `{args.root}/<study>/*.json`. "
             "Do not edit by hand — rerun the experiments and regenerate._", "",
             "Pre-audit numbers (September 2026 and earlier) are superseded and kept only in "
             "`experiments/archive/` for traceability; they must not be cited.", ""]
    found = False
    for name in args.studies:
        res = _load(os.path.join(args.root, name, "results.json"))
        if res is None:
            lines += [f"## {name.upper()}", "",
                      "No results generated with the corrected pipeline yet. Run "
                      f"`python -m biojepa.run_experiments --study {name}`, then regenerate.", ""]
            continue
        found = True
        lines += study_section(name, res,
                               _load(os.path.join(args.root, name, "hard_tests.json")),
                               _load(os.path.join(args.root, name, "latent_ablation.json")))
    if found:
        lines += ["## Interpretation limits", "",
                  "- Mean ± SD over (seed, fold) runs is descriptive; runs share patients.",
                  "- Inference uses patients as the unit after averaging over seeds.",
                  "- Conversion AUC uses one fixed window; it is discrimination, not calibrated risk.",
                  "- Lower rule-violation rates indicate agreement with programmed assumptions only.",
                  "- No external-cohort validation is included.", ""]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
