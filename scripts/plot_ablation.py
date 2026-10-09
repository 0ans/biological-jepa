"""Render comparison charts from *actual* repeated_ablation.py output.

Never displays placeholder bars or synthetic numbers as clinical results.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(source, destination):
    source = Path(source)
    data = json.loads(source.read_text(encoding="utf-8"))
    if data.get("comparison") != "identical v2 architecture with latent loss 0 vs 1":
        raise ValueError("Not a compatible repeated latent-ablation report")
    rows = data["per_seed"]
    if not rows:
        raise ValueError("No empirical results found")
    labels = [f"Seed {row['seed']}" for row in rows]
    diffs = [row["difference_a_minus_b"] for row in rows]
    low = [row["ci95"][0] for row in rows]
    high = [row["ci95"][1] for row in rows]
    fig, ax = plt.subplots(figsize=(9, max(4, len(rows) * 0.7 + 2)))
    x = list(range(len(rows)))
    ax.errorbar(diffs, x, xerr=[[d-l for d,l in zip(diffs,low)],
                                [h-d for d,h in zip(diffs,high)]],
                fmt="o", capsize=4)
    ax.axvline(0, linewidth=1, linestyle="--")
    ax.set_yticks(x, labels)
    ax.set_xlabel("MAE without latent loss minus with latent loss (positive favors latent)")
    ax.set_title(f"{data['study']} | {data['target']} at {data['horizon_years']}-year horizon")
    ax.set_ylabel("Independent random seeds, same patient cohort")
    ax.grid(axis="x", alpha=0.2)
    fig.text(.1, .02, "95% patient-cluster bootstrap CIs within each seed; exploratory, not external validation.",
             fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, 1))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=200)
    plt.close(fig)
    return str(destination)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input", default="experiments/repeated_ablation.json")
    p.add_argument("--output", default="experiments/repeated_ablation.png")
    args=p.parse_args()
    print(render(args.input,args.output))
