#!/usr/bin/env python
"""Cohort description (the paper's "Table 1") computed from the data file.

Writes experiments/<study>/cohort_summary.json: subjects, visits, follow-up,
baseline diagnoses, availability of every feature, and the conversion-design
flow (eligible cases / controls and each exclusion reason). Every cohort
number quoted in the paper should come from this file.

    PYTHONPATH=src python scripts/cohort_summary.py --study adni
"""
import argparse
import os
import sys
from collections import Counter
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from biojepa.data.dataset import check_event_timing  # noqa: E402
from biojepa.evaluate import CONVERSION_WINDOW_YEARS, conversion_queries  # noqa: E402
from biojepa.run_experiments import load_study, run_manifest, write_json  # noqa: E402


def summarize(study_name: str) -> dict:
    study = load_study(study_name)
    subs = study.subjects
    n_visits = np.array([len(s.times) for s in subs])
    follow = np.array([float(np.max(s.times)) for s in subs])
    pct = lambda k: round(100.0 * k / len(subs), 1)
    out = {
        "n_subjects": len(subs),
        "n_visit_rows": int(n_visits.sum()),
        "visits_per_subject": {"median": float(np.median(n_visits)), "max": int(n_visits.max())},
        "follow_up_years": {"median": round(float(np.median(follow)), 2),
                            "max": round(float(np.max(follow)), 2)},
        "baseline_group_counts": dict(Counter(s.group for s in subs)),
        "dementia_at_baseline": int(sum(s.dementia_at_baseline for s in subs)),
        "converters": int(sum(s.converts for s in subs)),
        "dynamic_feature_observed_at_baseline_pct": {
            spec.name: pct(sum(np.isfinite(s.dyn[0, d]) for s in subs))
            for d, spec in enumerate(study.dyn_specs)},
        "baseline_pathology_available_pct": {
            name: pct(sum(np.isfinite(s.patho[k]) for s in subs))
            for k, name in enumerate(study.patho_names)},
        "static_available_pct": {
            name: pct(sum(np.isfinite(s.static[k]) for s in subs))
            for k, name in enumerate(study.static_names)},
    }
    lite = SimpleNamespace(study=study, dyn_by_subject=[s.dyn for s in subs])
    _, labels, excl = conversion_queries(lite, range(len(subs)))
    out["conversion_design"] = {"window_years": CONVERSION_WINDOW_YEARS,
                                "eligible": int(len(labels)), "cases": int(labels.sum()),
                                "controls": int(len(labels) - labels.sum()), "excluded": excl,
                                "event_timing_check": check_event_timing(study)}
    out["manifest"] = run_manifest(study_name, study)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", choices=("adni", "oasis", "synthetic"), required=True)
    ap.add_argument("--out-root", default="experiments")
    args = ap.parse_args()
    summary = summarize(args.study)
    target = os.path.join(args.out_root, args.study, "cohort_summary.json")
    write_json(target, summary)
    print({k: v for k, v in summary.items() if k != "manifest"})
    print(f"wrote {target}")
