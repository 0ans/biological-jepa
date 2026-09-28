#!/usr/bin/env python
"""Downloads the real datasets used in this project and records provenance.

1. ADNI sample (primary): ADNIMERGE-style longitudinal table as redistributed
   with the `abaR` R package (github.com/ncullen93/abaR, processed from ADNI's
   public ADNIMERGE derivative). Real ADNI participants; for the full dataset
   and imaging, register at adni.loni.usc.edu — see docs/ADNI_ACCESS.md.
2. OASIS-2 longitudinal (secondary): 150 subjects / 373 visits
   (Marcus et al., 2010), public research-repository mirror.

Data files are stored under data/raw/ and are NOT committed to git.
"""
import hashlib
import os
import sys

import pandas as pd

SOURCES = {
    "adnimerge_sample.rda": (
        "https://raw.githubusercontent.com/ncullen93/abaR/HEAD/data/adnimerge.rda",
        "ADNI sample, long format (2,347 subjects), via abaR R package"),
    "oasis_longitudinal.csv": (
        "https://raw.githubusercontent.com/KrishKPs/Alzimers_Prediction/HEAD/data/oasis_longitudinal.csv",
        "OASIS-2 longitudinal CSV (150 subjects / 373 visits), public mirror"),
}


# Pinned hashes of the exact files used for the published results.
# If a source changes upstream, verification FAILS instead of silently
# producing results from different data.
EXPECTED_SHA256 = {
    "adnimerge_sample.rda": "9bea0c79e6401e983e546b7ed36aa38007d5ca5d708c43d2db18daa719dcdc8a",
    "oasis_longitudinal.csv": "d2f0a15ff35fa4a65c35c064b848fc1b396e9d2f4aec935c88d6e1588d31d40e",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs("data/raw", exist_ok=True)
    for fname, (url, desc) in SOURCES.items():
        dest = os.path.join("data", "raw", fname)
        if os.path.exists(dest):
            print(f"[skip] {fname} exists")
        else:
            print(f"[get ] {fname} — {desc}")
            import urllib.request
            urllib.request.urlretrieve(url, dest)
        got = sha256(dest)
        expected = EXPECTED_SHA256.get(fname)
        status = "OK" if (expected is None or got == expected) else "MISMATCH vs pinned hash"
        print(f"        sha256:{got}  ({os.path.getsize(dest)//1024} KB)  [{status}]")
        if expected is not None and got != expected:
            sys.exit(f"ERROR: {fname} does not match the pinned hash used for the published results.")

    # verify shapes so corruption is caught immediately
    o = pd.read_csv("data/raw/oasis_longitudinal.csv")
    assert o.shape[0] >= 370 and o["Subject ID"].nunique() >= 150, "OASIS file looks wrong"
    import pyreadr
    a = list(pyreadr.read_r("data/raw/adnimerge_sample.rda").values())[0]
    assert a["RID"].nunique() >= 2000 and {"MMSE", "ADAS13", "CDRSB", "CSF_ABETA_bl"} <= set(a.columns), \
        "ADNI sample looks wrong"
    print("[ok  ] both datasets verified (row counts + required columns present)")


if __name__ == "__main__":
    sys.exit(main())
