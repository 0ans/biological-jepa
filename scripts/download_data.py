#!/usr/bin/env python
"""Fetches / verifies the datasets used in this project and records provenance.

1. ADNI (primary). ADNI participant-level data are governed by the ADNI Data
   Use Agreement: obtain them through your own approved LONI/IDA account
   (adni.loni.usc.edu) — see docs/ADNI_ACCESS.md. This script therefore does
   NOT download ADNI data by default. It only verifies a file you placed at
   data/raw/adnimerge_sample.rda. The historical third-party copy (abaR R
   package) can be fetched with --adni-from-abar, solely if you have
   confirmed that your ADNI approval covers using that copy.
2. OASIS-2 longitudinal (secondary): 150 subjects / 373 visits
   (Marcus et al., 2010), public research-repository mirror of the widely
   used Kaggle derivative. Official access: NITRC + OASIS data use terms.

Data files are stored under data/raw/ and are NOT committed to git.
"""
import argparse
import hashlib
import os
import sys
import urllib.request

import pandas as pd

ADNI_FILE = "adnimerge_sample.rda"
OASIS_FILE = "oasis_longitudinal.csv"
SOURCES = {
    ADNI_FILE: (
        "https://raw.githubusercontent.com/ncullen93/abaR/HEAD/data/adnimerge.rda",
        "ADNI sample, long format (2,347 subjects), third-party abaR copy"),
    OASIS_FILE: (
        "https://raw.githubusercontent.com/KrishKPs/Alzimers_Prediction/HEAD/data/oasis_longitudinal.csv",
        "OASIS-2 longitudinal CSV (150 subjects / 373 visits), public mirror"),
}

# Pinned hashes of the exact files used so far. If a file differs, verification
# FAILS instead of silently producing results from different data.
EXPECTED_SHA256 = {
    ADNI_FILE: "9bea0c79e6401e983e546b7ed36aa38007d5ca5d708c43d2db18daa719dcdc8a",
    OASIS_FILE: "d2f0a15ff35fa4a65c35c064b848fc1b396e9d2f4aec935c88d6e1588d31d40e",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(fname):
    url, desc = SOURCES[fname]
    dest = os.path.join("data", "raw", fname)
    tmp = dest + ".part"
    print(f"[get ] {fname} — {desc}")
    urllib.request.urlretrieve(url, tmp)  # write to a temp name: no half file on failure
    os.replace(tmp, dest)


def verify(fname, allow_unpinned):
    dest = os.path.join("data", "raw", fname)
    got = sha256(dest)
    expected = EXPECTED_SHA256.get(fname)
    match = got == expected
    print(f"        {fname} sha256:{got}  ({os.path.getsize(dest) // 1024} KB)  "
          f"[{'matches pinned hash' if match else 'DIFFERENT from pinned hash'}]")
    if not match and not allow_unpinned:
        sys.exit(f"ERROR: {fname} does not match the pinned hash. If this is your own "
                 "licensed copy, re-run with --allow-unpinned and report its hash "
                 "(it is recorded in every results.json manifest).")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--adni-from-abar", action="store_true",
                    help="download the third-party abaR ADNI copy (only if your ADNI DUA permits it)")
    ap.add_argument("--allow-unpinned", action="store_true",
                    help="accept data files whose hash differs from the pinned one")
    args = ap.parse_args(argv)
    os.makedirs("data/raw", exist_ok=True)

    oasis = os.path.join("data", "raw", OASIS_FILE)
    if not os.path.exists(oasis):
        fetch(OASIS_FILE)
    verify(OASIS_FILE, args.allow_unpinned)
    o = pd.read_csv(oasis)
    assert o.shape[0] >= 370 and o["Subject ID"].nunique() >= 150, "OASIS file looks wrong"
    print("[ok  ] OASIS-2 verified")

    adni = os.path.join("data", "raw", ADNI_FILE)
    if not os.path.exists(adni):
        if args.adni_from_abar:
            print("[warn] Using the third-party abaR copy of ADNI data. You are responsible "
                  "for confirming that your ADNI Data Use Agreement covers this.")
            fetch(ADNI_FILE)
        else:
            print(f"[skip] ADNI: place your licensed file at {adni} (see docs/ADNI_ACCESS.md); "
                  "it is not downloaded automatically.")
            return 0
    verify(ADNI_FILE, args.allow_unpinned)
    import pyreadr
    a = list(pyreadr.read_r(adni).values())[0]
    required = {"RID", "VISCODE", "YEARS_bl", "DX_bl", "MMSE", "ADAS13", "CDRSB", "CSF_ABETA_bl",
                "ConvertedToDementia", "TimeUnderRiskDementia"}
    missing = required - set(a.columns)
    assert not missing, f"ADNI file lacks required columns: {sorted(missing)}"
    assert a["RID"].nunique() >= 2000, "ADNI sample looks wrong"
    print("[ok  ] ADNI file verified (hash + required columns)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
