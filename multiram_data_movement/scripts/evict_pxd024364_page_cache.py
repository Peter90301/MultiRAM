#!/usr/bin/env python3
"""Hint that selected RAW or derived MGF files should leave the Linux page cache."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


ROOT = Path("/mnt/hdd/tsunghan/raw-ms-dataset/proteomic_large_dataset")
MANIFEST = ROOT / "manifests/selected_raw_manifest.tsv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("raw", "mgf"))
    args = parser.parse_args()
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    advised = 0
    missing = 0
    for row in rows:
        relative = Path(row["relative_path"])
        if args.kind == "raw":
            path = ROOT / "raw" / relative
        else:
            path = ROOT / "derived/ms2_mgf" / relative.with_suffix(".mgf")
        if not path.exists():
            missing += 1
            continue
        with path.open("rb") as handle:
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        advised += 1
    print(f"kind={args.kind} advised={advised} missing={missing}")
    # Missing MGF files correspond to RAW files that the vendor parser could
    # not decode; downstream stages intentionally process the valid subset.
    return 0 if missing == 0 or args.kind == "mgf" else 1


if __name__ == "__main__":
    raise SystemExit(main())
