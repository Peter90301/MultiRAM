#!/usr/bin/env python3
"""Merge parsed one-row CSV files into a normalized metrics CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from parse_perf import FINAL_COLUMNS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Directory containing parsed CSV files.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = sorted(Path(args.input_dir).rglob("*.csv"))
    frames = []
    for path in paths:
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            print(f"[WARN] skipping {path}: {exc}")
            continue
        frame["parsed_source_csv"] = str(path)
        frames.append(frame)
    if frames:
        merged = pd.concat(frames, ignore_index=True, sort=False)
    else:
        merged = pd.DataFrame(columns=FINAL_COLUMNS)
    for col in FINAL_COLUMNS:
        if col not in merged.columns:
            merged[col] = pd.NA
    ordered = FINAL_COLUMNS + [c for c in merged.columns if c not in FINAL_COLUMNS]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged[ordered].to_csv(out, index=False)
    print(f"Wrote {out} rows={len(merged)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

