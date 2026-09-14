#!/usr/bin/env python3
"""CPU clustering surrogate for Falcon-style benchmarking.

This is a pure CPU baseline that clusters spectra by precursor m/z buckets.
It is used when Falcon binary is unavailable on host.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_mgf_precursors(path: Path) -> list[tuple[str, float, int]]:
    spectra: list[tuple[str, float, int]] = []
    in_block = False
    title = ""
    precursor = 0.0
    charge = 0
    idx = 0

    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            upper = line.upper()
            if upper == "BEGIN IONS":
                in_block = True
                title = ""
                precursor = 0.0
                charge = 0
                continue
            if upper == "END IONS":
                if in_block:
                    idx += 1
                    sid = title or f"scan_{idx}"
                    spectra.append((sid, precursor, charge))
                in_block = False
                continue
            if not in_block or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().upper()
            value = value.strip()
            if key == "TITLE":
                title = value
            elif key == "PEPMASS":
                try:
                    precursor = float(value.split()[0])
                except ValueError:
                    precursor = 0.0
            elif key == "CHARGE":
                digits = "".join(ch for ch in value if ch.isdigit())
                charge = int(digits) if digits else 0

    return spectra


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU clustering surrogate")
    parser.add_argument("--query", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bin-width", type=float, default=1.0)
    args = parser.parse_args()

    query = Path(args.query)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    spectra = parse_mgf_precursors(query)
    if not spectra:
        raise RuntimeError("No spectra parsed from query MGF.")

    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["identifier", "precursor_mz", "precursor_charge", "cluster"])
        for sid, precursor, charge in spectra:
            bucket = int(precursor / max(args.bin_width, 1e-6))
            cluster = f"z{charge}_b{bucket}"
            writer.writerow([sid, f"{precursor:.6f}", charge, cluster])

    print(f"Wrote {len(spectra)} clustered spectra to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
