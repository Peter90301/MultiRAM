#!/usr/bin/env python3
"""Print status for the Platinum NA12878 full minimap2 run."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path


ROOT = Path("/home/tsl012/multiomic/multiram_data_movement")
RAW = ROOT / "results_platinum_na12878_hg38/raw_logs/genomics_alignment/cpu/storage_bound/full/run_0"
PAIR_TIMING = RAW / "pair_timing.tsv"
MANIFEST = Path("/mnt/hdd/tsunghan/raw-ms-dataset/genomic_large_Dset/PRJEB3246_NA12878_paired_fastqs.tsv")
PID_FILE = RAW / "run.pid"


def main() -> int:
    manifest_rows = list(csv.DictReader(MANIFEST.open(newline="", encoding="utf-8"), delimiter="\t"))
    total_pairs = len(manifest_rows)
    total_bytes = sum(int(row["r1_bytes"]) + int(row["r2_bytes"]) for row in manifest_rows)

    completed_rows = []
    if PAIR_TIMING.exists():
        completed_rows = list(csv.DictReader(PAIR_TIMING.open(newline="", encoding="utf-8"), delimiter="\t"))
    completed_pairs = len(completed_rows)
    completed_bytes = sum(int(row.get("input_bytes") or 0) for row in completed_rows)
    completed_runtime = sum(float(row.get("runtime_s") or 0.0) for row in completed_rows)

    pid = PID_FILE.read_text(encoding="utf-8").strip() if PID_FILE.exists() else ""
    running = False
    if pid and Path(f"/proc/{pid}").exists():
        running = True

    eta = None
    if completed_bytes > 0 and completed_runtime > 0:
        remaining = max(0, total_bytes - completed_bytes)
        eta = remaining / (completed_bytes / completed_runtime)

    status = {
        "running": running,
        "pid": pid,
        "completed_pairs": completed_pairs,
        "total_pairs": total_pairs,
        "completed_input_bytes": completed_bytes,
        "total_input_bytes": total_bytes,
        "completed_percent": completed_bytes / total_bytes * 100 if total_bytes else 0,
        "completed_runtime_s": completed_runtime,
        "eta_s_from_completed_pairs": eta,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
