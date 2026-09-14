#!/usr/bin/env python3
"""Parse perf stat and /usr/bin/time -v logs into a one-row CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


FINAL_COLUMNS = [
    "workload",
    "stage",
    "platform",
    "dataset",
    "regime",
    "run_id",
    "run_mode",
    "total_runtime_s",
    "cold_runtime_s",
    "warm_runtime_s",
    "storage_overhead_s",
    "storage_bytes_read",
    "storage_bytes_written",
    "cache_references",
    "cache_misses",
    "llc_loads",
    "llc_load_misses",
    "llc_stores",
    "cycles",
    "instructions",
    "major_faults",
    "minor_faults",
    "h2d_bytes",
    "d2h_bytes",
    "h2d_time_s",
    "d2h_time_s",
    "gpu_dram_bytes",
    "gpu_dram_read_bytes",
    "gpu_dram_write_bytes",
    "avg_power_w",
    "energy_j",
    "memory_stall_time_s",
    "host_dram_bytes",
    "data_movement_fraction",
    "total_data_movement_bytes",
    "multiram_data_movement_bytes",
    "multiram_reduction_percent",
    "cache_drop_status",
    "raw_log_dir",
    "notes",
]


PERF_NAME_MAP = {
    "cycles": "cycles",
    "instructions": "instructions",
    "cache-references": "cache_references",
    "cache-misses": "cache_misses",
    "LLC-loads": "llc_loads",
    "LLC-load-misses": "llc_load_misses",
    "LLC-stores": "llc_stores",
    "major-faults": "major_faults",
    "minor-faults": "minor_faults",
}


def nan() -> float:
    return math.nan


def parse_number(text: str) -> float:
    cleaned = text.strip().replace(",", "")
    if cleaned in {"", "<not counted>", "<not supported>"}:
        return nan()
    try:
        return float(cleaned)
    except ValueError:
        return nan()


def parse_perf(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if not path.exists():
        return metrics
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        value = parse_number(parts[0])
        event = parts[2]
        if event in PERF_NAME_MAP:
            metrics[PERF_NAME_MAP[event]] = value
    return metrics


def parse_time_v(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if not path.exists():
        return metrics
    text = path.read_text(encoding="utf-8", errors="replace")
    patterns = {
        "user_time_s": r"User time \(seconds\):\s*([0-9.]+)",
        "system_time_s": r"System time \(seconds\):\s*([0-9.]+)",
        "max_rss_kb": r"Maximum resident set size \(kbytes\):\s*([0-9]+)",
        "major_faults": r"Major \(requiring I/O\) page faults:\s*([0-9]+)",
        "minor_faults": r"Minor \(reclaiming a frame\) page faults:\s*([0-9]+)",
        "fs_inputs": r"File system inputs:\s*([0-9]+)",
        "fs_outputs": r"File system outputs:\s*([0-9]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            metrics[key] = parse_number(match.group(1))
    return metrics


def load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_row(meta: dict[str, Any], perf: dict[str, float], time_v: dict[str, float], storage_block_size: int) -> dict[str, Any]:
    row = {col: math.nan for col in FINAL_COLUMNS}
    for key in ["workload", "stage", "platform", "dataset", "regime", "run_id", "run_mode", "cache_drop_status", "raw_log_dir"]:
        row[key] = meta.get(key, "")
    row["total_runtime_s"] = meta.get("total_runtime_s", math.nan)
    if meta.get("run_mode") == "cold":
        row["cold_runtime_s"] = row["total_runtime_s"]
    elif meta.get("run_mode") == "warm":
        row["warm_runtime_s"] = row["total_runtime_s"]

    row["storage_bytes_read"] = time_v.get("fs_inputs", math.nan) * storage_block_size if "fs_inputs" in time_v else math.nan
    row["storage_bytes_written"] = time_v.get("fs_outputs", math.nan) * storage_block_size if "fs_outputs" in time_v else math.nan

    for key, value in perf.items():
        row[key] = value
    for key in ["major_faults", "minor_faults"]:
        if key in time_v:
            row[key] = time_v[key]

    power = meta.get("power_w", math.nan)
    if isinstance(power, (int, float)) and not math.isnan(float(power)):
        row["avg_power_w"] = float(power)
        row["energy_j"] = float(power) * float(row["total_runtime_s"])
    row["notes"] = f"return_code={meta.get('return_code', '')}"
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perf-log", required=True)
    parser.add_argument("--time-log", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--storage-block-size", type=int, default=1024)
    args = parser.parse_args()

    meta = load_metadata(Path(args.metadata))
    row = build_row(meta, parse_perf(Path(args.perf_log)), parse_time_v(Path(args.time_log)), args.storage_block_size)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FINAL_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

