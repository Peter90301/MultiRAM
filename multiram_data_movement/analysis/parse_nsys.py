#!/usr/bin/env python3
"""Parse Nsight Systems/Compute and GPU power logs into a one-row CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

from parse_perf import FINAL_COLUMNS


def parse_float(text: str) -> float:
    cleaned = text.strip().replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return math.nan


def parse_nsys_memcpy(path: Path) -> dict[str, float]:
    metrics = {"h2d_bytes": math.nan, "d2h_bytes": math.nan, "h2d_time_s": math.nan, "d2h_time_s": math.nan}
    if not path.exists():
        return metrics
    text = path.read_text(encoding="utf-8", errors="replace")
    h2d_bytes = 0.0
    d2h_bytes = 0.0
    h2d_time_ns = 0.0
    d2h_time_ns = 0.0
    found = False
    for line in text.splitlines():
        lower = line.lower()
        if "memcpy" not in lower:
            continue
        numbers = [parse_float(x) for x in re.findall(r"[-+]?[0-9]*\.?[0-9]+", line)]
        if not numbers:
            continue
        # Nsight text tables vary by version. Use conservative pattern matching:
        # first numeric field is commonly total time in ns, one later field may be bytes.
        time_ns = numbers[0]
        byte_candidates = [n for n in numbers if n > 1024]
        bytes_value = byte_candidates[-1] if byte_candidates else math.nan
        if "h2d" in lower or "host to device" in lower:
            found = True
            h2d_time_ns += time_ns
            if not math.isnan(bytes_value):
                h2d_bytes += bytes_value
        elif "d2h" in lower or "device to host" in lower:
            found = True
            d2h_time_ns += time_ns
            if not math.isnan(bytes_value):
                d2h_bytes += bytes_value
    if found:
        metrics["h2d_bytes"] = h2d_bytes
        metrics["d2h_bytes"] = d2h_bytes
        metrics["h2d_time_s"] = h2d_time_ns * 1e-9
        metrics["d2h_time_s"] = d2h_time_ns * 1e-9
    return metrics


def parse_ncu_csv(path: Path) -> dict[str, float]:
    metrics = {"gpu_dram_read_bytes": math.nan, "gpu_dram_write_bytes": math.nan, "gpu_dram_bytes": math.nan}
    if not path.exists() or path.stat().st_size == 0:
        return metrics
    text = path.read_text(encoding="utf-8", errors="replace")
    read_total = 0.0
    write_total = 0.0
    for line in text.splitlines():
        lower = line.lower()
        value_matches = re.findall(r"[-+]?[0-9]*\.?[0-9]+", line)
        if not value_matches:
            continue
        value = parse_float(value_matches[-1])
        if math.isnan(value):
            continue
        if "dram__bytes_read" in lower:
            read_total += value
        elif "dram__bytes_write" in lower:
            write_total += value
    if read_total > 0:
        metrics["gpu_dram_read_bytes"] = read_total
    if write_total > 0:
        metrics["gpu_dram_write_bytes"] = write_total
    if read_total > 0 or write_total > 0:
        metrics["gpu_dram_bytes"] = read_total + write_total
    return metrics


def parse_power(path: Path, total_runtime_s: float) -> dict[str, float]:
    if not path.exists() or path.stat().st_size == 0:
        return {"avg_power_w": math.nan, "energy_j": math.nan}
    values = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
        parts = [p.strip() for p in line.split(",")]
        if not parts:
            continue
        value = parse_float(parts[-1])
        if not math.isnan(value):
            values.append(value)
    if not values:
        return {"avg_power_w": math.nan, "energy_j": math.nan}
    avg = sum(values) / len(values)
    return {"avg_power_w": avg, "energy_j": avg * total_runtime_s}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsys-stats", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ncu-csv", default="")
    parser.add_argument("--power-csv", default="")
    args = parser.parse_args()

    meta: dict[str, Any] = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    row = {col: math.nan for col in FINAL_COLUMNS}
    for key in ["workload", "stage", "platform", "dataset", "regime", "run_id", "run_mode", "cache_drop_status", "raw_log_dir"]:
        row[key] = meta.get(key, "")
    row["total_runtime_s"] = meta.get("total_runtime_s", math.nan)
    if meta.get("run_mode") == "cold":
        row["cold_runtime_s"] = row["total_runtime_s"]
    elif meta.get("run_mode") == "warm":
        row["warm_runtime_s"] = row["total_runtime_s"]

    row.update(parse_nsys_memcpy(Path(args.nsys_stats)))
    if args.ncu_csv:
        row.update(parse_ncu_csv(Path(args.ncu_csv)))
    if args.power_csv:
        row.update(parse_power(Path(args.power_csv), float(row["total_runtime_s"])))
    row["notes"] = f"return_code={meta.get('return_code', '')}"

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FINAL_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

