#!/usr/bin/env python3
"""Compute data-movement fractions and MultiRAM reduction percentages."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


NUMERIC_COLUMNS = [
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
    "h2d_bytes",
    "d2h_bytes",
    "h2d_time_s",
    "d2h_time_s",
    "gpu_dram_bytes",
    "avg_power_w",
    "energy_j",
    "memory_stall_time_s",
    "host_dram_bytes",
    "data_movement_fraction",
    "total_data_movement_bytes",
    "multiram_data_movement_bytes",
    "multiram_reduction_percent",
]


def clean_number(value) -> float:
    try:
        if pd.isna(value):
            return math.nan
        return float(value)
    except Exception:
        return math.nan


def multiram_bytes(config: dict, stage: str) -> float:
    stage_cfg = config.get("multiram_movement_bytes", {}).get(stage, {})
    return float(sum(clean_number(v) if not math.isnan(clean_number(v)) else 0.0 for v in stage_cfg.values()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--llc-miss-bytes",
        type=int,
        default=64,
        help="Fallback host DRAM byte estimate per LLC miss.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Pair cold/warm runs by workload identity when both exist.
    key_cols = ["workload", "stage", "platform", "dataset", "regime", "run_id"]
    if all(c in df.columns for c in key_cols + ["run_mode", "total_runtime_s"]):
        warm = df[df["run_mode"] == "warm"][key_cols + ["total_runtime_s"]].rename(columns={"total_runtime_s": "paired_warm_runtime_s"})
        df = df.merge(warm, on=key_cols, how="left")
        cold_mask = df["run_mode"] == "cold"
        df.loc[cold_mask, "warm_runtime_s"] = df.loc[cold_mask, "paired_warm_runtime_s"]
        df.loc[cold_mask, "cold_runtime_s"] = df.loc[cold_mask, "total_runtime_s"]
        df["storage_overhead_s"] = df["cold_runtime_s"] - df["warm_runtime_s"]
        df.loc[df["storage_overhead_s"] < 0, "storage_overhead_s"] = 0
        df = df.drop(columns=["paired_warm_runtime_s"])

    if "host_dram_bytes" not in df.columns:
        df["host_dram_bytes"] = pd.NA
    llc_misses = pd.to_numeric(df.get("llc_load_misses", pd.Series(index=df.index, dtype=float)), errors="coerce")
    df["host_dram_bytes"] = pd.to_numeric(df["host_dram_bytes"], errors="coerce")
    df["host_dram_bytes"] = df["host_dram_bytes"].fillna(llc_misses * args.llc_miss_bytes)

    for col in ["storage_bytes_read", "storage_bytes_written", "h2d_bytes", "d2h_bytes", "gpu_dram_bytes", "host_dram_bytes"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["total_data_movement_bytes"] = (
        df["storage_bytes_read"]
        + df["storage_bytes_written"]
        + df["host_dram_bytes"]
        + df["h2d_bytes"]
        + df["d2h_bytes"]
        + df["gpu_dram_bytes"]
    )

    df["multiram_data_movement_bytes"] = df["stage"].map(lambda s: multiram_bytes(config, str(s)))
    denom = df["total_data_movement_bytes"].replace(0, pd.NA)
    df["multiram_reduction_percent"] = (1.0 - df["multiram_data_movement_bytes"] / denom) * 100.0

    for col in ["storage_overhead_s", "memory_stall_time_s", "h2d_time_s", "d2h_time_s"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["data_movement_fraction"] = (
        df["storage_overhead_s"] + df["memory_stall_time_s"] + df["h2d_time_s"] + df["d2h_time_s"]
    ) / pd.to_numeric(df["total_runtime_s"], errors="coerce").replace(0, pd.NA)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out} rows={len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

