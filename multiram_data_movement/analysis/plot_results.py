#!/usr/bin/env python3
"""Generate publication-ready data-movement figures with matplotlib."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def label_series(df: pd.DataFrame) -> pd.Series:
    return df["stage"].astype(str) + "\n" + df["platform"].astype(str) + "\n" + df["regime"].astype(str)


def save(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=300)
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def plot_runtime_breakdown(df: pd.DataFrame, output_dir: Path) -> None:
    cols = ["storage_overhead_s", "memory_stall_time_s", "h2d_time_s", "d2h_time_s"]
    plot_df = df.copy()
    for col in cols:
        if col not in plot_df:
            plot_df[col] = 0.0
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce").fillna(0.0)
    plot_df["compute_or_other_s"] = (
        pd.to_numeric(plot_df["total_runtime_s"], errors="coerce").fillna(0.0)
        - plot_df[cols].sum(axis=1)
    ).clip(lower=0.0)
    labels = label_series(plot_df)
    fig, ax = plt.subplots(figsize=(max(8, len(plot_df) * 0.65), 5))
    bottom = None
    for col in ["compute_or_other_s"] + cols:
        values = plot_df[col].to_numpy()
        ax.bar(labels, values, bottom=bottom, label=col)
        bottom = values if bottom is None else bottom + values
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Runtime Breakdown")
    ax.tick_params(axis="x", labelrotation=45)
    ax.legend(fontsize=8)
    save(fig, output_dir, "runtime_breakdown")


def plot_movement_volume(df: pd.DataFrame, output_dir: Path) -> None:
    labels = label_series(df)
    baseline = pd.to_numeric(df.get("total_data_movement_bytes"), errors="coerce").fillna(0.0) / 1e9
    multiram = pd.to_numeric(df.get("multiram_data_movement_bytes"), errors="coerce").fillna(0.0) / 1e9
    x = range(len(df))
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.65), 5))
    ax.bar([i - 0.2 for i in x], baseline, width=0.4, label="CPU/GPU baseline")
    ax.bar([i + 0.2 for i in x], multiram, width=0.4, label="MultiRAM modeled")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Data movement (GB)")
    ax.set_title("Data Movement Volume")
    ax.legend()
    save(fig, output_dir, "data_movement_volume")


def plot_reduction(df: pd.DataFrame, output_dir: Path) -> None:
    labels = label_series(df)
    reduction = pd.to_numeric(df.get("multiram_reduction_percent"), errors="coerce")
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.65), 4))
    ax.bar(labels, reduction)
    ax.set_ylabel("Reduction (%)")
    ax.set_ylim(bottom=min(0, reduction.min(skipna=True) if reduction.notna().any() else 0), top=100)
    ax.set_title("MultiRAM Data-Movement Reduction")
    ax.tick_params(axis="x", labelrotation=45)
    save(fig, output_dir, "multiram_reduction_percent")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    out = Path(args.output_dir)
    if df.empty:
        print("[WARN] input CSV is empty; no plots generated")
        return 0
    plot_runtime_breakdown(df, out)
    plot_movement_volume(df, out)
    plot_reduction(df, out)
    print(f"Wrote plots to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

