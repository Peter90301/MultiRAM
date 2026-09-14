#!/usr/bin/env python3
"""Create example profiling commands for the MultiRAM data-movement matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


WORKLOADS = {
    ("genomics_alignment", "cpu"): ROOT / "workloads/genomics/genomics_minimap2.sh",
    ("proteomics_clustering", "cpu"): ROOT / "workloads/proteomics_clustering/proteomics_clustering_cpu.sh",
    ("proteomics_clustering", "gpu"): ROOT / "workloads/proteomics_clustering/proteomics_clustering_gpu.sh",
    ("oms", "cpu"): ROOT / "workloads/oms/oms_cpu.sh",
    ("oms", "gpu"): ROOT / "workloads/oms/oms_gpu.sh",
}


def q(path: Path | str) -> str:
    return "'" + str(path).replace("'", "'\"'\"'") + "'"


def dataset_name(config: dict, stage: str) -> str:
    if stage == "genomics_alignment":
        return config.get("workloads", {}).get("genomics", {}).get("dataset_name", "dataset")
    if stage == "proteomics_clustering":
        return config.get("workloads", {}).get("proteomics_clustering", {}).get("dataset_name", "dataset")
    if stage == "oms":
        return config.get("workloads", {}).get("oms", {}).get("dataset_name", "dataset")
    return "dataset"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["genomics_alignment", "proteomics_clustering", "oms"], required=True)
    parser.add_argument("--platform", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--regime", choices=["storage_bound", "dram_bound"], required=True)
    parser.add_argument("--run-id", default="0")
    parser.add_argument("--run-mode", choices=["cold", "warm"], default="warm")
    parser.add_argument("--execute", action="store_true", help="Execute instead of printing the command.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    wrapper = WORKLOADS.get((args.stage, args.platform))
    if wrapper is None:
        raise SystemExit(f"No wrapper for stage={args.stage} platform={args.platform}")

    profiler = ROOT / ("profiling/profile_cpu.sh" if args.platform == "cpu" else "profiling/profile_gpu.sh")
    command = [
        str(profiler),
        "--config",
        str(config_path),
        "--workload",
        args.stage,
        "--stage",
        args.stage,
        "--platform",
        args.platform,
        "--dataset",
        dataset_name(config, args.stage),
        "--regime",
        args.regime,
        "--run-id",
        args.run_id,
        "--run-mode",
        args.run_mode,
        "--",
        str(wrapper),
        str(config_path),
    ]
    rendered = " ".join(q(part) for part in command)
    if args.execute:
        import subprocess

        raise SystemExit(subprocess.call(command))
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

