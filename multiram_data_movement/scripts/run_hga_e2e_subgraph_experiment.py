#!/usr/bin/env python3
"""Benchmark minigraph and candidate-filtered HGA on matched MHC subgraphs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from run_data_movement_time_experiment import (
    MINIGRAPH,
    NSYS,
    Workload,
    machine_metadata,
    profile_gpu,
    run_once,
)


ROOT = Path("/home/tsl012/multiomic/multiram_data_movement")
DEFAULT_OUTPUT = ROOT / "results_hga_e2e_subgraph_comparison"
DEFAULT_MANIFEST = DEFAULT_OUTPUT / "inputs/manifest.json"
HGA_MAPPER = ROOT / "scripts/run_hga_candidate_mapper.py"
EVALUATOR = ROOT / "analysis/evaluate_hga_e2e_mappers.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--gpu-index", type=int, default=1)
    parser.add_argument("--skip-nsys", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    reference = Path(manifest["combined_subgraph_fasta"])
    reads = Path(manifest["reads_fastq"])
    truth = Path(manifest["truth_sam"])
    graphs = [Path(item["hga_graph"]) for item in manifest["subgraphs"]]
    for path in [reference, reads, truth, MINIGRAPH, HGA_MAPPER, *graphs]:
        if not path.exists():
            raise FileNotFoundError(path)

    runtime_path = args.output_dir / "runtime_records.jsonl"
    if runtime_path.exists():
        raise FileExistsError(
            f"{runtime_path} exists; use a fresh output directory"
        )
    if not args.skip_nsys and not NSYS.exists():
        raise FileNotFoundError(NSYS)

    def minigraph_command(run_dir: Path) -> list[str]:
        return [
            str(MINIGRAPH),
            "-x",
            "sr",
            "-c",
            "-t",
            "16",
            str(reference),
            str(reads),
            "-o",
            str(run_dir / "minigraph.gaf"),
        ]

    def hga_mapper_command(run_dir: Path) -> list[str]:
        return [
            sys.executable,
            str(HGA_MAPPER),
            "--manifest",
            str(args.manifest),
            "--output-dir",
            str(run_dir),
            "--threads",
            "16",
            "--gpu-blocks",
            "8",
            "--gpu-threads",
            "128",
        ]

    workloads = [
        Workload(
            "cpu_minigraph_e2e",
            "genomics",
            "cpu",
            [reference, reads],
            minigraph_command,
        ),
        Workload(
            "gpu_candidate_hga_e2e",
            "genomics",
            "gpu",
            [reference, reads, *graphs],
            hga_mapper_command,
        ),
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = machine_metadata(args.gpu_index)
    metadata.update(
        {
            "experiment": "end-to-end matched MHC subgraph mapper comparison",
            "manifest": manifest,
            "trials": args.trials,
            "cpu_pipeline": "minigraph -x sr -c",
            "gpu_pipeline": "minimap2 candidate routing + HGA DP + endpoint PAF",
            "accuracy_metric": "correct subgraph and endpoint within 20 bp",
            "hga_changes": [
                "PTX fallback for compute capability 12.0",
                "CUDA launch error checking",
                "kernel launched on the H2D stream",
                "packed-read layout correction",
                "score and endpoint D2H output",
            ],
        }
    )
    (args.output_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    for workload in workloads:
        for trial in range(1, args.trials + 1):
            run_once(workload, "cold", trial, args.output_dir, env)
            run_once(workload, "warm", trial, args.output_dir, env)
        if workload.platform == "gpu" and not args.skip_nsys:
            profile_gpu(workload, args.output_dir, env)

    quality_dir = args.output_dir / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)
    minigraph_gaf = (
        args.output_dir
        / "raw/cpu_minigraph_e2e/warm/trial_1/minigraph.gaf"
    )
    hga_paf = (
        args.output_dir
        / "raw/gpu_candidate_hga_e2e/warm/trial_1/hga_mapper.paf"
    )
    quality_path = quality_dir / "mapping_quality.json"
    quality_command = [
        sys.executable,
        str(EVALUATOR),
        "--truth-sam",
        str(truth),
        "--minigraph-gaf",
        str(minigraph_gaf),
        "--hga-paf",
        str(hga_paf),
        "--output",
        str(quality_path),
        "--tolerance",
        "20",
    ]
    completed = subprocess.run(
        quality_command, text=True, capture_output=True, check=False
    )
    (quality_dir / "evaluation.stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    (quality_dir / "evaluation.stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr)
    print(completed.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
