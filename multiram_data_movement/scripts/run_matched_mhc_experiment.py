#!/usr/bin/env python3
"""Run matched MHC minigraph CPU and HGA GPU measurements."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from run_data_movement_time_experiment import (
    HGA,
    MINIGRAPH,
    NSYS,
    Workload,
    machine_metadata,
    profile_gpu,
    run_once,
)


ROOT = Path("/home/tsl012/multiomic/multiram_data_movement")
DEFAULT_RESULTS = ROOT / "results_matched_mhc_minigraph_hga_pim"
DEFAULT_MANIFEST = DEFAULT_RESULTS / "inputs/matched_mhc_manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--gpu-index", type=int, default=1)
    parser.add_argument("--skip-nsys", action="store_true")
    args = parser.parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(
            f"Missing {args.manifest}; run prepare_matched_mhc_inputs.py first"
        )
    if not NSYS.exists() and not args.skip_nsys:
        raise FileNotFoundError(NSYS)
    if (args.output_dir / "runtime_records.jsonl").exists():
        raise FileExistsError(
            f"{args.output_dir}/runtime_records.jsonl already exists; "
            "use a fresh output directory to avoid mixing trials"
        )

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validation = manifest.get("validation", {})
    if not all(validation.values()):
        raise RuntimeError(f"Input validation did not pass: {validation}")

    reference = Path(manifest["reference_fasta"])
    query_fastq = Path(manifest["query_fastq"])
    hga_graph = Path(manifest["hga_graph"])
    hga_reads = Path(manifest["hga_reads"])
    for path in (reference, query_fastq, hga_graph, hga_reads, MINIGRAPH, HGA):
        if not path.exists():
            raise FileNotFoundError(path)

    def minigraph_command(_: Path) -> list[str]:
        return [
            str(MINIGRAPH),
            "-x",
            "sr",
            "-c",
            "-t",
            "16",
            str(reference),
            str(query_fastq),
            "-o",
            "/dev/null",
        ]

    def hga_command(_: Path) -> list[str]:
        return [
            str(HGA),
            "-g",
            str(hga_graph),
            "-r",
            str(hga_reads),
            "-m",
            "2",
            "-n",
            "2",
            "-o",
            "3",
            "-d",
            "1",
            "-b",
            "32",
            "-t",
            "128",
            "-c",
            "1",
        ]

    workloads = [
        Workload(
            "matched_mhc_cpu_minigraph",
            "genomics",
            "cpu",
            [reference, query_fastq],
            minigraph_command,
            True,
        ),
        Workload(
            "matched_mhc_gpu_hga",
            "genomics",
            "gpu",
            [hga_graph, hga_reads],
            hga_command,
        ),
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = machine_metadata(args.gpu_index)
    metadata.update(
        {
            "experiment": "matched MHC minigraph/HGA/PIM",
            "manifest_path": str(args.manifest.resolve()),
            "manifest": manifest,
            "trials": args.trials,
            "comparison_scope": (
                "same logical graph and reads; different algorithms/output semantics"
            ),
        }
    )
    (args.output_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu_index),
            "OMP_NUM_THREADS": "8",
            "OPENBLAS_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
            "NUMEXPR_NUM_THREADS": "8",
        }
    )
    for workload in workloads:
        for trial in range(1, args.trials + 1):
            run_once(workload, "cold", trial, args.output_dir, env)
            run_once(workload, "warm", trial, args.output_dir, env)
        if workload.platform == "gpu" and not args.skip_nsys:
            profile_gpu(workload, args.output_dir, env)

    quality_dir = args.output_dir / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)
    gaf = quality_dir / "minigraph_mappings.gaf"
    quality_command = [
        str(MINIGRAPH),
        "-x",
        "sr",
        "-c",
        "-t",
        "16",
        str(reference),
        str(query_fastq),
        "-o",
        str(gaf),
    ]
    quality = subprocess.run(
        quality_command,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    (quality_dir / "minigraph_stderr.log").write_text(
        quality.stderr, encoding="utf-8"
    )
    if quality.returncode != 0:
        raise RuntimeError("minigraph quality run failed")
    mapping_count = sum(1 for _ in gaf.open("r", encoding="utf-8"))
    quality_record = {
        "command": quality_command,
        "query_count": int(manifest["query_count"]),
        "gaf_mapping_records": mapping_count,
        "mapped_record_fraction_percent": (
            100.0 * mapping_count / int(manifest["query_count"])
        ),
        "note": (
            "GAF records are mappings, not a truth-labeled mapping-accuracy metric; "
            "a query may emit more than one record."
        ),
    }
    (quality_dir / "minigraph_quality_summary.json").write_text(
        json.dumps(quality_record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(quality_record, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
