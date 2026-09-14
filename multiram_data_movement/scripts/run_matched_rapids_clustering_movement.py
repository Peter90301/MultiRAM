#!/usr/bin/env python3
"""Measure matched RAPIDS spectral-clustering movement on a small PXD MGF."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/tsl012/multiomic/multiram_data_movement")
SCRIPT_DIR = ROOT / "scripts"
DEFAULT_OUTPUT = ROOT / "results_rapids_matched_clustering"
DEFAULT_MGF = Path(
    "/mnt/hdd/tsunghan/raw-ms-dataset/proteomic_large_dataset/derived/ms2_mgf/"
    "20151220_alr_CompleteHumanProteome_HUVEC_LysC_ETD_fr9.mgf"
)
PYTHON = Path("/home/tsl012/miniconda3/envs/multiomic/bin/python")
RUNNER = Path(
    "/home/tsl012/multiomic/proteomic_full_pipeline/"
    "run_hyperspec_gpu_cluster.py"
)

sys.path.insert(0, str(SCRIPT_DIR))
from run_data_movement_time_experiment import (  # noqa: E402
    Workload,
    machine_metadata,
    profile_gpu,
    run_once,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def count_spectra(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for line in handle:
            if line.startswith(b"BEGIN IONS"):
                count += 1
    return count


def package_version(module: str) -> str:
    result = subprocess.run(
        [str(PYTHON), "-c", f"import {module}; print({module}.__version__)"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() or "unknown"


def build_workload(mgf: Path) -> Workload:
    def command(run_dir: Path) -> list[str]:
        run_dir.mkdir(parents=True, exist_ok=True)
        return [
            str(PYTHON),
            str(RUNNER),
            "--input",
            str(mgf),
            "--output-prefix",
            str(run_dir / "clusters"),
            "--cpu-core-preprocess",
            "8",
            "--cpu-core-cluster",
            "8",
            "--batch-size",
            "5000",
        ]

    return Workload(
        name="proteomics_gpu_rapids",
        domain="proteomics_clustering",
        platform="gpu",
        input_paths=[mgf],
        command_factory=command,
    )


def summarize(
    output_dir: Path,
    mgf: Path,
    runtime_records: list[dict[str, object]],
    transfer: dict[str, object],
    gpu_index: int,
) -> dict[str, object]:
    cold = [
        float(record["wall_s"])
        for record in runtime_records
        if record["mode"] == "cold"
    ]
    warm = [
        float(record["wall_s"])
        for record in runtime_records
        if record["mode"] == "warm"
    ]
    cold_median = statistics.median(cold)
    warm_median = statistics.median(warm)
    storage_s = max(0.0, cold_median - warm_median)
    copy_s = float(transfer["copy_union_time_s"])

    copy_by_direction = transfer["copy_by_direction"]
    assert isinstance(copy_by_direction, dict)
    h2d_bytes = sum(
        int(value["bytes"])
        for key, value in copy_by_direction.items()
        if "Host-to-Device" in key
    )
    d2h_bytes = sum(
        int(value["bytes"])
        for key, value in copy_by_direction.items()
        if "Device-to-Host" in key
    )
    explicit_movement_s = storage_s + copy_s

    gpu = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()

    return {
        "status": "measured_small_input",
        "created_at": now(),
        "scope": {
            "input_mgf": str(mgf),
            "input_bytes": mgf.stat().st_size,
            "raw_spectra": count_spectra(mgf),
            "trials_per_cache_state": len(cold),
        },
        "software": {
            "rapids_component": "cuML DBSCAN, metric=precomputed",
            "rapids_version": package_version("cuml"),
            "cupy_version": package_version("cupy"),
            "pipeline_frontend": (
                "Hyper-Spec MGF preprocessing and 2048-bit HDC representation"
            ),
            "runner": str(RUNNER),
        },
        "measured_hardware": {
            "hostname": platform.node(),
            "gpu_index": gpu_index,
            "gpu": gpu,
            "note": (
                "Movement was measured on this local GPU. A100/H100 compute "
                "rows are projections from Fig. 18, not local hardware runs."
            ),
        },
        "runtime": {
            "cold_trials_s": cold,
            "warm_trials_s": warm,
            "cold_median_s": cold_median,
            "warm_median_s": warm_median,
        },
        "movement": {
            "storage_cold_minus_warm_s": storage_s,
            "cuda_copy_union_s": copy_s,
            "explicit_movement_s": explicit_movement_s,
            "explicit_movement_fraction_of_cold_percent": (
                100.0 * explicit_movement_s / cold_median
            ),
            "h2d_bytes": h2d_bytes,
            "d2h_bytes": d2h_bytes,
            "copy_by_direction": copy_by_direction,
            "boundary": (
                "Cold-minus-warm storage service plus the union of CUDA "
                "memcpy intervals. GPU HBM traffic inside kernels is excluded."
            ),
        },
        "artifacts": {
            "output_dir": str(output_dir),
            "runtime_records": str(output_dir / "runtime_records.jsonl"),
            "gpu_transfer_records": str(
                output_dir / "gpu_transfer_records.jsonl"
            ),
            "nsys_report": str(
                output_dir
                / "raw/proteomics_gpu_rapids/nsys/profile.nsys-rep"
            ),
        },
    }


def write_report(output_dir: Path, summary: dict[str, object]) -> None:
    scope = summary["scope"]
    runtime = summary["runtime"]
    movement = summary["movement"]
    hardware = summary["measured_hardware"]
    software = summary["software"]
    assert isinstance(scope, dict)
    assert isinstance(runtime, dict)
    assert isinstance(movement, dict)
    assert isinstance(hardware, dict)
    assert isinstance(software, dict)

    lines = [
        "# Matched RAPIDS Clustering Movement",
        "",
        "## Scope",
        "",
        f"- Input: `{scope['input_mgf']}`",
        f"- Input size: {int(scope['input_bytes']):,} bytes",
        f"- Raw spectra: {int(scope['raw_spectra']):,}",
        f"- RAPIDS component: {software['rapids_component']}",
        f"- cuML: {software['rapids_version']}",
        f"- GPU used for movement measurement: {hardware['gpu']}",
        "",
        "The runner uses Hyper-Spec only for MGF preprocessing, spectral HDC "
        "encoding, and bucketing. The clustering operation itself is RAPIDS "
        "cuML DBSCAN. This is therefore a matched RAPIDS measurement rather "
        "than a movement proxy from a different input.",
        "",
        "## Measured Results",
        "",
        "| Metric | Result |",
        "| --- | ---: |",
        f"| Cold median | {float(runtime['cold_median_s']):.6f} s |",
        f"| Warm median | {float(runtime['warm_median_s']):.6f} s |",
        (
            "| Storage service, cold - warm | "
            f"{float(movement['storage_cold_minus_warm_s']):.6f} s |"
        ),
        (
            "| CUDA copy union | "
            f"{float(movement['cuda_copy_union_s']):.6f} s |"
        ),
        (
            "| Explicit movement | "
            f"{float(movement['explicit_movement_s']):.6f} s |"
        ),
        (
            "| Movement / cold overall | "
            f"{float(movement['explicit_movement_fraction_of_cold_percent']):.3f}% |"
        ),
        f"| H2D bytes | {int(movement['h2d_bytes']):,} |",
        f"| D2H bytes | {int(movement['d2h_bytes']):,} |",
        "",
        "## Interpretation",
        "",
        "The movement value above is measured on the local RTX PRO 6000 "
        "Blackwell GPU. It is used as the matched movement anchor in Table VI. "
        "A100 and H100 compute times remain projections from Fig. 18 normalized "
        "throughput; they are not claimed as A100/H100 hardware measurements.",
        "",
        "GPU HBM traffic generated inside RAPIDS kernels is outside the measured "
        "boundary because hardware HBM counters were not collected.",
    ]
    (output_dir / "MATCHED_RAPIDS_MOVEMENT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_MGF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--gpu-index", type=int, default=1)
    args = parser.parse_args()

    mgf = args.input.resolve()
    output_dir = args.output_dir.resolve()
    if not mgf.exists():
        raise FileNotFoundError(mgf)
    output_dir.mkdir(parents=True, exist_ok=True)

    workload = build_workload(mgf)
    metadata = machine_metadata(args.gpu_index)
    metadata.update(
        {
            "input_mgf": str(mgf),
            "input_bytes": mgf.stat().st_size,
            "trials": args.trials,
            "rapids_version": package_version("cuml"),
            "pipeline_boundary": (
                "Hyper-Spec preprocessing/HDC plus RAPIDS cuML DBSCAN"
            ),
        }
    )
    (output_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu_index),
            "LOKY_MAX_CPU_COUNT": "8",
            "OMP_NUM_THREADS": "8",
            "OPENBLAS_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
            "NUMEXPR_NUM_THREADS": "8",
            "CCCL_IGNORE_DEPRECATED_CUDA_BELOW_12": "1",
            "CUPY_CACHE_IN_MEMORY": "0",
        }
    )

    runtime_records: list[dict[str, object]] = []
    for trial in range(1, args.trials + 1):
        runtime_records.append(
            run_once(workload, "cold", trial, output_dir, env)
        )
        runtime_records.append(
            run_once(workload, "warm", trial, output_dir, env)
        )
    transfer = profile_gpu(workload, output_dir, env)

    summary = summarize(
        output_dir, mgf, runtime_records, transfer, args.gpu_index
    )
    (output_dir / "matched_rapids_movement.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_report(output_dir, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
