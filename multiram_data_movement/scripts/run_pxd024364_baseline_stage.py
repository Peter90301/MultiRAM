#!/usr/bin/env python3
"""Run a resumable CPU or GPU proteomics baseline over converted PXD024364 MGF."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from run_pxd024364_raw_conversion import (
    parse_time_log,
    prior_outcomes,
    start_monitor,
    stop_monitor,
)


ROOT = Path("/mnt/hdd/tsunghan/raw-ms-dataset/proteomic_large_dataset")
MANIFEST = ROOT / "manifests/selected_raw_manifest.tsv"
MGF_ROOT = ROOT / "derived/ms2_mgf"
RESULT_ROOT = ROOT / "results"
FALCON = Path("/home/tsl012/miniconda3/envs/falconrun/bin/falcon")
HYPERSPEC_PYTHON = Path("/home/tsl012/miniconda3/envs/multiomic/bin/python")
HYPERSPEC_RUNNER = Path(
    "/home/tsl012/multiomic/proteomic_full_pipeline/run_hyperspec_gpu_cluster.py"
)
HOMSTC_PYTHON = Path("/home/tsl012/miniconda3/envs/homstc/bin/python")
MGF_TO_MSP = Path(
    "/home/tsl012/multiomic/proteomic_full_pipeline/convert_mgf_to_spectrast_msp.py"
)
SPECTRAST = Path("/home/tsl012/.local/tpp-5.0.0/bin/spectrast")
CPU_REF = Path(
    "/mnt/hdd/tsunghan/raw-ms-dataset/homs-tc-dataset/ref/"
    "massive_human_hcd_unique_targetdecoy_spectrast.splib"
)
ANNSOLO_PYTHON = Path("/home/tsl012/miniconda3/envs/annsolo/bin/python")
GPU_REF = Path(
    "/mnt/hdd/tsunghan/raw-ms-dataset/homs-tc-dataset/ref/"
    "massive_human_hcd_unique_targetdecoy.splib"
)
STAGES = ("cpu_clustering", "gpu_clustering", "cpu_oms", "gpu_oms")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def manifest_rows() -> list[dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def done_paths(result_path: Path) -> set[str]:
    done: set[str] = set()
    if not result_path.exists():
        return done
    for line in result_path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("exit_status") == 0:
            done.add(str(record["relative_path"]))
    return done


def count_lines(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    result = subprocess.run(
        ["rg", "-c", pattern, str(path)], text=True, capture_output=True, check=False
    )
    return int(result.stdout.strip() or 0) if result.returncode in (0, 1) else 0


def stage_command(
    stage: str, mgf: Path, output_dir: Path, gpu_index: int
) -> tuple[list[str], Path, Path | None]:
    token = output_dir.name
    if stage == "cpu_clustering":
        prefix = output_dir / "falcon_clusters.csv"
        return (
            [
                str(FALCON),
                "--precursor_tol", "20", "ppm",
                "--fragment_tol", "0.05",
                "--eps", "0.10",
                "--n_neighbors", "16",
                "--n_neighbors_ann", "32",
                "--n_probe", "8",
                "--batch_size", "32768",
                str(mgf),
                str(prefix),
            ],
            Path(str(prefix) + ".csv"),
            None,
        )
    if stage == "gpu_clustering":
        prefix = output_dir / "hyperspec_clusters"
        return (
            [
                "env", f"CUDA_VISIBLE_DEVICES={gpu_index}",
                str(HYPERSPEC_PYTHON), str(HYPERSPEC_RUNNER),
                "--input", str(mgf),
                "--output-prefix", str(prefix),
                "--cpu-core-preprocess", "8",
                "--cpu-core-cluster", "8",
                "--batch-size", "5000",
            ],
            prefix.with_suffix(".parquet"),
            None,
        )
    if stage == "cpu_oms":
        msp = output_dir / f"query_{token}.msp"
        output = output_dir / f"query_{token}.pep.xml"
        quote = lambda value: shlex.quote(str(value))
        shell = (
            f"{quote(HOMSTC_PYTHON)} {quote(MGF_TO_MSP)} "
            f"--input {quote(mgf)} --output {quote(msp)} && "
            f"{quote(SPECTRAST)} -sEpep.xml -sO{quote(output_dir)} "
            f"-sL{quote(CPU_REF)} {quote(msp)}"
        )
        return (["bash", "-lc", shell], output, msp)

    output = output_dir / "ann_solo_output.mztab"
    return (
        [
            "env", f"CUDA_VISIBLE_DEVICES={gpu_index}", "NUMBA_DISABLE_JIT=1",
            str(ANNSOLO_PYTHON), "-m", "ann_solo.ann_solo",
            str(GPU_REF), str(mgf), str(output),
            "--precursor_tolerance_mass", "5",
            "--precursor_tolerance_mode", "ppm",
            "--precursor_tolerance_mass_open", "500",
            "--precursor_tolerance_mode_open", "Da",
            "--fragment_mz_tolerance", "0.05",
            "--remove_precursor", "--remove_precursor_tolerance", "0.05",
            "--min_mz", "101", "--max_mz", "1500",
            "--min_intensity", "0.01", "--min_peaks", "10",
            "--min_mz_range", "250", "--max_peaks_used", "50",
            "--max_peaks_used_library", "50", "--scaling", "rank",
            "--fdr", "0.01", "--fdr_min_group_size", "20",
            "--mode", "ann", "--num_candidates", "1024",
            "--batch_size", "8000", "--num_list", "256",
            "--num_probe", "128", "--model", "none",
        ],
        output,
        None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--gpu-index", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=1,
        help="Maximum attempts per input before recording and skipping it.",
    )
    args = parser.parse_args()

    stage_dir = RESULT_ROOT / args.stage
    output_root = stage_dir / "outputs"
    log_root = stage_dir / "per_file_logs"
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    result_path = stage_dir / "metrics.jsonl"
    done, failure_counts = prior_outcomes(result_path)
    converted, _ = prior_outcomes(RESULT_ROOT / "raw_conversion/metrics.jsonl")
    rows = [
        row
        for row in manifest_rows()
        if row["relative_path"] in converted
        and row["relative_path"] not in done
        and failure_counts.get(row["relative_path"], 0) < args.max_attempts
    ]
    if args.limit:
        rows = rows[: args.limit]

    monitors = [
        start_monitor(["iostat", "-dx", "1"], stage_dir / "iostat.log"),
        start_monitor(["sar", "-u", "1"], stage_dir / "sar_cpu.log"),
    ]
    if args.stage.startswith("gpu_"):
        monitors.append(
            start_monitor(
                [
                    "nvidia-smi",
                    f"--id={args.gpu_index}",
                    "--query-gpu=timestamp,index,power.draw,memory.used,utilization.gpu,utilization.memory",
                    "--format=csv,noheader,nounits",
                    "-lms", "500",
                ],
                stage_dir / "gpu_samples.csv",
            )
        )

    try:
        for sequence, row in enumerate(rows, start=1):
            relative = row["relative_path"]
            mgf = MGF_ROOT / Path(relative).with_suffix(".mgf")
            if not mgf.exists() or mgf.stat().st_size == 0:
                raise RuntimeError(f"Missing converted input: {mgf}")
            token = row["path_sha256"][:16]
            output_dir = output_root / token
            output_dir.mkdir(parents=True, exist_ok=True)
            command, expected_output, temporary = stage_command(
                args.stage, mgf, output_dir, args.gpu_index
            )
            stdout_path = log_root / f"{token}.stdout.log"
            stderr_path = log_root / f"{token}.stderr.log"
            time_path = log_root / f"{token}.time_v.log"
            timed_command = ["/usr/bin/time", "-v", "-o", str(time_path), *command]
            env = os.environ.copy()
            env.update(
                {
                    "LOKY_MAX_CPU_COUNT": "8",
                    "OMP_NUM_THREADS": "8",
                    "OPENBLAS_NUM_THREADS": "8",
                    "MKL_NUM_THREADS": "8",
                    "NUMEXPR_NUM_THREADS": "8",
                }
            )
            started_at = utc_now()
            start = time.perf_counter()
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr:
                process = subprocess.run(
                    timed_command, stdout=stdout, stderr=stderr, text=True, env=env
                )
            wall_s = time.perf_counter() - start
            output_bytes = expected_output.stat().st_size if expected_output.exists() else 0
            effective_status = process.returncode or (1 if output_bytes == 0 else 0)
            record: dict[str, object] = {
                "sequence": sequence,
                "stage": args.stage,
                "relative_path": relative,
                "path_sha256": row["path_sha256"],
                "mgf_bytes": mgf.stat().st_size,
                "output_path": str(expected_output),
                "output_bytes": output_bytes,
                "started_at": started_at,
                "finished_at": utc_now(),
                "wall_s": wall_s,
                "exit_status": effective_status,
                "command": command,
            }
            if time_path.exists():
                record.update(parse_time_log(time_path))
            record["exit_status"] = effective_status
            if args.stage == "cpu_clustering":
                record["cluster_assignments"] = max(0, count_lines(expected_output, r"^[^#].*," ) - 1)
            elif args.stage == "gpu_clustering":
                record["cluster_output_rows"] = None
            elif args.stage == "cpu_oms":
                record["psm_rows"] = count_lines(expected_output, r"<spectrum_query ")
            else:
                record["psm_rows"] = count_lines(expected_output, r"^PSM\t")
            with result_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            print(json.dumps(record), flush=True)
            if effective_status == 0 and temporary is not None:
                temporary.unlink(missing_ok=True)
    finally:
        for monitor in monitors:
            stop_monitor(monitor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
