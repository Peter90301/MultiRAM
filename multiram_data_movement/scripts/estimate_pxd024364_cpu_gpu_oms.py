#!/usr/bin/env python3
"""Compare measured PXD024364 CPU/GPU OMS samples and extrapolate by bytes."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


DATA_ROOT = Path("/mnt/hdd/tsunghan/raw-ms-dataset/proteomic_large_dataset")
RESULT_ROOT = DATA_ROOT / "results"
REPORT_ROOT = Path(
    "/home/tsl012/multiomic/multiram_data_movement/results_pxd024364_1tb"
)
LOCAL_TZ = ZoneInfo("America/Los_Angeles")


def successful_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("exit_status") == 0:
            records.append(record)
    return records


def gpu_peak_hbm_mib(record: dict[str, object]) -> float:
    start = datetime.fromisoformat(str(record["started_at"])).astimezone(LOCAL_TZ)
    finish = datetime.fromisoformat(str(record["finished_at"])).astimezone(LOCAL_TZ)
    peak = 0.0
    samples = RESULT_ROOT / "gpu_oms/gpu_samples.csv"
    with samples.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 4:
                continue
            try:
                timestamp = datetime.strptime(
                    row[0].strip(), "%Y/%m/%d %H:%M:%S.%f"
                ).replace(tzinfo=LOCAL_TZ)
                memory_mib = float(row[3].strip())
            except ValueError:
                continue
            if start <= timestamp <= finish:
                peak = max(peak, memory_mib)
    return peak


def gib_from_kib(value: object) -> float:
    return int(value) / 1024**2


def io_gb(record: dict[str, object]) -> float:
    blocks = int(record.get("fs_input_blocks", 0)) + int(
        record.get("fs_output_blocks", 0)
    )
    return blocks * 512 / 1e9


def main() -> int:
    cpu_rows = successful_records(RESULT_ROOT / "cpu_oms/metrics.jsonl")
    gpu_rows = successful_records(RESULT_ROOT / "gpu_oms/metrics.jsonl")
    conversion_rows = successful_records(RESULT_ROOT / "raw_conversion/metrics.jsonl")
    cpu_by_path = {str(row["relative_path"]): row for row in cpu_rows}
    gpu_by_path = {str(row["relative_path"]): row for row in gpu_rows}
    common_paths = sorted(cpu_by_path.keys() & gpu_by_path.keys())
    if not common_paths:
        raise SystemExit("No identical successful CPU/GPU OMS input is available")

    # Use only identical inputs so the speedup denominator is directly comparable.
    cpu = cpu_by_path[common_paths[0]]
    gpu = gpu_by_path[common_paths[0]]
    input_bytes = int(cpu["mgf_bytes"])
    if input_bytes != int(gpu["mgf_bytes"]):
        raise SystemExit("Matched CPU/GPU records disagree on input size")

    full_bytes = sum(int(row["mgf_bytes"]) for row in conversion_rows)
    cpu_wall_s = float(cpu["wall_s"])
    gpu_wall_s = float(gpu["wall_s"])
    cpu_estimated_s = cpu_wall_s * full_bytes / input_bytes
    gpu_estimated_s = gpu_wall_s * full_bytes / input_bytes
    cpu_peak_gib = gib_from_kib(cpu["max_rss_kb"])
    gpu_host_peak_gib = gib_from_kib(gpu["max_rss_kb"])
    gpu_hbm_peak_mib = gpu_peak_hbm_mib(gpu)
    gpu_hbm_peak_gib = gpu_hbm_peak_mib / 1024
    gpu_combined_peak_gib = gpu_host_peak_gib + gpu_hbm_peak_gib
    cpu_io_gb = io_gb(cpu)
    gpu_io_gb = io_gb(gpu)

    report = {
        "dataset": "PXD024364 / MSV000086944 approximately 1 TB RAW subset",
        "stage": "open_modification_search",
        "status": "one_identical_file_measured_then_extrapolated_by_mgf_bytes",
        "matched_relative_path": common_paths[0],
        "matched_input_bytes": input_bytes,
        "full_converted_files": len(conversion_rows),
        "full_mgf_bytes": full_bytes,
        "cpu": {
            "baseline": "SpectraST",
            "measured_wall_s": cpu_wall_s,
            "estimated_full_wall_s": cpu_estimated_s,
            "estimated_full_days": cpu_estimated_s / 86400,
            "estimated_full_years_365d": cpu_estimated_s / (365 * 86400),
            "peak_host_rss_gib": cpu_peak_gib,
            "filesystem_io_gb": cpu_io_gb,
        },
        "gpu": {
            "baseline": "ANN-SoLo with default GPU indexing",
            "measured_wall_s": gpu_wall_s,
            "estimated_full_wall_s": gpu_estimated_s,
            "estimated_full_days": gpu_estimated_s / 86400,
            "estimated_full_years_365d": gpu_estimated_s / (365 * 86400),
            "peak_host_rss_gib": gpu_host_peak_gib,
            "peak_hbm_mib": gpu_hbm_peak_mib,
            "peak_hbm_gib": gpu_hbm_peak_gib,
            "conservative_host_plus_hbm_peak_gib": gpu_combined_peak_gib,
            "filesystem_io_gb": gpu_io_gb,
        },
        "comparison": {
            "gpu_speedup_over_cpu": cpu_wall_s / gpu_wall_s,
            "gpu_host_memory_ratio_vs_cpu": gpu_host_peak_gib / cpu_peak_gib,
            "gpu_host_plus_hbm_memory_ratio_vs_cpu": (
                gpu_combined_peak_gib / cpu_peak_gib
            ),
            "gpu_filesystem_io_ratio_vs_cpu": gpu_io_gb / cpu_io_gb,
            "gpu_filesystem_io_overhead_percent_vs_cpu": (
                (gpu_io_gb / cpu_io_gb) - 1
            )
            * 100,
        },
        "limitations": [
            "Full-dataset OMS runtimes are estimates, not measured full runs.",
            "Only one identical completed MGF input is used for extrapolation.",
            "SpectraST and ANN-SoLo are different baseline algorithms and produced different PSM counts.",
            "Host RSS and HBM peaks may occur at different instants; their sum is a conservative upper bound.",
            "Filesystem I/O uses Linux GNU-time block counters at 512 bytes per block and is not DRAM or GPU-HBM traffic.",
            "Exact H2D, D2H, and GPU-HBM traffic require Nsight profiling and are not included.",
        ],
    }

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_ROOT / "cpu_gpu_oms_extrapolation.json"
    md_path = REPORT_ROOT / "CPU_GPU_OMS_EXTRAPOLATION.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# PXD024364 CPU vs GPU OMS Extrapolation",
                "",
                f"Matched input: `{common_paths[0]}` ({input_bytes / 1e6:.3f} MB).",
                "",
                "| Metric | CPU SpectraST | GPU ANN-SoLo |",
                "| --- | ---: | ---: |",
                f"| Measured wall time | {cpu_wall_s:.3f} s | {gpu_wall_s:.3f} s |",
                f"| Measured wall time | {cpu_wall_s / 3600:.3f} h | {gpu_wall_s / 3600:.3f} h |",
                f"| Estimated full runtime | {cpu_estimated_s / 86400:.3f} days | {gpu_estimated_s / 86400:.3f} days |",
                f"| Peak host RSS | {cpu_peak_gib:.3f} GiB | {gpu_host_peak_gib:.3f} GiB |",
                f"| Peak GPU HBM | N/A | {gpu_hbm_peak_gib:.3f} GiB |",
                f"| Host + HBM conservative peak | {cpu_peak_gib:.3f} GiB | {gpu_combined_peak_gib:.3f} GiB |",
                f"| Filesystem I/O estimate | {cpu_io_gb:.3f} GB | {gpu_io_gb:.3f} GB |",
                "",
                f"GPU speedup over CPU: **{cpu_wall_s / gpu_wall_s:.3f}x**.",
                "",
                f"GPU host+HBM memory ratio over CPU: **{gpu_combined_peak_gib / cpu_peak_gib:.3f}x**.",
                "",
                f"GPU filesystem-I/O overhead over CPU: **{((gpu_io_gb / cpu_io_gb) - 1) * 100:.3f}%**.",
                "",
                "These full-dataset runtimes are one-file, byte-scaled estimates and must not be reported as measured full runs. SpectraST and ANN-SoLo are different algorithmic baselines.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
