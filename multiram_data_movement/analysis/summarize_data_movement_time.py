#!/usr/bin/env python3
"""Summarize cold/warm and CUDA-copy time as explicit movement fractions."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


DEFAULT_RESULTS = Path(
    "/home/tsl012/multiomic/multiram_data_movement/results_data_movement_time_minigraph"
)
OMS_RESULT_ROOT = Path(
    "/mnt/hdd/tsunghan/raw-ms-dataset/proteomic_large_dataset/results"
)


def load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def direction_bytes(record: dict[str, object], needle: str) -> int:
    total = 0
    directions = record.get("copy_by_direction", {})
    if not isinstance(directions, dict):
        return 0
    for label, values in directions.items():
        if needle.lower() in str(label).lower() and isinstance(values, dict):
            total += int(values.get("bytes", 0))
    return total


def fmt(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def fmt_percent(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}%"


def active_sda_read_bandwidth(results_dir: Path) -> float | None:
    values_kib_s: list[float] = []
    pattern = results_dir / "raw/genomics_cpu_minigraph/cold"
    for path in pattern.glob("trial_*/iostat.log"):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split()
            if fields and fields[0] == "sda" and len(fields) >= 3:
                read_kib_s = float(fields[2])
                if read_kib_s > 0:
                    values_kib_s.append(read_kib_s)
    return statistics.median(values_kib_s) * 1024 if values_kib_s else None


def first_success(path: Path) -> dict[str, object] | None:
    for record in load_jsonl(path):
        if int(record.get("exit_status", 1)) == 0:
            return record
    return None


def oms_storage_models(storage_bandwidth_bytes_s: float | None) -> list[dict[str, object]]:
    if not storage_bandwidth_bytes_s:
        return []
    specifications = (
        ("CPU SpectraST", OMS_RESULT_ROOT / "cpu_oms/metrics.jsonl"),
        ("GPU ANN-SoLo", OMS_RESULT_ROOT / "gpu_oms/metrics.jsonl"),
    )
    models: list[dict[str, object]] = []
    for platform_name, path in specifications:
        record = first_success(path)
        if not record:
            continue
        filesystem_read_bytes = int(record["fs_input_blocks"]) * 512
        wall_s = float(record["wall_s"])
        storage_service_s = filesystem_read_bytes / storage_bandwidth_bytes_s
        models.append({
            "platform": platform_name,
            "overall_runtime_s": wall_s,
            "filesystem_read_bytes": filesystem_read_bytes,
            "measured_active_hdd_bandwidth_bytes_s": storage_bandwidth_bytes_s,
            "modeled_storage_service_s": storage_service_s,
            "modeled_storage_fraction_percent": 100.0 * storage_service_s / wall_s,
            "gpu_transfer_time_s": None,
            "method": "filesystem bytes divided by measured active HDD bandwidth",
        })
    return models


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    runtimes = load_jsonl(args.results_dir / "runtime_records.jsonl")
    transfers = {
        str(item["workload"]): item
        for item in load_jsonl(args.results_dir / "gpu_transfer_records.jsonl")
    }
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in runtimes:
        if int(record.get("return_code", 1)) == 0:
            grouped[str(record["workload"])].append(record)

    storage_bandwidth = active_sda_read_bandwidth(args.results_dir)
    rows: list[dict[str, object]] = []
    for workload, records in sorted(grouped.items()):
        cold_records = [item for item in records if item["mode"] == "cold"]
        cold = [float(item["wall_s"]) for item in cold_records]
        warm = [float(item["wall_s"]) for item in records if item["mode"] == "warm"]
        if not cold or not warm:
            continue
        cold_median = statistics.median(cold)
        warm_median = statistics.median(warm)
        raw_storage_delta_s = cold_median - warm_median
        storage_s = max(0.0, raw_storage_delta_s)
        cold_fs_read_bytes = statistics.median(
            int(item.get("fs_input_blocks") or 0) * 512 for item in cold_records
        )
        storage_service_s = None
        if workload == "genomics_cpu_minigraph" and storage_bandwidth:
            storage_service_s = cold_fs_read_bytes / storage_bandwidth
        transfer = transfers.get(workload)
        gpu_copy_s = float(transfer["copy_union_time_s"]) if transfer else 0.0
        platform_name = str(records[0]["platform"])
        explicit_s = storage_s + gpu_copy_s
        rows.append({
            "workload": workload,
            "domain": records[0]["domain"],
            "platform": platform_name,
            "trials": min(len(cold), len(warm)),
            "cold_median_s": cold_median,
            "warm_median_s": warm_median,
            "cold_warm_raw_delta_s": raw_storage_delta_s,
            "storage_movement_s": storage_s,
            "storage_fraction_cold_percent": 100.0 * storage_s / cold_median,
            "cold_filesystem_read_bytes": int(cold_fs_read_bytes),
            "storage_service_equivalent_s": storage_service_s,
            "storage_service_equivalent_fraction_cold_percent": (
                100.0 * storage_service_s / cold_median
                if storage_service_s is not None else None
            ),
            "gpu_copy_union_s": gpu_copy_s if platform_name == "gpu" else None,
            "gpu_copy_fraction_warm_percent": (
                100.0 * gpu_copy_s / warm_median if platform_name == "gpu" else None
            ),
            "h2d_bytes": direction_bytes(transfer, "Host-to-Device") if transfer else None,
            "d2h_bytes": direction_bytes(transfer, "Device-to-Host") if transfer else None,
            "explicit_movement_s": explicit_s,
            "explicit_movement_fraction_cold_percent": 100.0 * explicit_s / cold_median,
        })

    output_json = args.results_dir / "data_movement_time_summary.json"
    output_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    oms_models = oms_storage_models(storage_bandwidth)
    (args.results_dir / "oms_storage_service_model.json").write_text(
        json.dumps(oms_models, indent=2) + "\n", encoding="utf-8"
    )
    metadata = json.loads(
        (args.results_dir / "experiment_metadata.json").read_text(encoding="utf-8")
    )
    lines = [
        "# Measured CPU/GPU Data-Movement Time",
        "",
        "## Hardware and Method",
        "",
        f"- CPU: {metadata['cpu_model']}",
        f"- GPU: {metadata['gpu']}",
        f"- minigraph: {metadata.get('minigraph_binary_version', 'unknown')} "
        f"(source `{metadata.get('minigraph_source_commit', 'unknown')}`)",
        f"- Nsight: {metadata['nsys']}",
        f"- Trials per cold/warm mode: {metadata['trials']}",
        f"- Cache eviction: selected-file POSIX_FADV_DONTNEED",
        f"- perf_event_paranoid: {metadata['perf_event_paranoid']}",
        "",
        "## Genomics Input",
        "",
        f"- NA12878 paired reads: {int(metadata['manifest']['records_per_mate']):,} per mate "
        f"({2 * int(metadata['manifest']['records_per_mate']):,} sequences total)",
        "- Reference/graph input: UCSC hg38 FASTA gzip",
        "- minigraph command shape: `minigraph -x sr -c -t 16 hg38.fa.gz R1.fastq R2.fastq -o /dev/null`",
        "- Scope: matched profiling subset; this is not the 614 GB full-dataset run",
        "",
        "## Results",
        "",
        "| Workload | Platform | Cold median (s) | Warm median (s) | Cold-warm delta (s) | Cold FS reads | Serialized storage service (s) | Service / cold | CUDA copy (s) | H2D | D2H | Critical explicit / cold |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        h2d = "N/A" if row["h2d_bytes"] is None else f"{int(row['h2d_bytes']) / 1e6:.3f} MB"
        d2h = "N/A" if row["d2h_bytes"] is None else f"{int(row['d2h_bytes']) / 1e6:.3f} MB"
        lines.append(
            f"| {row['workload']} | {row['platform']} | "
            f"{fmt(row['cold_median_s'])} | {fmt(row['warm_median_s'])} | "
            f"{fmt(row['cold_warm_raw_delta_s'])} | "
            f"{int(row['cold_filesystem_read_bytes']) / 1e9:.3f} GB | "
            f"{fmt(row['storage_service_equivalent_s'])} | "
            f"{fmt_percent(row['storage_service_equivalent_fraction_cold_percent'])} | "
            f"{fmt(row['gpu_copy_union_s'], 6)} | "
            f"{h2d} | {d2h} | "
            f"{fmt_percent(row['explicit_movement_fraction_cold_percent'])} |"
        )
    lines.extend([
        "",
        "## Meaning of the Numbers",
        "",
        "`Cold-warm delta` is the median cold runtime minus the median warm runtime. A negative value means the storage delay is smaller than run-to-run noise; it does not mean the cold run performed no I/O. `Serialized storage service` is cold filesystem-read bytes divided by the median active HDD throughput. It measures equivalent device service time, not additional critical-path delay, because I/O can overlap decompression and graph/index construction.",
        "",
        "These are conservative explicit-movement measurements. CPU DRAM-stall time is unavailable because hardware counters are blocked (`perf_event_paranoid=4`). GPU HBM accesses inside kernels are also excluded. Therefore the table must not be described as total memory-system stall time.",
        "",
        "The HGA row uses the existing MHC graph because an hg38 HGA graph exceeds practical GPU DP-memory capacity. It characterizes GPU transfer overhead and is not a direct runtime/accuracy comparison with the minigraph hg38 row.",
        "",
    ])
    if storage_bandwidth:
        lines.extend([
            "## I/O Validation",
            "",
            f"The median active `sda` read throughput in the three cold hg38 runs was {storage_bandwidth / 1e6:.3f} MB/s. The minigraph runtime records contain the exact selected input bytes and GNU-time filesystem-read blocks for each cold/warm pair.",
            "",
        ])
    if oms_models:
        lines.extend([
            "## Existing OMS Runs: Storage Model Only",
            "",
            "These OMS rows were completed previously, but no matched cold/warm or Nsight run exists. Storage time is modeled as filesystem-read bytes divided by the measured active HDD bandwidth above. It is not a direct timing decomposition.",
            "",
            "| OMS baseline | Overall runtime (s) | Filesystem reads | Modeled storage service (s) | Modeled storage / overall | GPU transfer |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for model in oms_models:
            lines.append(
                f"| {model['platform']} | {model['overall_runtime_s']:.3f} | "
                f"{model['filesystem_read_bytes'] / 1e9:.3f} GB | "
                f"{model['modeled_storage_service_s']:.3f} | "
                f"{model['modeled_storage_fraction_percent']:.3f}% | N/A |"
            )
        lines.extend([
            "",
            "The OMS percentages cover storage service only. They exclude CPU DRAM stalls, GPU H2D/D2H copies, and GPU HBM accesses, so they are lower-bound components rather than total data-movement fractions.",
            "",
        ])
    report = args.results_dir / "DATA_MOVEMENT_TIME_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(report)
    print(output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
