#!/usr/bin/env python3
"""Summarize the matched MHC CPU, GPU, and modeled PIM experiment."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


DEFAULT_RESULTS = Path(
    "/home/tsl012/multiomic/multiram_data_movement/"
    "results_matched_mhc_minigraph_hga_pim"
)


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def direction_bytes(record: dict[str, object], direction: str) -> int:
    values = record.get("copy_by_direction", {})
    if not isinstance(values, dict):
        return 0
    return sum(
        int(entry.get("bytes", 0))
        for label, entry in values.items()
        if direction.lower() in str(label).lower() and isinstance(entry, dict)
    )


def hga_kernel_median(results: Path, mode: str) -> float:
    values: list[float] = []
    for path in (results / f"raw/matched_mhc_gpu_hga/{mode}").glob(
        "trial_*/stdout.log"
    ):
        match = re.search(
            r"Time:\s*([0-9.eE+-]+)s",
            path.read_text(encoding="utf-8", errors="replace"),
        )
        if match:
            values.append(float(match.group(1)))
    if not values:
        raise RuntimeError(f"No HGA kernel timing for {mode}")
    return statistics.median(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    records = load_jsonl(args.results_dir / "runtime_records.jsonl")
    transfer = load_jsonl(args.results_dir / "gpu_transfer_records.jsonl")[0]
    metadata = json.loads(
        (args.results_dir / "experiment_metadata.json").read_text(encoding="utf-8")
    )
    manifest = metadata["manifest"]
    pim = json.loads(
        (args.results_dir / "pim_model_result.json").read_text(encoding="utf-8")
    )
    quality = json.loads(
        (args.results_dir / "quality/minigraph_quality_summary.json").read_text(
            encoding="utf-8"
        )
    )

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        if int(record["return_code"]) == 0:
            grouped[str(record["workload"])].append(record)

    measured: dict[str, dict[str, float | int | str]] = {}
    for workload, workload_records in grouped.items():
        cold_records = [row for row in workload_records if row["mode"] == "cold"]
        warm_records = [row for row in workload_records if row["mode"] == "warm"]
        cold_s = statistics.median(float(row["wall_s"]) for row in cold_records)
        warm_s = statistics.median(float(row["wall_s"]) for row in warm_records)
        storage_s = max(0.0, cold_s - warm_s)
        measured[workload] = {
            "cold_median_s": cold_s,
            "warm_median_s": warm_s,
            "cold_warm_raw_delta_s": cold_s - warm_s,
            "storage_critical_path_s": storage_s,
            "storage_fraction_cold_percent": 100.0 * storage_s / cold_s,
            "cold_filesystem_read_bytes": int(
                statistics.median(
                    int(row.get("fs_input_blocks") or 0) * 512
                    for row in cold_records
                )
            ),
            "peak_rss_bytes": int(
                statistics.median(
                    int(row.get("max_rss_kb") or 0) * 1024
                    for row in workload_records
                )
            ),
        }

    cpu = measured["matched_mhc_cpu_minigraph"]
    gpu = measured["matched_mhc_gpu_hga"]
    cuda_copy_s = float(transfer["copy_union_time_s"])
    h2d_bytes = direction_bytes(transfer, "Host-to-Device")
    d2h_bytes = direction_bytes(transfer, "Device-to-Host")
    gpu_explicit_s = float(gpu["storage_critical_path_s"]) + cuda_copy_s
    gpu_explicit_fraction = 100.0 * gpu_explicit_s / float(gpu["cold_median_s"])

    pim_cold_s = float(pim["modeled_cold_total_s"])
    pim_resident_s = float(pim["modeled_resident_total_s"])
    pim_ingress_s = float(pim["movement"]["cold_external_ingress_s"])
    cpu_cold_s = float(cpu["cold_median_s"])
    cpu_warm_s = float(cpu["warm_median_s"])
    gpu_cold_s = float(gpu["cold_median_s"])
    gpu_warm_s = float(gpu["warm_median_s"])
    hga_kernel_warm_s = hga_kernel_median(args.results_dir, "warm")

    summary = {
        "status": {
            "cpu_gpu": "measured",
            "pim": "modeled_not_hardware_measurement",
        },
        "matched_workload": {
            "reference": manifest["reference_header"],
            "vertices": manifest["reference_vertices"],
            "queries": manifest["query_count"],
            "query_length_histogram": manifest["query_length_histogram"],
            "graph_sequence_sha256": manifest["reference_sequence_sha256"],
            "query_sequence_sha256": manifest["query_sequence_sha256"],
            "validation": manifest["validation"],
        },
        "cpu_minigraph": {
            **cpu,
            "physical_input_bytes": (
                int(manifest["reference_file_bytes"])
                + int(manifest["query_fastq_bytes"])
            ),
            "gaf_mapping_records": quality["gaf_mapping_records"],
            "mapping_record_fraction_percent": quality[
                "mapped_record_fraction_percent"
            ],
        },
        "gpu_hga": {
            **gpu,
            "physical_input_bytes": (
                int(manifest["hga_graph_file_bytes"])
                + int(manifest["hga_reads_file_bytes"])
            ),
            "hga_graph_representation_expansion": manifest[
                "hga_graph_representation_expansion"
            ],
            "kernel_cold_median_s": hga_kernel_median(args.results_dir, "cold"),
            "kernel_warm_median_s": hga_kernel_warm_s,
            "h2d_bytes": h2d_bytes,
            "d2h_bytes": d2h_bytes,
            "cuda_copy_union_s": cuda_copy_s,
            "explicit_storage_plus_cuda_s": gpu_explicit_s,
            "explicit_storage_plus_cuda_fraction_cold_percent": (
                gpu_explicit_fraction
            ),
        },
        "pim_3ddram": pim,
        "same_input_runtime_comparisons": {
            "minigraph_faster_than_hga_cold_x": gpu_cold_s / cpu_cold_s,
            "minigraph_faster_than_hga_warm_x": gpu_warm_s / cpu_warm_s,
            "pim_speedup_vs_minigraph_cold_x": cpu_cold_s / pim_cold_s,
            "pim_speedup_vs_minigraph_resident_x": cpu_warm_s / pim_resident_s,
            "pim_speedup_vs_hga_cold_x": gpu_cold_s / pim_cold_s,
            "pim_speedup_vs_hga_resident_x": gpu_warm_s / pim_resident_s,
            "hga_kernel_speed_vs_pim_model_core_x": (
                pim_resident_s / hga_kernel_warm_s
            ),
        },
        "modeled_explicit_external_movement_time_reduction": {
            "pim_vs_cpu_cold_percent": (
                100.0
                * (float(cpu["storage_critical_path_s"]) - pim_ingress_s)
                / float(cpu["storage_critical_path_s"])
            ),
            "pim_vs_gpu_cold_percent": (
                100.0 * (gpu_explicit_s - pim_ingress_s) / gpu_explicit_s
            ),
            "note": (
                "PIM uses modeled FeNAND ingress; CPU/GPU use measured cold-warm "
                "critical-path and CUDA-copy time."
            ),
        },
    }
    summary_path = args.results_dir / "matched_mhc_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Matched MHC Minigraph, HGA, and MultiRAM Results",
        "",
        "## Workload Validation",
        "",
        f"- Logical graph: `{manifest['reference_header']}`",
        f"- Vertices/bases: {int(manifest['reference_vertices']):,}",
        f"- Queries: {int(manifest['query_count']):,} single-end reads",
        f"- Lengths: {manifest['query_length_histogram']}",
        f"- Graph SHA-256 match: `{manifest['validation']['graph_sequences_identical']}`",
        f"- Read SHA-256 match: `{manifest['validation']['query_sequences_identical']}`",
        "",
        "## Runtime",
        "",
        "| System | Status | Cold/end-to-end (s) | Warm/resident (s) | Kernel/model core (s) |",
        "| --- | --- | ---: | ---: | ---: |",
        f"| CPU minigraph | measured | {cpu_cold_s:.6f} | {cpu_warm_s:.6f} | N/A |",
        f"| GPU HGA | measured | {gpu_cold_s:.6f} | {gpu_warm_s:.6f} | {summary['gpu_hga']['kernel_warm_median_s']:.6f} |",
        f"| MultiRAM 3D-DRAM PIM | modeled | {pim_cold_s:.6f} | {pim_resident_s:.6f} | {float(pim['pim_pipeline_latency_s']):.6f} |",
        "",
        "PIM cold time adds reference/query ingress at 8.1 GB/s to the modeled PIM pipeline latency. The 55-second Python model-runner time is not accelerator latency.",
        "",
        "## Data Movement",
        "",
        "| System | Cold filesystem/external bytes | Internal/PCIe bytes | Movement time | Movement / cold |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| CPU minigraph | {int(cpu['cold_filesystem_read_bytes']) / 1e6:.3f} MB | unavailable DRAM traffic | {float(cpu['storage_critical_path_s']):.6f} s | {float(cpu['storage_fraction_cold_percent']):.2f}% |",
        f"| GPU HGA | {int(gpu['cold_filesystem_read_bytes']) / 1e6:.3f} MB | H2D {h2d_bytes / 1e6:.3f} MB; D2H {d2h_bytes / 1e6:.3f} MB | {gpu_explicit_s:.6f} s | {gpu_explicit_fraction:.2f}% |",
        f"| MultiRAM PIM | {int(pim['movement']['cold_external_ingress_bytes']) / 1e6:.3f} MB | 3D-DRAM internal {float(pim['movement']['internal_3ddram_read_bytes']) / 1e6:.3f} MB | {pim_ingress_s:.6f} s external | {float(pim['modeled_external_movement_fraction_cold_percent']):.2f}% |",
        "",
        f"The modeled 3D-DRAM internal service equivalent is {float(pim['movement']['internal_service_equivalent_s_at_30_34_tbps']) * 1e6:.3f}-{float(pim['movement']['internal_service_equivalent_s_at_19_01_tbps']) * 1e6:.3f} us at 30.34-19.01 TB/s. It is diagnostic and is not added again to PIM latency.",
        "",
        f"HGA's custom text graph is {float(manifest['hga_graph_representation_expansion']):.2f}x the one-byte-per-base logical graph. Its physical graph representation therefore dominates cold end-to-end time.",
        "",
        "## Same-Input Comparisons",
        "",
        f"- CPU minigraph is **{gpu_cold_s / cpu_cold_s:.3f}x faster** than HGA cold end-to-end.",
        f"- Modeled PIM speedup versus CPU minigraph: **{cpu_cold_s / pim_cold_s:.3f}x cold**, **{cpu_warm_s / pim_resident_s:.3f}x resident**.",
        f"- Modeled PIM speedup versus GPU HGA: **{gpu_cold_s / pim_cold_s:.3f}x cold**, **{gpu_warm_s / pim_resident_s:.3f}x resident**.",
        f"- Kernel/model core only: HGA is **{pim_resident_s / hga_kernel_warm_s:.3f}x faster** than the PIM model core ({hga_kernel_warm_s:.6f} s versus {pim_resident_s:.6f} s).",
        f"- Modeled PIM explicit external movement-time reduction: **{summary['modeled_explicit_external_movement_time_reduction']['pim_vs_cpu_cold_percent']:.2f}% vs CPU**, **{summary['modeled_explicit_external_movement_time_reduction']['pim_vs_gpu_cold_percent']:.2f}% vs GPU**.",
        "",
        "## Output and Scope",
        "",
        f"- minigraph emitted {int(quality['gaf_mapping_records']):,} GAF records ({float(quality['mapped_record_fraction_percent']):.2f}% of input-record count). This is not truth-labeled accuracy.",
        "- HGA reports score-only alignment and does not emit mapping coordinates or paths in this public build.",
        "- Runtime ratios are same-input system comparisons, not identical-output algorithmic speedups.",
        "- This controlled MHC regional experiment does not replace the full hg38 storage-capacity experiment.",
        "",
    ]
    report_path = args.results_dir / "MATCHED_MHC_RESULT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(report_path)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
