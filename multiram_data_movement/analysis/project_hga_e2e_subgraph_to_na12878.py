#!/usr/bin/env python3
"""Project the corrected matched-subgraph mapper experiment to full NA12878."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path("/home/tsl012/multiomic/multiram_data_movement")
SOURCE_DIR = ROOT / "results_hga_e2e_subgraph_comparison"
PLATINUM_CSV = (
    ROOT
    / "results_platinum_na12878_hg38/processed/"
    "platinum_na12878_data_movement.csv"
)
PIM_JSON = (
    ROOT
    / "results_matched_mhc_minigraph_hga_pim/"
    "PLATINUM_NA12878_PIM_PROJECTION.json"
)
CPU_PIM_JSON = (
    ROOT
    / "results_matched_mhc_minigraph_hga_pim/"
    "PLATINUM_NA12878_CPU_HGA_PIM_PROJECTION.json"
)
HG38_MMI = Path(
    "/mnt/hdd/tsunghan/raw-ms-dataset/references/hg38/hg38.mmi"
)

SOURCE_CPU_MAX_GHZ = 5.4762891
TARGET_CPU_MAX_GHZ = 3.8
SOURCE_GPU_HBM_GBPS = 1792.0
TARGET_GPU_HBM_GBPS = 1935.0
HDD_BANDWIDTH_BYTES_S = 109_051_904.0
PCIE_EFFECTIVE_BYTES_S = 24e9
HG38_LOGICAL_BASES = 3_209_286_105


def duration(seconds: float) -> str:
    if seconds >= 86400:
        return f"{seconds / 86400:.2f} days"
    if seconds >= 3600:
        return f"{seconds / 3600:.2f} h"
    if seconds >= 60:
        return f"{seconds / 60:.2f} min"
    return f"{seconds:.3f} s"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--platinum-csv", type=Path, default=PLATINUM_CSV)
    args = parser.parse_args()

    source = json.loads(
        (args.source_dir / "HGA_E2E_SUBGRAPH_RESULT.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        (args.source_dir / "inputs/manifest.json").read_text(encoding="utf-8")
    )
    with args.platinum_csv.open("r", encoding="utf-8") as handle:
        platinum = next(csv.DictReader(handle))
    pim = json.loads(PIM_JSON.read_text(encoding="utf-8"))[
        "projected_multiram"
    ]
    prior_cpu = json.loads(CPU_PIM_JSON.read_text(encoding="utf-8"))[
        "cpu_minigraph"
    ]

    source_reads = int(source["workload"]["reads"])
    source_query_bases = (
        source_reads * int(source["workload"]["read_length"])
    )
    full_reads = int(platinum["fastq_records"])
    full_bases = int(platinum["base_count"])
    read_scale = full_reads / source_reads
    base_scale = full_bases / source_query_bases
    cpu_scale = SOURCE_CPU_MAX_GHZ / TARGET_CPU_MAX_GHZ
    gpu_scale = SOURCE_GPU_HBM_GBPS / TARGET_GPU_HBM_GBPS

    cpu = source["cpu_minigraph"]
    gpu = source["gpu_candidate_hga"]
    phases = gpu["warm_median_phases_s"]

    # Direct scaling retains all fixed startup costs once per 4,000-read batch.
    source_cpu_cold_s = float(cpu["cold_median_s"]) * base_scale
    source_cpu_warm_s = float(cpu["warm_median_s"]) * base_scale
    target_cpu_cold_s = source_cpu_cold_s * cpu_scale
    target_cpu_warm_s = source_cpu_warm_s * cpu_scale

    kernel_batch_s = float(phases["hga_kernel_sum"])
    warm_batch_s = float(gpu["warm_median_s"])
    nonkernel_batch_s = max(0.0, warm_batch_s - kernel_batch_s)
    target_kernel_batch_s = kernel_batch_s * gpu_scale
    target_nonkernel_batch_s = nonkernel_batch_s * cpu_scale
    replay_compute_s = (
        target_kernel_batch_s * base_scale
        + target_nonkernel_batch_s * read_scale
    )

    # A persistent implementation keeps graph shards/processes alive. Its
    # streaming host work is routing, materialization, and output merge.
    streaming_host_batch_s = sum(
        float(phases[name])
        for name in (
            "candidate_routing",
            "materialize_groups",
            "merge_output",
        )
    )
    persistent_compute_s = (
        target_kernel_batch_s * base_scale
        + streaming_host_batch_s * cpu_scale * read_scale
    )

    graph_disk_bytes_per_base = sum(
        int(row["hga_graph_bytes"]) for row in manifest["subgraphs"]
    ) / sum(int(row["length"]) for row in manifest["subgraphs"])
    projected_hga_graph_disk_bytes = (
        graph_disk_bytes_per_base * HG38_LOGICAL_BASES
    )
    storage_bytes = (
        int(platinum["compressed_fastq_bytes"])
        + HG38_MMI.stat().st_size
        + projected_hga_graph_disk_bytes
    )
    storage_service_s = storage_bytes / HDD_BANDWIDTH_BYTES_S

    # This conservative PCIe model directly scales all measured H2D/D2H bytes,
    # including repeated graph transfers in the current four-process wrapper.
    source_pcie_bytes = int(gpu["h2d_bytes"]) + int(gpu["d2h_bytes"])
    replay_pcie_bytes = source_pcie_bytes * read_scale
    replay_pcie_service_s = replay_pcie_bytes / PCIE_EFFECTIVE_BYTES_S
    movement_service_s = storage_service_s + replay_pcie_service_s

    replay_overall_s = replay_compute_s + movement_service_s
    persistent_overall_s = persistent_compute_s + movement_service_s

    measured_cpu_overall_s = float(
        prior_cpu["overall_conservative_additive_s"]
    )
    measured_cpu_compute_s = float(prior_cpu["target_compute_s"])
    measured_cpu_movement_s = float(
        prior_cpu["movement_storage_service_s"]
    )
    pim_overall_s = float(pim["full_time_conservative_additive_s"])
    pim_resident_compute_s = float(pim["pim_pipeline_resident_s"])
    hga_kernel_only_s = target_kernel_batch_s * base_scale
    pim_movement_range_s = [
        float(pim["external_stream_pipeline_bottleneck_s"])
        + float(pim["internal_service_s_at_30_34_tbps"]),
        float(pim["external_stream_pipeline_bottleneck_s"])
        + float(pim["internal_service_s_at_19_01_tbps"]),
    ]

    output = {
        "status": "projection_from_corrected_measured_small_workload",
        "source": {
            "reads": source_reads,
            "read_length": int(source["workload"]["read_length"]),
            "search_space_bases": int(
                source["workload"]["search_space_bases"]
            ),
            "gpu": "NVIDIA RTX PRO 6000 Blackwell Max-Q",
            "cpu_minigraph_cold_s": float(cpu["cold_median_s"]),
            "cpu_minigraph_warm_s": float(cpu["warm_median_s"]),
            "gpu_candidate_hga_cold_s": float(gpu["cold_median_s"]),
            "gpu_candidate_hga_warm_s": warm_batch_s,
            "gpu_hga_kernel_s": kernel_batch_s,
            "gpu_explicit_movement_s": float(gpu["explicit_movement_s"]),
            "quality_is_source_only": gpu["quality"],
        },
        "target": {
            "dataset": "Platinum Genomes NA12878 PRJEB3246",
            "fastq_sequence_records": full_reads,
            "query_bases": full_bases,
            "average_read_length": full_bases / full_reads,
            "read_count_scale": read_scale,
            "query_base_scale": base_scale,
            "reference": "UCSC hg38 represented as candidate-routed shards",
            "cpu": "Intel Xeon 6 6515P, 16C/32T, up to 3.8 GHz",
            "gpu": "NVIDIA A100 80GB PCIe",
        },
        "direct_small_workload_scaling": {
            "source_cpu_minigraph_cold_s": source_cpu_cold_s,
            "source_cpu_minigraph_warm_s": source_cpu_warm_s,
            "xeon6515p_cpu_minigraph_cold_s": target_cpu_cold_s,
            "xeon6515p_cpu_minigraph_warm_s": target_cpu_warm_s,
            "warning": (
                "This direct row repeats the 100-kb graph experiment and does "
                "not model whole-hg38 graph construction/index behavior."
            ),
        },
        "paper_cpu_baseline": {
            "overall_s": measured_cpu_overall_s,
            "compute_s": measured_cpu_compute_s,
            "movement_s": measured_cpu_movement_s,
            "movement_fraction_percent": (
                100.0 * measured_cpu_movement_s / measured_cpu_overall_s
            ),
            "basis": (
                "Existing Xeon 6515P projection from real hg38 minigraph "
                "100k-pair phase measurements; preferred over direct 100-kb "
                "subgraph scaling for the full-genome CPU row."
            ),
        },
        "a100_candidate_hga": {
            "prototype_4000_read_batch_replay": {
                "compute_s": replay_compute_s,
                "movement_s": movement_service_s,
                "overall_s": replay_overall_s,
                "movement_fraction_percent": (
                    100.0 * movement_service_s / replay_overall_s
                ),
                "description": (
                    "Current wrapper behavior: four HGA processes and graph "
                    "setup are repeated for each 4,000-read batch."
                ),
            },
            "persistent_graph_streaming_projection": {
                "compute_s": persistent_compute_s,
                "movement_s": movement_service_s,
                "overall_s": persistent_overall_s,
                "movement_fraction_percent": (
                    100.0 * movement_service_s / persistent_overall_s
                ),
                "description": (
                    "Engineering target: candidate routing is streamed and HGA "
                    "graph shards/processes remain resident across batches."
                ),
            },
            "movement_model": {
                "compressed_fastq_bytes": int(
                    platinum["compressed_fastq_bytes"]
                ),
                "minimap2_hg38_index_bytes": HG38_MMI.stat().st_size,
                "projected_hga_shard_files_bytes": (
                    projected_hga_graph_disk_bytes
                ),
                "storage_bytes": storage_bytes,
                "storage_service_s": storage_service_s,
                "conservative_replayed_pcie_bytes": replay_pcie_bytes,
                "pcie_service_s": replay_pcie_service_s,
                "storage_bandwidth_bytes_s": HDD_BANDWIDTH_BYTES_S,
                "pcie_bandwidth_bytes_s": PCIE_EFFECTIVE_BYTES_S,
            },
        },
        "multiram": {
            "overall_s": pim_overall_s,
            "resident_compute_s": pim_resident_compute_s,
            "movement_s_range": pim_movement_range_s,
            "movement_fraction_percent_range": [
                100.0 * value / pim_overall_s
                for value in pim_movement_range_s
            ],
        },
        "core_compute_only": {
            "cpu_minigraph_xeon6515p_s": measured_cpu_compute_s,
            "gpu_hga_a100_kernel_s": hga_kernel_only_s,
            "multiram_resident_pipeline_s": pim_resident_compute_s,
            "hga_slower_than_cpu_x": (
                hga_kernel_only_s / measured_cpu_compute_s
            ),
            "cpu_slower_than_multiram_x": (
                measured_cpu_compute_s / pim_resident_compute_s
            ),
            "hga_slower_than_multiram_x": (
                hga_kernel_only_s / pim_resident_compute_s
            ),
        },
        "comparisons": {
            "multiram_speedup_vs_cpu_x": (
                measured_cpu_overall_s / pim_overall_s
            ),
            "multiram_speedup_vs_hga_prototype_x": (
                replay_overall_s / pim_overall_s
            ),
            "multiram_speedup_vs_hga_persistent_x": (
                persistent_overall_s / pim_overall_s
            ),
            "cpu_faster_than_hga_prototype_x": (
                replay_overall_s / measured_cpu_overall_s
            ),
            "cpu_faster_than_hga_persistent_x": (
                persistent_overall_s / measured_cpu_overall_s
            ),
            "multiram_movement_reduction_vs_cpu_percent_range": [
                100.0 * (1.0 - value / measured_cpu_movement_s)
                for value in reversed(pim_movement_range_s)
            ],
            "multiram_movement_reduction_vs_hga_percent_range": [
                100.0 * (1.0 - value / movement_service_s)
                for value in reversed(pim_movement_range_s)
            ],
        },
        "limits": [
            "These are projections, not executions on all NA12878 FASTQ files.",
            "The source workload uses four linear 25-kb MHC subgraphs; whole-genome repeats and graph branching can change candidate count and DP work.",
            "The source HGA quality metrics are not projected as hg38 accuracy.",
            "A100 kernel time is scaled by 1792/1935 HBM bandwidth; host phases are scaled by 5.4762891/3.8 CPU clock.",
            "HGA remains endpoint-only and does not emit minigraph-equivalent traceback, path, or CIGAR.",
            "Movement time is serialized service time and may overlap computation.",
        ],
    }

    json_path = args.source_dir / (
        "PLATINUM_NA12878_FROM_VALID_HGA_E2E_PROJECTION.json"
    )
    json_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    hga_replay = output["a100_candidate_hga"][
        "prototype_4000_read_batch_replay"
    ]
    hga_persistent = output["a100_candidate_hga"][
        "persistent_graph_streaming_projection"
    ]
    multiram = output["multiram"]
    comparisons = output["comparisons"]
    lines = [
        "# Full NA12878 Projection from Corrected HGA End-to-End Result",
        "",
        "> This report uses the corrected HGA run that produced nonzero scores and "
        "validated endpoints. It does not use the invalid earlier HGA timing.",
        "",
        "## Scaling Basis",
        "",
        f"- Source: {source_reads:,} single-end 100-bp reads over four 25-kb MHC subgraphs.",
        f"- Target: {full_reads:,} NA12878 FASTQ sequences and {full_bases:,} bases.",
        f"- Read-count scale: **{read_scale:,.2f}x**.",
        f"- Query-base scale used for DP kernels: **{base_scale:,.2f}x**.",
        "- Target platform: Xeon 6515P host plus NVIDIA A100 80GB PCIe.",
        "",
        "## Main Result",
        "",
        "| System | Overall | Movement service | Movement / overall | Status |",
        "| --- | ---: | ---: | ---: | --- |",
        f"| CPU minigraph, Xeon 6515P | **{measured_cpu_overall_s:,.1f} s ({duration(measured_cpu_overall_s)})** | {measured_cpu_movement_s:,.1f} s | {100 * measured_cpu_movement_s / measured_cpu_overall_s:.2f}% | hg38 measured-phase projection |",
        f"| A100 candidate HGA, current 4k-batch wrapper | **{hga_replay['overall_s']:,.1f} s ({duration(hga_replay['overall_s'])})** | {hga_replay['movement_s']:,.1f} s | {hga_replay['movement_fraction_percent']:.3f}% | conservative prototype projection |",
        f"| A100 candidate HGA, persistent graph/streaming | **{hga_persistent['overall_s']:,.1f} s ({duration(hga_persistent['overall_s'])})** | {hga_persistent['movement_s']:,.1f} s | {hga_persistent['movement_fraction_percent']:.3f}% | optimized engineering projection |",
        f"| MultiRAM PIM | **{pim_overall_s:,.1f} s ({duration(pim_overall_s)})** | {multiram['movement_s_range'][0]:.1f}-{multiram['movement_s_range'][1]:.1f} s | {multiram['movement_fraction_percent_range'][0]:.2f}-{multiram['movement_fraction_percent_range'][1]:.2f}% | analytical model |",
        "",
        "The current HGA wrapper projection is the conservative number to report "
        "for the code as it exists. The persistent row is achievable only after "
        "removing repeated process and graph setup.",
        "",
        "## Core Compute Only",
        "",
        "This view excludes storage, PCIe, candidate routing, graph parsing, "
        "process startup, materialization, and output merge.",
        "",
        "| System | Core compute time | Relative to MultiRAM |",
        "| --- | ---: | ---: |",
        f"| CPU minigraph, Xeon 6515P | **{measured_cpu_compute_s:,.1f} s ({duration(measured_cpu_compute_s)})** | {measured_cpu_compute_s / pim_resident_compute_s:.2f}x |",
        f"| GPU HGA CUDA kernel, A100 | **{hga_kernel_only_s:,.1f} s ({duration(hga_kernel_only_s)})** | {hga_kernel_only_s / pim_resident_compute_s:,.1f}x |",
        f"| MultiRAM resident pipeline | **{pim_resident_compute_s:,.1f} s ({duration(pim_resident_compute_s)})** | 1.00x |",
        "",
        f"Even after removing all wrapper and movement overhead, the projected HGA "
        f"kernel remains **{hga_kernel_only_s / measured_cpu_compute_s:.1f}x slower** "
        "than minigraph compute because HGA evaluates dense DP cells over each "
        "candidate subgraph while minigraph uses sparse seeding and chaining.",
        "",
        "## HGA Movement Model",
        "",
        f"- Compressed NA12878 FASTQ: {int(platinum['compressed_fastq_bytes']) / 1e9:.3f} GB.",
        f"- hg38 minimap2 index: {HG38_MMI.stat().st_size / 1e9:.3f} GB.",
        f"- Projected HGA shard files: {projected_hga_graph_disk_bytes / 1e9:.3f} GB.",
        f"- Storage-read service: {storage_service_s:,.1f} s at {HDD_BANDWIDTH_BYTES_S / 1e6:.3f} MB/s.",
        f"- Conservative replayed PCIe traffic: {replay_pcie_bytes / 1e9:.3f} GB, or {replay_pcie_service_s:.1f} s at 24 GB/s.",
        "",
        "## Comparisons",
        "",
        f"- MultiRAM speedup vs CPU minigraph: **{comparisons['multiram_speedup_vs_cpu_x']:.2f}x**.",
        f"- MultiRAM speedup vs current HGA wrapper: **{comparisons['multiram_speedup_vs_hga_prototype_x']:,.1f}x**.",
        f"- MultiRAM speedup vs persistent HGA projection: **{comparisons['multiram_speedup_vs_hga_persistent_x']:,.1f}x**.",
        f"- CPU minigraph is **{comparisons['cpu_faster_than_hga_prototype_x']:.1f}x** faster than the current HGA wrapper projection and **{comparisons['cpu_faster_than_hga_persistent_x']:.1f}x** faster than the persistent projection.",
        f"- MultiRAM movement-time reduction vs CPU: **{comparisons['multiram_movement_reduction_vs_cpu_percent_range'][0]:.2f}-{comparisons['multiram_movement_reduction_vs_cpu_percent_range'][1]:.2f}%**.",
        f"- MultiRAM movement-time reduction vs HGA: **{comparisons['multiram_movement_reduction_vs_hga_percent_range'][0]:.2f}-{comparisons['multiram_movement_reduction_vs_hga_percent_range'][1]:.2f}%**.",
        "",
        "## Direct Small-Test Scaling Check",
        "",
        "For transparency, mechanically scaling the matched 100-kb experiment gives:",
        "",
        f"- Current source CPU minigraph: cold {duration(source_cpu_cold_s)}, warm {duration(source_cpu_warm_s)}.",
        f"- Xeon 6515P clock-scaled minigraph: cold {duration(target_cpu_cold_s)}, warm {duration(target_cpu_warm_s)}.",
        "",
        "The main CPU row instead uses the existing real-hg38 phase projection, "
        "because graph construction and indexing are not represented by a 100-kb "
        "subgraph.",
        "",
        "## Accuracy and Scope",
        "",
        f"The small matched workload measured {gpu['quality']['correct_endpoint_rate_percent']:.2f}% correct endpoint/all reads and {gpu['quality']['correct_endpoint_precision_percent']:.2f}% correct endpoint/mapped. "
        "Those values validate the corrected HGA pipeline on MHC, but they are "
        "**not full-NA12878 accuracy estimates**.",
        "",
        "- Full hg38 must be candidate-routed and sharded; public HGA cannot load it as one DP graph.",
        "- Whole-genome repeat structure can increase candidate multiplicity and runtime.",
        "- HGA emits score/endpoints, not minigraph-equivalent paths or CIGAR.",
        "- Movement service may overlap computation and should not be described as measured critical-path delay.",
    ]
    report_path = args.source_dir / (
        "PLATINUM_NA12878_FROM_VALID_HGA_E2E_PROJECTION.md"
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "report": str(report_path),
        "json": str(json_path),
        "cpu_minigraph_s": measured_cpu_overall_s,
        "hga_prototype_s": replay_overall_s,
        "hga_persistent_s": persistent_overall_s,
        "multiram_s": pim_overall_s,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
