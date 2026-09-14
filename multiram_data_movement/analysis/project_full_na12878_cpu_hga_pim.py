#!/usr/bin/env python3
"""Project full NA12878 runtime and explicit movement time across three systems."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


ROOT = Path("/home/tsl012/multiomic/multiram_data_movement")
MATCHED_DIR = ROOT / "results_matched_mhc_minigraph_hga_pim"
PLATINUM_CSV = (
    ROOT
    / "results_platinum_na12878_hg38/processed/"
    "platinum_na12878_data_movement.csv"
)
MINIGRAPH_WARM_LOG_GLOB = (
    ROOT
    / "results_data_movement_time_minigraph/raw/"
    "genomics_cpu_minigraph/warm"
)
HG38_FASTA_GZ = Path(
    "/mnt/hdd/tsunghan/raw-ms-dataset/references/hg38/hg38.fa.gz"
)

HG38_LOGICAL_BASES = 3_209_286_105
MINIGRAPH_PROFILE_QUERY_BASES = 20_200_000
HDD_BANDWIDTH_BYTES_S = 109_051_904.0
PCIE_EFFECTIVE_BYTES_S = 24e9
SOURCE_CPU_MAX_GHZ = 5.4762891
TARGET_CPU_MAX_GHZ = 3.8
SOURCE_GPU_HBM_GBPS = 1792.0
TARGET_GPU_HBM_GBPS = 1935.0

PHASE_RE = re.compile(
    r"\[M::(?P<phase>[^:]+)::(?P<time>[0-9.]+)\*"
)


def read_minigraph_phases(log_dir: Path) -> dict[str, float | list[float]]:
    fixed_times = []
    mapping_times = []
    for log_path in sorted(log_dir.glob("trial_*/stderr.log")):
        phases: dict[str, float] = {}
        for line in log_path.read_text(encoding="utf-8").splitlines():
            match = PHASE_RE.search(line)
            if match:
                phases[match.group("phase")] = float(match.group("time"))
        fixed = phases["mg_opt_update"]
        mapped = phases["worker_pipeline"]
        fixed_times.append(fixed)
        mapping_times.append(mapped - fixed)
    if not fixed_times:
        raise RuntimeError(f"No minigraph warm logs found under {log_dir}")
    return {
        "fixed_phase_trials_s": fixed_times,
        "mapping_phase_trials_s": mapping_times,
        "fixed_phase_median_s": statistics.median(fixed_times),
        "mapping_phase_median_s": statistics.median(mapping_times),
    }


def hours(seconds: float) -> float:
    return seconds / 3600.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-dir", type=Path, default=MATCHED_DIR)
    parser.add_argument("--platinum-csv", type=Path, default=PLATINUM_CSV)
    parser.add_argument(
        "--minigraph-warm-log-dir",
        type=Path,
        default=MINIGRAPH_WARM_LOG_GLOB,
    )
    args = parser.parse_args()

    matched = json.loads(
        (args.matched_dir / "matched_mhc_summary.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        (
            args.matched_dir / "inputs/matched_mhc_manifest.json"
        ).read_text(encoding="utf-8")
    )
    pim_projection = json.loads(
        (
            args.matched_dir / "PLATINUM_NA12878_PIM_PROJECTION.json"
        ).read_text(encoding="utf-8")
    )
    with args.platinum_csv.open("r", encoding="utf-8") as handle:
        platinum = next(csv.DictReader(handle))

    full_bases = int(platinum["base_count"])
    full_sequences = int(platinum["fastq_records"])
    compressed_fastq_bytes = int(platinum["compressed_fastq_bytes"])
    physical_cpu_input_bytes = compressed_fastq_bytes + HG38_FASTA_GZ.stat().st_size
    query_base_scale = full_bases / int(manifest["query_total_bases"])

    cpu_phases = read_minigraph_phases(args.minigraph_warm_log_dir)
    minigraph_scale = full_bases / MINIGRAPH_PROFILE_QUERY_BASES
    cpu_source_compute_s = (
        float(cpu_phases["fixed_phase_median_s"])
        + float(cpu_phases["mapping_phase_median_s"]) * minigraph_scale
    )
    cpu_clock_scale = SOURCE_CPU_MAX_GHZ / TARGET_CPU_MAX_GHZ
    cpu_target_compute_s = cpu_source_compute_s * cpu_clock_scale
    cpu_storage_s = physical_cpu_input_bytes / HDD_BANDWIDTH_BYTES_S
    cpu_overlap_s = max(cpu_target_compute_s, cpu_storage_s)
    cpu_additive_s = cpu_target_compute_s + cpu_storage_s

    hga = matched["gpu_hga"]
    source_graph_vertices = int(manifest["reference_vertices"])
    graph_scale = HG38_LOGICAL_BASES / source_graph_vertices
    gpu_hbm_scale = SOURCE_GPU_HBM_GBPS / TARGET_GPU_HBM_GBPS
    hga_kernel_s = (
        float(hga["kernel_warm_median_s"])
        * graph_scale
        * query_base_scale
        * gpu_hbm_scale
    )

    graph_representation_expansion = float(
        manifest["hga_graph_representation_expansion"]
    )
    read_representation_expansion = (
        int(manifest["hga_reads_file_bytes"])
        / int(manifest["logical_query_payload_bytes"])
    )
    hga_graph_storage_bytes = HG38_LOGICAL_BASES * graph_representation_expansion
    hga_read_storage_bytes = full_bases * read_representation_expansion
    hga_storage_bytes = hga_graph_storage_bytes + hga_read_storage_bytes
    hga_storage_s = hga_storage_bytes / HDD_BANDWIDTH_BYTES_S

    source_logical_payload_bytes = (
        int(manifest["logical_graph_payload_bytes"])
        + int(manifest["logical_query_payload_bytes"])
    )
    target_logical_payload_bytes = HG38_LOGICAL_BASES + full_bases
    hga_h2d_bytes = (
        int(hga["h2d_bytes"])
        * target_logical_payload_bytes
        / source_logical_payload_bytes
    )
    hga_d2h_bytes = full_sequences * (
        int(hga["d2h_bytes"]) / int(manifest["hga_read_count"])
    )
    hga_pcie_s = (hga_h2d_bytes + hga_d2h_bytes) / PCIE_EFFECTIVE_BYTES_S
    hga_movement_s = hga_storage_s + hga_pcie_s
    hga_overlap_s = max(hga_kernel_s, hga_movement_s)
    hga_additive_s = hga_kernel_s + hga_movement_s

    pim = pim_projection["projected_multiram"]
    pim_external_s = float(pim["external_stream_pipeline_bottleneck_s"])
    pim_internal_fast_s = float(pim["internal_service_s_at_30_34_tbps"])
    pim_internal_slow_s = float(pim["internal_service_s_at_19_01_tbps"])
    pim_movement_fast_s = pim_external_s + pim_internal_fast_s
    pim_movement_slow_s = pim_external_s + pim_internal_slow_s
    pim_overlap_s = float(pim["full_time_perfect_stream_compute_overlap_s"])
    pim_additive_s = float(pim["full_time_conservative_additive_s"])

    output = {
        "status": "projection_not_full_dataset_measurement",
        "target_workload": {
            "dataset": "Platinum Genomes NA12878 PRJEB3246",
            "fastq_sequence_records": full_sequences,
            "query_bases": full_bases,
            "compressed_fastq_bytes": compressed_fastq_bytes,
            "reference": "UCSC hg38",
            "reference_bases": HG38_LOGICAL_BASES,
        },
        "target_platforms": {
            "cpu": "Intel Xeon 6 Granite Rapids 6515P, 16C/32T, up to 3.8 GHz",
            "gpu": "NVIDIA A100 80GB PCIe",
            "pim": "MultiRAM FeNAND + 3D-DRAM analytical model",
        },
        "cpu_minigraph": {
            "status": "projected_from_measured_hg38_100k_pair_phases",
            "minigraph_profile_query_bases": MINIGRAPH_PROFILE_QUERY_BASES,
            "fixed_graph_phase_median_s": cpu_phases["fixed_phase_median_s"],
            "mapping_phase_median_s": cpu_phases["mapping_phase_median_s"],
            "query_base_scale": minigraph_scale,
            "source_projected_compute_s": cpu_source_compute_s,
            "target_clock_scale": cpu_clock_scale,
            "target_compute_s": cpu_target_compute_s,
            "physical_storage_input_bytes": physical_cpu_input_bytes,
            "movement_storage_service_s": cpu_storage_s,
            "overall_perfect_overlap_s": cpu_overlap_s,
            "overall_conservative_additive_s": cpu_additive_s,
        },
        "gpu_hga": {
            "status": (
                "idealized_partitioned_projection; public HGA cannot directly "
                "represent or fit full hg38 on A100 80GB"
            ),
            "source_kernel_s": hga["kernel_warm_median_s"],
            "graph_scale": graph_scale,
            "query_base_scale": query_base_scale,
            "target_hbm_scale": gpu_hbm_scale,
            "target_kernel_s": hga_kernel_s,
            "projected_hga_graph_storage_bytes": hga_graph_storage_bytes,
            "projected_hga_read_storage_bytes": hga_read_storage_bytes,
            "movement_storage_service_s": hga_storage_s,
            "projected_h2d_bytes": hga_h2d_bytes,
            "projected_d2h_bytes": hga_d2h_bytes,
            "movement_pcie_service_s": hga_pcie_s,
            "movement_total_service_s": hga_movement_s,
            "overall_perfect_overlap_s": hga_overlap_s,
            "overall_conservative_additive_s": hga_additive_s,
        },
        "multiram_pim": {
            "status": "base_scaled_analytical_model",
            "movement_external_stream_s": pim_external_s,
            "movement_internal_3ddram_service_s_range": [
                pim_internal_fast_s,
                pim_internal_slow_s,
            ],
            "movement_total_service_s_range": [
                pim_movement_fast_s,
                pim_movement_slow_s,
            ],
            "overall_perfect_overlap_s": pim_overlap_s,
            "overall_conservative_additive_s": pim_additive_s,
        },
        "conservative_comparison": {
            "cpu_overall_speedup_from_multiram_x": cpu_additive_s / pim_additive_s,
            "hga_overall_speedup_from_multiram_x": hga_additive_s / pim_additive_s,
            "cpu_movement_fraction_percent": 100.0
            * cpu_storage_s
            / cpu_additive_s,
            "hga_movement_fraction_percent": 100.0
            * hga_movement_s
            / hga_additive_s,
            "pim_movement_fraction_percent_range": [
                100.0 * pim_movement_fast_s / pim_additive_s,
                100.0 * pim_movement_slow_s / pim_additive_s,
            ],
            "cpu_movement_time_reduction_with_multiram_percent_range": [
                100.0 * (1.0 - pim_movement_slow_s / cpu_storage_s),
                100.0 * (1.0 - pim_movement_fast_s / cpu_storage_s),
            ],
            "hga_movement_time_reduction_with_multiram_percent_range": [
                100.0 * (1.0 - pim_movement_slow_s / hga_movement_s),
                100.0 * (1.0 - pim_movement_fast_s / hga_movement_s),
            ],
        },
        "method_limits": [
            "All three full-dataset values are projections, not completed full runs.",
            "CPU movement is serialized storage service only; DRAM/cache stall time is unavailable.",
            "CPU clock scaling is a simple throughput model and does not model Xeon IPC or memory-channel differences.",
            "HGA kernel work is scaled as graph vertices times query bases and assumes ideal graph partitioning.",
            "Public HGA uses 32-bit graph indices and its full dynamic-programming state cannot fit hg38 on A100 80GB.",
            "HGA is score-only while minigraph is a mapper, so runtime is not an accuracy-equivalent algorithm comparison.",
            "PIM movement includes FeNAND external-stream service plus diagnostic internal 3D-DRAM service; internal service is not added twice to overall runtime.",
        ],
    }

    json_path = args.matched_dir / "PLATINUM_NA12878_CPU_HGA_PIM_PROJECTION.json"
    json_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    c = output["conservative_comparison"]
    report_lines = [
        "# Platinum NA12878 CPU Minigraph, GPU HGA, and MultiRAM Projection",
        "",
        "All values below use the full NA12878 workload: "
        f"{full_bases:,} query bases ({full_sequences:,} FASTQ sequences) "
        "against UCSC hg38.",
        "",
        "## Main Result",
        "",
        "| System | Projection status | Overall time, conservative | Overall time, perfect overlap | Explicit movement time | Movement / overall |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        f"| CPU minigraph, Xeon 6515P | Measured-phase projection | **{cpu_additive_s:,.1f} s ({hours(cpu_additive_s):.2f} h)** | {cpu_overlap_s:,.1f} s ({hours(cpu_overlap_s):.2f} h) | **{cpu_storage_s:,.1f} s ({hours(cpu_storage_s):.2f} h)** | **{c['cpu_movement_fraction_percent']:.2f}%** |",
        f"| GPU HGA, A100 80GB | Idealized partitioned projection | **{hga_additive_s:,.1f} s ({hours(hga_additive_s):.2f} h)** | {hga_overlap_s:,.1f} s ({hours(hga_overlap_s):.2f} h) | **{hga_movement_s:,.1f} s ({hours(hga_movement_s):.2f} h)** | **{c['hga_movement_fraction_percent']:.2f}%** |",
        f"| MultiRAM PIM | Analytical model | **{pim_additive_s:,.1f} s ({pim_additive_s / 60:.2f} min)** | {pim_overlap_s:,.1f} s ({pim_overlap_s / 60:.2f} min) | **{pim_movement_fast_s:,.1f}-{pim_movement_slow_s:,.1f} s ({pim_movement_fast_s / 60:.2f}-{pim_movement_slow_s / 60:.2f} min)** | **{c['pim_movement_fraction_percent_range'][0]:.2f}-{c['pim_movement_fraction_percent_range'][1]:.2f}%** |",
        "",
        "`Conservative` adds movement and compute. `Perfect overlap` takes their maximum. "
        "The conservative column is the safer single number to report.",
        "",
        "## Movement Breakdown",
        "",
        f"- CPU storage: {physical_cpu_input_bytes / 1e9:.3f} GB / {HDD_BANDWIDTH_BYTES_S / 1e6:.3f} MB/s = **{cpu_storage_s:,.1f} s**.",
        f"- GPU HGA storage: {hga_storage_bytes / 1e9:.3f} GB / {HDD_BANDWIDTH_BYTES_S / 1e6:.3f} MB/s = **{hga_storage_s:,.1f} s**.",
        f"- GPU HGA PCIe: {(hga_h2d_bytes + hga_d2h_bytes) / 1e9:.3f} GB / 24 GB/s = **{hga_pcie_s:,.1f} s**.",
        f"- MultiRAM FeNAND bottleneck: **{pim_external_s:,.1f} s**; internal 3D-DRAM service: **{pim_internal_fast_s:.3f}-{pim_internal_slow_s:.3f} s**.",
        "",
        "## Conservative Comparison",
        "",
        f"- MultiRAM overall speedup versus CPU minigraph: **{c['cpu_overall_speedup_from_multiram_x']:.2f}x**.",
        f"- MultiRAM overall speedup versus idealized GPU HGA: **{c['hga_overall_speedup_from_multiram_x']:.2f}x**.",
        f"- MultiRAM movement-time reduction versus CPU: **{c['cpu_movement_time_reduction_with_multiram_percent_range'][0]:.2f}-{c['cpu_movement_time_reduction_with_multiram_percent_range'][1]:.2f}%**.",
        f"- MultiRAM movement-time reduction versus GPU HGA: **{c['hga_movement_time_reduction_with_multiram_percent_range'][0]:.2f}-{c['hga_movement_time_reduction_with_multiram_percent_range'][1]:.2f}%**.",
        "",
        "## Important Qualification",
        "",
        "The CPU number is projected from real hg38 minigraph phase measurements. "
        "The HGA number is theoretical: public HGA cannot directly represent or "
        "fit full hg38 on an A100 80GB, and HGA returns alignment scores rather "
        "than minigraph-compatible mappings. Do not label the HGA row as a "
        "completed full-NA12878 GPU run.",
        "",
        "CPU movement excludes unmeasured DRAM/cache stalls, so it is an explicit "
        "storage-service lower bound. The PIM result is a base-scaled architecture "
        "model and can change if whole-genome anchor or DP-cell density differs from MHC. "
        "The CPU compute projection also does not add gzip decompression cost because "
        "the measured 100k-pair profiling subset was uncompressed.",
        "",
    ]
    report_path = args.matched_dir / "PLATINUM_NA12878_CPU_HGA_PIM_PROJECTION.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(report_path)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
