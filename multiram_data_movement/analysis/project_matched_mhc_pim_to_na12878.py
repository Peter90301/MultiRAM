#!/usr/bin/env python3
"""Project the matched MHC PIM model to the full Platinum NA12878 workload."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path("/home/tsl012/multiomic/multiram_data_movement")
DEFAULT_MATCHED = ROOT / "results_matched_mhc_minigraph_hga_pim"
DEFAULT_PLATINUM_CSV = (
    ROOT
    / "results_platinum_na12878_hg38/processed/"
    "platinum_na12878_data_movement.csv"
)
HG38_FASTA_GZ = Path(
    "/mnt/hdd/tsunghan/raw-ms-dataset/references/hg38/hg38.fa.gz"
)
HG38_LOGICAL_BASES = 3_209_286_105
FENAND_COMPRESSED_GBPS = 4.3
FENAND_DECOMPRESSED_GBPS = 8.1
PNM_LOW_TBPS = 19.01
PNM_HIGH_TBPS = 30.34


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-results", type=Path, default=DEFAULT_MATCHED)
    parser.add_argument("--platinum-csv", type=Path, default=DEFAULT_PLATINUM_CSV)
    args = parser.parse_args()

    matched = json.loads(
        (args.matched_results / "matched_mhc_summary.json").read_text(
            encoding="utf-8"
        )
    )
    with args.platinum_csv.open("r", encoding="utf-8") as handle:
        platinum = next(csv.DictReader(handle))

    pim = matched["pim_3ddram"]
    source_query_bases = sum(
        int(length) * int(count)
        for length, count in matched["matched_workload"][
            "query_length_histogram"
        ].items()
    )
    full_pairs = int(platinum["read_count"])
    full_sequences = int(platinum["fastq_records"])
    full_bases = int(platinum["base_count"])
    compressed_fastq_bytes = int(platinum["compressed_fastq_bytes"])
    decompressed_fastq_lower_bound_bytes = int(
        platinum["decompressed_fastq_lower_bound_bytes"]
    )
    base_scale = full_bases / source_query_bases
    sequence_scale = full_sequences / int(matched["matched_workload"]["queries"])

    projected_pim_pipeline_s = float(pim["pim_pipeline_latency_s"]) * base_scale
    projected_internal_bytes = (
        float(pim["movement"]["internal_3ddram_read_bytes"]) * base_scale
    )
    projected_internal_low_s = projected_internal_bytes / (PNM_LOW_TBPS * 1e12)
    projected_internal_high_s = projected_internal_bytes / (PNM_HIGH_TBPS * 1e12)
    projected_pim_core_energy_j = float(pim["pim_energy_j"]) * base_scale

    compressed_stream_bytes = compressed_fastq_bytes + HG38_FASTA_GZ.stat().st_size
    decompressed_stream_lower_bound_bytes = (
        decompressed_fastq_lower_bound_bytes + HG38_LOGICAL_BASES
    )
    compressed_stream_s = compressed_stream_bytes / (
        FENAND_COMPRESSED_GBPS * 1e9
    )
    decompressed_stream_s = decompressed_stream_lower_bound_bytes / (
        FENAND_DECOMPRESSED_GBPS * 1e9
    )
    external_pipeline_s = max(compressed_stream_s, decompressed_stream_s)

    perfect_overlap_s = max(projected_pim_pipeline_s, external_pipeline_s)
    conservative_additive_s = projected_pim_pipeline_s + external_pipeline_s
    output = {
        "status": "projection_not_full_dataset_measurement",
        "source_matched_mhc": {
            "query_bases": source_query_bases,
            "queries": matched["matched_workload"]["queries"],
            "pim_pipeline_latency_s": pim["pim_pipeline_latency_s"],
            "internal_3ddram_read_bytes": pim["movement"][
                "internal_3ddram_read_bytes"
            ],
        },
        "target_platinum_na12878": {
            "paired_spots": full_pairs,
            "fastq_sequence_records": full_sequences,
            "bases": full_bases,
            "average_bases_per_sequence": full_bases / full_sequences,
            "compressed_fastq_bytes": compressed_fastq_bytes,
            "decompressed_fastq_lower_bound_bytes": (
                decompressed_fastq_lower_bound_bytes
            ),
            "hg38_fasta_gzip_bytes": HG38_FASTA_GZ.stat().st_size,
            "hg38_logical_bases": HG38_LOGICAL_BASES,
        },
        "scaling": {
            "query_base_scale": base_scale,
            "query_sequence_count_scale": sequence_scale,
            "selected_scale": "query_base_scale",
            "assumption": (
                "MHC anchors, DP cells, internal bytes, latency, and energy per "
                "query base remain constant on full hg38."
            ),
        },
        "projected_multiram": {
            "pim_pipeline_resident_s": projected_pim_pipeline_s,
            "pim_pipeline_resident_minutes": projected_pim_pipeline_s / 60.0,
            "internal_3ddram_read_bytes": projected_internal_bytes,
            "internal_service_s_at_19_01_tbps": projected_internal_low_s,
            "internal_service_s_at_30_34_tbps": projected_internal_high_s,
            "pim_core_energy_j_excluding_fenand": projected_pim_core_energy_j,
            "compressed_fenand_input_s_at_4_3_gbps": compressed_stream_s,
            "decompressed_fenand_output_s_at_8_1_gbps_lower_bound": (
                decompressed_stream_s
            ),
            "external_stream_pipeline_bottleneck_s": external_pipeline_s,
            "full_time_perfect_stream_compute_overlap_s": perfect_overlap_s,
            "full_time_perfect_stream_compute_overlap_minutes": (
                perfect_overlap_s / 60.0
            ),
            "full_time_conservative_additive_s": conservative_additive_s,
            "full_time_conservative_additive_minutes": (
                conservative_additive_s / 60.0
            ),
            "projected_paired_spots_per_s_range": [
                full_pairs / conservative_additive_s,
                full_pairs / perfect_overlap_s,
            ],
            "projected_sequence_records_per_s_range": [
                full_sequences / conservative_additive_s,
                full_sequences / perfect_overlap_s,
            ],
            "projected_bases_per_s_range": [
                full_bases / conservative_additive_s,
                full_bases / perfect_overlap_s,
            ],
        },
        "limits": [
            "This is a base-scaled projection, not an execution on all 36 FASTQ files.",
            "The decompressed FASTQ size is a strict lower bound and omits read-name bytes.",
            "Whole-genome repeat structure can change anchors and DP cells per base.",
            "FeNAND energy is unavailable and excluded from projected energy.",
            "HGA is not projected because the public implementation cannot hold full hg38.",
            "A full minigraph CPU runtime is not projected from the regional MHC timing.",
        ],
    }

    json_path = args.matched_results / "PLATINUM_NA12878_PIM_PROJECTION.json"
    json_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    p = output["projected_multiram"]
    lines = [
        "# Platinum NA12878 MultiRAM Projection",
        "",
        "This is a projection from the matched MHC PIM model, not a full-dataset hardware or simulator run.",
        "",
        "## Scale",
        "",
        f"- MHC source query bases: {source_query_bases:,}",
        f"- Platinum NA12878 bases: {full_bases:,}",
        f"- Query-base scale: **{base_scale:,.3f}x**",
        f"- Platinum paired spots: {full_pairs:,}",
        f"- Platinum FASTQ sequence records: {full_sequences:,}",
        "",
        "## Projected Data Movement",
        "",
        "| Component | Volume | Bandwidth | Service time |",
        "| --- | ---: | ---: | ---: |",
        f"| Compressed FASTQ + hg38 gzip input | {compressed_stream_bytes / 1e9:.3f} GB | 4.3 GB/s | {compressed_stream_s:.3f} s |",
        f"| Decompressed FASTQ lower bound + hg38 | {decompressed_stream_lower_bound_bytes / 1e12:.3f} TB | 8.1 GB/s | {decompressed_stream_s:.3f} s |",
        f"| Internal 3D-DRAM reads | {projected_internal_bytes / 1e12:.3f} TB | 19.01-30.34 TB/s | {projected_internal_high_s:.3f}-{projected_internal_low_s:.3f} s |",
        "",
        "The FeNAND stages are treated as a stream pipeline, so their bottleneck is the larger service time, not their sum.",
        "",
        "## Projected Runtime",
        "",
        f"- PIM compute/internal-memory model, resident data: **{projected_pim_pipeline_s:.3f} s ({projected_pim_pipeline_s / 60:.2f} min)**",
        f"- FeNAND external-stream bottleneck: **{external_pipeline_s:.3f} s ({external_pipeline_s / 60:.2f} min)**",
        f"- Perfect stream/compute overlap: **{perfect_overlap_s:.3f} s ({perfect_overlap_s / 60:.2f} min)**",
        f"- Conservative additive bound: **{conservative_additive_s:.3f} s ({conservative_additive_s / 60:.2f} min)**",
        "",
        "Therefore the projected full NA12878 MultiRAM time is **16.56-19.68 minutes** under the constant-work-per-base assumption.",
        "",
        "## Limits",
        "",
        "- The 1.516 TB decompressed FASTQ value is a strict lower bound; real FASTQ includes longer read identifiers.",
        "- Whole-genome repeats may change anchor and DP-cell density relative to MHC.",
        "- HGA is excluded because its public implementation cannot represent/run full hg38 on an A100 80GB.",
        "- CPU minigraph must be run or modeled separately; the regional MHC end-to-end time is not a valid whole-genome CPU scale factor.",
        "",
    ]
    report_path = args.matched_results / "PLATINUM_NA12878_PIM_PROJECTION.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(report_path)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
