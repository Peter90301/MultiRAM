#!/usr/bin/env python3
"""Align Table VI and full-PXD projections with Fig. 18/19 throughput."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/home/tsl012/multiomic/multiram_data_movement")
OUTPUT_DIR = ROOT / "results_pxd024364_1tb"
RAPIDS_RESULT = (
    ROOT
    / "results_rapids_matched_clustering"
    / "matched_rapids_movement.json"
)
PIM_RESULT = OUTPUT_DIR / "PXD024364_PIM_RUNTIME_PROJECTION.json"

# Existing Table VI CPU anchors for the matched PXD024364 inputs.
CLUSTERING_CPU_COMPUTE_S = 95.691
CLUSTERING_CPU_MOVEMENT_S = 0.391868
CLUSTERING_PIM_MOVEMENT_S = 0.008447097617669753
CLUSTERING_SMALL_BYTES = 68_172_703
CLUSTERING_SMALL_SPECTRA = 30_242

OMS_CPU_COMPUTE_S = 57_348.019
OMS_CPU_MOVEMENT_S = 44.615
OMS_GPU_MOVEMENT_LOWER_BOUND_S = 85.418
OMS_PIM_MOVEMENT_S = 0.031119772739776237
OMS_SMALL_BYTES = 153_197_038
OMS_SMALL_VALID_QUERIES = 29_835

# Means of the normalized speedups printed in Fig. 18 and Fig. 19.
CLUSTERING_FACTORS = {
    "a100_rapids": 581.4,
    "h100_rapids": 840.6,
    "multiram": 16_393.0,
}
OMS_FACTORS = {
    "a100_ann_solo": 508.0,
    "multiram": 14_226.2,
}


def row(
    compute_s: float,
    movement_s: float,
    cpu_overall_s: float,
) -> dict[str, float]:
    overall_s = compute_s + movement_s
    return {
        "compute_s": compute_s,
        "movement_s": movement_s,
        "overall_s": overall_s,
        "movement_fraction_percent": 100.0 * movement_s / overall_s,
        "overall_speedup_vs_cpu_x": cpu_overall_s / overall_s,
    }


def format_duration(seconds: float) -> str:
    if seconds >= 86_400:
        return f"{seconds:,.1f} s ({seconds / 86_400:.2f} d)"
    if seconds >= 3_600:
        return f"{seconds:,.1f} s ({seconds / 3_600:.2f} h)"
    if seconds >= 60:
        return f"{seconds:,.1f} s ({seconds / 60:.2f} min)"
    return f"{seconds:,.3f} s"


def make_rows(
    clustering_cpu_compute_s: float,
    clustering_cpu_movement_s: float,
    clustering_rapids_movement_s: float,
    clustering_pim_movement_s: float,
    oms_cpu_compute_s: float,
    oms_cpu_movement_s: float,
    oms_gpu_movement_s: float,
    oms_pim_movement_s: float,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    clustering_cpu_overall_s = (
        clustering_cpu_compute_s + clustering_cpu_movement_s
    )
    oms_cpu_overall_s = oms_cpu_compute_s + oms_cpu_movement_s

    clustering = {
        "cpu_falcon": row(
            clustering_cpu_compute_s,
            clustering_cpu_movement_s,
            clustering_cpu_overall_s,
        ),
        "a100_rapids": row(
            clustering_cpu_compute_s / CLUSTERING_FACTORS["a100_rapids"],
            clustering_rapids_movement_s,
            clustering_cpu_overall_s,
        ),
        "h100_rapids": row(
            clustering_cpu_compute_s / CLUSTERING_FACTORS["h100_rapids"],
            clustering_rapids_movement_s,
            clustering_cpu_overall_s,
        ),
        "multiram": row(
            clustering_cpu_compute_s / CLUSTERING_FACTORS["multiram"],
            clustering_pim_movement_s,
            clustering_cpu_overall_s,
        ),
    }
    oms = {
        "cpu_spectrast": row(
            oms_cpu_compute_s,
            oms_cpu_movement_s,
            oms_cpu_overall_s,
        ),
        "a100_ann_solo": row(
            oms_cpu_compute_s / OMS_FACTORS["a100_ann_solo"],
            oms_gpu_movement_s,
            oms_cpu_overall_s,
        ),
        "multiram": row(
            oms_cpu_compute_s / OMS_FACTORS["multiram"],
            oms_pim_movement_s,
            oms_cpu_overall_s,
        ),
    }
    return clustering, oms


def append_rows(
    lines: list[str],
    workload: str,
    rows: dict[str, dict[str, float]],
    labels: dict[str, str],
    full_duration: bool = False,
) -> None:
    for key, item in rows.items():
        if full_duration:
            compute = format_duration(item["compute_s"])
            movement = format_duration(item["movement_s"])
            overall = format_duration(item["overall_s"])
        else:
            compute = f"{item['compute_s']:,.3f} s"
            movement = f"{item['movement_s']:,.3f} s"
            overall = f"{item['overall_s']:,.3f} s"
        lines.append(
            f"| {workload} | {labels[key]} | {compute} | {movement} | "
            f"{overall} | {item['movement_fraction_percent']:.3f}% | "
            f"{item['overall_speedup_vs_cpu_x']:,.2f}x |"
        )


def main() -> int:
    if not RAPIDS_RESULT.exists():
        raise FileNotFoundError(
            f"Run run_matched_rapids_clustering_movement.py first: "
            f"{RAPIDS_RESULT}"
        )
    rapids_result = json.loads(RAPIDS_RESULT.read_text(encoding="utf-8"))
    pim_result = json.loads(PIM_RESULT.read_text(encoding="utf-8"))
    rapids_movement_s = float(
        rapids_result["movement"]["explicit_movement_s"]
    )

    small_clustering, small_oms = make_rows(
        CLUSTERING_CPU_COMPUTE_S,
        CLUSTERING_CPU_MOVEMENT_S,
        rapids_movement_s,
        CLUSTERING_PIM_MOVEMENT_S,
        OMS_CPU_COMPUTE_S,
        OMS_CPU_MOVEMENT_S,
        OMS_GPU_MOVEMENT_LOWER_BOUND_S,
        OMS_PIM_MOVEMENT_S,
    )

    full_scope = pim_result["full_converted_batch_scope"]
    full_bytes = int(full_scope["mgf_bytes"])
    full_spectra = int(full_scope["spectra"])
    full_oms_queries = int(full_scope["oms_valid_query_spectra"])
    clustering_compute_scale = full_spectra / CLUSTERING_SMALL_SPECTRA
    clustering_movement_scale = full_bytes / CLUSTERING_SMALL_BYTES
    oms_compute_scale = full_oms_queries / OMS_SMALL_VALID_QUERIES
    oms_movement_scale = full_bytes / OMS_SMALL_BYTES

    full_clustering, full_oms = make_rows(
        CLUSTERING_CPU_COMPUTE_S * clustering_compute_scale,
        CLUSTERING_CPU_MOVEMENT_S * clustering_movement_scale,
        rapids_movement_s * clustering_movement_scale,
        float(full_scope["clustering"]["movement_s"]),
        OMS_CPU_COMPUTE_S * oms_compute_scale,
        OMS_CPU_MOVEMENT_S * oms_movement_scale,
        OMS_GPU_MOVEMENT_LOWER_BOUND_S * oms_movement_scale,
        float(full_scope["oms"]["movement_s"]),
    )

    output = {
        "status": "component_throughput_aligned_projection",
        "method": {
            "compute": (
                "Matched CPU compute divided by the mean normalized component "
                "speedup printed in Fig. 18 or Fig. 19."
            ),
            "clustering_rapids_movement": (
                "Measured on the matched small MGF as cold-minus-warm storage "
                "service plus Nsight CUDA memcpy union."
            ),
            "full_compute": (
                "Small clustering compute scaled by spectrum count; small OMS "
                "compute scaled by valid query count."
            ),
            "full_movement": (
                "CPU/GPU movement scaled by MGF bytes. MultiRAM movement comes "
                "from the existing FeRAM/3D-DRAM analytical model."
            ),
            "overall": "Compute + Movement.",
        },
        "figure_factors": {
            "clustering": CLUSTERING_FACTORS,
            "oms": OMS_FACTORS,
        },
        "matched_rapids_measurement": {
            "source": str(RAPIDS_RESULT),
            "measured_gpu": rapids_result["measured_hardware"]["gpu"],
            "software": rapids_result["software"],
            "movement": rapids_result["movement"],
        },
        "small_matched_scope": {
            "clustering": small_clustering,
            "oms": small_oms,
        },
        "full_pxd024364_projection": {
            "scope": {
                "files": int(full_scope["files"]),
                "mgf_bytes": full_bytes,
                "spectra": full_spectra,
                "oms_valid_queries": full_oms_queries,
                "clustering_compute_scale": clustering_compute_scale,
                "clustering_movement_scale": clustering_movement_scale,
                "oms_compute_scale": oms_compute_scale,
                "oms_movement_scale": oms_movement_scale,
            },
            "clustering": full_clustering,
            "oms": full_oms,
        },
        "validity": {
            "a100_h100_compute": (
                "Architecture-level projections from Fig. 18/19 normalized "
                "throughput, not A100/H100 hardware measurements."
            ),
            "clustering_movement": (
                "Matched RAPIDS cuML measurement on a local RTX PRO 6000 "
                "Blackwell GPU. HBM traffic internal to kernels is excluded."
            ),
            "full_dataset": (
                "Linear projection from the small matched inputs; the 645.197 "
                "GB MGF batch was not rerun for this table."
            ),
            "oms_gpu_movement": (
                "Storage-service lower bound; H2D/D2H and HBM traffic remain "
                "unmeasured."
            ),
        },
    }

    json_path = OUTPUT_DIR / "TABLE_VI_COMPONENT_THROUGHPUT_ALIGNMENT.json"
    json_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    clustering_labels = {
        "cpu_falcon": "Xeon + Falcon",
        "a100_rapids": "A100 + RAPIDS",
        "h100_rapids": "H100 + RAPIDS",
        "multiram": "MultiRAM",
    }
    oms_labels = {
        "cpu_spectrast": "Xeon + SpectraST",
        "a100_ann_solo": "A100 + ANN-SoLo",
        "multiram": "MultiRAM",
    }
    lines = [
        "# Table VI Alignment with Fig. 18 and Fig. 19",
        "",
        "## Method",
        "",
        "Compute time is projected from the normalized component throughput in "
        "Fig. 18 and Fig. 19:",
        "",
        "```text",
        "T_accelerator_compute = T_CPU_compute / mean_figure_speedup",
        "T_overall = T_compute + T_movement",
        "```",
        "",
        "PXD024364 is not one of the five figure datasets, so these are "
        "architecture-level projections anchored to the matched PXD inputs.",
        "",
        "| Workload | A100 | H100 | MultiRAM |",
        "| --- | ---: | ---: | ---: |",
        "| Clustering, Fig. 18 mean | 581.4x RAPIDS | 840.6x RAPIDS | 16,393x |",
        "| OMS, Fig. 19 mean | 508.0x ANN-SoLo | N/A | 14,226.2x |",
        "",
        "## Updated Small-Input Table VI",
        "",
        "| Workload | Platform | Compute | Movement | Overall | Fraction | Speedup |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    append_rows(
        lines, "Clustering", small_clustering, clustering_labels
    )
    append_rows(lines, "OMS", small_oms, oms_labels)
    lines.extend(
        [
            "",
            "The clustering movement star is removed because `0.436784 s` now "
            "comes from a matched RAPIDS cuML run on the same 68.173 MB MGF: "
            "`0.412073 s` cold-minus-warm storage plus `0.024711 s` CUDA-copy "
            "union. The trace used cuML 26.02 DBSCAN with a Hyper-Spec HDC "
            "frontend on a local RTX PRO 6000 Blackwell GPU.",
            "",
            "The OMS GPU movement value remains a lower bound because its GPU "
            "copy and HBM traffic were not profiled.",
            "",
            "## Full PXD024364 Projection",
            "",
            "Scope: 853 converted files, 645.197 GB MGF, 55,630,950 spectra, "
            "and 50,659,262 valid OMS queries. No full RAPIDS or OMS run was "
            "started for this update.",
            "",
            "| Workload | Platform | Compute | Movement | Overall | Fraction | Speedup |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    append_rows(
        lines,
        "Clustering",
        full_clustering,
        clustering_labels,
        full_duration=True,
    )
    append_rows(
        lines, "OMS", full_oms, oms_labels, full_duration=True
    )
    lines.extend(
        [
            "",
            "## Projection Boundary",
            "",
            "- Clustering compute scales by spectrum count; OMS compute scales "
            "by valid query count.",
            "- CPU/GPU movement scales by MGF bytes from the matched input.",
            "- MultiRAM movement uses the existing FeRAM/3D-DRAM model rather "
            "than byte-scaling the small row.",
            "- A100/H100 compute is derived from Fig. 18/19 normalized "
            "throughput. It is not an A100/H100 hardware measurement.",
            "- RAPIDS movement excludes kernel-internal HBM traffic.",
            "",
            "## Caption Text",
            "",
            "`Accelerator compute times are projected from the mean normalized "
            "component-throughput speedups in Fig. 18 and Fig. 19, anchored to "
            "matched Xeon compute. RAPIDS clustering movement is measured on "
            "the matched PXD024364 subset using cold/warm storage timing and "
            "Nsight CUDA-copy traces; full-dataset values are linear "
            "projections. MultiRAM movement is obtained from the FeRAM/3D-DRAM "
            "analytical model.`",
        ]
    )
    report_path = OUTPUT_DIR / "TABLE_VI_COMPONENT_THROUGHPUT_ALIGNMENT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "report": str(report_path),
                "json": str(json_path),
                "rapids_movement_s": rapids_movement_s,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
