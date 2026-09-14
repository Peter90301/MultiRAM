#!/usr/bin/env python3
"""Project measured movement-time rows to Xeon 6515P and A100 80GB PCIe."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


DEFAULT_RESULTS = Path(
    "/home/tsl012/multiomic/multiram_data_movement/results_data_movement_time_minigraph"
)
SOURCE_CPU_MAX_GHZ = 5.4762891
TARGET_CPU_MAX_GHZ = 3.8
SOURCE_GPU_HBM_GBPS = 1792.0
TARGET_GPU_HBM_GBPS = 1935.0


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def hga_gpu_active_s(results_dir: Path) -> float:
    values: list[float] = []
    for path in (results_dir / "raw/genomics_gpu_hga/warm").glob(
        "trial_*/stdout.log"
    ):
        match = re.search(
            r"Time:\s*([0-9.eE+-]+)s",
            path.read_text(encoding="utf-8", errors="replace"),
        )
        if match:
            values.append(float(match.group(1)))
    if not values:
        raise RuntimeError("No HGA GPU-active timing found")
    return statistics.median(values)


def hyperspec_gpu_active_s(results_dir: Path) -> float:
    values: list[float] = []
    pattern = re.compile(r"GPU clustering in ([0-9.]+) s")
    for path in (results_dir / "raw/proteomics_gpu_hyperspec/warm").glob(
        "trial_*/stderr.log"
    ):
        total = sum(
            float(value)
            for value in pattern.findall(
                path.read_text(encoding="utf-8", errors="replace")
            )
        )
        if total:
            values.append(total)
    if not values:
        raise RuntimeError("No Hyper-Spec GPU-active timing found")
    return statistics.median(values)


def project_cpu(row: dict[str, object], cpu_scale: float) -> dict[str, object]:
    storage_s = float(row["storage_movement_s"])
    target_warm_s = float(row["warm_median_s"]) * cpu_scale
    target_cold_s = target_warm_s + storage_s
    return {
        **row,
        "source_warm_s": row["warm_median_s"],
        "source_cold_s": row["cold_median_s"],
        "projected_warm_s": target_warm_s,
        "projected_cold_s": target_cold_s,
        "projected_explicit_movement_s": storage_s,
        "projected_explicit_movement_fraction_percent": 100.0
        * storage_s
        / target_cold_s,
        "projection_partition": {
            "host_phase_source_s": row["warm_median_s"],
            "storage_phase_s_unchanged": storage_s,
        },
    }


def project_gpu(
    row: dict[str, object],
    gpu_active_source_s: float,
    cpu_scale: float,
    gpu_scale: float,
) -> dict[str, object]:
    source_warm_s = float(row["warm_median_s"])
    storage_s = float(row["storage_movement_s"])
    copy_s = float(row["gpu_copy_union_s"])
    host_source_s = max(0.0, source_warm_s - gpu_active_source_s)
    gpu_compute_source_s = max(0.0, gpu_active_source_s - copy_s)
    target_host_s = host_source_s * cpu_scale
    target_gpu_compute_s = gpu_compute_source_s * gpu_scale
    # The measured copies are highly fragmented, so peak-PCIe scaling would be
    # misleading. Retain their measured service time as a conservative estimate.
    target_copy_s = copy_s
    target_warm_s = target_host_s + target_gpu_compute_s + target_copy_s
    target_cold_s = target_warm_s + storage_s
    movement_s = storage_s + target_copy_s
    return {
        **row,
        "source_warm_s": source_warm_s,
        "source_cold_s": row["cold_median_s"],
        "projected_warm_s": target_warm_s,
        "projected_cold_s": target_cold_s,
        "projected_explicit_movement_s": movement_s,
        "projected_explicit_movement_fraction_percent": 100.0
        * movement_s
        / target_cold_s,
        "projection_partition": {
            "host_phase_source_s": host_source_s,
            "gpu_active_source_s": gpu_active_source_s,
            "gpu_compute_source_s": gpu_compute_source_s,
            "cuda_copy_source_and_target_s": target_copy_s,
            "storage_phase_s_unchanged": storage_s,
            "target_host_phase_s": target_host_s,
            "target_gpu_compute_phase_s": target_gpu_compute_s,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    measured = load_json(args.results_dir / "data_movement_time_summary.json")
    assert isinstance(measured, list)
    cpu_scale = SOURCE_CPU_MAX_GHZ / TARGET_CPU_MAX_GHZ
    gpu_scale = SOURCE_GPU_HBM_GBPS / TARGET_GPU_HBM_GBPS
    workload_names = {str(item["workload"]) for item in measured}
    gpu_active: dict[str, float] = {}
    if "genomics_gpu_hga" in workload_names:
        gpu_active["genomics_gpu_hga"] = hga_gpu_active_s(args.results_dir)
    if "proteomics_gpu_hyperspec" in workload_names:
        gpu_active["proteomics_gpu_hyperspec"] = hyperspec_gpu_active_s(args.results_dir)
    projected: list[dict[str, object]] = []
    for item in measured:
        row = dict(item)
        if row["platform"] == "cpu":
            projected.append(project_cpu(row, cpu_scale))
        else:
            projected.append(
                project_gpu(
                    row,
                    gpu_active[str(row["workload"])],
                    cpu_scale,
                    gpu_scale,
                )
            )

    oms_source = load_json(args.results_dir / "oms_storage_service_model.json")
    assert isinstance(oms_source, list)
    oms_projection: list[dict[str, object]] = []
    oms_target_runtime = {
        "CPU SpectraST": 57392.634,
        "GPU ANN-SoLo": 21854.578,
    }
    for item in oms_source:
        model = dict(item)
        target_wall_s = oms_target_runtime[str(model["platform"])]
        movement_s = float(model["modeled_storage_service_s"])
        oms_projection.append(
            {
                **model,
                "projected_overall_runtime_s": target_wall_s,
                "projected_storage_fraction_percent": 100.0
                * movement_s
                / target_wall_s,
                "status": "storage-only model; no cold/warm or A100 CUDA trace",
            }
        )

    projected_by_name = {str(row["workload"]): row for row in projected}
    clustering_speedup = None
    if {
        "proteomics_cpu_falcon",
        "proteomics_gpu_hyperspec",
    }.issubset(projected_by_name):
        clustering_speedup = (
            float(projected_by_name["proteomics_cpu_falcon"]["projected_cold_s"])
            / float(
                projected_by_name["proteomics_gpu_hyperspec"]["projected_cold_s"]
            )
        )
    oms_by_name = {str(row["platform"]): row for row in oms_projection}
    oms_speedup = None
    if {"CPU SpectraST", "GPU ANN-SoLo"}.issubset(oms_by_name):
        oms_speedup = (
            float(oms_by_name["CPU SpectraST"]["projected_overall_runtime_s"])
            / float(oms_by_name["GPU ANN-SoLo"]["projected_overall_runtime_s"])
        )

    output = {
        "status": "hardware_projection_not_target_hardware_measurement",
        "source_platform": {
            "cpu": "AMD Ryzen Threadripper PRO 9985WX",
            "gpu": "NVIDIA RTX PRO 6000 Blackwell Max-Q",
        },
        "target_platform": {
            "cpu": "Intel Xeon 6 Granite Rapids 6515P, 16C/32T, up to 3.8 GHz",
            "gpu": "NVIDIA A100 80GB PCIe",
        },
        "scales": {
            "host_cpu_time": cpu_scale,
            "gpu_active_time": gpu_scale,
            "storage_time": 1.0,
            "cuda_copy_time": 1.0,
        },
        "assumptions": [
            "The target uses the same measured HDD; storage service time is unchanged.",
            "Host phases scale by source maximum clock divided by 3.8 GHz.",
            "GPU-active compute scales by 1792/1935 GB/s HBM bandwidth.",
            "CUDA copy time is retained because thousands of small transfers are not peak-PCIe-bandwidth limited.",
            "The target has enough cores for the measured 16-thread minigraph and 8-thread Hyper-Spec host phases.",
        ],
        "workloads": projected,
        "oms_storage_only": oms_projection,
        "comparisons": {
            "proteomics_clustering_gpu_speedup": clustering_speedup,
            "oms_gpu_speedup": oms_speedup,
            "genomics_gpu_speedup": None,
            "genomics_note": "Not computed because minigraph uses hg38 and HGA uses the MHC graph.",
        },
    }
    json_path = args.results_dir / "GNR6515P_A100_DATA_MOVEMENT_TIME_PROJECTION.json"
    json_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Xeon 6515P and A100 80GB Data-Movement Time Projection",
        "",
        "This is a phase-based projection from the Threadripper PRO 9985WX and RTX PRO 6000 measurements. It is not a target-hardware run.",
        "",
        "## Scaling",
        "",
        f"- Host/CPU time scale: `{SOURCE_CPU_MAX_GHZ} / {TARGET_CPU_MAX_GHZ} = {cpu_scale:.6f}`",
        f"- GPU-active time scale: `{SOURCE_GPU_HBM_GBPS} / {TARGET_GPU_HBM_GBPS} = {gpu_scale:.6f}`",
        "- Storage time scale: `1.0` (same HDD assumption)",
        "- CUDA copy time scale: `1.0` (fragmented-copy conservative assumption)",
        "",
        "## Projected Subset Results",
        "",
        "| Workload | Target | Projected cold overall (s) | Projected warm (s) | Critical-path explicit movement (s) | Movement / cold |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in projected:
        target = "Xeon 6515P" if row["platform"] == "cpu" else "Xeon 6515P + A100 80GB"
        lines.append(
            f"| {row['workload']} | {target} | {row['projected_cold_s']:.3f} | "
            f"{row['projected_warm_s']:.3f} | {row['projected_explicit_movement_s']:.6f} | "
            f"{row['projected_explicit_movement_fraction_percent']:.2f}% |"
        )
    service_rows = [
        row for row in projected
        if row.get("storage_service_equivalent_s") is not None
    ]
    if service_rows:
        lines.extend([
            "",
            "## Storage-Service Context",
            "",
            "The cold-warm delta above is critical-path overhead. The following serialized service equivalent is read bytes divided by measured active HDD throughput; it overlaps graph loading/index construction and must not be added to overall runtime.",
            "",
            "| Workload | Storage service equivalent (s) | Service / projected cold |",
            "| --- | ---: | ---: |",
        ])
        for row in service_rows:
            service_s = float(row["storage_service_equivalent_s"])
            service_fraction = 100.0 * service_s / float(row["projected_cold_s"])
            lines.append(
                f"| {row['workload']} | {service_s:.3f} | {service_fraction:.2f}% |"
            )
    lines.extend(
        [
            "",
            "## Existing OMS Runs: Storage-Only Projection",
            "",
            "| OMS baseline | Projected overall (s) | Storage service (s) | Storage / overall | GPU transfer |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in oms_projection:
        lines.append(
            f"| {row['platform']} | {row['projected_overall_runtime_s']:.3f} | "
            f"{row['modeled_storage_service_s']:.3f} | "
            f"{row['projected_storage_fraction_percent']:.3f}% | N/A |"
        )
    lines.extend(
        [
            "",
            "## Projected GPU Speedup",
            "",
            (
                f"- Proteomics clustering, Hyper-Spec versus Falcon: **{clustering_speedup:.3f}x**"
                if clustering_speedup is not None
                else "- Proteomics clustering: unchanged; see the prior complete report."
            ),
            (
                f"- OMS, ANN-SoLo versus SpectraST: **{oms_speedup:.3f}x**"
                if oms_speedup is not None
                else "- OMS: no complete paired projection is available."
            ),
            "- Genomics speedup is not reported because the CPU and GPU rows use different references.",
        ]
    )
    lines.extend(
        [
            "",
            "## Limits",
            "",
            "- Critical-path explicit movement includes cold/warm storage overhead and measured CUDA copies only.",
            "- CPU DRAM stalls and GPU HBM traffic inside kernels remain unavailable.",
            "- The GPU was not exclusively reserved during the source measurement.",
            "- A different target storage device changes both cold runtime and movement fraction.",
            "- The HGA row uses the MHC graph; minigraph uses hg38, so they are not direct runtime/accuracy peers.",
            "",
        ]
    )
    report_path = args.results_dir / "GNR6515P_A100_DATA_MOVEMENT_TIME_PROJECTION.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(report_path)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
