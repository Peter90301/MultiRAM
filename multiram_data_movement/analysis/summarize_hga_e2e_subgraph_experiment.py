#!/usr/bin/env python3
"""Summarize the matched end-to-end minigraph and HGA mapper experiment."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


ROOT = Path("/home/tsl012/multiomic/multiram_data_movement")
DEFAULT_RESULTS = ROOT / "results_hga_e2e_subgraph_comparison"


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def median_runtime(
    records: list[dict], workload: str, mode: str
) -> tuple[float, list[float]]:
    values = [
        float(record["wall_s"])
        for record in records
        if record["workload"] == workload and record["mode"] == mode
    ]
    return statistics.median(values), values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    runtime_records = load_jsonl(args.results_dir / "runtime_records.jsonl")
    gpu_transfer = load_jsonl(args.results_dir / "gpu_transfer_records.jsonl")[0]
    quality = json.loads(
        (args.results_dir / "quality/mapping_quality.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        (args.results_dir / "inputs/manifest.json").read_text(encoding="utf-8")
    )

    cpu_cold, cpu_cold_trials = median_runtime(
        runtime_records, "cpu_minigraph_e2e", "cold"
    )
    cpu_warm, cpu_warm_trials = median_runtime(
        runtime_records, "cpu_minigraph_e2e", "warm"
    )
    gpu_cold, gpu_cold_trials = median_runtime(
        runtime_records, "gpu_candidate_hga_e2e", "cold"
    )
    gpu_warm, gpu_warm_trials = median_runtime(
        runtime_records, "gpu_candidate_hga_e2e", "warm"
    )
    cpu_storage_s = max(0.0, cpu_cold - cpu_warm)
    gpu_storage_s = max(0.0, gpu_cold - gpu_warm)
    cuda_copy_s = float(gpu_transfer["copy_union_time_s"])
    gpu_movement_s = gpu_storage_s + cuda_copy_s

    warm_hga_timings = []
    for trial in range(1, 4):
        path = (
            args.results_dir
            / f"raw/gpu_candidate_hga_e2e/warm/trial_{trial}/"
            "hga_mapper_timing.json"
        )
        warm_hga_timings.append(json.loads(path.read_text(encoding="utf-8")))
    phase_names = (
        "candidate_routing",
        "materialize_groups",
        "hga_process_wall_sum",
        "hga_kernel_sum",
        "merge_output",
        "end_to_end",
    )
    median_phases = {
        name: statistics.median(
            float(item["phases_s"][name]) for item in warm_hga_timings
        )
        for name in phase_names
    }

    cpu_record = next(
        record
        for record in runtime_records
        if record["workload"] == "cpu_minigraph_e2e"
        and record["mode"] == "cold"
    )
    gpu_record = next(
        record
        for record in runtime_records
        if record["workload"] == "gpu_candidate_hga_e2e"
        and record["mode"] == "cold"
    )
    output = {
        "status": "completed_measured_experiment",
        "workload": {
            "subgraphs": len(manifest["subgraphs"]),
            "bases_per_subgraph": manifest["subgraphs"][0]["length"],
            "search_space_bases": sum(
                int(item["length"]) for item in manifest["subgraphs"]
            ),
            "reads": manifest["read_count"],
            "read_length": manifest["read_length"],
            "simulated_error_rate_nominal_percent": 5.0,
        },
        "cpu_minigraph": {
            "cold_trial_s": cpu_cold_trials,
            "warm_trial_s": cpu_warm_trials,
            "cold_median_s": cpu_cold,
            "warm_median_s": cpu_warm,
            "storage_critical_delta_s": cpu_storage_s,
            "movement_fraction_cold_percent": (
                100.0 * cpu_storage_s / cpu_cold
            ),
            "cold_filesystem_read_bytes": int(
                cpu_record["fs_input_blocks"]
            )
            * 512,
            "quality": quality["minigraph"],
        },
        "gpu_candidate_hga": {
            "cold_trial_s": gpu_cold_trials,
            "warm_trial_s": gpu_warm_trials,
            "cold_median_s": gpu_cold,
            "warm_median_s": gpu_warm,
            "storage_critical_delta_s": gpu_storage_s,
            "cuda_copy_union_s": cuda_copy_s,
            "explicit_movement_s": gpu_movement_s,
            "movement_fraction_cold_percent": (
                100.0 * gpu_movement_s / gpu_cold
            ),
            "cold_filesystem_read_bytes": int(
                gpu_record["fs_input_blocks"]
            )
            * 512,
            "h2d_bytes": gpu_transfer["copy_by_direction"][
                "Host-to-Device"
            ]["bytes"],
            "d2h_bytes": gpu_transfer["copy_by_direction"][
                "Device-to-Host"
            ]["bytes"],
            "warm_median_phases_s": median_phases,
            "quality": quality["candidate_hga"],
        },
        "comparisons": {
            "minigraph_faster_cold_x": gpu_cold / cpu_cold,
            "minigraph_faster_warm_x": gpu_warm / cpu_warm,
            "candidate_hga_endpoint_recall_minus_minigraph_percentage_points": (
                float(
                    quality["candidate_hga"][
                        "correct_endpoint_rate_percent"
                    ]
                )
                - float(quality["minigraph"]["correct_endpoint_rate_percent"])
            ),
        },
        "validity_notes": [
            "Both pipelines use the same four subgraph sequences and the same truth-labeled reads.",
            "GPU end-to-end time includes minimap2 candidate routing, HGA input materialization, four HGA processes, DP kernels, D2H endpoints, and PAF merge.",
            "HGA reports score and endpoint but not traceback/CIGAR, so the common quality task is localization rather than full path equivalence.",
            "Cold-warm deltas on this small workload are noise-sensitive and represent explicit storage critical-path estimates only.",
            "The HGA optimized kernel uses 8-bit DP state; the experiment therefore uses 100 bp reads.",
        ],
        "invalidated_prior_hga_results": {
            "status": "invalid",
            "reason": (
                "The old binary contained only sm_75 machine code on a "
                "compute-capability 12.0 GPU and did not check kernel launch "
                "errors. It returned zero scores while reporting near-zero time."
            ),
        },
    }
    json_path = args.results_dir / "HGA_E2E_SUBGRAPH_RESULT.json"
    json_path.write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )

    c = output["cpu_minigraph"]
    g = output["gpu_candidate_hga"]
    comp = output["comparisons"]
    p = g["warm_median_phases_s"]
    lines = [
        "# End-to-End Minigraph vs Candidate-Filtered HGA",
        "",
        "## Matched Workload",
        "",
        f"- Graph search space: {output['workload']['subgraphs']} MHC subgraphs x "
        f"{output['workload']['bases_per_subgraph']:,} bases = "
        f"{output['workload']['search_space_bases']:,} bases.",
        f"- Reads: {output['workload']['reads']:,} single-end, "
        f"{output['workload']['read_length']} bp.",
        "- Truth: Mason SAM; nominal sequencing error is 5%.",
        "- Common correctness metric: correct subgraph and target endpoint within 20 bp.",
        "",
        "## End-to-End Result",
        "",
        "| Pipeline | Cold median | Warm median | Explicit movement | Movement / cold | Mapping rate | Correct endpoint / all reads | Correct endpoint / mapped |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| CPU minigraph | **{c['cold_median_s']:.6f} s** | {c['warm_median_s']:.6f} s | {c['storage_critical_delta_s']:.6f} s | {c['movement_fraction_cold_percent']:.2f}% | {c['quality']['mapping_rate_percent']:.2f}% | **{c['quality']['correct_endpoint_rate_percent']:.2f}%** | {c['quality']['correct_endpoint_precision_percent']:.2f}% |",
        f"| GPU minimap2 routing + HGA | **{g['cold_median_s']:.6f} s** | {g['warm_median_s']:.6f} s | {g['explicit_movement_s']:.6f} s | {g['movement_fraction_cold_percent']:.2f}% | {g['quality']['mapping_rate_percent']:.2f}% | **{g['quality']['correct_endpoint_rate_percent']:.2f}%** | {g['quality']['correct_endpoint_precision_percent']:.2f}% |",
        "",
        f"Minigraph is **{comp['minigraph_faster_cold_x']:.2f}x faster cold** "
        f"and **{comp['minigraph_faster_warm_x']:.2f}x faster warm** on this small workload. "
        "The candidate-HGA pipeline has higher endpoint recall by "
        f"**{comp['candidate_hga_endpoint_recall_minus_minigraph_percentage_points']:.2f} percentage points**.",
        "",
        "## GPU Phase Breakdown",
        "",
        "| Phase | Warm median |",
        "| --- | ---: |",
        f"| minimap2 candidate routing | {p['candidate_routing']:.6f} s |",
        f"| Group/materialize HGA reads | {p['materialize_groups']:.6f} s |",
        f"| Four HGA process wall-time sum | {p['hga_process_wall_sum']:.6f} s |",
        f"| HGA kernels inside those processes | {p['hga_kernel_sum']:.6f} s |",
        f"| Merge endpoint PAF | {p['merge_output']:.6f} s |",
        f"| End-to-end pipeline | {p['end_to_end']:.6f} s |",
        "",
        "Nsight measured:",
        "",
        f"- H2D: {g['h2d_bytes']:,} bytes.",
        f"- D2H: {g['d2h_bytes']:,} bytes.",
        f"- CUDA copy union: {g['cuda_copy_union_s'] * 1e6:.3f} us.",
        "",
        "The GPU pipeline is dominated by HGA DP and per-shard process/setup overhead, "
        "not PCIe copies.",
        "",
        "## HGA Correctness Fixes",
        "",
        "The public HGA code required the following fixes before this experiment:",
        "",
        "- Added PTX fallback because the installed binary targeted only sm_75 while this GPU is compute capability 12.0.",
        "- Added CUDA launch-error checking; the old runs silently launched no usable kernel.",
        "- Put kernels on the same stream as H2D copies.",
        "- Corrected packed-read indexing.",
        "- Restored score and row/column endpoint D2H output.",
        "",
        "**All earlier HGA runtimes produced by the old binary are invalid.** "
        "They returned zero scores and must not be used in the paper.",
        "",
        "## Remaining Scope Difference",
        "",
        "This is now a fair end-to-end **localization** comparison: same search space, "
        "reads, truth, timing boundary, and endpoint metric. HGA still lacks traceback "
        "and CIGAR/path output, so it is not yet a fully output-equivalent replacement "
        "for minigraph.",
        "",
    ]
    report_path = args.results_dir / "HGA_E2E_SUBGRAPH_RESULT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(report_path)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
