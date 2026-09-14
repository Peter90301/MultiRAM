#!/usr/bin/env python3
"""Aggregate completed PXD024364 per-file metrics into JSON and Markdown."""

from __future__ import annotations

import csv
import json
from pathlib import Path


DATA_ROOT = Path("/mnt/hdd/tsunghan/raw-ms-dataset/proteomic_large_dataset")
RESULT_ROOT = DATA_ROOT / "results"
REPORT_ROOT = Path(
    "/home/tsl012/multiomic/multiram_data_movement/results_pxd024364_1tb"
)
SELECTION = DATA_ROOT / "manifests/selection_summary.json"
STAGES = ("raw_conversion", "cpu_clustering", "gpu_clustering", "cpu_oms", "gpu_oms")


def records(path: Path) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    if not path.exists():
        return output
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("exit_status") == 0:
            output.append(record)
    return output


def gpu_power(stage_dir: Path) -> dict[str, float | int | None]:
    path = stage_dir / "gpu_samples.csv"
    if not path.exists():
        return {"samples": 0, "average_power_w": None, "peak_memory_mib": None}
    powers: list[float] = []
    memories: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 4:
                continue
            try:
                powers.append(float(row[2].strip()))
                memories.append(float(row[3].strip()))
            except ValueError:
                continue
    return {
        "samples": len(powers),
        "average_power_w": sum(powers) / len(powers) if powers else None,
        "peak_memory_mib": max(memories) if memories else None,
    }


def main() -> int:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    total_files = int(selection["selected_files"])
    summary: dict[str, object] = {"selection": selection, "stages": {}}
    stage_summary: dict[str, dict[str, object]] = {}
    for stage in STAGES:
        rows = records(RESULT_ROOT / stage / "metrics.jsonl")
        wall_s = sum(float(row.get("wall_s", 0)) for row in rows)
        input_bytes = sum(
            int(row.get("raw_bytes", row.get("mgf_bytes", 0))) for row in rows
        )
        item: dict[str, object] = {
            "completed_files": len(rows),
            "total_files": total_files,
            "completed_percent": 100.0 * len(rows) / total_files,
            "wall_s_sum": wall_s,
            "user_s_sum": sum(float(row.get("user_s", 0)) for row in rows),
            "system_s_sum": sum(float(row.get("system_s", 0)) for row in rows),
            "input_bytes_sum": input_bytes,
            "output_bytes_sum": sum(int(row.get("output_bytes", row.get("mgf_bytes", 0))) for row in rows),
            "max_peak_rss_kb": max((int(row.get("max_rss_kb", 0)) for row in rows), default=0),
            "throughput_input_MB_s": input_bytes / wall_s / 1e6 if wall_s else None,
        }
        if stage == "raw_conversion":
            item["spectra_sum"] = sum(int(row.get("spectra", 0)) for row in rows)
        if stage.startswith("gpu_"):
            item.update(gpu_power(RESULT_ROOT / stage))
            if item.get("average_power_w") is not None:
                item["sampled_energy_j_estimate"] = wall_s * float(item["average_power_w"])
        stage_summary[stage] = item
    summary["stages"] = stage_summary

    conversion_s = float(stage_summary["raw_conversion"]["wall_s_sum"])
    summary["sequential_pipeline_wall_s"] = {
        "cpu": conversion_s
        + float(stage_summary["cpu_clustering"]["wall_s_sum"])
        + float(stage_summary["cpu_oms"]["wall_s_sum"]),
        "gpu": conversion_s
        + float(stage_summary["gpu_clustering"]["wall_s_sum"])
        + float(stage_summary["gpu_oms"]["wall_s_sum"]),
    }
    (REPORT_ROOT / "aggregate_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# PXD024364 1 TB Aggregate Results",
        "",
        f"Selected input: {selection['selected_files']} RAW files, "
        f"{selection['selected_bytes']:,} bytes.",
        "",
        "| Stage | Completed | Wall time (s) | Input bytes | Throughput (MB/s) | Peak RSS (KiB) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stage in STAGES:
        item = stage_summary[stage]
        throughput = item["throughput_input_MB_s"]
        throughput_text = f"{float(throughput):.3f}" if throughput is not None else "N/A"
        lines.append(
            f"| {stage} | {item['completed_files']}/{total_files} | "
            f"{float(item['wall_s_sum']):.3f} | {int(item['input_bytes_sum']):,} | "
            f"{throughput_text} | {int(item['max_peak_rss_kb']):,} |"
        )
    lines.extend(
        [
            "",
            "CPU/GPU values are measured per-file batched sums. MultiRAM modeled "
            "time and movement reduction are intentionally reported separately after "
            "the full spectrum and transfer counts are available.",
            "",
        ]
    )
    (REPORT_ROOT / "AGGREGATE_RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
