#!/usr/bin/env python3
"""Estimate full PXD024364 CPU OMS time from completed SpectraST files."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


DATA_ROOT = Path("/mnt/hdd/tsunghan/raw-ms-dataset/proteomic_large_dataset")
RESULT_ROOT = DATA_ROOT / "results"
REPORT_ROOT = Path(
    "/home/tsl012/multiomic/multiram_data_movement/results_pxd024364_1tb"
)


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


def main() -> int:
    cpu_rows = successful_records(RESULT_ROOT / "cpu_oms/metrics.jsonl")
    mgf_rows = successful_records(RESULT_ROOT / "raw_conversion/metrics.jsonl")
    if not cpu_rows:
        raise SystemExit("No successful CPU OMS measurements are available")

    measured_bytes = sum(int(row["mgf_bytes"]) for row in cpu_rows)
    measured_wall_s = sum(float(row["wall_s"]) for row in cpu_rows)
    total_bytes = sum(int(row["mgf_bytes"]) for row in mgf_rows)
    if measured_bytes <= 0 or measured_wall_s <= 0:
        raise SystemExit("CPU OMS measurements contain no usable bytes or time")

    throughput_bytes_s = measured_bytes / measured_wall_s
    estimated_wall_s = total_bytes / throughput_bytes_s
    report = {
        "dataset": "PXD024364 / MSV000086944 approximately 1 TB RAW subset",
        "stage": "cpu_oms",
        "baseline": "SpectraST",
        "status": "estimated_from_completed_files",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "linear extrapolation by successfully converted MGF input bytes",
        "measured_files": len(cpu_rows),
        "measured_input_bytes": measured_bytes,
        "measured_wall_s": measured_wall_s,
        "measured_throughput_bytes_s": throughput_bytes_s,
        "full_dataset_files": len(mgf_rows),
        "full_dataset_input_bytes": total_bytes,
        "estimated_full_wall_s": estimated_wall_s,
        "estimated_full_days": estimated_wall_s / 86400,
        "estimated_full_years_365d": estimated_wall_s / (365 * 86400),
        "limitations": [
            "This is an extrapolation, not a measured full-dataset runtime.",
            "Only completed CPU OMS files contribute to the throughput estimate.",
            "Runtime may not scale linearly because query composition, library-search cost, caching, and per-file startup overhead vary.",
            "The interrupted in-progress file is excluded.",
        ],
    }

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_ROOT / "cpu_oms_full_dataset_estimate.json"
    md_path = REPORT_ROOT / "CPU_OMS_FULL_DATASET_ESTIMATE.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# PXD024364 CPU OMS Full-Dataset Estimate",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| Measured completed files | {len(cpu_rows)} |",
                f"| Measured MGF input | {measured_bytes / 1e9:.6f} GB |",
                f"| Measured wall time | {measured_wall_s:.3f} s |",
                f"| Measured throughput | {throughput_bytes_s / 1e3:.3f} kB/s |",
                f"| Full converted files | {len(mgf_rows)} |",
                f"| Full MGF input | {total_bytes / 1e9:.6f} GB |",
                f"| Estimated full wall time | {estimated_wall_s:.3f} s |",
                f"| Estimated full wall time | {estimated_wall_s / 86400:.3f} days |",
                f"| Estimated full wall time | {estimated_wall_s / (365 * 86400):.3f} years |",
                "",
                "## Method",
                "",
                "The estimate linearly scales the aggregate wall time of successfully "
                "completed SpectraST files by MGF input bytes. The interrupted file is "
                "excluded.",
                "",
                "## Reporting Constraint",
                "",
                "This value is a one-sample extrapolation and must not be labeled as a "
                "measured full-dataset runtime. Query composition, cache state, library "
                "search behavior, and per-file startup overhead can make scaling nonlinear.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
