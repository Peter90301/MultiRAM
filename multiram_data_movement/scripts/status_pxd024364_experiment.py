#!/usr/bin/env python3
"""Report download and per-stage progress for the PXD024364 1 TB experiment."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path


ROOT = Path("/mnt/hdd/tsunghan/raw-ms-dataset/proteomic_large_dataset")
MANIFEST = ROOT / "manifests/selected_raw_manifest.tsv"
RESULT_ROOT = ROOT / "results"
UNIT = "multiram-pxd024364-experiment.service"
STAGES = ("raw_conversion", "cpu_clustering", "gpu_clustering", "cpu_oms", "gpu_oms")


def unit_property(name: str) -> str:
    result = subprocess.run(
        ["systemctl", "--user", "show", UNIT, f"--property={name}", "--value"],
        text=True,
        capture_output=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def records_by_path(path: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    if not path.exists():
        return {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        records[str(record.get("relative_path", ""))] = record
    return records


def main() -> int:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    total_files = len(rows)
    result: dict[str, object] = {
        "unit": UNIT,
        "active_state": unit_property("ActiveState") or "not_started",
        "result": unit_property("Result") or "not_started",
        "main_pid": int(unit_property("MainPID") or 0),
        "total_files": total_files,
    }
    stages: dict[str, object] = {}
    for stage in STAGES:
        latest = records_by_path(RESULT_ROOT / stage / "metrics.jsonl")
        records = [record for record in latest.values() if record.get("exit_status") == 0]
        failed = [record for record in latest.values() if record.get("exit_status") not in (None, 0)]
        stages[stage] = {
            "completed_files": len(records),
            "failed_files": len(failed),
            "completed_percent": 100.0 * len(records) / total_files,
            "wall_s_sum": sum(float(record.get("wall_s", 0)) for record in records),
            "input_bytes_sum": sum(
                int(record.get("raw_bytes", record.get("mgf_bytes", 0)))
                for record in records
            ),
        }
    result["stages"] = stages
    log_path = RESULT_ROOT / "orchestrator/experiment.log"
    if log_path.exists():
        result["last_log_lines"] = log_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()[-5:]
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
