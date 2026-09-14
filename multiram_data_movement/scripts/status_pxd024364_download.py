#!/usr/bin/env python3
"""Report verified-by-size progress for the selected PXD024364 subset."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path


ROOT = Path("/mnt/hdd/tsunghan/raw-ms-dataset/proteomic_large_dataset")
MANIFEST = ROOT / "manifests/selected_raw_manifest.tsv"
RAW_DIR = ROOT / "raw"
UNIT = "multiram-pxd024364-download.service"


def unit_property(name: str) -> str:
    result = subprocess.run(
        ["systemctl", "--user", "show", UNIT, f"--property={name}", "--value"],
        text=True,
        capture_output=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def main() -> None:
    result: dict[str, object] = {
        "dataset": "PXD024364 / MSV000086944",
        "unit": UNIT,
        "active_state": unit_property("ActiveState") or "not_started",
        "main_pid": int(unit_property("MainPID") or 0),
    }
    if not MANIFEST.exists():
        result["status"] = "manifest_not_created"
        print(json.dumps(result, indent=2))
        return

    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    expected_bytes = sum(int(row["size_bytes"]) for row in rows)
    completed_bytes = 0
    completed_files = 0
    partial_bytes = 0
    mismatched_files = 0
    for row in rows:
        expected_size = int(row["size_bytes"])
        path = RAW_DIR / row["relative_path"]
        if path.exists():
            actual_size = path.stat().st_size
            if actual_size == expected_size:
                completed_files += 1
                completed_bytes += actual_size
            else:
                mismatched_files += 1
        # rclone may insert a random token before the configured partial suffix.
        partial_candidates = list(path.parent.glob(path.name + ".*.partial"))
        partial_candidates.extend(path.parent.glob(path.name + ".partial"))
        partial_bytes += sum(
            candidate.stat().st_size
            for candidate in set(partial_candidates)
            if candidate.is_file()
        )

    result.update(
        {
            "selected_files": len(rows),
            "selected_bytes": expected_bytes,
            "selected_TB_decimal": expected_bytes / 1e12,
            "completed_files": completed_files,
            "completed_bytes_verified_by_size": completed_bytes,
            "partial_bytes": partial_bytes,
            "mismatched_files": mismatched_files,
            "completed_percent": 100.0 * completed_bytes / expected_bytes,
            "on_disk_percent_including_partials": 100.0
            * (completed_bytes + partial_bytes)
            / expected_bytes,
        }
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
