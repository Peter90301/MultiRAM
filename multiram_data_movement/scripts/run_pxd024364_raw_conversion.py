#!/usr/bin/env python3
"""Convert the selected PXD024364 RAW files to MS2 MGF with resumable metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/mnt/hdd/tsunghan/raw-ms-dataset/proteomic_large_dataset")
MANIFEST = ROOT / "manifests/selected_raw_manifest.tsv"
RAW_DIR = ROOT / "raw"
MGF_DIR = ROOT / "derived/ms2_mgf"
RESULT_DIR = ROOT / "results/raw_conversion"
PARSER = Path(
    "/home/tsl012/multiomic/external/ThermoRawFileParser-v2/ThermoRawFileParser"
)


TIME_PATTERNS = {
    "user_s": r"User time \(seconds\):\s*(\S+)",
    "system_s": r"System time \(seconds\):\s*(\S+)",
    "cpu_percent": r"Percent of CPU this job got:\s*(\S+)%",
    "max_rss_kb": r"Maximum resident set size \(kbytes\):\s*(\d+)",
    "major_page_faults": r"Major \(requiring I/O\) page faults:\s*(\d+)",
    "minor_page_faults": r"Minor \(reclaiming a frame\) page faults:\s*(\d+)",
    "fs_input_blocks": r"File system inputs:\s*(\d+)",
    "fs_output_blocks": r"File system outputs:\s*(\d+)",
    "exit_status": r"Exit status:\s*(\d+)",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_rows() -> list[dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def prior_outcomes(result_path: Path) -> tuple[set[str], dict[str, int]]:
    completed: set[str] = set()
    failures: dict[str, int] = {}
    if not result_path.exists():
        return completed, failures
    for line in result_path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("exit_status") == 0 and row.get("mgf_bytes", 0) > 0:
            completed.add(str(row["relative_path"]))
        elif row.get("exit_status") not in (None, 0):
            relative = str(row["relative_path"])
            failures[relative] = failures.get(relative, 0) + 1
    return completed, failures


def parse_time_log(path: Path) -> dict[str, float | int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    values: dict[str, float | int] = {}
    integer_fields = {
        "max_rss_kb",
        "major_page_faults",
        "minor_page_faults",
        "fs_input_blocks",
        "fs_output_blocks",
        "exit_status",
    }
    for name, pattern in TIME_PATTERNS.items():
        match = re.search(pattern, text)
        if match:
            values[name] = int(match.group(1)) if name in integer_fields else float(match.group(1))
    return values


def count_spectra(path: Path) -> int:
    result = subprocess.run(
        ["rg", "-c", "^BEGIN IONS$", str(path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"Failed to count spectra in {path}: {result.stderr}")
    return int(result.stdout.strip() or 0)


def start_monitor(command: list[str], output: Path) -> subprocess.Popen[str]:
    handle = output.open("a", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    process._multiram_output_handle = handle  # type: ignore[attr-defined]
    return process


def stop_monitor(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    process._multiram_output_handle.close()  # type: ignore[attr-defined]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="Process at most N pending files.")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=1,
        help="Maximum conversion attempts per RAW file before recording and skipping it.",
    )
    args = parser.parse_args()

    MGF_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = RESULT_DIR / "per_file_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    result_path = RESULT_DIR / "metrics.jsonl"
    done, failure_counts = prior_outcomes(result_path)
    rows = read_rows()
    pending = [
        row
        for row in rows
        if row["relative_path"] not in done
        and failure_counts.get(row["relative_path"], 0) < args.max_attempts
    ]
    if args.limit > 0:
        pending = pending[: args.limit]

    iostat = start_monitor(["iostat", "-dx", "1"], RESULT_DIR / "iostat.log")
    sar = start_monitor(["sar", "-u", "1"], RESULT_DIR / "sar_cpu.log")
    stage_start = time.perf_counter()
    try:
        for row in pending:
            relative = row["relative_path"]
            raw_path = RAW_DIR / relative
            expected_bytes = int(row["size_bytes"])
            if not raw_path.exists() or raw_path.stat().st_size != expected_bytes:
                raise RuntimeError(f"Missing or wrong-size input: {raw_path}")

            output_dir = MGF_DIR / Path(relative).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            mgf_path = output_dir / (Path(relative).stem + ".mgf")
            token = row["path_sha256"][:16]
            stdout_path = log_dir / f"{token}.stdout.log"
            stderr_path = log_dir / f"{token}.stderr.log"
            time_path = log_dir / f"{token}.time_v.log"
            command = [
                "/usr/bin/time",
                "-v",
                "-o",
                str(time_path),
                str(PARSER),
                "-i",
                str(raw_path),
                "-o",
                str(output_dir),
                "-f=0",
                "-m=0",
                "-L=2",
            ]
            started_at = utc_now()
            start = time.perf_counter()
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr:
                process = subprocess.run(command, stdout=stdout, stderr=stderr, text=True)
            wall_s = time.perf_counter() - start
            metrics: dict[str, object] = {
                "sequence": int(row["selection_rank"]),
                "relative_path": relative,
                "path_sha256": row["path_sha256"],
                "raw_bytes": expected_bytes,
                "mgf_path": str(mgf_path),
                "mgf_bytes": mgf_path.stat().st_size if mgf_path.exists() else 0,
                "spectra": count_spectra(mgf_path) if process.returncode == 0 and mgf_path.exists() else 0,
                "started_at": started_at,
                "finished_at": utc_now(),
                "wall_s": wall_s,
                "exit_status": process.returncode,
                "command": command,
            }
            if time_path.exists():
                metrics.update(parse_time_log(time_path))
            with result_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            print(json.dumps(metrics), flush=True)
            # Some public Thermo RAW files are internally truncated despite
            # matching the official remote size. Preserve the failure and let
            # the remaining independent files continue.
    finally:
        stop_monitor(iostat)
        stop_monitor(sar)
        (RESULT_DIR / "last_stage_runtime_s.txt").write_text(
            f"{time.perf_counter() - stage_start:.6f}\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
