#!/usr/bin/env python3
"""Run reproducible cold/warm CPU and GPU data-movement measurements."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/tsl012/multiomic/multiram_data_movement")
DEFAULT_RESULTS = ROOT / "results_data_movement_time_minigraph"
DEFAULT_MANIFEST = Path(
    "/mnt/hdd/tsunghan/raw-ms-dataset/genomic_large_Dset/"
    "profiling_subsets/na12878_100k_pairs/manifest.json"
)
MINIGRAPH = Path(
    "/home/tsl012/multiomic/genomic_proteomic_CPU_test/minigraph_source/minigraph"
)
HG38_FASTA = Path("/mnt/hdd/tsunghan/raw-ms-dataset/references/hg38/hg38.fa.gz")
HGA = Path("/home/tsl012/multiomic/external/hga/bin/hga")
FALCON = Path("/home/tsl012/miniconda3/envs/falconrun/bin/falcon")
HYPERSPEC_PYTHON = Path("/home/tsl012/miniconda3/envs/multiomic/bin/python")
HYPERSPEC_RUNNER = Path(
    "/home/tsl012/multiomic/proteomic_full_pipeline/run_hyperspec_gpu_cluster.py"
)
MGF = Path(
    "/mnt/hdd/tsunghan/raw-ms-dataset/proteomic_large_dataset/derived/ms2_mgf/"
    "20151220_alr_CompleteHumanProteome_HUVEC_LysC_ETD_fr9.mgf"
)
NSYS = Path(
    "/home/tsl012/.local/opt/nsight-systems-2025.6.3/opt/nvidia/"
    "nsight-systems/2025.6.3/target-linux-x64/nsys"
)


@dataclass
class Workload:
    name: str
    domain: str
    platform: str
    input_paths: list[Path]
    command_factory: object
    discard_stdout: bool = False


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_time_v(path: Path) -> dict[str, float | int | None]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    patterns: dict[str, tuple[str, object]] = {
        "user_s": (r"User time \(seconds\):\s*([0-9.]+)", float),
        "system_s": (r"System time \(seconds\):\s*([0-9.]+)", float),
        "cpu_percent": (r"Percent of CPU this job got:\s*([0-9.]+)%", float),
        "max_rss_kb": (r"Maximum resident set size \(kbytes\):\s*(\d+)", int),
        "major_page_faults": (r"Major \(requiring I/O\) page faults:\s*(\d+)", int),
        "minor_page_faults": (r"Minor \(reclaiming a frame\) page faults:\s*(\d+)", int),
        "fs_input_blocks": (r"File system inputs:\s*(\d+)", int),
        "fs_output_blocks": (r"File system outputs:\s*(\d+)", int),
    }
    result: dict[str, float | int | None] = {}
    for key, (pattern, converter) in patterns.items():
        match = re.search(pattern, text)
        result[key] = converter(match.group(1)) if match else None
    return result


def evict_files(paths: list[Path]) -> dict[str, object]:
    advised: list[str] = []
    failed: list[dict[str, str]] = []
    for path in paths:
        try:
            with path.open("rb") as handle:
                os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            advised.append(str(path))
        except OSError as exc:
            failed.append({"path": str(path), "error": str(exc)})
    return {"method": "POSIX_FADV_DONTNEED", "advised": advised, "failed": failed}


def start_iostat(path: Path) -> subprocess.Popen[str] | None:
    if not shutil_which("iostat"):
        return None
    handle = path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        ["iostat", "-dx", "-y", "1"], stdout=handle, stderr=subprocess.STDOUT, text=True
    )
    process._movement_log_handle = handle  # type: ignore[attr-defined]
    return process


def stop_iostat(process: subprocess.Popen[str] | None) -> None:
    if process is None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    handle = getattr(process, "_movement_log_handle", None)
    if handle is not None:
        handle.close()


def shutil_which(command: str) -> str | None:
    result = subprocess.run(
        ["bash", "-lc", f"command -v {shlex.quote(command)}"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() or None


def run_once(
    workload: Workload,
    mode: str,
    trial: int,
    output_root: Path,
    env: dict[str, str],
) -> dict[str, object]:
    run_dir = output_root / "raw" / workload.name / mode / f"trial_{trial}"
    run_dir.mkdir(parents=True, exist_ok=True)
    command = workload.command_factory(run_dir)
    cache = evict_files(workload.input_paths) if mode == "cold" else {
        "method": "not_requested", "advised": [], "failed": []
    }
    time_log = run_dir / "time_v.log"
    stderr_log = run_dir / "stderr.log"
    stdout_log = run_dir / "stdout.log"
    iostat_process = start_iostat(run_dir / "iostat.log")
    started_at = now()
    start = time.perf_counter()
    with stderr_log.open("w", encoding="utf-8") as stderr, (
        open(os.devnull, "w", encoding="utf-8")
        if workload.discard_stdout
        else stdout_log.open("w", encoding="utf-8")
    ) as stdout:
        process = subprocess.run(
            ["/usr/bin/time", "-v", "-o", str(time_log), *command],
            stdout=stdout,
            stderr=stderr,
            text=True,
            env=env,
            check=False,
        )
    wall_s = time.perf_counter() - start
    stop_iostat(iostat_process)
    record: dict[str, object] = {
        "record_type": "runtime",
        "workload": workload.name,
        "domain": workload.domain,
        "platform": workload.platform,
        "mode": mode,
        "trial": trial,
        "started_at": started_at,
        "finished_at": now(),
        "wall_s": wall_s,
        "return_code": process.returncode,
        "command": command,
        "input_paths": [str(path) for path in workload.input_paths],
        "input_bytes": sum(path.stat().st_size for path in workload.input_paths),
        "cache_eviction": cache,
        "run_dir": str(run_dir),
        **parse_time_v(time_log),
    }
    with (output_root / "runtime_records.jsonl").open("a", encoding="utf-8") as out:
        out.write(json.dumps(record) + "\n")
    if process.returncode != 0:
        raise RuntimeError(f"{workload.name} {mode} trial {trial} failed: {stderr_log}")
    print(f"{workload.name}: {mode} trial {trial}: {wall_s:.3f} s", flush=True)
    return record


def union_duration_ns(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    merged = 0
    start, end = sorted(intervals)[0]
    for next_start, next_end in sorted(intervals)[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            merged += end - start
            start, end = next_start, next_end
    return merged + end - start


def profile_gpu(
    workload: Workload,
    output_root: Path,
    env: dict[str, str],
) -> dict[str, object]:
    profile_dir = output_root / "raw" / workload.name / "nsys"
    profile_dir.mkdir(parents=True, exist_ok=True)
    command = workload.command_factory(profile_dir / "output")
    report_base = profile_dir / "profile"
    profile_command = [
        str(NSYS), "profile", "--force-overwrite=true", "--sample=none",
        "--cpuctxsw=none", "--resolve-symbols=false", "--trace=cuda",
        f"--output={report_base}", *command,
    ]
    with (profile_dir / "stdout.log").open("w", encoding="utf-8") as stdout, \
            (profile_dir / "stderr.log").open("w", encoding="utf-8") as stderr:
        start = time.perf_counter()
        process = subprocess.run(
            profile_command, stdout=stdout, stderr=stderr, text=True, env=env, check=False
        )
        profiled_wall_s = time.perf_counter() - start
    report = report_base.with_suffix(".nsys-rep")
    sqlite_path = report_base.with_suffix(".sqlite")
    if process.returncode != 0 or not report.exists():
        raise RuntimeError(f"Nsight failed for {workload.name}: {profile_dir / 'stderr.log'}")
    export = subprocess.run(
        [str(NSYS), "export", "--force-overwrite=true", "--type=sqlite",
         f"--output={sqlite_path}", str(report)],
        text=True, capture_output=True, check=False,
    )
    (profile_dir / "export.log").write_text(
        export.stdout + export.stderr, encoding="utf-8"
    )
    if export.returncode != 0 or not sqlite_path.exists():
        raise RuntimeError(f"Nsight export failed for {workload.name}")

    connection = sqlite3.connect(sqlite_path)
    enum_rows = connection.execute(
        "SELECT id, label FROM ENUM_CUDA_MEMCPY_OPER"
    ).fetchall()
    labels = {int(row[0]): str(row[1]) for row in enum_rows}
    copy_rows = connection.execute(
        "SELECT start, end, bytes, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY"
    ).fetchall()
    connection.close()
    by_direction: dict[str, dict[str, int]] = {}
    intervals: list[tuple[int, int]] = []
    for start_ns, end_ns, byte_count, copy_kind in copy_rows:
        label = labels.get(int(copy_kind), f"kind_{copy_kind}")
        entry = by_direction.setdefault(label, {"bytes": 0, "count": 0, "sum_time_ns": 0})
        entry["bytes"] += int(byte_count)
        entry["count"] += 1
        entry["sum_time_ns"] += int(end_ns) - int(start_ns)
        intervals.append((int(start_ns), int(end_ns)))
    record = {
        "record_type": "gpu_transfer",
        "workload": workload.name,
        "domain": workload.domain,
        "platform": workload.platform,
        "profiled_wall_s": profiled_wall_s,
        "copy_operations": len(copy_rows),
        "copy_by_direction": by_direction,
        "copy_sum_time_s": sum(end - start for start, end in intervals) / 1e9,
        "copy_union_time_s": union_duration_ns(intervals) / 1e9,
        "report": str(report),
        "sqlite": str(sqlite_path),
        "command": command,
    }
    with (output_root / "gpu_transfer_records.jsonl").open("a", encoding="utf-8") as out:
        out.write(json.dumps(record) + "\n")
    print(
        f"{workload.name}: CUDA copies={len(copy_rows)}, "
        f"union={record['copy_union_time_s']:.6f} s",
        flush=True,
    )
    return record


def machine_metadata(gpu_index: int) -> dict[str, object]:
    cpu = subprocess.run(
        ["lscpu"], text=True, capture_output=True, check=False
    ).stdout
    cpu_model = next(
        (line.split(":", 1)[1].strip() for line in cpu.splitlines() if line.startswith("Model name:")),
        "unknown",
    )
    gpu = subprocess.run(
        ["nvidia-smi", f"--id={gpu_index}",
         "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=False,
    ).stdout.strip()
    minigraph_version = subprocess.run(
        [str(MINIGRAPH), "--version"], text=True, capture_output=True, check=False
    ).stdout.strip()
    minigraph_source = MINIGRAPH.parent
    minigraph_commit = subprocess.run(
        ["git", "-C", str(minigraph_source), "rev-parse", "HEAD"],
        text=True, capture_output=True, check=False,
    ).stdout.strip()
    minigraph_commit_subject = subprocess.run(
        ["git", "-C", str(minigraph_source), "log", "-1", "--format=%cs %s"],
        text=True, capture_output=True, check=False,
    ).stdout.strip()
    return {
        "created_at": now(),
        "hostname": platform.node(),
        "kernel": platform.release(),
        "cpu_model": cpu_model,
        "gpu_index": gpu_index,
        "gpu": gpu,
        "minigraph_binary_version": minigraph_version,
        "minigraph_source_commit": minigraph_commit,
        "minigraph_source_commit_subject": minigraph_commit_subject,
        "perf_event_paranoid": Path("/proc/sys/kernel/perf_event_paranoid").read_text().strip(),
        "nsys": subprocess.run(
            [str(NSYS), "--version"], text=True, capture_output=True, check=False
        ).stdout.strip(),
        "method": "cold/warm POSIX_FADV_DONTNEED plus CUDA activity timeline",
    }


def build_workloads(manifest: dict[str, object]) -> list[Workload]:
    r1 = Path(str(manifest["r1_fastq"]))
    r2 = Path(str(manifest["r2_fastq"]))
    hga_graph = Path(str(manifest["hga_graph"]))
    hga_reads = Path(str(manifest["hga_reads"]))

    def minigraph_command(_: Path) -> list[str]:
        return [
            str(MINIGRAPH), "-x", "sr", "-c", "-t", "16",
            str(HG38_FASTA), str(r1), str(r2), "-o", "/dev/null",
        ]

    def hga_command(_: Path) -> list[str]:
        return [str(HGA), "-g", str(hga_graph), "-r", str(hga_reads),
                "-m", "2", "-n", "2", "-o", "3", "-d", "1",
                "-b", "32", "-t", "128", "-c", "1"]

    def falcon_command(run_dir: Path) -> list[str]:
        run_dir.mkdir(parents=True, exist_ok=True)
        prefix = run_dir / "falcon_clusters.csv"
        return [str(FALCON), "--precursor_tol", "20", "ppm",
                "--fragment_tol", "0.05", "--eps", "0.10",
                "--n_neighbors", "16", "--n_neighbors_ann", "32",
                "--n_probe", "8", "--batch_size", "32768",
                str(MGF), str(prefix)]

    def hyperspec_command(run_dir: Path) -> list[str]:
        run_dir.mkdir(parents=True, exist_ok=True)
        return [str(HYPERSPEC_PYTHON), str(HYPERSPEC_RUNNER),
                "--input", str(MGF), "--output-prefix", str(run_dir / "clusters"),
                "--cpu-core-preprocess", "8", "--cpu-core-cluster", "8",
                "--batch-size", "5000"]

    return [
        Workload("genomics_cpu_minigraph", "genomics", "cpu",
                 [HG38_FASTA, r1, r2], minigraph_command, True),
        Workload("genomics_gpu_hga", "genomics", "gpu",
                 [hga_graph, hga_reads], hga_command),
        Workload("proteomics_cpu_falcon", "proteomics_clustering", "cpu",
                 [MGF], falcon_command),
        Workload("proteomics_gpu_hyperspec", "proteomics_clustering", "gpu",
                 [MGF], hyperspec_command),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--gpu-index", type=int, default=1)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--skip-nsys", action="store_true")
    args = parser.parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(
            f"Missing {args.manifest}; run prepare_data_movement_time_inputs.py first"
        )
    if not NSYS.exists() and not args.skip_nsys:
        raise FileNotFoundError(NSYS)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    workloads = build_workloads(manifest)
    if args.only:
        requested = set(args.only)
        workloads = [item for item in workloads if item.name in requested]
        missing = requested - {item.name for item in workloads}
        if missing:
            raise ValueError(f"Unknown workload(s): {sorted(missing)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = machine_metadata(args.gpu_index)
    metadata["manifest"] = manifest
    metadata["trials"] = args.trials
    (args.output_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(args.gpu_index),
        "LOKY_MAX_CPU_COUNT": "8",
        "OMP_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "NUMEXPR_NUM_THREADS": "8",
        "CCCL_IGNORE_DEPRECATED_CUDA_BELOW_12": "1",
        "CUPY_CACHE_IN_MEMORY": "0",
    })
    for workload in workloads:
        for trial in range(1, args.trials + 1):
            run_once(workload, "cold", trial, args.output_dir, env)
            run_once(workload, "warm", trial, args.output_dir, env)
        if workload.platform == "gpu" and not args.skip_nsys:
            profile_gpu(workload, args.output_dir, env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
