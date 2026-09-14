import argparse
import copy
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional

from config_unified import CfgDRAM3D, CfgPIM
from graph_aligner_minigraph_like import parse_fasta, parse_fastq
from main_analysis import run_seq_to_graph_alignment
from simulator_unified import DatabaseManager, Final_PIM_Simulator


# Realistic defaults for direct execution on modern datacenter GPUs (e.g., A100).
REALISTIC_DEFAULTS = {
    "max_queries": 0,
    "gpu_repeats": 20,
    "baseline_mode": "paper",
    "pim_timing_model": "conservative",
    "pim_seeding_overlap_coeff": 0.6,
    "gpu_divergence_penalty": 1.30,
    "gpu_cache_miss_penalty": 1.40,
    "gpu_candidate_amplification": 1.40,
    "gpu_scheduler_overhead_us": 50.0,
    "gpu_host_merge_us_per_query": 0.05,
}

READ_PROFILES = {
    "auto": {"aligner_k": None, "aligner_w": None, "aligner_band": 96, "aligner_max_occ": 256},
    "short": {"aligner_k": 15, "aligner_w": 10, "aligner_band": 96, "aligner_max_occ": 256},
    "long": {"aligner_k": 19, "aligner_w": 20, "aligner_band": 256, "aligner_max_occ": 128},
}

SHORT_LENGTH_BUCKETS_BP = [100, 150, 250]
LONG_LENGTH_BUCKETS_BP = [2000, 4000, 6000, 8000, 10000]


def _extract_numeric_metric(text: str, key: str) -> Optional[float]:
    m = re.search(rf"{re.escape(key)}=([0-9eE+\-.]+)", text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _summarize_lengths(seqs: List[str]) -> Dict[str, float]:
    if not seqs:
        return {"count": 0, "min": 0, "avg": 0.0, "max": 0}
    lens = [len(s) for s in seqs]
    return {
        "count": len(lens),
        "min": min(lens),
        "avg": sum(lens) / len(lens),
        "max": max(lens),
    }


def _run_length_study(args, here: str) -> None:
    script_path = os.path.abspath(__file__)

    groups = []
    if args.study_preset in ("short", "all"):
        groups.append(("short", args.short_query_fastq or args.query_fastq, SHORT_LENGTH_BUCKETS_BP))
    if args.study_preset in ("long", "all"):
        groups.append(("long", args.long_query_fastq or args.query_fastq, LONG_LENGTH_BUCKETS_BP))

    if not groups:
        return

    for profile, query_fastq, buckets in groups:
        if not query_fastq:
            print(f"[WARN] Skip {profile} study: query FASTQ not provided")
            continue

        if not os.path.exists(query_fastq):
            print(f"[WARN] Skip {profile} study: dataset not found: {query_fastq}")
            continue

        try:
            seqs = parse_fastq(query_fastq)
        except Exception as exc:
            print(f"[WARN] Skip {profile} study: failed to parse FASTQ {query_fastq}: {exc}")
            continue

        stats = _summarize_lengths(seqs)
        if stats["count"] == 0:
            print(f"[WARN] Skip {profile} study: no reads parsed from {query_fastq}")
            continue

        print("\n" + "=" * 72)
        print(f"Length Study: {profile.upper()} reads")
        print(f"Dataset: {query_fastq}")
        print(
            "Preflight: "
            f"count={int(stats['count'])}, min={int(stats['min'])}, "
            f"avg={stats['avg']:.1f}, max={int(stats['max'])}"
        )
        for b in buckets:
            kept = sum(1 for s in seqs if len(s) >= b)
            print(f"  reads_len>={b}: {kept}")

        if int(stats["max"]) < int(min(buckets)):
            print(
                f"[WARN] Skip {profile} study: max read length {int(stats['max'])} "
                f"is smaller than smallest bucket {min(buckets)}"
            )
            continue

        print("=" * 72)

        rows = []
        labels = [str(v) for v in buckets] + ["average"]
        for label in labels:
            cmd = [
                sys.executable,
                script_path,
                "--study-preset",
                "none",
                "--read-profile",
                profile,
                "--query-fastq",
                query_fastq,
                "--max-queries",
                str(args.max_queries),
                "--target-total-bases",
                str(args.target_total_bases),
                "--gpu-repeats",
                str(args.gpu_repeats),
                "--gpu-index",
                str(args.gpu_index),
                "--gpu-tdp-w",
                str(args.gpu_tdp_w),
                "--gpu-bin",
                args.gpu_bin,
                "--gpu-pcie-gbps",
                str(args.gpu_pcie_gbps),
                "--gpu-launch-overhead-us",
                str(args.gpu_launch_overhead_us),
                "--gpu-idle-power-w",
                str(args.gpu_idle_power_w),
                "--cpu-prepost-power-w",
                str(args.cpu_prepost_power_w),
                "--gpu-indexing-gops",
                str(args.gpu_indexing_gops),
                "--gpu-seeding-gops",
                str(args.gpu_seeding_gops),
                "--gpu-filtering-gops",
                str(args.gpu_filtering_gops),
                "--gpu-filter-ratio",
                str(args.gpu_filter_ratio),
                "--gpu-divergence-penalty",
                str(args.gpu_divergence_penalty),
                "--gpu-cache-miss-penalty",
                str(args.gpu_cache_miss_penalty),
                "--gpu-candidate-amplification",
                str(args.gpu_candidate_amplification),
                "--gpu-scheduler-overhead-us",
                str(args.gpu_scheduler_overhead_us),
                "--gpu-host-merge-us-per-query",
                str(args.gpu_host_merge_us_per_query),
                "--pim-seeding-overlap-coeff",
                str(args.pim_seeding_overlap_coeff),
                "--pim-timing-model",
                args.pim_timing_model,
                "--baseline-mode",
                args.baseline_mode,
            ]

            if args.gfa:
                cmd += ["--gfa", args.gfa]
            if args.ref_fasta:
                cmd += ["--ref-fasta", args.ref_fasta]
            if args.query_fasta:
                cmd += ["--query-fasta", args.query_fasta]
            if args.aligner_k is not None:
                cmd += ["--aligner-k", str(args.aligner_k)]
            if args.aligner_w is not None:
                cmd += ["--aligner-w", str(args.aligner_w)]
            if args.aligner_max_occ is not None:
                cmd += ["--aligner-max-occ", str(args.aligner_max_occ)]
            if args.aligner_band is not None:
                cmd += ["--aligner-band", str(args.aligner_band)]
            if label != "average":
                cmd += ["--read-length-target", label]

            run = subprocess.run(cmd, text=True, capture_output=True, cwd=here)
            if run.returncode != 0:
                print(f"[FAIL] {profile}:{label}bp run failed")
                print(run.stderr[-800:])
                continue

            out = run.stdout
            avg_len = _extract_numeric_metric(out, "avg_query_len")
            speedup = _extract_numeric_metric(out, "speedup_pim_over_gpu_nomap")
            e_ratio = _extract_numeric_metric(out, "energy_ratio_gpu_nomap_over_pim")
            queries = _extract_numeric_metric(out, "num_queries")

            rows.append(
                {
                    "bucket": (label + "bp") if label != "average" else "average",
                    "avg_len": avg_len,
                    "num_queries": queries,
                    "speedup": speedup,
                    "energy_ratio": e_ratio,
                }
            )

        if not rows:
            print(f"[WARN] No successful runs for {profile} study")
            continue

        print("bucket\tavg_len\tnum_queries\tspeedup_pim_over_gpu\tenergy_ratio_gpu_over_pim")
        for r in rows:
            avg_len_txt = f"{r['avg_len']:.1f}" if r["avg_len"] is not None else "NA"
            nq_txt = f"{int(r['num_queries'])}" if r["num_queries"] is not None else "NA"
            sp_txt = f"{r['speedup']:.4f}" if r["speedup"] is not None else "NA"
            er_txt = f"{r['energy_ratio']:.4f}" if r["energy_ratio"] is not None else "NA"
            print(f"{r['bucket']}\t{avg_len_txt}\t{nq_txt}\t{sp_txt}\t{er_txt}")


def _avg_query_len(query_fasta: Optional[str], query_fastq: Optional[str], max_queries: int) -> int:
    if query_fastq:
        seqs = parse_fastq(query_fastq)
    elif query_fasta:
        seqs = parse_fasta(query_fasta)
    else:
        return 200

    if not seqs:
        return 200

    if max_queries > 0 and len(seqs) > max_queries:
        seqs = seqs[:max_queries]

    return max(8, int(sum(len(s) for s in seqs) / len(seqs)))


def _get_gpu_power_once(gpu_index: int) -> Optional[float]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None

    cmd = [
        nvidia_smi,
        "--query-gpu=index,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None

    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            idx = int(parts[0])
            power = float(parts[1])
        except ValueError:
            continue
        if idx == gpu_index:
            return power
    return None


def _get_gpu_name_and_power_limit(gpu_index: int) -> tuple[Optional[str], Optional[float]]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None, None

    cmd = [
        nvidia_smi,
        "--query-gpu=index,name,power.limit",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None, None

    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        if idx != gpu_index:
            continue

        name = parts[1] if parts[1] else None
        try:
            p_limit = float(parts[2])
        except ValueError:
            p_limit = None
        return name, p_limit

    return None, None


def _ensure_gpu_binary(source_path: str, binary_path: str) -> None:
    need_build = (not os.path.exists(binary_path)) or (
        os.path.getmtime(source_path) > os.path.getmtime(binary_path)
    )
    if not need_build:
        _ensure_executable(binary_path)
        return

    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError(
            "nvcc not found, and GPU benchmark binary needs rebuild. "
            "Please install CUDA toolkit (nvcc) or provide a prebuilt up-to-date binary via --gpu-bin."
        )

    cmd = [nvcc, source_path, "-O3", "-o", binary_path]
    subprocess.run(cmd, check=True)

    # Ensure the output binary is executable for direct subprocess execution.
    if os.path.exists(binary_path):
        cur_mode = os.stat(binary_path).st_mode
        os.chmod(binary_path, cur_mode | 0o111)


def _ensure_executable(binary_path: str) -> None:
    if not os.path.exists(binary_path):
        raise RuntimeError(f"GPU benchmark binary not found: {binary_path}")

    if os.access(binary_path, os.X_OK):
        return

    cur_mode = os.stat(binary_path).st_mode
    os.chmod(binary_path, cur_mode | 0o111)

    if not os.access(binary_path, os.X_OK):
        raise RuntimeError(
            "GPU benchmark binary exists but is not executable. "
            f"Try: chmod +x {binary_path}"
        )


def _run_gpu_benchmark(
    binary_path: str,
    num_queries: int,
    avg_len: int,
    ops_per_task: int,
    repeats: int,
    gpu_index: int,
    gpu_tdp_w: float,
) -> Dict[str, float]:
    _ensure_executable(binary_path)

    # GPU side is treated as no-mapping baseline (no tier-aware/modulo-aware placement).
    cmd = [
        binary_path,
        str(num_queries),
        str(avg_len),
        str(ops_per_task),
        str(repeats),
    ]

    power_samples: List[float] = []
    stop_flag = {"done": False}

    def sampler() -> None:
        while not stop_flag["done"]:
            p = _get_gpu_power_once(gpu_index)
            if p is not None:
                power_samples.append(p)
            time.sleep(0.2)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "GPU benchmark execution failed. "
            f"cmd={cmd}, returncode={e.returncode}\n"
            f"stdout_tail:\n{(e.stdout or '')[-1200:]}\n"
            f"stderr_tail:\n{(e.stderr or '')[-1200:]}"
        ) from e
    except PermissionError as e:
        raise RuntimeError(
            "Cannot execute GPU benchmark binary due to permission/noexec policy: "
            f"{binary_path}. If this directory is mounted with noexec, set --gpu-bin "
            "to an executable path such as /tmp/rapidx_bench."
        ) from e
    wall_s = time.time() - t0
    stop_flag["done"] = True
    t.join(timeout=1.0)

    stdout = proc.stdout
    m = re.search(r"GPU_BENCH_RESULT\s+elapsed_s=([0-9eE+\-.]+)\s+total_queries=([0-9eE+\-.]+)\s+total_ops=([0-9eE+\-.]+)", stdout)
    if not m:
        raise RuntimeError("Failed to parse GPU benchmark output.\n" + stdout)

    elapsed_s = float(m.group(1))
    total_queries = float(m.group(2))
    total_ops = float(m.group(3))

    # Guardrail for short kernel runs: low-frequency sampling can miss peak load.
    # If no samples are collected, or sampled power is too low (<100W), fall back
    # to a reasonable GPU full-load proxy (TDP-like value).
    sampled_avg_power_w = (sum(power_samples) / len(power_samples)) if power_samples else None
    if sampled_avg_power_w is None or sampled_avg_power_w < 100.0:
        avg_power_w = max(100.0, gpu_tdp_w)
        power_mode = "tdp_fallback"
    else:
        avg_power_w = sampled_avg_power_w
        power_mode = "sampled"

    energy_j = avg_power_w * elapsed_s

    return {
        "elapsed_s": elapsed_s,
        "wall_s": wall_s,
        "total_queries": total_queries,
        "total_ops": total_ops,
        "avg_power_w": avg_power_w,
        "sampled_avg_power_w": sampled_avg_power_w if sampled_avg_power_w is not None else -1.0,
        "power_mode": power_mode,
        "energy_j": energy_j,
        "power_samples": float(len(power_samples)),
    }


def _gpu_full_pipeline_model(
    num_queries: int,
    avg_len: int,
    ops_per_seed_task: int,
    ops_per_align_task: int,
    kernel_batch_time_s: float,
    gpu_active_power_w: float,
    pcie_gbps: float,
    launch_overhead_us: float,
    gpu_idle_power_w: float,
    cpu_prepost_power_w: float,
    indexing_gops: float,
    seeding_gops: float,
    filtering_gops: float,
    filter_ratio: float,
    divergence_penalty: float,
    cache_miss_penalty: float,
    candidate_amplification: float,
    scheduler_overhead_us: float,
    host_merge_us_per_query: float,
) -> Dict[str, float]:
    # Transfer model (host->device for reads and metadata; device->host for scores/locations)
    h2d_bytes = num_queries * (avg_len + 16)
    d2h_bytes = num_queries * 8
    transfer_s = (h2d_bytes + d2h_bytes) / max(1.0, pcie_gbps * 1e9)

    # Host-side preprocessing model.
    divergence_penalty = max(1.0, divergence_penalty)
    cache_miss_penalty = max(1.0, cache_miss_penalty)
    candidate_amplification = max(1.0, candidate_amplification)

    indexing_ops = num_queries * avg_len * 32.0 * cache_miss_penalty
    seeding_ops = num_queries * ops_per_seed_task * candidate_amplification * cache_miss_penalty
    filtering_ops = (
        num_queries
        * ops_per_seed_task
        * max(0.0, min(1.0, filter_ratio))
        * candidate_amplification
        * divergence_penalty
    )

    t_index_s = indexing_ops / max(1.0, indexing_gops * 1e9)
    t_seed_s = seeding_ops / max(1.0, seeding_gops * 1e9)
    t_filter_s = filtering_ops / max(1.0, filtering_gops * 1e9)
    t_launch_s = max(0.0, launch_overhead_us) * 1e-6
    t_sched_s = max(0.0, scheduler_overhead_us) * 1e-6
    t_merge_s = max(0.0, host_merge_us_per_query) * 1e-6 * num_queries

    # Kernel-side degradation from divergence and candidate blow-up.
    kernel_effective_s = kernel_batch_time_s * divergence_penalty * (1.0 + 0.5 * (candidate_amplification - 1.0))

    # Full GPU pipeline time for one batch.
    total_batch_time_s = kernel_effective_s + transfer_s + t_launch_s + t_sched_s + t_index_s + t_seed_s + t_filter_s + t_merge_s

    # System energy model: GPU active on kernel; GPU idle during transfer/launch;
    # CPU handles preprocessing stages.
    gpu_energy_j = gpu_active_power_w * kernel_effective_s + gpu_idle_power_w * (transfer_s + t_launch_s + t_sched_s)
    cpu_energy_j = cpu_prepost_power_w * (t_index_s + t_seed_s + t_filter_s + t_merge_s)
    total_energy_j = gpu_energy_j + cpu_energy_j

    return {
        "kernel_batch_time_s": kernel_batch_time_s,
        "kernel_effective_s": kernel_effective_s,
        "transfer_s": transfer_s,
        "launch_s": t_launch_s,
        "sched_s": t_sched_s,
        "index_s": t_index_s,
        "seed_s": t_seed_s,
        "filter_s": t_filter_s,
        "merge_s": t_merge_s,
        "total_batch_time_s": total_batch_time_s,
        "gpu_energy_j": gpu_energy_j,
        "cpu_energy_j": cpu_energy_j,
        "total_energy_j": total_energy_j,
    }


def _run_pim(
    num_queries: int,
    ops_per_seed_task: int,
    ops_per_align_task: int,
    mapping_summary: Optional[dict] = None,
) -> Dict[str, float]:
    cfg_dram = CfgDRAM3D()
    cfg_pim = CfgPIM()
    db_manager = DatabaseManager(cfg_dram, cfg_pim)
    db_manager.map_genome_data(ptr_size_gb=4.0, cal_size_gb=13.4, ref_genome_size_gb=3.0)

    sim_bio = Final_PIM_Simulator(application="bio", database_manager=db_manager)
    if hasattr(sim_bio.cfg_app, "ops_per_seed_task"):
        sim_bio.cfg_app.ops_per_seed_task = ops_per_seed_task
    if hasattr(sim_bio.cfg_app, "ops_per_align_task"):
        sim_bio.cfg_app.ops_per_align_task = ops_per_align_task

    t_s, e = sim_bio.run_bio_pipeline(num_queries=num_queries, mapping_summary=mapping_summary)
    p_w = e.sum() / t_s if t_s > 0 else 0.0
    return {
        "elapsed_s": t_s,
        "energy_j": e.sum(),
        "avg_power_w": p_w,
    }


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = "/root/mhc_test"
    default_long_query_fastq = os.path.join(default_data_dir, "long_reads.fq")

    parser = argparse.ArgumentParser(description="Compare GPU vs PIM for sequence-to-graph alignment workload")
    parser.add_argument("--gfa", type=str, default=None)
    parser.add_argument("--ref-fasta", type=str, default=os.path.join(default_data_dir, "mhc_ref.fa"))
    parser.add_argument("--query-fasta", type=str, default=None)
    parser.add_argument("--query-fastq", type=str, default=os.path.join(default_data_dir, "query_reads.fq"))
    parser.add_argument("--short-query-fastq", type=str, default=None, help="Optional dataset for short-read sweep")
    parser.add_argument("--long-query-fastq", type=str, default=default_long_query_fastq, help="Optional dataset for long-read sweep")
    parser.add_argument("--study-preset", type=str, choices=["none", "short", "long", "all"], default="none", help="Run automatic read-length sweep and summary table")
    parser.add_argument("--max-queries", type=int, default=REALISTIC_DEFAULTS["max_queries"])
    parser.add_argument("--target-total-bases", type=int, default=0, help="If >0, trim batch by total bases for fair long-vs-short comparison")
    parser.add_argument("--read-length-target", type=int, default=0, help="If >0, crop/filter reads to this fixed length before alignment")
    parser.add_argument("--read-profile", type=str, choices=["auto", "short", "long"], default="auto", help="Preset alignment knobs tuned for read length regime")
    parser.add_argument("--aligner-k", type=int, default=None, help="Override k-mer length for minimizer seeding")
    parser.add_argument("--aligner-w", type=int, default=None, help="Override minimizer window length")
    parser.add_argument("--aligner-max-occ", type=int, default=None, help="Override max repetitive seed occurrences")
    parser.add_argument("--aligner-band", type=int, default=None, help="Override DP band width")
    parser.add_argument("--gpu-repeats", type=int, default=REALISTIC_DEFAULTS["gpu_repeats"])
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--gpu-tdp-w", type=float, default=None, help="Fallback GPU power (W) when sampling misses load; default auto-detected via nvidia-smi")
    parser.add_argument("--gpu-bin", type=str, default=os.path.join(here, "rapidx_bench"))
    parser.add_argument("--gpu-pcie-gbps", type=float, default=24.0, help="Effective host-device bandwidth (GB/s)")
    parser.add_argument("--gpu-launch-overhead-us", type=float, default=30.0, help="Per-batch launch/runtime overhead on GPU")
    parser.add_argument("--gpu-idle-power-w", type=float, default=70.0, help="GPU power when not in active kernel execution")
    parser.add_argument("--cpu-prepost-power-w", type=float, default=200.0, help="CPU power for pre/post pipeline stages")
    parser.add_argument("--gpu-indexing-gops", type=float, default=120.0, help="Effective indexing throughput (GOPS)")
    parser.add_argument("--gpu-seeding-gops", type=float, default=80.0, help="Effective seeding throughput (GOPS)")
    parser.add_argument("--gpu-filtering-gops", type=float, default=140.0, help="Effective filtering throughput (GOPS)")
    parser.add_argument("--gpu-filter-ratio", type=float, default=0.35, help="Fraction of seeding ops spent in filtering")
    parser.add_argument("--gpu-divergence-penalty", type=float, default=1.25, help="Warp/control-flow divergence penalty multiplier")
    parser.add_argument("--gpu-cache-miss-penalty", type=float, default=1.30, help="Cache miss/memory irregularity penalty multiplier")
    parser.add_argument("--gpu-candidate-amplification", type=float, default=1.40, help="Candidate explosion multiplier for seeding/filtering")
    parser.add_argument("--gpu-scheduler-overhead-us", type=float, default=120.0, help="Runtime scheduler overhead per batch (us)")
    parser.add_argument("--gpu-host-merge-us-per-query", type=float, default=0.12, help="Host-side merge/postprocess overhead per query (us)")
    parser.add_argument(
        "--pim-seeding-overlap-coeff",
        type=float,
          default=REALISTIC_DEFAULTS["pim_seeding_overlap_coeff"],
        help="[0,1] fraction of seeding-memory latency hidden by pipeline overlap in PIM",
    )
    parser.add_argument(
        "--pim-timing-model",
        type=str,
        choices=["paper_like", "conservative"],
        default=REALISTIC_DEFAULTS["pim_timing_model"],
        help="PIM timing model: paper_like uses seeding/alignment overlap; conservative uses additive penalties",
    )
    parser.add_argument(
        "--baseline-mode",
        type=str,
        choices=["worst_case", "paper"],
        default=REALISTIC_DEFAULTS["baseline_mode"],
        help="No-mapping baseline severity for GPU comparison",
    )
    parser.set_defaults(
        gpu_divergence_penalty=REALISTIC_DEFAULTS["gpu_divergence_penalty"],
        gpu_cache_miss_penalty=REALISTIC_DEFAULTS["gpu_cache_miss_penalty"],
        gpu_candidate_amplification=REALISTIC_DEFAULTS["gpu_candidate_amplification"],
        gpu_scheduler_overhead_us=REALISTIC_DEFAULTS["gpu_scheduler_overhead_us"],
        gpu_host_merge_us_per_query=REALISTIC_DEFAULTS["gpu_host_merge_us_per_query"],
    )
    args = parser.parse_args()

    if args.long_query_fastq and not os.path.exists(args.long_query_fastq):
        if os.path.exists(default_long_query_fastq):
            print(
                "[WARN] Requested --long-query-fastq not found: "
                f"{args.long_query_fastq}; fallback to {default_long_query_fastq}"
            )
            args.long_query_fastq = default_long_query_fastq
        else:
            print(f"[WARN] long-read FASTQ not found: {args.long_query_fastq}")

    gpu_name, gpu_power_limit_w = _get_gpu_name_and_power_limit(args.gpu_index)
    if args.gpu_tdp_w is None:
        if gpu_power_limit_w is not None and gpu_power_limit_w > 0:
            args.gpu_tdp_w = gpu_power_limit_w
        elif gpu_name and "A100" in gpu_name.upper():
            args.gpu_tdp_w = 400.0
        else:
            args.gpu_tdp_w = 250.0

    prof = READ_PROFILES.get(args.read_profile, READ_PROFILES["auto"])
    if args.aligner_k is None:
        args.aligner_k = prof["aligner_k"]
    if args.aligner_w is None:
        args.aligner_w = prof["aligner_w"]
    if args.aligner_max_occ is None:
        args.aligner_max_occ = prof["aligner_max_occ"]
    if args.aligner_band is None:
        args.aligner_band = prof["aligner_band"]

    if args.study_preset != "none":
        _run_length_study(args, here)
        return

    if args.gfa is None and not os.path.exists(args.ref_fasta):
        args.ref_fasta = None
    if args.query_fastq and not os.path.exists(args.query_fastq):
        args.query_fastq = None

    print("=== Deriving workload from sequence-to-graph alignment ===")
    bio = run_seq_to_graph_alignment(
        gfa_path=args.gfa,
        ref_fasta=args.ref_fasta,
        query_fasta=args.query_fasta,
        query_fastq=args.query_fastq,
        max_queries=args.max_queries,
        target_total_bases=args.target_total_bases,
        read_length_target=args.read_length_target,
        aligner_k=args.aligner_k,
        aligner_w=args.aligner_w,
        aligner_max_occ=args.aligner_max_occ,
        aligner_band=args.aligner_band,
    )

    avg_len = int(bio.get("avg_query_len", _avg_query_len(args.query_fasta, args.query_fastq, args.max_queries)))
    print("\nDerived workload:")
    print(f"  num_queries={bio['num_queries']}")
    print(f"  ops_per_seed_task={bio['ops_per_seed_task']}")
    print(f"  ops_per_align_task={bio['ops_per_align_task']}")
    print(f"  avg_query_len={avg_len}")
    print(
        "  read_profile="
        f"{args.read_profile} (k={args.aligner_k}, w={args.aligner_w}, max_occ={args.aligner_max_occ}, band={args.aligner_band})"
    )
    if args.read_length_target > 0:
        print(f"  read_length_target={args.read_length_target}bp")
    print(
        f"  gpu_detected={gpu_name if gpu_name else 'unknown'} "
        f"(fallback_power={args.gpu_tdp_w:.1f}W)"
    )

    print("\n=== Running PIM model ===")
    mapping_summary = bio.get("mapping_summary")
    if mapping_summary is not None:
        mapping_summary["timing_model"] = args.pim_timing_model
        mapping_summary["seeding_memory_overlap_coeff"] = args.pim_seeding_overlap_coeff
    pim = _run_pim(
        num_queries=bio["num_queries"],
        ops_per_seed_task=bio["ops_per_seed_task"],
        ops_per_align_task=bio["ops_per_align_task"],
        mapping_summary=mapping_summary,
    )

    print("\n=== Running PIM no-mapping baseline ===")
    baseline_summary = copy.deepcopy(mapping_summary) if mapping_summary else {}
    baseline_summary["mode"] = "baseline"
    baseline_summary["timing_model"] = args.pim_timing_model
    baseline_summary["seeding_memory_overlap_coeff"] = args.pim_seeding_overlap_coeff
    if args.baseline_mode == "worst_case":
        # Worst-case no-mapping: no tier-aware placement and no modulo-aware distribution.
        # Increase read latency, channel conflicts, and tier-hop penalties.
        baseline_summary["baseline_policy"] = "no-tier-aware + no-modulo-aware (worst_case)"
        baseline_summary["baseline_node_tier_read_ns"] = 22.88
        baseline_summary["baseline_node_channel_switches_per_query"] = max(
            1.0, float(baseline_summary.get("avg_node_accesses_per_query", 1.0)) * 1.0
        )
        baseline_summary["baseline_channel_switch_penalty_ns"] = 1.2
        baseline_summary["baseline_node_tier_hops_per_query"] = max(
            2.0, float(baseline_summary.get("avg_node_accesses_per_query", 1.0)) * 7.0
        )

        lookups = max(1.0, float(baseline_summary.get("ptr_cal_lookups_per_query", 1.0)))
        baseline_summary["baseline_ptr_cal_tier_read_ns"] = 22.88
        baseline_summary["baseline_ptr_cal_channel_switches_per_query"] = lookups * 1.0
        baseline_summary["baseline_ptr_cal_avg_tier_hops_per_query"] = lookups * 7.0
        baseline_summary["baseline_no_modulo_conflict_penalty_ns"] = lookups * 1.5

    pim_nomap = _run_pim(
        num_queries=bio["num_queries"],
        ops_per_seed_task=bio["ops_per_seed_task"],
        ops_per_align_task=bio["ops_per_align_task"],
        mapping_summary=baseline_summary,
    )

    print("\n=== Running GPU benchmark ===")
    source = os.path.join(here, "rapidx.cu")
    _ensure_gpu_binary(source, args.gpu_bin)
    gpu = _run_gpu_benchmark(
        binary_path=args.gpu_bin,
        num_queries=bio["num_queries"],
        avg_len=avg_len,
        ops_per_task=bio["ops_per_align_task"],
        repeats=args.gpu_repeats,
        gpu_index=args.gpu_index,
        gpu_tdp_w=args.gpu_tdp_w,
    )

    # Apple-to-apple comparison on one batch (= num_queries reads).
    # GPU raw benchmark runs `repeats` batches, so normalize by repeats.
    gpu_elapsed_batch = gpu["elapsed_s"] / max(1, args.gpu_repeats)
    gpu_full = _gpu_full_pipeline_model(
        num_queries=bio["num_queries"],
        avg_len=avg_len,
        ops_per_seed_task=bio["ops_per_seed_task"],
        ops_per_align_task=bio["ops_per_align_task"],
        kernel_batch_time_s=gpu_elapsed_batch,
        gpu_active_power_w=gpu["avg_power_w"],
        pcie_gbps=args.gpu_pcie_gbps,
        launch_overhead_us=args.gpu_launch_overhead_us,
        gpu_idle_power_w=args.gpu_idle_power_w,
        cpu_prepost_power_w=args.cpu_prepost_power_w,
        indexing_gops=args.gpu_indexing_gops,
        seeding_gops=args.gpu_seeding_gops,
        filtering_gops=args.gpu_filtering_gops,
        filter_ratio=args.gpu_filter_ratio,
        divergence_penalty=args.gpu_divergence_penalty,
        cache_miss_penalty=args.gpu_cache_miss_penalty,
        candidate_amplification=args.gpu_candidate_amplification,
        scheduler_overhead_us=args.gpu_scheduler_overhead_us,
        host_merge_us_per_query=args.gpu_host_merge_us_per_query,
    )
    gpu_time_batch = gpu_full["total_batch_time_s"]
    gpu_energy_batch = gpu_full["total_energy_j"]

    print("\n=== Comparison: GPU vs PIM ===")
    print(f"Batch definition: {bio['num_queries']} queries")
    print(f"PIM avg_operating_power_w={pim['avg_power_w']:.2f}")
    print(f"PIM batch_elapsed_s={pim['elapsed_s']:.6f}")
    print(f"PIM batch_energy_j={pim['energy_j']:.6e}")
    sampled = gpu["sampled_avg_power_w"]
    sampled_txt = f"{sampled:.2f}" if sampled >= 0 else "N/A"
    print(f"GPU avg_operating_power_w={gpu['avg_power_w']:.2f} (mode={gpu['power_mode']}, sampled={sampled_txt}, samples={int(gpu['power_samples'])})")
    print(f"GPU kernel_batch_s={gpu_elapsed_batch:.6f} (raw_total_s={gpu['elapsed_s']:.6f}, repeats={args.gpu_repeats})")
    print(f"GPU effective_kernel_s={gpu_full['kernel_effective_s']:.6f} (div={args.gpu_divergence_penalty}, miss={args.gpu_cache_miss_penalty}, cand={args.gpu_candidate_amplification})")
    print(f"GPU full_batch_s={gpu_time_batch:.6f} [transfer={gpu_full['transfer_s']:.6f}, launch={gpu_full['launch_s']:.6f}, sched={gpu_full['sched_s']:.6f}, index={gpu_full['index_s']:.6f}, seed={gpu_full['seed_s']:.6f}, filter={gpu_full['filter_s']:.6f}, merge={gpu_full['merge_s']:.6f}]")
    print(f"GPU batch_energy_j={gpu_energy_batch:.6e}")

    print("\n=== PIM Mapping Benefit (Mapped vs No-Mapping Baseline) ===")
    print(f"PIM timing model: {args.pim_timing_model}")
    if mapping_summary and mapping_summary.get("baseline_policy"):
        print(f"Baseline policy: {mapping_summary['baseline_policy']}")
    print(f"PIM(mapped)   batch_elapsed_s={pim['elapsed_s']:.6f}, batch_energy_j={pim['energy_j']:.6e}")
    print(f"PIM(baseline) batch_elapsed_s={pim_nomap['elapsed_s']:.6f}, batch_energy_j={pim_nomap['energy_j']:.6e}")
    map_speedup = (pim_nomap["elapsed_s"] / pim["elapsed_s"]) if pim["elapsed_s"] > 0 else float("inf")
    map_energy_gain = (pim_nomap["energy_j"] / pim["energy_j"]) if pim["energy_j"] > 0 else float("inf")
    print(f"mapping_speedup_over_baseline={map_speedup:.4f}")
    print("  (definition: PIM baseline time / PIM mapped time, >1 => mapping accelerates PIM)")
    print(f"mapping_energy_gain_over_baseline={map_energy_gain:.4f}")
    print("  (definition: PIM baseline energy / PIM mapped energy, >1 => mapping saves PIM energy)")

    # Primary metric requested by user: mapped PIM vs GPU without mapping.
    speedup_pim_mapped_over_gpu_nomap = (
        gpu_time_batch / pim["elapsed_s"]
    ) if pim["elapsed_s"] > 0 else float("inf")
    energy_ratio_gpu_nomap_over_pim_mapped = (
        gpu_energy_batch / pim["energy_j"]
    ) if pim["energy_j"] > 0 else float("inf")

    # Keep PIM no-mapping baseline only as an ablation result.
    speedup_pim_baseline_over_gpu_nomap = (
        gpu_time_batch / pim_nomap["elapsed_s"]
    ) if pim_nomap["elapsed_s"] > 0 else float("inf")
    energy_ratio_gpu_nomap_over_pim_baseline = (
        gpu_energy_batch / pim_nomap["energy_j"]
    ) if pim_nomap["energy_j"] > 0 else float("inf")

    print("\n=== Ratios ===")
    print("primary_comparison = mapped_pim_vs_gpu_no_mapping")
    print(f"speedup_pim_over_gpu_nomap={speedup_pim_mapped_over_gpu_nomap:.4f}")
    print("  (definition: GPU(no-mapping) batch time / PIM(mapped) batch time, >1 => PIM faster)")
    print(f"energy_ratio_gpu_nomap_over_pim={energy_ratio_gpu_nomap_over_pim_mapped:.4f}")
    print("  (definition: GPU(no-mapping) batch energy / PIM(mapped) batch energy, >1 => PIM saves energy)")

    print("\n--- Ablation (not primary comparison) ---")
    print(f"baseline_mode_for_pim_ablation={args.baseline_mode}")
    print(f"speedup_pim_baseline_over_gpu_nomap={speedup_pim_baseline_over_gpu_nomap:.4f}")
    print(f"energy_ratio_gpu_nomap_over_pim_baseline={energy_ratio_gpu_nomap_over_pim_baseline:.4f}")


if __name__ == "__main__":
    main()
