#!/usr/bin/env python3
"""Run the MultiRAM timing stages and summarize PIM/PNM latency.

Stages:
1. Genomic sequence-to-graph simulation on GenDP/3D-DRAM PNM.
2. Proteomic clustering on FeRAM PIM.
3. Proteomic HyperOMS/FeRAM PIM simulation.
4. Package-level transfer model.
5. MoE inference timing from the downstream model script or cached JSON.

The paper's full-pipeline timing is a composed timeline, so this script keeps
each stage latency visible and also reports their sum.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent

GENOMIC_DIR = ROOT / "GenDP/GenDRAM"
GENOMIC_SCRIPT = GENOMIC_DIR / "main_analysis.py"
DEFAULT_REF_FASTA = ROOT / "GenDP/mhc_test/mhc_ref.fa"
DEFAULT_QUERY_FASTQ = ROOT / "GenDP/mhc_test/query_reads_100bp.fq"
HGA_ROOT = ROOT / "external/hga"
HGA_BIN = HGA_ROOT / "bin/hga"

PROTEOMIC_DIR = ROOT / "proteomic_full_pipeline"
PIM_OMS_SCRIPT = PROTEOMIC_DIR / "pim_hyperoms_estimator.py"
HOMSTC_PYTHON = Path(os.environ.get("MULTIRAM_HOMSTC_PYTHON", sys.executable))
MULTIOMIC_PYTHON = Path(os.environ.get("MULTIRAM_PYTHON", sys.executable))
DEFAULT_PROTEOMIC_QUERY = (
    ROOT / "sumukh_proteomic_test/pxd000561_subsets/PXD000561_100.mgf"
)
DEFAULT_PROTEOMIC_REF = ROOT / "data/proteomic/reference.splib"
DEFAULT_HOMSTC_CONFIG = ROOT / "sumukh_proteomic_test/homs-tc/configs/hek293.ini"

MOE_DIR = ROOT / "TRPCA-MOE-2/TRPCA_MoE"
MOE_SCRIPT = MOE_DIR / "test_moe.py"
MOE_JSON = MOE_DIR / "moe_results.json"


@dataclass
class StageTiming:
    stage: str
    command: list[str] | str
    status: str
    latency_sec: float | None
    wall_time_sec: float
    output_path: str | None = None
    log_path: str | None = None
    notes: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "pim_full_pipeline_timing_output"),
        help="Directory for logs and summary files.",
    )

    parser.add_argument("--skip-genomic", action="store_true")
    parser.add_argument(
        "--genomic-backend",
        choices=["hga", "gendp"],
        default="hga",
        help="hga runs the RapidsAtHKUST/HGA GPU aligner; gendp runs the previous simulator.",
    )
    parser.add_argument("--genomic-max-queries", type=int, default=1000)
    parser.add_argument("--genomic-ref-fasta", default=str(DEFAULT_REF_FASTA))
    parser.add_argument("--genomic-query-fastq", default=str(DEFAULT_QUERY_FASTQ))
    parser.add_argument("--hga-root", default=str(HGA_ROOT))
    parser.add_argument("--hga-bin", default=str(HGA_BIN))
    parser.add_argument("--hga-devices", type=int, default=1)
    parser.add_argument("--hga-blocks", type=int, default=68)
    parser.add_argument("--hga-threads", type=int, default=128)
    parser.add_argument("--hga-cpu-threads", type=int, default=1)
    parser.add_argument("--hga-match", type=int, default=2)
    parser.add_argument("--hga-mismatch", type=int, default=2)
    parser.add_argument("--hga-gap", type=int, default=3)
    parser.add_argument(
        "--no-build-hga",
        action="store_true",
        help="Do not run make if the HGA binary is missing.",
    )

    parser.add_argument("--skip-proteomic", action="store_true")
    parser.add_argument("--skip-proteomic-clustering", action="store_true")
    parser.add_argument("--skip-proteomic-oms", action="store_true")
    parser.add_argument("--proteomic-query", default=str(DEFAULT_PROTEOMIC_QUERY))
    parser.add_argument("--proteomic-ref", default=str(DEFAULT_PROTEOMIC_REF))
    parser.add_argument("--proteomic-config", default=str(DEFAULT_HOMSTC_CONFIG))
    parser.add_argument("--proteomic-python", default=str(HOMSTC_PYTHON))
    parser.add_argument("--proteomic-clustering-python", default=str(MULTIOMIC_PYTHON))
    parser.add_argument(
        "--proteomic-oms-backend",
        choices=["gpu", "cpu"],
        default="gpu",
        help="Functional backend used by pim_hyperoms_estimator.py for OMS.",
    )
    parser.add_argument(
        "--proteomic-max-queries",
        type=int,
        default=0,
        help="0 means full query; use a small value, e.g. 200, for smoke tests.",
    )
    parser.add_argument("--gpu-chunk-size", type=int, default=16384)
    parser.add_argument(
        "--proteomic-candidate-cap",
        type=int,
        default=0,
        help="Optional OMS candidate cap passed to pim_hyperoms_estimator.py; 0 means no cap.",
    )
    parser.add_argument("--progress-interval", type=int, default=2000)

    parser.add_argument("--skip-moe", action="store_true")
    parser.add_argument(
        "--reuse-moe-json",
        action="store_true",
        help="Read existing moe_results.json instead of running test_moe.py.",
    )
    parser.add_argument("--moe-python", default=sys.executable)

    parser.add_argument("--skip-transfer", action="store_true")
    parser.add_argument(
        "--transfer-mode",
        choices=["auto", "manual", "none"],
        default="auto",
        help="auto estimates bytes/bandwidth, manual uses --transfer-sec, none adds no transfer stage.",
    )
    parser.add_argument(
        "--transfer-model",
        choices=["multiram", "simple"],
        default="multiram",
        help="multiram uses per-component bandwidths; simple uses total bytes / --package-bandwidth-gbps.",
    )
    parser.add_argument("--transfer-sec", type=float, default=0.0)
    parser.add_argument(
        "--package-bandwidth-gbps",
        type=float,
        default=1024.0,
        help="Effective package bandwidth for auto transfer, in GB/s.",
    )
    parser.add_argument(
        "--package-link-energy-pj-per-bit",
        type=float,
        default=0.8,
        help=(
            "Assumed package-link energy in pJ/bit. This is a modeling "
            "assumption, not a measured value."
        ),
    )
    parser.add_argument("--host-package-gbps", type=float, default=64.0)
    parser.add_argument("--ucie-gbps", type=float, default=256.0)
    parser.add_argument("--dram-external-gbps", type=float, default=819.2)
    parser.add_argument("--dram-internal-pnm-gbps", type=float, default=19010.0)
    parser.add_argument("--dram-ring-gbps", type=float, default=2048.0)
    parser.add_argument("--feram-external-gbps", type=float, default=256.0)
    parser.add_argument("--fenand-compressed-gbps", type=float, default=4.3)
    parser.add_argument("--fenand-decompressed-gbps", type=float, default=8.1)
    parser.add_argument(
        "--transfer-include-reference",
        action="store_true",
        help="Include reference/database file sizes in auto transfer bytes.",
    )
    parser.add_argument(
        "--moe-feature-bytes",
        type=int,
        default=4096,
        help="Modeled fused feature payload sent to MoE in auto transfer.",
    )
    return parser.parse_args()


def run_command(cmd: list[str], cwd: Path, log_path: Path) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return proc.returncode, time.perf_counter() - start


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def extract_ms_latency(log_text: str, label: str) -> float | None:
    pattern = rf"{re.escape(label)}\s*:\s*([0-9.]+)\s*ms"
    match = re.search(pattern, log_text)
    if not match:
        return None
    return float(match.group(1)) / 1000.0


def extract_hga_latency(log_text: str) -> float | None:
    match = re.search(r"Time:\s*([0-9.eE+-]+)\s*s", log_text)
    if not match:
        return None
    return float(match.group(1))


def extract_last_json(log_text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    best_obj = None
    best_end = -1
    for match in re.finditer(r"\{", log_text):
        try:
            obj, end = decoder.raw_decode(log_text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            absolute_end = match.start() + end
            if "result" in obj:
                return obj
            if absolute_end > best_end:
                best_obj = obj
                best_end = absolute_end
    return best_obj


def read_first_fasta_sequence(path: Path) -> str:
    chunks: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        in_first = False
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if in_first:
                    break
                in_first = True
                continue
            if in_first:
                chunks.append(line.upper())
    seq = "".join(chunks)
    if not seq:
        raise RuntimeError(f"No FASTA sequence found in {path}")
    return seq


def iter_reads(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        first = fh.readline()
        if not first:
            return
        fh.seek(0)
        if first.startswith("@"):
            while True:
                header = fh.readline().strip()
                if not header:
                    break
                seq = fh.readline().strip().upper()
                fh.readline()
                fh.readline()
                if seq:
                    yield header[1:] if header.startswith("@") else header, seq
        else:
            header = None
            seq_chunks: list[str] = []
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if header is not None and seq_chunks:
                        yield header, "".join(seq_chunks).upper()
                    header = line[1:]
                    seq_chunks = []
                else:
                    seq_chunks.append(line)
            if header is not None and seq_chunks:
                yield header, "".join(seq_chunks).upper()


def write_hga_linear_graph(ref_fasta: Path, graph_path: Path) -> int:
    seq = read_first_fasta_sequence(ref_fasta)
    num_v = len(seq)
    num_e = max(0, num_v - 1)
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    with graph_path.open("w", encoding="utf-8") as fh:
        fh.write(f"{num_v} {num_e}\n")
        for source in range(num_e):
            fh.write(f"{source}\n")
        fh.write("0\n")
        for offset in range(num_v):
            fh.write(f"{offset}\n")
        for target in range(1, num_v):
            fh.write(f"{target}\n")
        fh.write("0\n")
        for offset in range(1, num_v + 1):
            fh.write(f"{num_e if offset == num_v else offset}\n")
        for base in seq:
            fh.write(f"{base if base in 'ATCG' else 'N'}\n")
    return num_v


def write_hga_reads(read_path: Path, hga_reads_path: Path, max_reads: int) -> tuple[int, int]:
    hga_reads_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    max_len = 0
    with hga_reads_path.open("w", encoding="utf-8") as out:
        for header, seq in iter_reads(read_path):
            if max_reads > 0 and count >= max_reads:
                break
            clean = "".join(base if base in "ATCG" else "N" for base in seq.upper())
            if not clean:
                continue
            out.write(f"{header or ('read_' + str(count))}\n{clean}\n")
            count += 1
            max_len = max(max_len, len(clean))
    if count == 0:
        raise RuntimeError(f"No reads converted from {read_path}")
    return count, max_len


def ensure_hga_binary(args: argparse.Namespace, log_path: Path) -> Path:
    hga_bin = Path(args.hga_bin)
    if hga_bin.exists():
        return hga_bin
    if args.no_build_hga:
        raise FileNotFoundError(f"HGA binary not found: {hga_bin}")
    hga_root = Path(args.hga_root)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[build] HGA binary missing, running make all in {hga_root}\n")
        proc = subprocess.run(
            ["make", "all"],
            cwd=str(hga_root),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0 or not hga_bin.exists():
        raise RuntimeError(f"Failed to build HGA; see {log_path}")
    return hga_bin


def run_genomic_hga(args: argparse.Namespace, out_dir: Path) -> StageTiming:
    log_path = out_dir / "logs/genomic_hga_gpu.log"
    hga_input_dir = out_dir / "hga_inputs"
    graph_path = hga_input_dir / "linear_graph.hga"
    reads_path = hga_input_dir / "reads.hga"
    try:
        num_v = write_hga_linear_graph(Path(args.genomic_ref_fasta), graph_path)
        num_reads, max_read_len = write_hga_reads(
            Path(args.genomic_query_fastq),
            reads_path,
            int(args.genomic_max_queries),
        )
        hga_bin = ensure_hga_binary(args, log_path)
    except Exception as exc:
        return StageTiming(
            stage="genomic_hga_gpu",
            command="prepare HGA inputs/build HGA",
            status="failed_prepare",
            latency_sec=None,
            wall_time_sec=0.0,
            output_path=str(hga_input_dir),
            log_path=str(log_path),
            notes=str(exc),
        )

    cmd = [
        str(hga_bin),
        "-g",
        str(graph_path),
        "-r",
        str(reads_path),
        "-m",
        str(args.hga_match),
        "-n",
        str(args.hga_mismatch),
        "-o",
        str(args.hga_gap),
        "-d",
        str(args.hga_devices),
        "-b",
        str(args.hga_blocks),
        "-t",
        str(args.hga_threads),
        "-c",
        str(args.hga_cpu_threads),
    ]
    rc, wall = run_command(cmd, Path(args.hga_root), log_path)
    log_text = read_text(log_path)
    latency = extract_hga_latency(log_text)
    return StageTiming(
        stage="genomic_hga_gpu",
        command=cmd,
        status="ok" if rc == 0 and latency is not None else f"failed_or_unparsed_rc_{rc}",
        latency_sec=latency,
        wall_time_sec=wall,
        output_path=str(hga_input_dir),
        log_path=str(log_path),
        notes=f"num_vertices={num_v}; num_reads={num_reads}; max_read_len={max_read_len}",
    )


def run_genomic_gendp(args: argparse.Namespace, out_dir: Path) -> StageTiming:
    log_path = out_dir / "logs/genomic_pnm.log"
    cmd = [
        sys.executable,
        str(GENOMIC_SCRIPT),
        "--ref-fasta",
        str(Path(args.genomic_ref_fasta)),
        "--query-fastq",
        str(Path(args.genomic_query_fastq)),
        "--max-queries",
        str(args.genomic_max_queries),
    ]
    rc, wall = run_command(cmd, GENOMIC_DIR, log_path)
    log_text = read_text(log_path)
    latency = extract_ms_latency(log_text, "Total Latency (Pipelined)")
    return StageTiming(
        stage="genomic_3ddram_pnm",
        command=cmd,
        status="ok" if rc == 0 and latency is not None else f"failed_or_unparsed_rc_{rc}",
        latency_sec=latency,
        wall_time_sec=wall,
        log_path=str(log_path),
        notes="Parsed Bioinformatics PPA Total Latency (Pipelined).",
    )


def run_genomic(args: argparse.Namespace, out_dir: Path) -> StageTiming:
    if args.genomic_backend == "hga":
        return run_genomic_hga(args, out_dir)
    return run_genomic_gendp(args, out_dir)


def run_proteomic_clustering(args: argparse.Namespace, out_dir: Path) -> StageTiming:
    log_path = out_dir / "logs/proteomic_pim_clustering.log"
    cmd = [
        str(Path(args.proteomic_clustering_python)),
        str(PIM_OMS_SCRIPT),
        "--mode",
        "clustering",
        "--query",
        str(Path(args.proteomic_query)),
    ]
    rc, wall = run_command(cmd, PROTEOMIC_DIR, log_path)
    payload = extract_last_json(read_text(log_path))
    latency = None
    notes = None
    summary_path = out_dir / "proteomic_pim/pim_clustering_summary.json"
    if payload is not None:
        result = payload.get("result", {})
        if "feram_cluster_core_s" in result:
            latency = float(result["feram_cluster_core_s"])
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        notes = (
            f"num_spectra={payload.get('num_spectra')}; "
            f"num_buckets={result.get('num_buckets')}; "
            f"energy_mj={result.get('feram_cluster_energy_mj')}"
        )
    return StageTiming(
        stage="proteomic_feram_pim_clustering",
        command=cmd,
        status="ok" if rc == 0 and latency is not None else f"failed_or_unparsed_rc_{rc}",
        latency_sec=latency,
        wall_time_sec=wall,
        output_path=str(summary_path) if summary_path.exists() else None,
        log_path=str(log_path),
        notes=notes,
    )


def run_proteomic_oms(args: argparse.Namespace, out_dir: Path) -> StageTiming:
    proteomic_out = out_dir / "proteomic_pim/pim_hyperoms_output.mztab"
    log_path = out_dir / "logs/proteomic_pim_oms.log"
    proteomic_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(Path(args.proteomic_python)),
        str(PIM_OMS_SCRIPT),
        "--mode",
        "oms",
        "--backend",
        str(args.proteomic_oms_backend),
        "--query",
        str(Path(args.proteomic_query)),
        "--ref",
        str(Path(args.proteomic_ref)),
        "--config",
        str(Path(args.proteomic_config)),
        "--hdc-codebook",
        "random",
        "--score-mode",
        "zscore_margin",
        "--margin-weight",
        "1.0",
        "--gpu-chunk-size",
        str(args.gpu_chunk_size),
        "--progress-interval",
        str(args.progress_interval),
        "--output",
        str(proteomic_out),
    ]
    if args.proteomic_candidate_cap > 0:
        cmd.extend(["--candidate-cap", str(args.proteomic_candidate_cap)])
    if args.proteomic_max_queries > 0:
        cmd.extend(["--max-queries", str(args.proteomic_max_queries)])

    rc, wall = run_command(cmd, PROTEOMIC_DIR, log_path)
    summary_path = proteomic_out.with_suffix(proteomic_out.suffix + ".summary.json")
    latency = None
    notes = None
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        latency = float(summary["simulator_wall_time_sec"])
        notes = (
            f"num_query_spectra={summary.get('num_query_spectra')}; "
            f"backend={summary.get('backend')}; counts={summary.get('counts')}"
        )
    return StageTiming(
        stage=f"proteomic_oms_{args.proteomic_oms_backend}",
        command=cmd,
        status="ok" if rc == 0 and latency is not None else f"failed_or_unparsed_rc_{rc}",
        latency_sec=latency,
        wall_time_sec=wall,
        output_path=str(summary_path) if summary_path.exists() else str(proteomic_out),
        log_path=str(log_path),
        notes=notes,
    )


def _parse_moe_json(path: Path) -> tuple[float | None, str]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    candidates = [
        data.get("test_metrics", {}).get("inference_time"),
        data.get("moe", {}).get("test_metrics", {}).get("inference_time"),
        data.get("moe", {}).get("test_metrics", {}).get("inference_time_sec"),
    ]
    for value in candidates:
        if value is not None:
            return float(value), f"Parsed inference time from {path.name}."
    return None, f"Could not find test_metrics.inference_time in {path}."


def run_moe(args: argparse.Namespace, out_dir: Path) -> StageTiming:
    log_path = out_dir / "logs/moe_inference.log"
    if args.reuse_moe_json:
        latency, notes = _parse_moe_json(MOE_JSON)
        return StageTiming(
            stage="moe_3ddram_pnm_inference",
            command=f"reuse {MOE_JSON}",
            status="ok" if latency is not None else "failed_or_unparsed_cached_json",
            latency_sec=latency,
            wall_time_sec=0.0,
            output_path=str(MOE_JSON),
            log_path=None,
            notes=notes,
        )

    cmd = [str(Path(args.moe_python)), str(MOE_SCRIPT)]
    rc, wall = run_command(cmd, MOE_DIR, log_path)
    latency = None
    notes = None
    if MOE_JSON.exists():
        latency, notes = _parse_moe_json(MOE_JSON)
    return StageTiming(
        stage="moe_3ddram_pnm_inference",
        command=cmd,
        status="ok" if rc == 0 and latency is not None else f"failed_or_unparsed_rc_{rc}",
        latency_sec=latency,
        wall_time_sec=wall,
        output_path=str(MOE_JSON) if MOE_JSON.exists() else None,
        log_path=str(log_path),
        notes=notes,
    )


def safe_size(path: str | Path | None) -> int:
    if path is None:
        return 0
    p = Path(path)
    try:
        return p.stat().st_size if p.exists() else 0
    except OSError:
        return 0


def transfer_time_sec(num_bytes: int | float, bandwidth_gbps: float) -> float:
    if num_bytes <= 0:
        return 0.0
    return float(num_bytes) / (max(float(bandwidth_gbps), 1e-12) * 1e9)


def transfer_energy_j(num_bytes: int | float, energy_pj_per_bit: float) -> float:
    if num_bytes <= 0:
        return 0.0
    return float(num_bytes) * 8.0 * max(float(energy_pj_per_bit), 0.0) * 1e-12


def run_transfer(args: argparse.Namespace, out_dir: Path, stages: list[StageTiming]) -> StageTiming:
    breakdown: dict[str, dict[str, float | int]] = {}
    energy_j: float | None = None
    if args.transfer_mode == "none":
        latency = 0.0
        notes = "Transfer disabled by --transfer-mode none."
        command: list[str] | str = "transfer disabled"
    elif args.transfer_mode == "manual":
        latency = float(args.transfer_sec)
        notes = "Manual transfer latency from --transfer-sec."
        command = f"manual transfer-sec={latency}"
    else:
        genomic_query_bytes = 0
        proteomic_query_bytes = 0
        if not args.skip_genomic:
            genomic_query_bytes = safe_size(args.genomic_query_fastq)
        if not args.skip_proteomic:
            proteomic_query_bytes = safe_size(args.proteomic_query)
        query_bytes = genomic_query_bytes + proteomic_query_bytes
        output_bytes = sum(safe_size(stage.output_path) for stage in stages)
        feature_bytes = max(0, int(args.moe_feature_bytes))
        genomic_reference_bytes = 0
        proteomic_reference_bytes = 0
        if args.transfer_include_reference:
            if not args.skip_genomic:
                genomic_reference_bytes = safe_size(args.genomic_ref_fasta)
            if not args.skip_proteomic:
                proteomic_reference_bytes = safe_size(args.proteomic_ref)
        reference_bytes = genomic_reference_bytes + proteomic_reference_bytes
        total_bytes = query_bytes + output_bytes + feature_bytes + reference_bytes

        if args.transfer_model == "simple":
            latency = transfer_time_sec(total_bytes, args.package_bandwidth_gbps)
            breakdown = {
                "simple_package_transfer": {
                    "bytes": total_bytes,
                    "bandwidth_GBps": float(args.package_bandwidth_gbps),
                    "latency_sec": latency,
                }
            }
            command = "simple transfer = bytes / package_bandwidth"
        else:
            # Segment model follows the bandwidth table used in the paper draft.
            # Independent links are modeled as additive package-level transfer costs.
            breakdown = {
                "host_to_multiram_package": {
                    "bytes": query_bytes,
                    "bandwidth_GBps": float(args.host_package_gbps),
                    "latency_sec": transfer_time_sec(query_bytes, args.host_package_gbps),
                },
                "fenand_compressed_input_stream": {
                    "bytes": reference_bytes,
                    "bandwidth_GBps": float(args.fenand_compressed_gbps),
                    "latency_sec": transfer_time_sec(reference_bytes, args.fenand_compressed_gbps),
                },
                "fenand_decompressed_output_stream": {
                    "bytes": reference_bytes,
                    "bandwidth_GBps": float(args.fenand_decompressed_gbps),
                    "latency_sec": transfer_time_sec(reference_bytes, args.fenand_decompressed_gbps),
                },
                "feram_oms_external_link": {
                    "bytes": proteomic_query_bytes,
                    "bandwidth_GBps": float(args.feram_external_gbps),
                    "latency_sec": transfer_time_sec(proteomic_query_bytes, args.feram_external_gbps),
                },
                "dram_external_interface": {
                    "bytes": genomic_query_bytes + feature_bytes,
                    "bandwidth_GBps": float(args.dram_external_gbps),
                    "latency_sec": transfer_time_sec(genomic_query_bytes + feature_bytes, args.dram_external_gbps),
                },
                "dram_internal_pnm": {
                    "bytes": genomic_query_bytes + feature_bytes,
                    "bandwidth_GBps": float(args.dram_internal_pnm_gbps),
                    "latency_sec": transfer_time_sec(genomic_query_bytes + feature_bytes, args.dram_internal_pnm_gbps),
                },
                "dram_on_chip_ring": {
                    "bytes": feature_bytes + output_bytes,
                    "bandwidth_GBps": float(args.dram_ring_gbps),
                    "latency_sec": transfer_time_sec(feature_bytes + output_bytes, args.dram_ring_gbps),
                },
                "chiplet_to_chiplet_ucie": {
                    "bytes": feature_bytes + output_bytes,
                    "bandwidth_GBps": float(args.ucie_gbps),
                    "latency_sec": transfer_time_sec(feature_bytes + output_bytes, args.ucie_gbps),
                },
            }
            latency = sum(float(item["latency_sec"]) for item in breakdown.values())
            command = "multiram segmented transfer = sum(bytes_i / bandwidth_i)"

        for item in breakdown.values():
            item["energy_j"] = transfer_energy_j(
                item["bytes"], args.package_link_energy_pj_per_bit
            )
        energy_j = sum(float(item["energy_j"]) for item in breakdown.values())

        notes = (
            f"model={args.transfer_model}; total_bytes={total_bytes}; "
            f"genomic_query_bytes={genomic_query_bytes}; "
            f"proteomic_query_bytes={proteomic_query_bytes}; "
            f"output_bytes={output_bytes}; feature_bytes={feature_bytes}; "
            f"reference_bytes={reference_bytes}; "
            f"assumed_package_link_energy_pj_per_bit="
            f"{args.package_link_energy_pj_per_bit}"
        )

    summary = {
        "mode": args.transfer_mode,
        "model": args.transfer_model,
        "latency_sec": latency,
        "energy_j": energy_j,
        "energy_model": {
            "package_link_energy_pj_per_bit": float(
                args.package_link_energy_pj_per_bit
            ),
            "package_link_energy_pj_per_byte": float(
                args.package_link_energy_pj_per_bit * 8.0
            ),
            "provenance": "assumed package-link energy; not measured",
        },
        "notes": notes,
        "breakdown": breakdown,
    }
    summary_path = out_dir / "transfer_model_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return StageTiming(
        stage="package_level_transfer",
        command=command,
        status="ok",
        latency_sec=latency,
        wall_time_sec=0.0,
        output_path=str(summary_path),
        log_path=None,
        notes=notes,
    )


def write_reports(stages: list[StageTiming], args: argparse.Namespace, out_dir: Path) -> None:
    stage_sum = sum(s.latency_sec for s in stages if s.latency_sec is not None)
    total = stage_sum
    payload = {
        "definition": (
            "T_total = T_genomic + T_proteomic_clustering + T_proteomic_OMS "
            "+ T_transfer + T_MoE."
        ),
        "stage_latency_sum_sec": stage_sum,
        "total_pim_pipeline_time_sec": total,
        "stages": [asdict(s) for s in stages],
    }
    json_path = out_dir / "pim_full_pipeline_timing_summary.json"
    md_path = out_dir / "pim_full_pipeline_timing_summary.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with md_path.open("w", encoding="utf-8") as fh:
        fh.write("# PIM Full Pipeline Timing Summary\n\n")
        fh.write("| Stage | Status | Latency (s) | Runner wall time (s) |\n")
        fh.write("|---|---|---:|---:|\n")
        for stage in stages:
            latency = "NA" if stage.latency_sec is None else f"{stage.latency_sec:.6f}"
            fh.write(
                f"| {stage.stage} | {stage.status} | {latency} | "
                f"{stage.wall_time_sec:.3f} |\n"
            )
        fh.write(f"\nTotal PIM pipeline time: `{total:.6f}` s\n")

    print(json.dumps(payload, indent=2))
    print(f"\nWrote:\n  {json_path}\n  {md_path}")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stages: list[StageTiming] = []
    if not args.skip_genomic:
        stages.append(run_genomic(args, out_dir))
    if not args.skip_proteomic and not args.skip_proteomic_clustering:
        stages.append(run_proteomic_clustering(args, out_dir))
    if not args.skip_proteomic and not args.skip_proteomic_oms:
        stages.append(run_proteomic_oms(args, out_dir))
    if not args.skip_transfer and args.transfer_mode != "none":
        stages.append(run_transfer(args, out_dir, stages))
    if not args.skip_moe:
        stages.append(run_moe(args, out_dir))

    write_reports(stages, args, out_dir)
    return 0 if all(stage.status == "ok" for stage in stages) else 1


if __name__ == "__main__":
    raise SystemExit(main())
