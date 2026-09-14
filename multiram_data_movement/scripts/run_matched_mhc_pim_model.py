#!/usr/bin/env python3
"""Run the matched MHC 3D-DRAM PIM model and quantify modeled movement."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from run_data_movement_time_experiment import parse_time_v


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = (
    ROOT / "multiram_data_movement/results_matched_mhc_minigraph_hga_pim"
)
DEFAULT_MANIFEST = DEFAULT_RESULTS / "inputs/matched_mhc_manifest.json"
GENDRAM_DIR = ROOT / "GenDP/GenDRAM"
PIM_SCRIPT = GENDRAM_DIR / "main_analysis.py"
FENAND_DECOMPRESSED_GBPS = 8.1
DRAM_INTERNAL_LOW_TBPS = 19.01
DRAM_INTERNAL_HIGH_TBPS = 30.34


def metric(text: str, label: str) -> float:
    match = re.search(rf"{re.escape(label)}\s*:\s*([0-9.eE+-]+)", text)
    if not match:
        raise ValueError(f"Missing metric {label!r}")
    return float(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not all(manifest["validation"].values()):
        raise RuntimeError("Matched input validation did not pass")
    reference = Path(manifest["reference_fasta"])
    query_fastq = Path(manifest["query_fastq"])
    query_count = int(manifest["query_count"])

    run_dir = args.output_dir / "raw/pim_model"
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    time_path = run_dir / "time_v.log"
    python_setup = (
        "import random,runpy;"
        f"random.seed({args.seed});"
        f"runpy.run_path({str(PIM_SCRIPT)!r},run_name='__main__')"
    )
    command = [
        sys.executable,
        "-c",
        python_setup,
        "--ref-fasta",
        str(reference),
        "--query-fastq",
        str(query_fastq),
        "--max-queries",
        str(query_count),
    ]
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(args.seed)
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.run(
            ["/usr/bin/time", "-v", "-o", str(time_path), *command],
            cwd=GENDRAM_DIR,
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )
    runner_wall_s = time.perf_counter() - started
    if process.returncode != 0:
        raise RuntimeError(f"PIM model failed; see {stderr_path}")

    output = stdout_path.read_text(encoding="utf-8", errors="replace")
    queries_aligned = int(metric(output, "Queries aligned"))
    avg_anchors = metric(output, "Avg anchors/query")
    avg_chain_anchors = metric(output, "Avg chain anchors/query")
    avg_dp_cells = metric(output, "Avg DP cells/query")
    avg_edit_distance = metric(output, "Avg edit distance")
    avg_node_accesses = metric(output, "Avg node accesses/query")
    avg_tier_read_ns = metric(output, "Avg tier read latency")
    pim_latency_s = metric(output, "Total Latency (Pipelined)") / 1000.0
    avg_power_w = metric(output, "Average Power")

    alignment_read_bytes_per_query = max(4096.0, avg_dp_cells * 0.5)
    internal_bytes_per_query = (
        avg_node_accesses * 1024.0
        + avg_anchors * 64.0
        + alignment_read_bytes_per_query
    )
    internal_read_bytes = queries_aligned * internal_bytes_per_query
    reference_bytes = int(manifest["reference_file_bytes"])
    query_bytes = int(manifest["query_fastq_bytes"])
    ingress_bytes = reference_bytes + query_bytes
    ingress_s = ingress_bytes / (FENAND_DECOMPRESSED_GBPS * 1e9)
    internal_service_low_s = internal_read_bytes / (DRAM_INTERNAL_LOW_TBPS * 1e12)
    internal_service_high_s = internal_read_bytes / (DRAM_INTERNAL_HIGH_TBPS * 1e12)

    record = {
        "status": "modeled_not_hardware_measurement",
        "command": command,
        "seed": args.seed,
        "runner_wall_s": runner_wall_s,
        "runner_resource_usage": parse_time_v(time_path),
        "queries_aligned": queries_aligned,
        "avg_anchors_per_query": avg_anchors,
        "avg_chain_anchors_per_query": avg_chain_anchors,
        "avg_dp_cells_per_query": avg_dp_cells,
        "avg_edit_distance": avg_edit_distance,
        "avg_node_accesses_per_query": avg_node_accesses,
        "avg_tier_read_latency_ns": avg_tier_read_ns,
        "pim_pipeline_latency_s": pim_latency_s,
        "pim_average_power_w": avg_power_w,
        "pim_energy_j": pim_latency_s * avg_power_w,
        "movement": {
            "reference_stream_bytes": reference_bytes,
            "query_stream_bytes": query_bytes,
            "cold_external_ingress_bytes": ingress_bytes,
            "fenand_decompressed_bandwidth_gbps": FENAND_DECOMPRESSED_GBPS,
            "cold_external_ingress_s": ingress_s,
            "internal_read_bytes_per_query": internal_bytes_per_query,
            "internal_3ddram_read_bytes": internal_read_bytes,
            "internal_pnm_bandwidth_tbps": [
                DRAM_INTERNAL_LOW_TBPS,
                DRAM_INTERNAL_HIGH_TBPS,
            ],
            "internal_service_equivalent_s_at_19_01_tbps": internal_service_low_s,
            "internal_service_equivalent_s_at_30_34_tbps": internal_service_high_s,
        },
        "modeled_resident_total_s": pim_latency_s,
        "modeled_cold_total_s": pim_latency_s + ingress_s,
        "modeled_external_movement_fraction_cold_percent": (
            100.0 * ingress_s / (pim_latency_s + ingress_s)
        ),
        "interpretation": (
            "PIM model latency already includes modeled compute and internal-memory "
            "effects. Internal bandwidth service equivalents are diagnostic and are "
            "not added again to total latency."
        ),
    }
    output_path = args.output_dir / "pim_model_result.json"
    output_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(output_path)
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
