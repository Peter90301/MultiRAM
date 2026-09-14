#!/usr/bin/env python3
"""Summarize Platinum Genomes NA12878 genomics data movement.

The script deliberately separates measured facts from modeled lower bounds.
Measured facts come from the ENA manifest and local file sizes. Lower bounds
are used where full decompression or hardware counters are unavailable.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def parse_mem_total_bytes() -> int | None:
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"^MemTotal:\s+(\d+)\s+kB", text, re.MULTILINE)
    if not match:
        return None
    return int(match.group(1)) * 1024


def bytes_or_zero(path: str) -> int:
    if not path:
        return 0
    p = Path(path)
    return p.stat().st_size if p.exists() else 0


def fmt_bytes(value: int | float) -> str:
    return f"{value:,.0f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--reference", required=True, help="Reference FASTA or minimap2 .mmi used by the run.")
    parser.add_argument("--pair-timing", default="", help="Optional pair_timing.tsv from genomics_minimap2_pairs.sh.")
    parser.add_argument("--config", default="")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    rows = read_manifest(Path(args.manifest))
    fastq_compressed_bytes = sum(sum(int(x) for x in row["fastq_bytes"].split(";")) for row in rows)
    read_count = sum(int(row["read_count"]) for row in rows)
    base_count = sum(int(row["base_count"]) for row in rows)
    fastq_records = read_count * 2
    reference_bytes = bytes_or_zero(args.reference)
    all_paired = all(row["library_layout"] == "PAIRED" and len(row["fastq_ftp"].split(";")) == 2 for row in rows)
    all_na12878 = all("NA12878" in row["sample_title"] for row in rows)
    any_na12877 = any("NA12877" in row["sample_title"] for row in rows)

    # FASTQ lower bound: sequence + quality + minimal FASTQ syntax overhead.
    # Each record needs at least: header "@\\n" (2), sequence newline (1),
    # plus line "+\\n" (2), quality newline (1) = 6 bytes, excluding the
    # real read name text. This is a strict lower bound.
    decompressed_fastq_lower_bound = (2 * base_count) + (6 * fastq_records)
    host_dram_lower_bound = decompressed_fastq_lower_bound + reference_bytes
    cpu_storage_read_lower_bound = fastq_compressed_bytes + reference_bytes
    cpu_total_movement_lower_bound = cpu_storage_read_lower_bound + host_dram_lower_bound

    # GPU lower bound assumes a conventional pipeline must read compressed
    # FASTQ from storage, materialize reads/reference in host memory, and move
    # at least reads/reference once across the host-device boundary.
    gpu_h2d_lower_bound = decompressed_fastq_lower_bound + reference_bytes
    gpu_total_movement_lower_bound = (
        cpu_storage_read_lower_bound + host_dram_lower_bound + gpu_h2d_lower_bound
    )

    # MultiRAM model: keep the compressed dataset in-package and avoid host DRAM
    # and PCIe/H2D movement of expanded reads. Include one reference movement if
    # the reference is supplied as an external file/index.
    multiram_movement_bytes = fastq_compressed_bytes + reference_bytes
    reduction_vs_cpu = (1.0 - multiram_movement_bytes / cpu_total_movement_lower_bound) * 100.0
    reduction_vs_gpu = (1.0 - multiram_movement_bytes / gpu_total_movement_lower_bound) * 100.0

    run_runtime_s = ""
    pair_count_completed = 0
    pair_runtime_sum_s = 0.0
    pair_input_completed_bytes = 0
    if args.pair_timing and Path(args.pair_timing).exists():
        with Path(args.pair_timing).open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                pair_count_completed += 1
                pair_runtime_sum_s += float(row.get("runtime_s") or 0.0)
                pair_input_completed_bytes += int(row.get("input_bytes") or 0)
        run_runtime_s = f"{pair_runtime_sum_s:.6f}"

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_rows = [
        {
            "workload": "genomics_alignment",
            "dataset": "Platinum_Genomes_NA12878_PRJEB3246",
            "reference": args.reference,
            "runs": len(rows),
            "fastq_files": len(rows) * 2,
            "all_paired": all_paired,
            "all_na12878": all_na12878,
            "any_na12877": any_na12877,
            "compressed_fastq_bytes": fastq_compressed_bytes,
            "reference_bytes": reference_bytes,
            "read_count": read_count,
            "base_count": base_count,
            "fastq_records": fastq_records,
            "decompressed_fastq_lower_bound_bytes": decompressed_fastq_lower_bound,
            "host_dram_lower_bound_bytes": host_dram_lower_bound,
            "cpu_storage_read_lower_bound_bytes": cpu_storage_read_lower_bound,
            "cpu_total_movement_lower_bound_bytes": cpu_total_movement_lower_bound,
            "gpu_h2d_lower_bound_bytes": gpu_h2d_lower_bound,
            "gpu_total_movement_lower_bound_bytes": gpu_total_movement_lower_bound,
            "multiram_movement_bytes": multiram_movement_bytes,
            "multiram_reduction_vs_cpu_percent": reduction_vs_cpu,
            "multiram_reduction_vs_gpu_percent": reduction_vs_gpu,
            "pair_count_completed": pair_count_completed,
            "pair_input_completed_bytes": pair_input_completed_bytes,
            "pair_runtime_sum_s": run_runtime_s,
        }
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    mem_total = parse_mem_total_bytes()
    cfg_text = ""
    if args.config:
        cfg_text = f"- Config: `{args.config}`\n"
    md = f"""# Platinum Genomes NA12878 Genomics Data Movement

## Dataset

- Source manifest: `{args.manifest}`
{cfg_text}- Reference/index: `{args.reference}`
- NA12878 paired runs: {len(rows)}
- FASTQ files: {len(rows) * 2}
- All rows paired-end: {all_paired}
- All rows contain NA12878: {all_na12878}
- Any row contains NA12877: {any_na12877}
- Compressed FASTQ bytes: {fmt_bytes(fastq_compressed_bytes)}
- Reference/index bytes: {fmt_bytes(reference_bytes)}
- ENA read count sum: {fmt_bytes(read_count)}
- ENA base count sum: {fmt_bytes(base_count)}
- FASTQ records, paired files combined: {fmt_bytes(fastq_records)}

## Storage-Bound Regime Check

- Host DRAM bytes: {fmt_bytes(mem_total) if mem_total else "unknown"}
- FASTQ decompressed lower bound bytes: {fmt_bytes(decompressed_fastq_lower_bound)}
- Lower-bound decompressed FASTQ / host DRAM: {(decompressed_fastq_lower_bound / mem_total):.3f}x
  """ + ("" if mem_total else "  (host DRAM unavailable)\n") + f"""

The lower bound includes only sequence bases, quality characters, and minimal FASTQ syntax. Real decompressed FASTQ is larger because read identifiers are not counted in this lower bound.

## Data Movement Lower Bounds

| Baseline | Components counted | Bytes |
| --- | --- | ---: |
| CPU storage read | compressed FASTQ + reference/index | {fmt_bytes(cpu_storage_read_lower_bound)} |
| CPU host DRAM | decompressed FASTQ lower bound + reference/index | {fmt_bytes(host_dram_lower_bound)} |
| CPU total movement lower bound | storage + host DRAM | {fmt_bytes(cpu_total_movement_lower_bound)} |
| GPU H2D lower bound | decompressed reads + reference/index copied to GPU once | {fmt_bytes(gpu_h2d_lower_bound)} |
| GPU total movement lower bound | CPU movement + H2D | {fmt_bytes(gpu_total_movement_lower_bound)} |
| MultiRAM modeled movement | compressed FASTQ stream + reference/index | {fmt_bytes(multiram_movement_bytes)} |

## MultiRAM Reduction

- Reduction vs CPU lower bound: {reduction_vs_cpu:.3f}%
- Reduction vs GPU lower bound: {reduction_vs_gpu:.3f}%

These are conservative lower-bound reductions because the CPU/GPU decompressed FASTQ byte count excludes real read identifier text, repeated cache-line movement, writebacks, and intermediate data structures.

## minimap2 Run Status

- Pair runs completed in `pair_timing.tsv`: {pair_count_completed}
- Completed pair input bytes: {fmt_bytes(pair_input_completed_bytes)}
- Sum of completed per-pair runtimes: {run_runtime_s or "not available"} s

## Notes

- Hardware performance counters were blocked on this machine by Linux `perf_event_paranoid=4`, so cache-miss based DRAM traffic is not available from `perf`.
- The genomics experiment is valid for Platinum Genomes NA12878. Proteomics clustering and OMS require separate proteomics/MS datasets; Platinum Genomes FASTQ cannot be used for those stages.
"""
    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")
    print(json.dumps(csv_rows[0], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
