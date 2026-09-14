#!/usr/bin/env python3
"""Prepare fixed genomics subsets and HGA text inputs for movement profiling."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


DEFAULT_R1 = Path(
    "/mnt/hdd/tsunghan/raw-ms-dataset/genomic_large_Dset/ERR174324_1.fastq.gz"
)
DEFAULT_R2 = Path(
    "/mnt/hdd/tsunghan/raw-ms-dataset/genomic_large_Dset/ERR174324_2.fastq.gz"
)
DEFAULT_OUTPUT = Path(
    "/mnt/hdd/tsunghan/raw-ms-dataset/genomic_large_Dset/"
    "profiling_subsets/na12878_100k_pairs"
)
DEFAULT_HGA_GRAPH = Path(
    "/home/tsl012/multiomic/genomic_compare_hga_1000/hga_inputs/linear_graph.hga"
)


def copy_fastq_records(source: Path, destination: Path, records: int) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    emitted = 0
    with gzip.open(source, "rt", encoding="utf-8", errors="replace") as src, \
            destination.open("w", encoding="utf-8") as dst:
        while emitted < records:
            record = [src.readline() for _ in range(4)]
            if not record[0]:
                break
            if any(line == "" for line in record):
                raise RuntimeError(f"Truncated FASTQ record in {source}")
            dst.writelines(record)
            emitted += 1
    return emitted


def fastq_to_hga_reads(source: Path, destination: Path) -> tuple[int, int]:
    count = 0
    max_read_len = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open(encoding="utf-8", errors="replace") as src, \
            destination.open("w", encoding="utf-8") as dst:
        while True:
            header = src.readline()
            if not header:
                break
            sequence = src.readline().strip().upper()
            plus = src.readline()
            quality = src.readline()
            if not plus or not quality:
                raise RuntimeError(f"Truncated FASTQ record in {source}")
            clean = "".join(base if base in "ATCG" else "N" for base in sequence)
            dst.write(f"{header.rstrip()}\n{clean}\n")
            count += 1
            max_read_len = max(max_read_len, len(clean))
    return count, max_read_len


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1", type=Path, default=DEFAULT_R1)
    parser.add_argument("--r2", type=Path, default=DEFAULT_R2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--records", type=int, default=100_000)
    parser.add_argument("--hga-graph", type=Path, default=DEFAULT_HGA_GRAPH)
    args = parser.parse_args()

    for source in (args.r1, args.r2, args.hga_graph):
        if not source.exists():
            raise FileNotFoundError(source)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    r1_out = args.output_dir / "NA12878_100k_R1.fastq"
    r2_out = args.output_dir / "NA12878_100k_R2.fastq"
    hga_reads = args.output_dir / "NA12878_100k_R1.hga"

    r1_count = copy_fastq_records(args.r1, r1_out, args.records)
    r2_count = copy_fastq_records(args.r2, r2_out, args.records)
    if r1_count != args.records or r2_count != args.records:
        raise RuntimeError(
            f"Requested {args.records} pairs, emitted R1={r1_count}, R2={r2_count}"
        )
    hga_count, max_read_len = fastq_to_hga_reads(r1_out, hga_reads)

    manifest = {
        "source_r1": str(args.r1),
        "source_r2": str(args.r2),
        "records_per_mate": args.records,
        "r1_fastq": str(r1_out),
        "r2_fastq": str(r2_out),
        "r1_bytes": r1_out.stat().st_size,
        "r2_bytes": r2_out.stat().st_size,
        "hga_graph": str(args.hga_graph),
        "hga_graph_bytes": args.hga_graph.stat().st_size,
        "hga_reads": str(hga_reads),
        "hga_reads_bytes": hga_reads.stat().st_size,
        "hga_read_count": hga_count,
        "hga_max_read_len": max_read_len,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

