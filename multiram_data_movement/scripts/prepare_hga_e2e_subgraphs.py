#!/usr/bin/env python3
"""Prepare truth-labeled MHC subgraphs and reads for an end-to-end HGA mapper."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path("/home/tsl012/multiomic")
DEFAULT_REFERENCE = ROOT / "GenDP/mhc_test/mhc_ref.fa"
DEFAULT_OUTPUT = (
    ROOT
    / "multiram_data_movement/results_hga_e2e_subgraph_comparison/inputs"
)
MASON = Path("/home/tsl012/miniconda3/bin/mason_simulator")
DEFAULT_STARTS = (400_000, 1_500_000, 2_700_000, 4_000_000)


def read_first_fasta(path: Path) -> str:
    sequence = []
    active = False
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith(">"):
            if active:
                break
            active = True
        elif active:
            sequence.append(line.strip().upper())
    result = "".join(sequence)
    if not result:
        raise ValueError(f"No FASTA sequence found in {path}")
    return result


def write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="ascii") as handle:
        for name, sequence in records:
            handle.write(f">{name}\n")
            for offset in range(0, len(sequence), 80):
                handle.write(sequence[offset : offset + 80] + "\n")


def write_linear_hga_graph(path: Path, sequence: str) -> None:
    vertices = len(sequence)
    edges = vertices - 1
    with path.open("w", encoding="ascii") as handle:
        handle.write(f"{vertices} {edges}\n")
        for value in range(edges):
            handle.write(f"{value}\n")
        handle.write("0\n0\n")
        for value in range(1, vertices):
            handle.write(f"{value}\n")
        for value in range(1, vertices):
            handle.write(f"{value}\n")
        for value in range(vertices):
            handle.write(f"{value}\n")
        handle.write(f"{edges}\n")
        for base in sequence:
            handle.write((base if base in "ACGT" else "N") + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_fastq(path: Path) -> int:
    lines = sum(1 for _ in path.open("rb"))
    if lines % 4:
        raise ValueError(f"Malformed FASTQ line count in {path}")
    return lines // 4


def count_primary_sam(path: Path) -> int:
    count = 0
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            if line.startswith("@"):
                continue
            fields = line.split("\t")
            flag = int(fields[1])
            if not flag & (0x100 | 0x800 | 0x4):
                count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--subgraph-length", type=int, default=25_000)
    parser.add_argument("--reads", type=int, default=4_000)
    parser.add_argument("--read-length", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.read_length > 120:
        raise ValueError("HGA optimized 8-bit DP requires reads of at most 120 bp")
    if not MASON.exists():
        raise FileNotFoundError(MASON)

    sequence = read_first_fasta(args.reference)
    records: list[tuple[str, str]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    graph_dir = args.output_dir / "hga_graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    subgraphs = []
    for index, start in enumerate(DEFAULT_STARTS):
        end = start + args.subgraph_length
        if end > len(sequence):
            raise ValueError(f"Subgraph {index} exceeds reference length")
        name = f"mhc_sg{index}_source_{start}_{end}"
        subsequence = sequence[start:end]
        records.append((name, subsequence))
        graph_path = graph_dir / f"{name}.hga"
        write_linear_hga_graph(graph_path, subsequence)
        subgraphs.append(
            {
                "name": name,
                "source_start": start,
                "source_end": end,
                "length": len(subsequence),
                "sequence_sha256": hashlib.sha256(
                    subsequence.encode("ascii")
                ).hexdigest(),
                "hga_graph": str(graph_path.resolve()),
                "hga_graph_bytes": graph_path.stat().st_size,
            }
        )

    combined_fasta = args.output_dir / "mhc_subgraphs.fa"
    reads_fastq = args.output_dir / "reads.fq"
    truth_sam = args.output_dir / "truth.sam"
    write_fasta(combined_fasta, records)

    command = [
        str(MASON),
        "-ir",
        str(combined_fasta),
        "-n",
        str(args.reads),
        "-o",
        str(reads_fastq),
        "-oa",
        str(truth_sam),
        "--illumina-read-length",
        str(args.read_length),
        "--illumina-prob-mismatch",
        "0.03",
        "--illumina-prob-insert",
        "0.01",
        "--illumina-prob-deletion",
        "0.01",
        "--seed",
        str(args.seed),
        "--embed-read-info",
        "--num-threads",
        "1",
    ]
    completed = subprocess.run(
        command, text=True, capture_output=True, check=False
    )
    (args.output_dir / "mason.stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    (args.output_dir / "mason.stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Mason failed with exit code {completed.returncode}: "
            f"{completed.stderr[-1000:]}"
        )

    read_count = count_fastq(reads_fastq)
    truth_count = count_primary_sam(truth_sam)
    if read_count != args.reads or truth_count != args.reads:
        raise RuntimeError(
            f"Expected {args.reads} reads/truths, got {read_count}/{truth_count}"
        )

    manifest = {
        "status": "prepared_truth_labeled_workload",
        "source_reference": str(args.reference.resolve()),
        "combined_subgraph_fasta": str(combined_fasta.resolve()),
        "reads_fastq": str(reads_fastq.resolve()),
        "truth_sam": str(truth_sam.resolve()),
        "read_count": read_count,
        "read_length": args.read_length,
        "seed": args.seed,
        "error_model": {
            "mismatch_probability": 0.03,
            "insertion_probability": 0.01,
            "deletion_probability": 0.01,
        },
        "subgraphs": subgraphs,
        "files": {
            "combined_fasta_bytes": combined_fasta.stat().st_size,
            "reads_fastq_bytes": reads_fastq.stat().st_size,
            "truth_sam_bytes": truth_sam.stat().st_size,
            "combined_fasta_sha256": sha256(combined_fasta),
            "reads_fastq_sha256": sha256(reads_fastq),
            "truth_sam_sha256": sha256(truth_sam),
        },
        "mason_command": command,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(manifest_path)
    print(json.dumps({"reads": read_count, "subgraphs": len(subgraphs)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
