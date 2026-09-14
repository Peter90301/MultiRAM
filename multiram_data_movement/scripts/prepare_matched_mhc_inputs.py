#!/usr/bin/env python3
"""Prepare and validate one logical MHC workload for minigraph, HGA, and PIM."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path("/home/tsl012/multiomic")
DEFAULT_REFERENCE = ROOT / "GenDP/mhc_test/mhc_ref.fa"
DEFAULT_FASTQ = ROOT / "GenDP/mhc_test/query_reads.fq"
DEFAULT_HGA_GRAPH = ROOT / "genomic_compare_hga_1000/hga_inputs/linear_graph.hga"
DEFAULT_OUTPUT = (
    ROOT / "multiram_data_movement/results_matched_mhc_minigraph_hga_pim/inputs"
)


def fasta_sequence(path: Path) -> str:
    chunks: list[str] = []
    in_first = False
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if in_first:
                break
            in_first = True
        elif in_first:
            chunks.append(line.upper())
    sequence = "".join(chunks)
    if not sequence:
        raise ValueError(f"No sequence in {path}")
    return sequence


def iter_fastq(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        while True:
            header = handle.readline()
            if not header:
                return
            sequence = handle.readline().strip().upper()
            plus = handle.readline()
            quality = handle.readline().strip()
            if not plus or len(sequence) != len(quality):
                raise ValueError(f"Malformed FASTQ record near {header.strip()} in {path}")
            yield header.strip()[1:].split()[0], sequence


def digest_sequences(sequences) -> tuple[int, int, Counter[int], str]:
    count = 0
    bases = 0
    lengths: Counter[int] = Counter()
    digest = hashlib.sha256()
    for sequence in sequences:
        count += 1
        bases += len(sequence)
        lengths[len(sequence)] += 1
        digest.update(sequence.encode("ascii"))
        digest.update(b"\n")
    return count, bases, lengths, digest.hexdigest()


def iter_hga_sequences(path: Path):
    with path.open("r", encoding="ascii") as handle:
        while True:
            header = handle.readline()
            if not header:
                return
            sequence = handle.readline().strip().upper()
            if not sequence:
                raise ValueError(f"Missing sequence after {header.strip()} in {path}")
            yield sequence


def hga_graph_digest(path: Path) -> tuple[int, int, str]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        num_vertices, num_edges = map(int, handle.readline().split())
        values_to_skip = num_edges + num_vertices + 1 + num_edges + num_vertices + 1
        for _ in range(values_to_skip):
            if not handle.readline():
                raise ValueError(f"Truncated HGA graph: {path}")
        digest = hashlib.sha256()
        labels = 0
        for raw in handle:
            label = raw.strip().upper()
            if label not in {"A", "T", "C", "G", "N"}:
                raise ValueError(f"Invalid graph label {label!r} in {path}")
            digest.update(label.encode("ascii"))
            labels += 1
    if labels != num_vertices:
        raise ValueError(f"Expected {num_vertices} labels, found {labels}")
    return num_vertices, num_edges, digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--fastq", type=Path, default=DEFAULT_FASTQ)
    parser.add_argument("--hga-graph", type=Path, default=DEFAULT_HGA_GRAPH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    for path in (args.reference, args.fastq, args.hga_graph):
        if not path.exists():
            raise FileNotFoundError(path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hga_reads = args.output_dir / "matched_mhc_reads.hga"

    reference = fasta_sequence(args.reference)
    reference_sha256 = hashlib.sha256(reference.encode("ascii")).hexdigest()
    num_vertices, num_edges, graph_sha256 = hga_graph_digest(args.hga_graph)

    fastq_records = list(iter_fastq(args.fastq))
    read_count, total_query_bases, lengths, read_sha256 = digest_sequences(
        sequence for _, sequence in fastq_records
    )
    with hga_reads.open("w", encoding="ascii") as output:
        for index, (_, sequence) in enumerate(fastq_records):
            clean = "".join(base if base in "ATCG" else "N" for base in sequence)
            output.write(f"read_{index}\n{clean}\n")
    hga_count, hga_bases, hga_lengths, hga_sha256 = digest_sequences(
        iter_hga_sequences(hga_reads)
    )

    graph_match = (
        num_vertices == len(reference) and graph_sha256 == reference_sha256
    )
    reads_match = (
        hga_count == read_count
        and hga_bases == total_query_bases
        and hga_lengths == lengths
        and hga_sha256 == read_sha256
    )
    manifest = {
        "logical_workload": "linear MHC graph plus single-end simulated MHC reads",
        "reference_fasta": str(args.reference.resolve()),
        "reference_header": args.reference.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()[0],
        "reference_vertices": len(reference),
        "reference_sequence_sha256": reference_sha256,
        "reference_file_bytes": args.reference.stat().st_size,
        "query_fastq": str(args.fastq.resolve()),
        "query_count": read_count,
        "query_total_bases": total_query_bases,
        "query_length_histogram": {
            str(length): count for length, count in sorted(lengths.items())
        },
        "query_sequence_sha256": read_sha256,
        "query_fastq_bytes": args.fastq.stat().st_size,
        "hga_graph": str(args.hga_graph.resolve()),
        "hga_graph_vertices": num_vertices,
        "hga_graph_edges": num_edges,
        "hga_graph_label_sha256": graph_sha256,
        "hga_graph_file_bytes": args.hga_graph.stat().st_size,
        "hga_reads": str(hga_reads.resolve()),
        "hga_read_count": hga_count,
        "hga_read_sequence_sha256": hga_sha256,
        "hga_reads_file_bytes": hga_reads.stat().st_size,
        "logical_graph_payload_bytes": len(reference),
        "logical_query_payload_bytes": total_query_bases,
        "hga_graph_representation_expansion": (
            args.hga_graph.stat().st_size / len(reference)
        ),
        "validation": {
            "graph_sequences_identical": graph_match,
            "query_sequences_identical": reads_match,
        },
    }
    if not graph_match or not reads_match:
        raise RuntimeError(f"Matched-input validation failed: {manifest['validation']}")

    manifest_path = args.output_dir / "matched_mhc_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)
    print(json.dumps(manifest["validation"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
