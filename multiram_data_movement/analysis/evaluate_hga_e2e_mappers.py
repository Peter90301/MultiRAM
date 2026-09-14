#!/usr/bin/env python3
"""Evaluate minigraph GAF and HGA mapper PAF against Mason SAM truth."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


CIGAR_RE = re.compile(r"(\d+)([MIDNSHP=X])")


def parse_truth(path: Path) -> dict[str, dict[str, int | str]]:
    truth = {}
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            if line.startswith("@"):
                continue
            fields = line.rstrip().split("\t")
            flag = int(fields[1])
            if flag & (0x4 | 0x100 | 0x800):
                continue
            start = int(fields[3]) - 1
            reference_span = sum(
                int(length)
                for length, op in CIGAR_RE.findall(fields[5])
                if op in "MDN=X"
            )
            truth[fields[0]] = {
                "target": fields[2],
                "start": start,
                "end": start + reference_span,
                "strand": "-" if flag & 0x10 else "+",
            }
    return truth


def parse_mapping(path: Path) -> dict[str, dict[str, int | str]]:
    mappings = {}
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            fields = line.rstrip().split("\t")
            if len(fields) < 12:
                continue
            query = fields[0]
            record = {
                "strand": fields[4],
                "target": fields[5].lstrip("><"),
                "start": int(fields[7]),
                "end": int(fields[8]),
                "mapq": int(fields[11]),
            }
            if query not in mappings or int(record["mapq"]) > int(
                mappings[query]["mapq"]
            ):
                mappings[query] = record
    return mappings


def evaluate(
    truth: dict[str, dict[str, int | str]],
    mappings: dict[str, dict[str, int | str]],
    tolerance: int,
) -> dict[str, float | int]:
    target_correct = 0
    endpoint_correct = 0
    strand_correct = 0
    endpoint_errors = []
    for query, expected in truth.items():
        observed = mappings.get(query)
        if observed is None:
            continue
        same_target = observed["target"] == expected["target"]
        if same_target:
            target_correct += 1
            endpoint_error = abs(int(observed["end"]) - int(expected["end"]))
            endpoint_errors.append(endpoint_error)
            if endpoint_error <= tolerance:
                endpoint_correct += 1
        if observed["strand"] == expected["strand"]:
            strand_correct += 1

    total = len(truth)
    mapped = sum(1 for query in truth if query in mappings)
    return {
        "truth_reads": total,
        "mapped_reads": mapped,
        "mapping_rate_percent": 100.0 * mapped / total,
        "correct_subgraph_reads": target_correct,
        "correct_subgraph_rate_percent": 100.0 * target_correct / total,
        "correct_endpoint_reads": endpoint_correct,
        "correct_endpoint_rate_percent": 100.0 * endpoint_correct / total,
        "correct_endpoint_precision_percent": (
            100.0 * endpoint_correct / mapped if mapped else 0.0
        ),
        "correct_strand_reads": strand_correct,
        "correct_strand_rate_percent": 100.0 * strand_correct / total,
        "median_endpoint_error_bp_on_correct_subgraph": (
            statistics.median(endpoint_errors) if endpoint_errors else None
        ),
        "mean_endpoint_error_bp_on_correct_subgraph": (
            statistics.fmean(endpoint_errors) if endpoint_errors else None
        ),
        "endpoint_tolerance_bp": tolerance,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth-sam", type=Path, required=True)
    parser.add_argument("--minigraph-gaf", type=Path, required=True)
    parser.add_argument("--hga-paf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=int, default=20)
    args = parser.parse_args()

    truth = parse_truth(args.truth_sam)
    output = {
        "metric": (
            "Correct endpoint requires the correct subgraph and absolute "
            "target-end error within the configured tolerance."
        ),
        "minigraph": evaluate(
            truth, parse_mapping(args.minigraph_gaf), args.tolerance
        ),
        "candidate_hga": evaluate(
            truth, parse_mapping(args.hga_paf), args.tolerance
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
