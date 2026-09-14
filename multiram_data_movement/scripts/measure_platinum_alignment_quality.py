#!/usr/bin/env python3
"""Measure mapping-quality proxies for Platinum NA12878 without storing SAM."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, TextIO


CIGAR_RE = re.compile(r"(\d+)([MIDNSHP=X])")


@dataclass
class AlignmentCounts:
    sam_records: int = 0
    primary_reads: int = 0
    primary_mapped_reads: int = 0
    primary_unmapped_reads: int = 0
    properly_paired_reads: int = 0
    singleton_reads: int = 0
    mapq_ge_20_reads: int = 0
    mapq_ge_30_reads: int = 0
    secondary_records: int = 0
    supplementary_records: int = 0
    mapped_reads_with_nm: int = 0
    edit_distance_sum: int = 0
    alignment_span_sum: int = 0

    def add(self, other: "AlignmentCounts") -> None:
        for field in self.__dataclass_fields__:
            setattr(self, field, getattr(self, field) + getattr(other, field))


def cigar_alignment_span(cigar: str) -> int:
    """Return reference/query alignment columns used by the NM identity proxy."""
    return sum(
        int(length)
        for length, operation in CIGAR_RE.findall(cigar)
        if operation in {"M", "I", "D", "=", "X"}
    )


def parse_nm(optional_fields: list[str]) -> int | None:
    for field in optional_fields:
        if field.startswith("NM:i:"):
            try:
                return int(field[5:])
            except ValueError:
                return None
    return None


def collect_sam_metrics(lines: Iterable[str]) -> AlignmentCounts:
    counts = AlignmentCounts()
    for line in lines:
        if not line or line.startswith("@"):
            continue
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 11:
            continue

        counts.sam_records += 1
        flag = int(fields[1])
        is_secondary = bool(flag & 0x100)
        is_supplementary = bool(flag & 0x800)
        if is_secondary:
            counts.secondary_records += 1
        if is_supplementary:
            counts.supplementary_records += 1
        if is_secondary or is_supplementary:
            continue

        counts.primary_reads += 1
        is_mapped = not bool(flag & 0x4)
        if not is_mapped:
            counts.primary_unmapped_reads += 1
            continue

        counts.primary_mapped_reads += 1
        mapq = int(fields[4])
        if mapq >= 20:
            counts.mapq_ge_20_reads += 1
        if mapq >= 30:
            counts.mapq_ge_30_reads += 1
        if flag & 0x2:
            counts.properly_paired_reads += 1
        if flag & 0x8:
            counts.singleton_reads += 1

        nm = parse_nm(fields[11:])
        span = cigar_alignment_span(fields[5])
        if nm is not None and span > 0:
            counts.mapped_reads_with_nm += 1
            counts.edit_distance_sum += nm
            counts.alignment_span_sum += span
    return counts


def percentage(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return 100.0 * numerator / denominator


def counts_with_rates(counts: AlignmentCounts) -> dict[str, int | float | None]:
    result: dict[str, int | float | None] = asdict(counts)
    result.update(
        {
            "mapped_rate_percent": percentage(
                counts.primary_mapped_reads, counts.primary_reads
            ),
            "properly_paired_rate_percent": percentage(
                counts.properly_paired_reads, counts.primary_reads
            ),
            "singleton_rate_percent": percentage(
                counts.singleton_reads, counts.primary_reads
            ),
            "mapq_ge_20_rate_percent": percentage(
                counts.mapq_ge_20_reads, counts.primary_reads
            ),
            "mapq_ge_30_rate_percent": percentage(
                counts.mapq_ge_30_reads, counts.primary_reads
            ),
            "alignment_identity_proxy_percent": (
                100.0
                * (1.0 - counts.edit_distance_sum / counts.alignment_span_sum)
                if counts.alignment_span_sum
                else None
            ),
        }
    )
    return result


def write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def run_pair(
    row: dict[str, str],
    minimap2: Path,
    reference: Path,
    reads_dir: Path,
    threads: int,
    stderr_path: Path,
) -> tuple[AlignmentCounts, float, int]:
    command = [
        str(minimap2),
        "-a",
        "-t",
        str(threads),
        "-x",
        "sr",
        "--secondary=no",
        str(reference),
        str(reads_dir / row["r1_fastq"]),
        str(reads_dir / row["r2_fastq"]),
    ]
    started = time.monotonic()
    with stderr_path.open("w", encoding="utf-8") as stderr_handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=stderr_handle,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1024 * 1024,
        )
        assert process.stdout is not None
        counts = collect_sam_metrics(process.stdout)
        process.stdout.close()
        return_code = process.wait()
    return counts, time.monotonic() - started, return_code


def write_pair_table(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reads-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--minimap2", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--max-pairs", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for required in (args.manifest, args.reference, args.minimap2):
        if not required.exists():
            raise FileNotFoundError(required)
    if not args.reads_dir.is_dir():
        raise NotADirectoryError(args.reads_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stderr_dir = args.output_dir / "minimap2_stderr"
    stderr_dir.mkdir(exist_ok=True)
    manifest_rows = read_manifest(args.manifest)
    if args.max_pairs > 0:
        manifest_rows = manifest_rows[: args.max_pairs]

    aggregate = AlignmentCounts()
    pair_results: list[dict[str, object]] = []
    run_started = time.time()
    state_path = args.output_dir / "alignment_quality_summary.json"
    table_path = args.output_dir / "per_pair_alignment_quality.csv"

    for pair_index, row in enumerate(manifest_rows, start=1):
        accession = row["run_accession"]
        print(
            f"[{pair_index}/{len(manifest_rows)}] measuring {accession}",
            flush=True,
        )
        counts, runtime_s, return_code = run_pair(
            row,
            args.minimap2,
            args.reference,
            args.reads_dir,
            args.threads,
            stderr_dir / f"{accession}.log",
        )
        if return_code != 0:
            raise RuntimeError(f"minimap2 failed for {accession}: rc={return_code}")

        aggregate.add(counts)
        result: dict[str, object] = {
            "run_accession": accession,
            "runtime_s": runtime_s,
            **counts_with_rates(counts),
        }
        pair_results.append(result)
        write_pair_table(table_path, pair_results)
        write_json_atomic(
            state_path,
            {
                "status": "running",
                "metric_definition": (
                    "Mapping-quality proxies on real NA12878 reads; not ground-truth "
                    "positional accuracy. Identity proxy = 1 - sum(NM)/sum(alignment span)."
                ),
                "completed_pairs": pair_index,
                "total_pairs": len(manifest_rows),
                "elapsed_s": time.time() - run_started,
                "aggregate": counts_with_rates(aggregate),
                "pairs": pair_results,
            },
        )
        print(
            json.dumps(
                {
                    "run_accession": accession,
                    "mapped_rate_percent": result["mapped_rate_percent"],
                    "properly_paired_rate_percent": result[
                        "properly_paired_rate_percent"
                    ],
                    "identity_proxy_percent": result[
                        "alignment_identity_proxy_percent"
                    ],
                    "runtime_s": runtime_s,
                }
            ),
            flush=True,
        )

    write_json_atomic(
        state_path,
        {
            "status": "complete",
            "metric_definition": (
                "Mapping-quality proxies on real NA12878 reads; not ground-truth "
                "positional accuracy. Identity proxy = 1 - sum(NM)/sum(alignment span)."
            ),
            "completed_pairs": len(pair_results),
            "total_pairs": len(manifest_rows),
            "elapsed_s": time.time() - run_started,
            "aggregate": counts_with_rates(aggregate),
            "pairs": pair_results,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
