#!/usr/bin/env python3
"""Run minimap2 candidate routing followed by HGA subgraph alignment."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path("/home/tsl012/multiomic")
MINIMAP2 = ROOT / "external/minimap2/minimap2"
HGA = ROOT / "external/hga/bin/hga"
TIME_RE = re.compile(r"^Time:\s*([0-9.eE+-]+)s")


def read_fastq(path: Path) -> dict[str, tuple[str, str]]:
    records = {}
    with path.open("r", encoding="ascii") as handle:
        while True:
            header = handle.readline()
            if not header:
                break
            sequence = handle.readline().strip().upper()
            plus = handle.readline()
            quality = handle.readline().strip()
            if not plus or len(sequence) != len(quality):
                raise ValueError(f"Malformed FASTQ near {header.strip()}")
            name = header[1:].split()[0]
            records[name] = (sequence, quality)
    return records


def reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def parse_candidates(path: Path) -> dict[str, dict[str, int | str]]:
    candidates: dict[str, dict[str, int | str]] = {}
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            fields = line.rstrip().split("\t")
            if len(fields) < 12:
                continue
            query = fields[0]
            record = {
                "query_length": int(fields[1]),
                "strand": fields[4],
                "target": fields[5],
                "target_length": int(fields[6]),
                "target_start": int(fields[7]),
                "target_end": int(fields[8]),
                "mapq": int(fields[11]),
            }
            old = candidates.get(query)
            if old is None or (
                int(record["mapq"]),
                int(record["target_end"]) - int(record["target_start"]),
            ) > (
                int(old["mapq"]),
                int(old["target_end"]) - int(old["target_start"]),
            ):
                candidates[query] = record
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--gpu-blocks", type=int, default=8)
    parser.add_argument("--gpu-threads", type=int, default=128)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    reference = Path(manifest["combined_subgraph_fasta"])
    reads_path = Path(manifest["reads_fastq"])
    graph_by_name = {
        item["name"]: Path(item["hga_graph"])
        for item in manifest["subgraphs"]
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()
    candidate_path = args.output_dir / "candidate_routes.paf"
    candidate_command = [
        str(MINIMAP2),
        "-x",
        "sr",
        "--secondary=no",
        "-t",
        str(args.threads),
        str(reference),
        str(reads_path),
    ]
    phase_start = time.perf_counter()
    with candidate_path.open("w", encoding="ascii") as stdout:
        candidate_run = subprocess.run(
            candidate_command,
            stdout=stdout,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    candidate_s = time.perf_counter() - phase_start
    (args.output_dir / "candidate_stderr.log").write_text(
        candidate_run.stderr, encoding="utf-8"
    )
    if candidate_run.returncode != 0:
        raise RuntimeError(
            f"Candidate routing failed: {candidate_run.stderr[-1000:]}"
        )

    phase_start = time.perf_counter()
    reads = read_fastq(reads_path)
    candidates = parse_candidates(candidate_path)
    grouped: dict[str, list[tuple[str, str, str, int]]] = defaultdict(list)
    for query, candidate in candidates.items():
        if query not in reads or candidate["target"] not in graph_by_name:
            continue
        sequence = reads[query][0]
        strand = str(candidate["strand"])
        if strand == "-":
            sequence = reverse_complement(sequence)
        grouped[str(candidate["target"])].append(
            (query, sequence, strand, int(candidate["mapq"]))
        )
    materialize_s = time.perf_counter() - phase_start

    hga_process_wall_s = 0.0
    hga_kernel_s = 0.0
    hga_input_bytes = 0
    hga_graph_bytes = 0
    results: dict[str, dict[str, int | str]] = {}
    shard_records = []
    for target in sorted(grouped):
        records = grouped[target]
        reads_file = args.output_dir / f"{target}.reads.hga"
        with reads_file.open("w", encoding="ascii") as handle:
            for local_id, (_, sequence, _, _) in enumerate(records):
                handle.write(f"read_{local_id}\n{sequence}\n")
        graph_path = graph_by_name[target]
        stdout_path = args.output_dir / f"{target}.hga.stdout"
        stderr_path = args.output_dir / f"{target}.hga.stderr"
        command = [
            str(HGA),
            "-g",
            str(graph_path),
            "-r",
            str(reads_file),
            "-m",
            "2",
            "-n",
            "2",
            "-o",
            "3",
            "-d",
            "1",
            "-b",
            str(args.gpu_blocks),
            "-t",
            str(args.gpu_threads),
            "-c",
            "1",
            "-x",
            "1",
            "-p",
        ]
        run_start = time.perf_counter()
        completed = subprocess.run(
            command, text=True, capture_output=True, check=False
        )
        run_wall_s = time.perf_counter() - run_start
        hga_process_wall_s += run_wall_s
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"HGA failed for {target}: {completed.stderr[-1000:]}"
            )

        kernel_s = None
        result_count = 0
        for line in completed.stdout.splitlines():
            time_match = TIME_RE.match(line)
            if time_match:
                kernel_s = float(time_match.group(1))
                continue
            fields = line.split()
            if len(fields) != 4 or not all(
                field.lstrip("-").isdigit() for field in fields
            ):
                continue
            local_id, score, row, col = map(int, fields)
            if local_id < 0 or local_id >= len(records):
                raise ValueError(f"Invalid HGA local read id: {local_id}")
            query, sequence, strand, mapq = records[local_id]
            results[query] = {
                "target": target,
                "strand": strand,
                "score": score,
                "query_end": row + 1,
                "target_end": col + 1,
                "query_length": len(sequence),
                "mapq": mapq,
            }
            result_count += 1
        if kernel_s is None or result_count != len(records):
            raise RuntimeError(
                f"Missing HGA output for {target}: "
                f"kernel={kernel_s}, results={result_count}/{len(records)}"
            )
        hga_kernel_s += kernel_s
        hga_input_bytes += reads_file.stat().st_size
        hga_graph_bytes += graph_path.stat().st_size
        shard_records.append(
            {
                "target": target,
                "reads": len(records),
                "graph_bytes": graph_path.stat().st_size,
                "hga_reads_bytes": reads_file.stat().st_size,
                "kernel_s": kernel_s,
                "process_wall_s": run_wall_s,
                "command": command,
            }
        )

    phase_start = time.perf_counter()
    mapper_paf = args.output_dir / "hga_mapper.paf"
    with mapper_paf.open("w", encoding="ascii") as handle:
        for query in reads:
            result = results.get(query)
            if result is None:
                continue
            query_length = int(result["query_length"])
            target_end = int(result["target_end"])
            target_start = max(0, target_end - query_length)
            matches = min(query_length, max(0, int(result["score"]) // 2))
            target_length = next(
                int(item["length"])
                for item in manifest["subgraphs"]
                if item["name"] == result["target"]
            )
            handle.write(
                "\t".join(
                    map(
                        str,
                        (
                            query,
                            query_length,
                            0,
                            query_length,
                            result["strand"],
                            result["target"],
                            target_length,
                            target_start,
                            target_end,
                            matches,
                            query_length,
                            result["mapq"],
                        ),
                    )
                )
                + "\n"
            )
    merge_s = time.perf_counter() - phase_start
    total_s = time.perf_counter() - total_start

    packed_words = (int(manifest["read_length"]) + 1 + 7) // 8
    modeled_h2d_bytes = 0
    modeled_d2h_bytes = 0
    for shard in shard_records:
        vertices = next(
            int(item["length"])
            for item in manifest["subgraphs"]
            if item["name"] == shard["target"]
        )
        reads_in_shard = int(shard["reads"])
        modeled_h2d_bytes += (
            reads_in_shard * packed_words * 4
            + (vertices // 32 + 1) * 4
            + 4
            + (vertices // 8 + 1) * 4
        )
        modeled_d2h_bytes += reads_in_shard * 3 * 4

    timing = {
        "status": "completed",
        "input_reads": len(reads),
        "routed_reads": len(candidates),
        "hga_mapped_reads": len(results),
        "phases_s": {
            "candidate_routing": candidate_s,
            "materialize_groups": materialize_s,
            "hga_process_wall_sum": hga_process_wall_s,
            "hga_kernel_sum": hga_kernel_s,
            "merge_output": merge_s,
            "end_to_end": total_s,
        },
        "movement": {
            "candidate_reference_bytes": reference.stat().st_size,
            "candidate_fastq_bytes": reads_path.stat().st_size,
            "hga_graph_file_bytes": hga_graph_bytes,
            "hga_grouped_read_file_bytes": hga_input_bytes,
            "modeled_h2d_bytes_from_hga_allocations": modeled_h2d_bytes,
            "modeled_d2h_bytes_from_hga_results": modeled_d2h_bytes,
        },
        "candidate_command": candidate_command,
        "shards": shard_records,
        "output_paf": str(mapper_paf.resolve()),
    }
    timing_path = args.output_dir / "hga_mapper_timing.json"
    timing_path.write_text(
        json.dumps(timing, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(timing, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
