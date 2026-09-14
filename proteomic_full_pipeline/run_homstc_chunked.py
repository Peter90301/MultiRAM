#!/usr/bin/env python3
"""Run homs-tc in query chunks and merge mzTab outputs.

Why:
- Large query + large library may trigger GPU OOM in a single run.
- Chunking query spectra reduces per-run GPU memory pressure.

What it does:
1) Split input MGF into chunk MGF files by spectrum count.
2) Run homs-tc per chunk.
3) Merge successful chunk mzTab files into one mzTab.
4) Print and save summary (counts, failures, total PSM rows).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chunked homs-tc runner")
    p.add_argument("--query", required=True, help="Input query MGF")
    p.add_argument("--ref", required=True, help="Reference SPLIB")
    p.add_argument("--config", required=True, help="homs-tc ini config")
    p.add_argument("--output-dir", required=True, help="Output directory")
    p.add_argument("--chunk-size", type=int, default=2000, help="Spectra per chunk")
    p.add_argument(
        "--homs-tc-python",
        default="/home/tsl012/miniconda3/envs/homstc/bin/python",
        help="Python executable for homs-tc environment",
    )
    p.add_argument(
        "--homs-tc-run-py",
        default="/home/tsl012/multiomic/sumukh_proteomic_test/homs-tc/run.py",
        help="Path to homs-tc run.py",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining chunks even when a chunk fails",
    )
    return p.parse_args()


def split_mgf_by_spectra(query: Path, out_dir: Path, chunk_size: int) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks: List[Path] = []
    chunk_idx = 1
    spectra_in_chunk = 0
    in_block = False
    buffer: List[str] = []

    chunk_path = out_dir / f"chunk_{chunk_idx:04d}.mgf"
    fout = chunk_path.open("w", encoding="utf-8")

    def rotate_chunk() -> None:
        nonlocal chunk_idx, spectra_in_chunk, chunk_path, fout
        fout.close()
        chunks.append(chunk_path)
        chunk_idx += 1
        spectra_in_chunk = 0
        chunk_path = out_dir / f"chunk_{chunk_idx:04d}.mgf"
        fout = chunk_path.open("w", encoding="utf-8")

    with query.open("r", encoding="utf-8", errors="replace") as fin:
        for line in fin:
            s = line.strip().upper()
            if s == "BEGIN IONS":
                in_block = True
                buffer = [line]
                continue

            if in_block:
                buffer.append(line)
                if s == "END IONS":
                    # Need a new file before writing this spectrum if chunk is full.
                    if spectra_in_chunk >= chunk_size:
                        rotate_chunk()
                    fout.writelines(buffer)
                    spectra_in_chunk += 1
                    in_block = False
                    buffer = []
                continue

            # Preserve non-spectrum lines if any appear in the MGF.
            fout.write(line)

    fout.close()
    if chunk_path.exists() and chunk_path.stat().st_size > 0:
        chunks.append(chunk_path)
    else:
        chunk_path.unlink(missing_ok=True)

    return chunks


def run_one_chunk(
    chunk_mgf: Path,
    ref: Path,
    config: Path,
    out_mztab: Path,
    log_file: Path,
    homstc_python: str,
    homstc_run_py: str,
) -> int:
    run_py_path = Path(homstc_run_py).resolve()
    run_cwd = run_py_path.parent
    cmd = [
        homstc_python,
        str(run_py_path),
        "--ref",
        str(ref),
        "--query",
        str(chunk_mgf),
        "--config",
        str(config),
        "--output",
        str(out_mztab),
    ]
    with log_file.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(
            cmd,
            cwd=str(run_cwd),
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return proc.returncode


def parse_mztab_psm_count(mztab: Path) -> int:
    count = 0
    with mztab.open("r", encoding="utf-8", errors="replace") as fin:
        for line in fin:
            if line.startswith("PSM\t"):
                count += 1
    return count


def merge_mztab(inputs: List[Path], output: Path) -> Tuple[int, int]:
    """Merge mzTab files.

    Returns:
      total_psm_rows, files_merged
    """
    if not inputs:
        return 0, 0

    total_psm = 0
    with output.open("w", encoding="utf-8") as fout:
        wrote_header = False
        for i, p in enumerate(inputs):
            with p.open("r", encoding="utf-8", errors="replace") as fin:
                for line in fin:
                    if line.startswith("MTD\t"):
                        if i == 0:
                            fout.write(line)
                        continue
                    if line.startswith("PSH\t"):
                        if not wrote_header:
                            fout.write(line)
                            wrote_header = True
                        continue
                    if line.startswith("PSM\t"):
                        fout.write(line)
                        total_psm += 1
    return total_psm, len(inputs)


def main() -> int:
    args = parse_args()

    query = Path(args.query)
    ref = Path(args.ref)
    config = Path(args.config)
    out_dir = Path(args.output_dir)

    if not query.exists():
        raise FileNotFoundError(f"Query not found: {query}")
    if not ref.exists():
        raise FileNotFoundError(f"Reference not found: {ref}")
    if not config.exists():
        raise FileNotFoundError(f"Config not found: {config}")

    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = out_dir / "chunks"
    run_dir = out_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    chunks = split_mgf_by_spectra(query, chunk_dir, args.chunk_size)
    if not chunks:
        raise RuntimeError("No chunks generated from query MGF")

    results = []
    ok_mztab_files: List[Path] = []

    for idx, chunk in enumerate(chunks, start=1):
        chunk_name = chunk.stem
        out_mztab = run_dir / f"{chunk_name}.mztab"
        log_file = run_dir / f"{chunk_name}.log"

        rc = run_one_chunk(
            chunk_mgf=chunk,
            ref=ref,
            config=config,
            out_mztab=out_mztab,
            log_file=log_file,
            homstc_python=args.homs_tc_python,
            homstc_run_py=args.homs_tc_run_py,
        )

        psm_count = parse_mztab_psm_count(out_mztab) if (rc == 0 and out_mztab.exists()) else 0
        if rc == 0 and out_mztab.exists():
            ok_mztab_files.append(out_mztab)

        results.append(
            {
                "chunk_index": idx,
                "chunk_file": str(chunk),
                "returncode": rc,
                "mztab_file": str(out_mztab),
                "log_file": str(log_file),
                "psm_count": psm_count,
            }
        )

        if rc != 0 and not args.continue_on_error:
            break

    merged_mztab = out_dir / "merged_chunked.mztab"
    merged_psm_count, merged_files = merge_mztab(ok_mztab_files, merged_mztab)

    summary = {
        "query": str(query),
        "ref": str(ref),
        "config": str(config),
        "chunk_size": args.chunk_size,
        "chunks_total": len(chunks),
        "chunks_succeeded": len(ok_mztab_files),
        "chunks_merged": merged_files,
        "merged_mztab": str(merged_mztab),
        "merged_psm_count": merged_psm_count,
        "results": results,
    }

    summary_file = out_dir / "chunked_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))

    return 0 if len(ok_mztab_files) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
