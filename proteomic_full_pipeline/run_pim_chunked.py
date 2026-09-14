#!/usr/bin/env python3
"""Chunked runner for PIM OMS using the existing homs-tc chunk utilities.

This script reuses the MGF splitting and mzTab merging helpers from
`run_homstc_chunked.py` and runs `pim_hyperoms_estimator.py --mode oms`
per chunk using a specified Python environment (default: homstc env).

Outputs:
- per-chunk mztab files under `<output-dir>/runs`
- merged mzTab `<output-dir>/merged_chunked.mztab`
- summary JSON `<output-dir>/chunked_summary.json`
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import List

from run_homstc_chunked import split_mgf_by_spectra, parse_mztab_psm_count, merge_mztab


def parse_args():
    p = argparse.ArgumentParser(description="Chunked PIM OMS runner")
    p.add_argument("--query", required=True, help="Input query MGF")
    p.add_argument("--ref", required=True, help="Reference SPLIB")
    p.add_argument("--config", required=True, help="homs-tc ini config (used to read tolerances)")
    p.add_argument("--output-dir", required=True, help="Output directory")
    p.add_argument("--chunk-size", type=int, default=2000, help="Spectra per chunk")
    p.add_argument(
        "--pim-python",
        default="/home/tsl012/miniconda3/envs/homstc/bin/python",
        help="Python executable to run pim_hyperoms_estimator.py",
    )
    p.add_argument(
        "--pim-script",
        default="/home/tsl012/multiomic/proteomic_full_pipeline/pim_hyperoms_estimator.py",
        help="Path to pim_hyperoms_estimator.py",
    )
    p.add_argument("--pim-std-ppm", type=float, default=None, help="Override --std-ppm for pim_hyperoms_estimator.py")
    p.add_argument("--pim-open-da", type=float, default=None, help="Override --open-da for pim_hyperoms_estimator.py")
    p.add_argument("--pim-candidate-cap", type=int, default=None, help="Override --candidate-cap for pim_hyperoms_estimator.py")
    p.add_argument("--pim-top-candidates", type=int, default=None, help="Backward-compatible alias for --pim-candidate-cap")
    p.add_argument("--pim-fdr-threshold", type=float, default=None, help="Override --fdr-threshold for pim_hyperoms_estimator.py")
    p.add_argument("--pim-hv-flip-bits", type=int, default=None, help="Override --hv-flip-bits for pim_hyperoms_estimator.py")
    p.add_argument("--pim-hv-seed", type=int, default=None, help="Override --hv-seed for pim_hyperoms_estimator.py")
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining chunks even when a chunk fails",
    )
    return p.parse_args()


def run_one_chunk_pim(pim_python: str, pim_script: str, chunk_mgf: Path, ref: Path, config: Path, out_mztab: Path, log_file: Path, extra_args: List[str] | None = None) -> int:
    cmd = [
        pim_python,
        str(pim_script),
        "--mode",
        "oms",
        "--query",
        str(chunk_mgf),
        "--ref",
        str(ref),
        "--config",
        str(config),
        "--output",
        str(out_mztab),
    ]
    if extra_args:
        cmd += extra_args
    with log_file.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


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

    results: List[dict] = []
    ok_mztab_files: List[Path] = []

    for idx, chunk in enumerate(chunks, start=1):
        chunk_name = chunk.stem
        out_mztab = run_dir / f"{chunk_name}.mztab"
        log_file = run_dir / f"{chunk_name}.log"

        extra_args: List[str] = []
        if args.pim_std_ppm is not None:
            extra_args += ["--std-ppm", str(args.pim_std_ppm)]
        if args.pim_open_da is not None:
            extra_args += ["--open-da", str(args.pim_open_da)]
        candidate_cap = args.pim_candidate_cap if args.pim_candidate_cap is not None else args.pim_top_candidates
        if candidate_cap is not None:
            extra_args += ["--candidate-cap", str(candidate_cap)]
        if args.pim_fdr_threshold is not None:
            extra_args += ["--fdr-threshold", str(args.pim_fdr_threshold)]
        if args.pim_hv_flip_bits is not None:
            extra_args += ["--hv-flip-bits", str(args.pim_hv_flip_bits)]
        if args.pim_hv_seed is not None:
            extra_args += ["--hv-seed", str(args.pim_hv_seed)]

        rc = run_one_chunk_pim(
            pim_python=args.pim_python,
            pim_script=args.pim_script,
            chunk_mgf=chunk,
            ref=ref,
            config=config,
            out_mztab=out_mztab,
            log_file=log_file,
            extra_args=extra_args,
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
