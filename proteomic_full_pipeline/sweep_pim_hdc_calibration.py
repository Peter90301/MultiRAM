#!/usr/bin/env python3
"""Sweep PIM HDC OMS calibration settings on a query subset."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


PIPELINE_ROOT = Path("/home/tsl012/multiomic/proteomic_full_pipeline")
PIM_RUNNER = PIPELINE_ROOT / "pim_hyperoms_estimator.py"
HOMSTC_PYTHON = Path("/home/tsl012/miniconda3/envs/homstc/bin/python")
DEFAULT_QUERY = Path("/mnt/hdd/tsunghan/raw-ms-dataset/homs-tc-dataset/query/b1926_293T_proteinID_06A_QE3_122212.mgf")
DEFAULT_REF = Path("/mnt/hdd/tsunghan/raw-ms-dataset/homs-tc-dataset/ref/massive_human_hcd_unique_targetdecoy.splib")
DEFAULT_CONFIG = Path("/home/tsl012/multiomic/sumukh_proteomic_test/homs-tc/configs/hek293.ini")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default=str(DEFAULT_QUERY))
    parser.add_argument("--ref", default=str(DEFAULT_REF))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(PIPELINE_ROOT / "pim_hdc_sweep_b1926"))
    parser.add_argument("--max-queries", type=int, default=1000)
    parser.add_argument("--candidate-cap", type=int, default=512)
    parser.add_argument("--gpu-chunk-size", type=int, default=8192)
    parser.add_argument("--flip-bits", nargs="+", type=int, default=[64, 128, 256, 512, 1024, 4096])
    parser.add_argument("--score-modes", nargs="+", default=["normalized", "zscore", "zscore_margin"])
    parser.add_argument("--codebooks", nargs="+", default=["random", "locality"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for codebook in args.codebooks:
        for score_mode in args.score_modes:
            for flip_bits in args.flip_bits:
                label = f"{codebook}_{score_mode}_flip{flip_bits}"
                output = out_dir / f"{label}.mztab"
                cmd = [
                    str(HOMSTC_PYTHON),
                    str(PIM_RUNNER),
                    "--mode",
                    "oms",
                    "--backend",
                    "gpu",
                    "--query",
                    str(args.query),
                    "--ref",
                    str(args.ref),
                    "--config",
                    str(args.config),
                    "--hdc-codebook",
                    codebook,
                    "--score-mode",
                    score_mode,
                    "--hv-flip-bits",
                    str(flip_bits),
                    "--candidate-cap",
                    str(args.candidate_cap),
                    "--max-queries",
                    str(args.max_queries),
                    "--gpu-chunk-size",
                    str(args.gpu_chunk_size),
                    "--progress-interval",
                    "0",
                    "--output",
                    str(output),
                ]
                print(f"=== {label} ===", flush=True)
                process = subprocess.run(cmd)
                summary_path = output.with_suffix(output.suffix + ".summary.json")
                if process.returncode != 0 or not summary_path.exists():
                    rows.append(
                        {
                            "label": label,
                            "returncode": process.returncode,
                            "final_identifications": None,
                            "summary_json": str(summary_path),
                        }
                    )
                    continue
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                counts = summary.get("counts", {})
                rows.append(
                    {
                        "label": label,
                        "codebook": codebook,
                        "score_mode": score_mode,
                        "flip_bits": flip_bits,
                        "returncode": process.returncode,
                        "max_queries": args.max_queries,
                        "candidate_cap": args.candidate_cap,
                        "standard_raw": counts.get("standard_raw"),
                        "standard_accepted": counts.get("standard_accepted"),
                        "open_raw": counts.get("open_raw"),
                        "open_accepted": counts.get("open_accepted"),
                        "final_identifications": counts.get("final_identifications"),
                        "simulator_wall_time_sec": summary.get("simulator_wall_time_sec"),
                        "summary_json": str(summary_path),
                    }
                )

                csv_path = out_dir / "sweep_results.csv"
                with csv_path.open("w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=sorted({key for row in rows for key in row}))
                    writer.writeheader()
                    writer.writerows(rows)

    print(f"Results written to {out_dir / 'sweep_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
