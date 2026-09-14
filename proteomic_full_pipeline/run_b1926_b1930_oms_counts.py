#!/usr/bin/env python3
"""Run CPU/GPU/PIM OMS counts for HEK293 b1926-b1930 queries."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path


PIPELINE_ROOT = Path("/home/tsl012/multiomic/proteomic_full_pipeline")
HOMSTC_ROOT = Path("/home/tsl012/multiomic/sumukh_proteomic_test/homs-tc")
HOMSTC_PYTHON = Path("/home/tsl012/miniconda3/envs/homstc/bin/python")
ANNSOLO_PYTHON = Path("/home/tsl012/miniconda3/envs/annsolo/bin/python")
SPECTRAST = Path("/home/tsl012/.local/tpp-5.0.0/bin/spectrast")
PEPTIDEPROPHET = Path("/home/tsl012/.local/tpp-5.0.0/bin/PeptideProphetParser")
PREPARE_SPECTRAST_QUERY = PIPELINE_ROOT / "prepare_spectrast_query.py"
PIM_RUNNER = PIPELINE_ROOT / "pim_hyperoms_estimator.py"
CLUSTERING_OMS = PIPELINE_ROOT / "clustering_OMS.py"

QUERY_DIR = Path("/mnt/hdd/tsunghan/raw-ms-dataset/homs-tc-dataset/query")
REF = Path("/mnt/hdd/tsunghan/raw-ms-dataset/homs-tc-dataset/ref/massive_human_hcd_unique_targetdecoy.splib")
CPU_REF = Path("/mnt/hdd/tsunghan/raw-ms-dataset/homs-tc-dataset/ref/massive_human_hcd_unique_targetdecoy_spectrast.splib")
HOMSTC_HEK293_CONFIG = HOMSTC_ROOT / "configs/hek293.ini"

QUERY_MAP = {
    "b1926": QUERY_DIR / "b1926_293T_proteinID_06A_QE3_122212.mgf",
    "b1927": QUERY_DIR / "b1927_293T_proteinID_07A_QE3_122212.mgf",
    "b1928": QUERY_DIR / "b1928_293T_proteinID_08A_QE3_122212.mgf",
    "b1929": QUERY_DIR / "b1929_293T_proteinID_09A_QE3_122212.mgf",
    "b1930": QUERY_DIR / "b1930_293T_proteinID_10A_QE3_122212.mgf",
}


def load_count_helpers():
    spec = importlib.util.spec_from_file_location("clustering_oms_helpers", CLUSTERING_OMS)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {CLUSTERING_OMS}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["clustering_oms_helpers"] = module
    spec.loader.exec_module(module)
    return module


def run_logged(cmd: list[str], log_path: Path, cwd: Path | None = None) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as fh:
        process = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = time.perf_counter() - start
    if process.returncode != 0:
        raise RuntimeError(f"Command failed ({process.returncode}): {' '.join(cmd)}; log={log_path}")
    return elapsed


def run_cpu(query: Path, out_dir: Path) -> tuple[Path, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "cpu_spectrast_query"
    prepare_cmd = [
        str(HOMSTC_PYTHON),
        str(PREPARE_SPECTRAST_QUERY),
        "--input",
        str(query),
        "--output-prefix",
        str(prefix),
    ]
    prepare_start = time.perf_counter()
    prepared = subprocess.check_output(prepare_cmd, text=True).strip()
    prepare_elapsed = time.perf_counter() - prepare_start

    search_elapsed = run_logged(
        [
            str(SPECTRAST),
            "-sEpep.xml",
            f"-sO{out_dir}",
            f"-sL{CPU_REF}",
            prepared,
        ],
        out_dir / "logs/cpu_spectrast.log",
    )
    pepxml = out_dir / "cpu_spectrast_query.pep.xml"
    prophet_elapsed = run_logged(
        [
            str(PEPTIDEPROPHET),
            str(pepxml),
            str(pepxml),
            "NONE",
        ],
        out_dir / "logs/cpu_peptideprophet.log",
    )
    (out_dir / "logs/cpu_prepare_time.json").write_text(
        json.dumps(
            {
                "prepare_sec": prepare_elapsed,
                "spectrast_sec": search_elapsed,
                "peptideprophet_sec": prophet_elapsed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return pepxml, prepare_elapsed + search_elapsed + prophet_elapsed


def run_gpu(query: Path, out_dir: Path) -> tuple[Path, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "ann_solo_output.mztab"
    elapsed = run_logged(
        [
            "env",
            "NUMBA_DISABLE_JIT=1",
            str(ANNSOLO_PYTHON),
            "-m",
            "ann_solo.ann_solo",
            str(REF),
            str(query),
            str(output),
            "--precursor_tolerance_mass",
            "5",
            "--precursor_tolerance_mode",
            "ppm",
            "--precursor_tolerance_mass_open",
            "500",
            "--precursor_tolerance_mode_open",
            "Da",
            "--fragment_mz_tolerance",
            "0.05",
            "--remove_precursor",
            "--remove_precursor_tolerance",
            "0.05",
            "--min_mz",
            "101",
            "--max_mz",
            "1500",
            "--min_intensity",
            "0.01",
            "--min_peaks",
            "10",
            "--min_mz_range",
            "250",
            "--max_peaks_used",
            "50",
            "--max_peaks_used_library",
            "50",
            "--scaling",
            "rank",
            "--fdr",
            "0.01",
            "--fdr_min_group_size",
            "20",
            "--mode",
            "ann",
            "--num_candidates",
            "1024",
            "--batch_size",
            "8000",
            "--num_list",
            "256",
            "--num_probe",
            "128",
            "--model",
            "none",
        ],
        out_dir / "logs/gpu_annsolo.log",
    )
    return output, elapsed


def run_pim(query: Path, out_dir: Path) -> tuple[Path, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "pim_hyperoms_output.mztab"
    elapsed = run_logged(
        [
            str(HOMSTC_PYTHON),
            str(PIM_RUNNER),
            "--mode",
            "oms",
            "--backend",
            "gpu",
            "--query",
            str(query),
            "--ref",
            str(REF),
            "--config",
            str(HOMSTC_HEK293_CONFIG),
            "--hdc-codebook",
            "random",
            "--score-mode",
            "normalized",
            "--hv-flip-bits",
            "64",
            "--candidate-cap",
            "512",
            "--gpu-chunk-size",
            "8192",
            "--progress-interval",
            "5000",
            "--output",
            str(output),
        ],
        out_dir / "logs/pim_hdc_oms.log",
    )
    return output, elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(QUERY_MAP), choices=sorted(QUERY_MAP))
    parser.add_argument(
        "--output-root",
        default=str(PIPELINE_ROOT / "benchmark_output_oms_counts_b1926_b1930"),
    )
    parser.add_argument("--force", action="store_true", help="Rerun even when output files already exist.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    helpers = load_count_helpers()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for dataset in args.datasets:
        query = QUERY_MAP[dataset]
        if not query.exists():
            raise FileNotFoundError(query)
        print(f"=== {dataset} ===", flush=True)
        dataset_out = output_root / dataset

        cpu_out = dataset_out / "cpu"
        cpu_pepxml = cpu_out / "cpu_spectrast_query.pep.xml"
        if args.force or not cpu_pepxml.exists():
            cpu_pepxml, cpu_elapsed = run_cpu(query, cpu_out)
        else:
            cpu_elapsed = None
        cpu_strict = helpers._count_pepxml_psms(cpu_pepxml, 0.01, "DECOY_")
        cpu_pp = helpers._pepxml_peptideprophet_metrics(cpu_pepxml, 0.01)
        cpu_paper = int(round(float(cpu_pp["peptideprophet_est_tot_num_correct"])))

        gpu_out = dataset_out / "gpu"
        gpu_mztab = gpu_out / "ann_solo_output.mztab"
        if args.force or not gpu_mztab.exists():
            gpu_mztab, gpu_elapsed = run_gpu(query, gpu_out)
        else:
            gpu_elapsed = None
        gpu_count = helpers._count_mztab_psms(gpu_mztab, 0.01)

        pim_out = dataset_out / "pim"
        pim_mztab = pim_out / "pim_hyperoms_output.mztab"
        if args.force or not pim_mztab.exists():
            pim_mztab, pim_elapsed = run_pim(query, pim_out)
        else:
            pim_elapsed = None
        pim_count = helpers._count_mztab_psms(pim_mztab, 0.01)

        row = {
            "dataset": dataset,
            "cpu_spectrast_peptideprophet_est_correct": cpu_paper,
            "cpu_spectrast_strict_target_decoy_1pct": cpu_strict,
            "gpu_annsolo_1pct": gpu_count,
            "pim_hdc_1pct": pim_count,
            "cpu_elapsed_sec": cpu_elapsed,
            "gpu_elapsed_sec": gpu_elapsed,
            "pim_elapsed_sec": pim_elapsed,
            "cpu_source": str(cpu_pepxml),
            "gpu_source": str(gpu_mztab),
            "pim_source": str(pim_mztab),
        }
        rows.append(row)
        print(
            f"{dataset}: CPU={cpu_paper} (strict={cpu_strict}), "
            f"GPU={gpu_count}, PIM={pim_count}",
            flush=True,
        )

        csv_path = output_root / "oms_counts_summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"Summary written to {output_root / 'oms_counts_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
