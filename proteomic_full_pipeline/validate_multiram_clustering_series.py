#!/usr/bin/env python3
"""Validate optimized MultiRAM clustering on a series of HEK293 runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import evaluate_clustering_quality as quality


ROOT = Path(__file__).resolve().parent
DEFAULT_QUERY_DIR = Path(
    "/mnt/hdd/tsunghan/raw-ms-dataset/homs-tc-dataset/query"
)
DEFAULT_TRUTH_ROOT = ROOT / "benchmark_output_oms_counts_b1926_b1930"
DEFAULT_OUTPUT = ROOT / "clustering_quality_b1926_b1930_optimized"
DEFAULT_B1926_TRUTH = ROOT / (
    "benchmark_output_paper_hek293_b1926_calibrated/b1926/run_1/cpu/"
    "cpu_spectrast_query.pep.xml"
)


def evaluate_sample(
    sample: str,
    query_dir: Path,
    truth_root: Path,
    b1926_truth: Path,
    threshold_ratio: float,
    bucket_width: float,
    seed: int,
) -> dict[str, object]:
    matches = sorted(query_dir.glob(f"{sample}_*.mgf"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one MGF for {sample}, found {len(matches)}")
    mgf_path = matches[0]
    truth_path = (
        b1926_truth
        if sample == "b1926"
        else truth_root / sample / "cpu/cpu_spectrast_query.pep.xml"
    )
    if not truth_path.is_file():
        raise FileNotFoundError(truth_path)

    min_probability = quality.peptideprophet_threshold(truth_path, 0.01)
    truth = quality.read_peptide_truth(truth_path, min_probability)
    all_spectra = quality.parse_mgf_file(
        str(mgf_path), max_peaks=50, mz_max=2000.0
    )
    selected, eligible_spectra, truth_clusters = quality.choose_subset(
        truth, set(range(len(all_spectra))), 0, 0
    )
    spectra = [all_spectra[index] for index in selected]
    truth_labels = [str(truth[index]["truth_label"]) for index in selected]
    truth_charges = [int(truth[index]["charge"]) for index in selected]
    parsed_charges = [spectrum.get("precursor_charge") for _name, spectrum in spectra]
    runtime_charges = [
        int(parsed) if parsed is not None else expected
        for parsed, expected in zip(parsed_charges, truth_charges)
    ]

    predicted, details = quality.run_multiram_functional_clustering(
        spectra,
        runtime_charges,
        threshold_ratio=threshold_ratio,
        bucket_width=bucket_width,
        seed=seed,
    )
    metrics = quality.quality_metrics(truth_labels, predicted)
    return {
        "sample": sample,
        "mgf": str(mgf_path),
        "truth_pepxml": str(truth_path),
        "peptideprophet_min_probability": min_probability,
        "eligible_spectra": eligible_spectra,
        "truth_clusters": truth_clusters,
        "charge_fallback_count": sum(value is None for value in parsed_charges),
        "charge_truth_mismatch_count": sum(
            value is not None and int(value) != expected
            for value, expected in zip(parsed_charges, truth_charges)
        ),
        "functional_model": details,
        "quality": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples", nargs="+", default=["b1926", "b1927", "b1928", "b1929", "b1930"]
    )
    parser.add_argument("--query-dir", type=Path, default=DEFAULT_QUERY_DIR)
    parser.add_argument("--truth-root", type=Path, default=DEFAULT_TRUTH_ROOT)
    parser.add_argument("--b1926-truth", type=Path, default=DEFAULT_B1926_TRUTH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold-ratio", type=float, default=0.455)
    parser.add_argument("--bucket-width", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = [
        evaluate_sample(
            sample,
            args.query_dir,
            args.truth_root,
            args.b1926_truth,
            args.threshold_ratio,
            args.bucket_width,
            args.seed,
        )
        for sample in args.samples
    ]
    rows = []
    for result in results:
        metrics = result["quality"]
        rows.append(
            {
                "sample": result["sample"],
                "evaluated_spectra": metrics["spectra"],
                "truth_clusters": result["truth_clusters"],
                "incorrect_spectra": metrics["incorrect_clustered_spectra"],
                "incorrect_clustering_ratio": metrics["incorrect_clustering_ratio"],
                "completeness": metrics["completeness"],
                "clustered_ratio": metrics["clustered_ratio"],
                "pairwise_precision": metrics["pairwise_precision"],
                "pairwise_recall": metrics["pairwise_recall"],
                "charge_truth_mismatch_count": result["charge_truth_mismatch_count"],
            }
        )

    total_spectra = sum(row["evaluated_spectra"] for row in rows)
    total_incorrect = sum(row["incorrect_spectra"] for row in rows)
    weighted_completeness = sum(
        row["evaluated_spectra"] * row["completeness"] for row in rows
    ) / total_spectra
    summary = {
        "configuration": {
            "bucket_key": "MGF precursor charge + precursor m/z bin",
            "bucket_width_da": args.bucket_width,
            "threshold_ratio": args.threshold_ratio,
            "hv_dimension": 2048,
            "seed": args.seed,
        },
        "aggregate": {
            "evaluated_spectra": total_spectra,
            "incorrect_spectra": total_incorrect,
            "incorrect_clustering_ratio": total_incorrect / total_spectra,
            "weighted_mean_completeness": weighted_completeness,
        },
        "samples": results,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "quality_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "quality_summary.csv", index=False)

    table_rows = [
        "| {sample} | {evaluated_spectra:,} | {incorrect_clustering_ratio:.4%} | "
        "{completeness:.4%} | {clustered_ratio:.4%} | {pairwise_precision:.4%} | "
        "{pairwise_recall:.4%} |".format(**row)
        for row in rows
    ]
    report = f"""# MultiRAM Clustering Quality Validation

- Bucket key: MGF precursor charge + {args.bucket_width:g} Da precursor-m/z bin
- Hamming cutoff: {args.threshold_ratio:g} x 2048 bits
- Runtime metadata only; peptide truth is used only for offline evaluation

| Sample | Evaluated spectra | Incorrect ratio | Completeness | Clustered ratio | Pair precision | Pair recall |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

Across {total_spectra:,} spectra, {total_incorrect:,} were assigned against the
majority peptide in their predicted cluster: **{total_incorrect / total_spectra:.4%}**.
The spectrum-weighted mean completeness is **{weighted_completeness:.4%}**.

The parameters were selected using b1926. Runs b1927-b1930 are independent
validation runs and were not used to choose the operating point.
"""
    (args.output_dir / "MULTIRAM_CLUSTERING_QUALITY_VALIDATION.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(summary["aggregate"], indent=2))
    print(f"Report: {args.output_dir / 'MULTIRAM_CLUSTERING_QUALITY_VALIDATION.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
