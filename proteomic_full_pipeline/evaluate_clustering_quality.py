#!/usr/bin/env python3
"""Evaluate Falcon, RAPIDS, and MultiRAM clustering against peptide truth."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    normalized_mutual_info_score,
)


ROOT = Path(__file__).resolve().parent
GPU_BENCHMARK_DIR = ROOT.parent / "sumukh_proteomic_test"
if str(GPU_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(GPU_BENCHMARK_DIR))

from gpu_benchmark import (  # noqa: E402
    SpecHDFeRAMSystem,
    _hamming_distance_matrix_numpy,
    nn_chain_hac_from_distance,
    parse_mgf_file,
)


PEPXML_NS = {"pep": "http://regis-web.systemsbiology.net/pepXML"}
DEFAULT_MGF = Path(
    "/mnt/hdd/tsunghan/raw-ms-dataset/homs-tc-dataset/query/"
    "b1926_293T_proteinID_06A_QE3_122212.mgf"
)
DEFAULT_PEPXML = ROOT / (
    "benchmark_output_paper_hek293_b1926_calibrated/b1926/run_1/cpu/"
    "cpu_spectrast_query.pep.xml"
)
DEFAULT_FALCON = ROOT / (
    "benchmark_output_paper_hek293_b1926_calibrated/b1926/run_1/cpu/"
    "cpu_falcon.csv.csv"
)
DEFAULT_RAPIDS = ROOT / (
    "benchmark_output_paper_hek293_b1926_calibrated/b1926/run_1/gpu/"
    "gpu_hyperspec_clusters.parquet"
)
DEFAULT_OUTPUT = ROOT / "clustering_quality_b1926_multiram"


def peptideprophet_threshold(path: Path, error_threshold: float) -> float:
    best_error = -1.0
    probability = None
    in_all = False
    error_tag = f"{{{PEPXML_NS['pep']}}}error_point"
    roc_tag = f"{{{PEPXML_NS['pep']}}}roc_error_data"
    for event, elem in ET.iterparse(path, events=("start", "end")):
        if event == "start" and elem.tag == roc_tag:
            in_all = elem.attrib.get("charge") == "all"
        elif event == "end" and elem.tag == error_tag and in_all:
            try:
                error = float(elem.attrib["error"])
                min_prob = float(elem.attrib["min_prob"])
            except (KeyError, ValueError):
                continue
            if error <= error_threshold and error >= best_error:
                best_error = error
                probability = min_prob
        elif event == "end" and elem.tag == roc_tag:
            in_all = False
            elem.clear()
    if probability is None:
        raise RuntimeError(f"No PeptideProphet threshold <= {error_threshold} in {path}")
    return probability


def read_peptide_truth(path: Path, min_probability: float) -> dict[int, dict[str, Any]]:
    truth: dict[int, dict[str, Any]] = {}
    query_tag = f"{{{PEPXML_NS['pep']}}}spectrum_query"
    for _event, elem in ET.iterparse(path, events=("end",)):
        if elem.tag != query_tag:
            continue
        hit = elem.find("./pep:search_result/pep:search_hit[@hit_rank='1']", PEPXML_NS)
        if hit is None:
            elem.clear()
            continue
        result = hit.find(
            "./pep:analysis_result[@analysis='peptideprophet']/pep:peptideprophet_result",
            PEPXML_NS,
        )
        try:
            probability = float(result.attrib["probability"]) if result is not None else -1.0
            raw_index = int(elem.attrib["start_scan"]) - 1
            charge = int(elem.attrib["assumed_charge"])
        except (KeyError, TypeError, ValueError):
            elem.clear()
            continue
        if probability < min_probability or raw_index < 0:
            elem.clear()
            continue

        peptide = hit.attrib.get("peptide", "")
        modification = hit.find("./pep:modification_info", PEPXML_NS)
        modified = (
            modification.attrib.get("modified_peptide", peptide)
            if modification is not None
            else peptide
        )
        if modified:
            truth[raw_index] = {
                "truth_label": f"z{charge}:{modified}",
                "peptide": modified,
                "charge": charge,
                "probability": probability,
            }
        elem.clear()
    return truth


def read_falcon(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, comment="#")
    extracted = frame["identifier"].str.extract(r"index:(\d+)$", expand=False)
    frame = frame.loc[extracted.notna()].copy()
    frame["raw_index"] = extracted.loc[extracted.notna()].astype(int)
    frame["precursor_charge"] = frame["precursor_charge"].astype(int)
    return frame


def metadata_key(charge: Any, retention_time: Any, precursor_mz: Any) -> tuple[int, int, int]:
    return (
        int(charge),
        int(round(float(retention_time) * 100.0)),
        int(round(float(precursor_mz) * 1000.0)),
    )


def crosswalk_rapids(rapids_path: Path, falcon: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rapids = pd.read_parquet(rapids_path)
    key_to_index: dict[tuple[int, int, int], int] = {}
    duplicate_keys: set[tuple[int, int, int]] = set()
    for row in falcon.itertuples(index=False):
        key = metadata_key(row.precursor_charge, row.retention_time, row.precursor_mz)
        if key in key_to_index:
            duplicate_keys.add(key)
        else:
            key_to_index[key] = int(row.raw_index)
    for key in duplicate_keys:
        key_to_index.pop(key, None)

    raw_indices = []
    for row in rapids.itertuples(index=False):
        key = metadata_key(row.precursor_charge, row.retention_time, row.precursor_mz)
        raw_indices.append(key_to_index.get(key))
    rapids = rapids.assign(raw_index=raw_indices)
    matched = rapids["raw_index"].notna()
    rapids = rapids.loc[matched].copy()
    rapids["raw_index"] = rapids["raw_index"].astype(int)
    return rapids, {
        "rapids_rows": len(raw_indices),
        "matched_rows": int(matched.sum()),
        "matched_percent": 100.0 * float(matched.mean()),
        "ambiguous_falcon_metadata_keys": len(duplicate_keys),
        "metadata_key": "charge + retention time rounded 0.01 s + precursor m/z rounded 0.001",
    }


def namespace_labels(labels: list[Any], charges: list[int], prefix: str) -> list[str]:
    output = []
    for index, (label, charge) in enumerate(zip(labels, charges)):
        try:
            is_noise = int(label) == -1
        except (TypeError, ValueError):
            is_noise = str(label).strip() == "-1"
        if is_noise:
            output.append(f"{prefix}:noise:{index}")
        else:
            output.append(f"z{int(charge)}:{label}")
    return output


def choose_subset(
    truth: dict[int, dict[str, Any]],
    available_indices: set[int],
    max_spectra: int,
    max_per_truth_cluster: int,
) -> tuple[list[int], int, int]:
    groups: dict[str, list[int]] = defaultdict(list)
    for raw_index, record in truth.items():
        if raw_index in available_indices:
            groups[str(record["truth_label"])].append(raw_index)
    eligible = [
        (label, sorted(indices))
        for label, indices in groups.items()
        if len(indices) >= 2
    ]
    eligible.sort(key=lambda item: (-len(item[1]), item[0]))
    eligible_spectra = sum(len(indices) for _label, indices in eligible)

    selected: list[int] = []
    for _label, indices in eligible:
        members = indices if max_per_truth_cluster == 0 else indices[:max_per_truth_cluster]
        if max_spectra and len(selected) + len(members) > max_spectra:
            continue
        selected.extend(members)
        if max_spectra and len(selected) >= max_spectra:
            break
    return sorted(selected), eligible_spectra, len(eligible)


def run_multiram_functional_clustering(
    spectra: list[tuple[str, dict[str, Any]]],
    charges: list[int],
    threshold_ratio: float,
    bucket_width: float,
    seed: int,
) -> tuple[list[str], dict[str, Any]]:
    started = time.perf_counter()
    accelerator = SpecHDFeRAMSystem(d_dim=2048, q_levels=16, seed=seed)
    encoded = np.stack(
        [
            accelerator.encode_spectrum(spectrum["mz"], spectrum["intensity"])
            for _identifier, spectrum in spectra
        ]
    )
    encoding_s = time.perf_counter() - started

    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, ((_identifier, spectrum), charge) in enumerate(zip(spectra, charges)):
        precursor = float(spectrum.get("precursor_mz") or 0.0)
        bucket = int(precursor // bucket_width) if bucket_width > 0 else 0
        buckets[(int(charge), bucket)].append(index)

    labels = [""] * len(spectra)
    next_cluster = 0
    clustering_started = time.perf_counter()
    bucket_sizes = []
    for key in sorted(buckets):
        indices = buckets[key]
        bucket_sizes.append(len(indices))
        if len(indices) == 1:
            local_clusters = [[0]]
        else:
            distances = _hamming_distance_matrix_numpy(encoded[indices])
            local_clusters = nn_chain_hac_from_distance(
                distances,
                threshold_ratio=threshold_ratio,
                hv_dimension=encoded.shape[1],
            )
        for cluster in local_clusters:
            label = f"pim:{next_cluster}"
            next_cluster += 1
            for local_index in cluster:
                labels[indices[local_index]] = label
    clustering_s = time.perf_counter() - clustering_started
    return labels, {
        "encoding_s": encoding_s,
        "functional_clustering_s": clustering_s,
        "num_buckets": len(buckets),
        "max_bucket_size": max(bucket_sizes, default=0),
        "mean_bucket_size": float(np.mean(bucket_sizes)) if bucket_sizes else 0.0,
        "threshold_ratio": threshold_ratio,
        "bucket_width_da": bucket_width,
        "hv_dimension": 2048,
        "algorithm": "SpecHD bipolar HDC + charge/precursor buckets + NN-chain complete-linkage HAC",
    }


def comb2(value: int) -> int:
    return value * (value - 1) // 2


def quality_metrics(truth: list[str], predicted: list[str]) -> dict[str, Any]:
    if len(truth) != len(predicted) or not truth:
        raise ValueError("truth and predicted labels must have equal non-zero length")
    true_names = {label: idx for idx, label in enumerate(sorted(set(truth)))}
    pred_names = {label: idx for idx, label in enumerate(sorted(set(predicted)))}
    contingency = np.zeros((len(true_names), len(pred_names)), dtype=np.int64)
    for true_label, pred_label in zip(truth, predicted):
        contingency[true_names[true_label], pred_names[pred_label]] += 1

    rows, cols = linear_sum_assignment(-contingency)
    assignment_correct = int(contingency[rows, cols].sum())
    tp = int(sum(comb2(int(value)) for value in contingency.ravel()))
    pred_pairs = int(sum(comb2(int(value)) for value in contingency.sum(axis=0)))
    true_pairs = int(sum(comb2(int(value)) for value in contingency.sum(axis=1)))
    fp = pred_pairs - tp
    fn = true_pairs - tp
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    pairwise_f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    true_sets: dict[str, set[int]] = defaultdict(set)
    pred_sets: dict[str, set[int]] = defaultdict(set)
    for index, (true_label, pred_label) in enumerate(zip(truth, predicted)):
        true_sets[true_label].add(index)
        pred_sets[pred_label].add(index)
    predicted_memberships = {frozenset(members) for members in pred_sets.values()}
    exact_truth_clusters = sum(
        frozenset(members) in predicted_memberships for members in true_sets.values()
    )
    true_count = len(true_sets)
    pred_count = len(pred_sets)
    count_agreement = min(true_count, pred_count) / max(true_count, pred_count)

    clustered_mask = [
        str(label) != "-1" and ":noise:" not in str(label)
        for label in predicted
    ]
    clustered_count = sum(clustered_mask)
    clustered_groups: dict[str, list[str]] = defaultdict(list)
    for true_label, pred_label, is_clustered in zip(
        truth, predicted, clustered_mask
    ):
        if is_clustered:
            clustered_groups[pred_label].append(true_label)
    incorrect_count = sum(
        len(labels) - Counter(labels).most_common(1)[0][1]
        for labels in clustered_groups.values()
    )

    return {
        "spectra": len(truth),
        "adjusted_rand_index": float(adjusted_rand_score(truth, predicted)),
        "normalized_mutual_information": float(
            normalized_mutual_info_score(truth, predicted, average_method="arithmetic")
        ),
        "cluster_assignment_agreement": assignment_correct / len(truth),
        "pairwise_precision": precision,
        "pairwise_recall": recall,
        "pairwise_f1": pairwise_f1,
        "true_positive_pairs": tp,
        "false_positive_pairs": fp,
        "false_negative_pairs": fn,
        "truth_cluster_count": true_count,
        "predicted_cluster_count": pred_count,
        "cluster_count_agreement": count_agreement,
        "exact_correct_clusters": exact_truth_clusters,
        "correct_cluster_ratio": exact_truth_clusters / true_count,
        "num_clustered": clustered_count,
        "num_noise": len(truth) - clustered_count,
        "clustered_ratio": clustered_count / len(truth),
        "incorrect_clustered_spectra": incorrect_count,
        "incorrect_clustering_ratio": (
            incorrect_count / clustered_count if clustered_count else 0.0
        ),
        "completeness": float(completeness_score(truth, predicted)),
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    rows = []
    hyperspec_rows = []
    for platform in ("falcon_cpu", "rapids_gpu", "multiram_pim_functional"):
        result = summary["quality"][platform]
        rows.append(
            "| {platform} | {adjusted_rand_index:.4f} | "
            "{normalized_mutual_information:.4f} | "
            "{cluster_assignment_agreement:.2%} | {pairwise_precision:.2%} | "
            "{pairwise_recall:.2%} | {cluster_count_agreement:.2%} | "
            "{correct_cluster_ratio:.2%} | {predicted_cluster_count} |".format(
                platform=platform, **result
            )
        )
        hyperspec_rows.append(
            "| {platform} | {incorrect_clustering_ratio:.4%} | "
            "{completeness:.4%} | {clustered_ratio:.4%} | "
            "{incorrect_clustered_spectra} / {num_clustered} |".format(
                platform=platform, **result
            )
        )
    scope = summary["scope"]
    crosswalk = summary["crosswalk"]
    pim = summary["multiram_functional_model"]
    if scope["selection_is_exhaustive"]:
        selection = (
            "all matched spectra from truth clusters containing at least two "
            "high-confidence identifications"
        )
    else:
        per_cluster = (
            "unlimited" if scope["max_per_truth_cluster"] == 0
            else str(scope["max_per_truth_cluster"])
        )
        overall = (
            "unlimited" if scope["max_spectra"] == 0
            else f"{scope['max_spectra']:,}"
        )
        selection = (
            "truth clusters with at least two identified spectra, capped at "
            f"{per_cluster} spectra per truth cluster and {overall} spectra overall"
        )
    text = f"""# MultiRAM Clustering Quality Evaluation

## Scope

- Dataset: `{scope['input_mgf']}`
- Truth source: `{scope['truth_pepxml']}`
- Truth definition: modified peptide sequence + precursor charge
- PeptideProphet error threshold: {scope['peptideprophet_error_threshold']:.2%}
- PeptideProphet minimum probability: {scope['peptideprophet_min_probability']:.4f}
- Matched high-confidence PSMs: {scope['matched_high_confidence_psms']:,}
- Eligible repeated-cluster spectra: {scope['eligible_spectra']:,} in
  {scope['eligible_truth_clusters']:,} truth clusters
- Evaluated spectra: {scope['evaluated_spectra']:,}
- Truth clusters: {scope['truth_clusters']:,}
- Selection: {selection}
- RAPIDS metadata crosswalk: {crosswalk['matched_rows']:,} / {crosswalk['rapids_rows']:,}
  ({crosswalk['matched_percent']:.3f}%)

The peptide assignments are a high-confidence experimental reference, not
absolute biological ground truth. Quality is measured only on spectra passing
the stated PeptideProphet threshold.

## Results

| Platform | ARI | NMI | Assignment agreement | Pair precision | Pair recall | Cluster-count agreement | Correct cluster ratio | Predicted clusters |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

### Hyper-Spec-Compatible Metrics

| Platform | Incorrect clustering ratio | Completeness | Clustered ratio | Incorrect / clustered spectra |
|---|---:|---:|---:|---:|
{chr(10).join(hyperspec_rows)}

## Metric Definitions

- **ARI**: adjusted agreement between all same/different-cluster spectrum pairs.
- **NMI**: normalized mutual information between truth and predicted labels.
- **Assignment agreement**: spectrum accuracy after optimal one-to-one Hungarian
  matching of predicted and truth cluster IDs.
- **Pair precision/recall**: precision and recall of predicted same-cluster pairs.
- **Cluster-count agreement**: `min(K_pred, K_truth) / max(K_pred, K_truth)`.
- **Correct cluster ratio**: truth clusters whose complete member set exactly
  matches one predicted cluster, divided by all evaluated truth clusters.
- **Incorrect clustering ratio**: spectra differing from the majority peptide
  assignment in each non-noise cluster, divided by all non-noise spectra.
- **Completeness**: whether spectra with the same peptide assignment are placed
  in the same predicted cluster.
- **Clustered ratio**: spectra not assigned the noise label `-1`, divided by all
  evaluated spectra. Singleton clusters count as clustered for compatibility
  with the Hyper-Spec evaluator.
- Falcon noise label `-1` is converted to one singleton cluster per spectrum.
- Cluster labels are charge-namespaced before comparison.

## MultiRAM Functional Boundary

- Algorithm: {pim['algorithm']}
- HDC dimension: {pim['hv_dimension']}
- Threshold ratio: {pim['threshold_ratio']} of the fixed {pim['hv_dimension']}-bit HV width
- Bucket width: {pim['bucket_width_da']} Da, additionally separated by charge
- Input-charge fallback count: {scope['input_charge_fallback_count']:,}
- Input-charge/truth mismatch count: {scope['input_charge_truth_mismatch_count']:,}
- Functional encoding time: {pim['encoding_s']:.6f} s
- Functional clustering time: {pim['functional_clustering_s']:.6f} s

This functional execution produces MultiRAM algorithm labels for quality
measurement. Hardware latency and energy remain outputs of the separate FeRAM
analytical/NeuroSIM model; these host execution times are not PIM performance.

## Artifacts

- `clustering_quality_summary.json`: complete metrics and configuration
- `clustering_quality_summary.csv`: compact platform table
- `matched_assignments.csv`: per-spectrum truth and predicted assignments
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-mgf", type=Path, default=DEFAULT_MGF)
    parser.add_argument("--truth-pepxml", type=Path, default=DEFAULT_PEPXML)
    parser.add_argument("--falcon-csv", type=Path, default=DEFAULT_FALCON)
    parser.add_argument("--rapids-parquet", type=Path, default=DEFAULT_RAPIDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--error-threshold", type=float, default=0.01)
    parser.add_argument(
        "--max-spectra", type=int, default=0,
        help="maximum evaluated spectra; 0 evaluates all eligible spectra",
    )
    parser.add_argument(
        "--max-per-truth-cluster", type=int, default=0,
        help="maximum spectra per truth cluster; 0 keeps all members",
    )
    parser.add_argument("--threshold-ratio", type=float, default=0.455)
    parser.add_argument("--bucket-width", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for path in (args.input_mgf, args.truth_pepxml, args.falcon_csv, args.rapids_parquet):
        if not path.is_file():
            parser.error(f"required input not found: {path}")
    if args.max_spectra == 1 or args.max_spectra < 0:
        parser.error("--max-spectra must be 0 (unlimited) or at least 2")
    if args.max_per_truth_cluster == 1 or args.max_per_truth_cluster < 0:
        parser.error("--max-per-truth-cluster must be 0 (unlimited) or at least 2")

    min_probability = peptideprophet_threshold(args.truth_pepxml, args.error_threshold)
    truth = read_peptide_truth(args.truth_pepxml, min_probability)
    falcon = read_falcon(args.falcon_csv)
    rapids, crosswalk = crosswalk_rapids(args.rapids_parquet, falcon)
    falcon_by_index = falcon.set_index("raw_index", drop=False)
    rapids_by_index = rapids.set_index("raw_index", drop=False)
    available = set(falcon_by_index.index) & set(rapids_by_index.index)
    selected, eligible_spectra, eligible_truth_clusters = choose_subset(
        truth,
        available,
        max_spectra=args.max_spectra,
        max_per_truth_cluster=args.max_per_truth_cluster,
    )
    if not selected:
        raise RuntimeError("No repeated peptide-truth clusters matched all platforms")
    matched_high_confidence = sum(raw_index in available for raw_index in truth)

    all_spectra = parse_mgf_file(str(args.input_mgf), max_peaks=50, mz_max=2000.0)
    if max(selected) >= len(all_spectra):
        raise RuntimeError("Truth spectrum index exceeds parsed MGF spectrum count")
    spectra = [all_spectra[index] for index in selected]
    truth_charges = [int(truth[index]["charge"]) for index in selected]
    parsed_charges = [spectrum.get("precursor_charge") for _name, spectrum in spectra]
    charges = [
        int(parsed) if parsed is not None else truth_charge
        for parsed, truth_charge in zip(parsed_charges, truth_charges)
    ]
    charge_fallback_count = sum(charge is None for charge in parsed_charges)
    charge_mismatch_count = sum(
        parsed is not None and int(parsed) != truth_charge
        for parsed, truth_charge in zip(parsed_charges, truth_charges)
    )
    truth_labels = [str(truth[index]["truth_label"]) for index in selected]

    falcon_rows = falcon_by_index.loc[selected]
    rapids_rows = rapids_by_index.loc[selected]
    falcon_labels = namespace_labels(
        falcon_rows["cluster"].tolist(), charges, "falcon"
    )
    rapids_labels = namespace_labels(
        rapids_rows["cluster"].tolist(), charges, "rapids"
    )
    pim_labels, pim_details = run_multiram_functional_clustering(
        spectra,
        charges,
        threshold_ratio=args.threshold_ratio,
        bucket_width=args.bucket_width,
        seed=args.seed,
    )

    quality = {
        "falcon_cpu": quality_metrics(truth_labels, falcon_labels),
        "rapids_gpu": quality_metrics(truth_labels, rapids_labels),
        "multiram_pim_functional": quality_metrics(truth_labels, pim_labels),
    }
    summary = {
        "scope": {
            "input_mgf": str(args.input_mgf.resolve()),
            "truth_pepxml": str(args.truth_pepxml.resolve()),
            "falcon_csv": str(args.falcon_csv.resolve()),
            "rapids_parquet": str(args.rapids_parquet.resolve()),
            "peptideprophet_error_threshold": args.error_threshold,
            "peptideprophet_min_probability": min_probability,
            "high_confidence_psms": len(truth),
            "matched_high_confidence_psms": matched_high_confidence,
            "eligible_spectra": eligible_spectra,
            "eligible_truth_clusters": eligible_truth_clusters,
            "evaluated_spectra": len(selected),
            "truth_clusters": len(set(truth_labels)),
            "max_spectra": args.max_spectra,
            "max_per_truth_cluster": args.max_per_truth_cluster,
            "selection_is_exhaustive": len(selected) == eligible_spectra,
            "input_charge_fallback_count": charge_fallback_count,
            "input_charge_truth_mismatch_count": charge_mismatch_count,
        },
        "crosswalk": crosswalk,
        "multiram_functional_model": pim_details,
        "quality": quality,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "clustering_quality_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    compact = pd.DataFrame(
        [{"platform": platform, **metrics} for platform, metrics in quality.items()]
    )
    compact.to_csv(args.output_dir / "clustering_quality_summary.csv", index=False)
    assignments = pd.DataFrame(
        {
            "raw_index": selected,
            "peptideprophet_probability": [truth[index]["probability"] for index in selected],
            "truth_label": truth_labels,
            "falcon_label": falcon_labels,
            "rapids_label": rapids_labels,
            "multiram_label": pim_labels,
        }
    )
    assignments.to_csv(args.output_dir / "matched_assignments.csv", index=False)
    write_report(args.output_dir / "MULTIRAM_CLUSTERING_QUALITY_REPORT.md", summary)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Report: {args.output_dir / 'MULTIRAM_CLUSTERING_QUALITY_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
