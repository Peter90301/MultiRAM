#!/usr/bin/env python3
"""Project FeRAM clustering and HyperOMS time for PXD024364."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path("/home/tsl012/multiomic")
HMP2_ROOT = Path("/mnt/hdd/tsunghan/raw-ms-dataset/HMP2_IBDMDB")
RESULTS = ROOT / "multiram_data_movement/results_pxd024364_1tb"
AGGREGATE_JSON = RESULTS / "aggregate_metrics.json"
CLUSTER_MODEL_JSON = (
    HMP2_ROOT
    / "timing_full_pim_hmp2_sample_10k_fullproteomics_gpu1/"
    "proteomic_pim/pim_clustering_summary.json"
)
OMS_MODEL_JSON = (
    HMP2_ROOT
    / "timing_full_pim_hmp2_sample_10k_fullproteomics_gpu1/"
    "proteomic_pim/pim_model_latency_estimate.json"
)

TABLE_CLUSTER_MGF = Path(
    "/mnt/hdd/tsunghan/raw-ms-dataset/proteomic_large_dataset/derived/"
    "ms2_mgf/20151220_alr_CompleteHumanProteome_HUVEC_LysC_ETD_fr9.mgf"
)
TABLE_OMS_MGF = Path(
    "/mnt/hdd/tsunghan/raw-ms-dataset/proteomic_large_dataset/derived/"
    "ms2_mgf/20151201_alr_CompleteHumanProteome_HepG2_ETD_LysN_fr15.mgf"
)

FENAND_DECOMPRESSED_BYTES_S = 8.1e9
FERAM_EXTERNAL_BYTES_S = 256e9
CLUSTER_HV_BYTES = 2048 // 8
OMS_HV_BYTES = 8192 // 8
RESULT_ID_BYTES = 4
OMS_REFERENCE_SPECTRA = 2_992_672
TABLE_OMS_VALID_QUERIES = 29_835


def count_mgf_spectra(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for line in handle:
            if line.rstrip() == b"BEGIN IONS":
                count += 1
    return count


def clustering_projection(
    *,
    mgf_bytes: int,
    spectra: int,
    core_seconds_per_spectrum: float,
) -> dict[str, float | int]:
    core_s = spectra * core_seconds_per_spectrum
    mgf_ingress_s = mgf_bytes / FENAND_DECOMPRESSED_BYTES_S
    encoded_hv_bytes = spectra * CLUSTER_HV_BYTES
    result_bytes = spectra * RESULT_ID_BYTES
    feram_link_s = (encoded_hv_bytes + result_bytes) / FERAM_EXTERNAL_BYTES_S
    movement_s = mgf_ingress_s + feram_link_s
    overall_s = core_s + movement_s
    return {
        "mgf_bytes": mgf_bytes,
        "spectra": spectra,
        "core_s": core_s,
        "mgf_ingress_s": mgf_ingress_s,
        "encoded_hv_bytes": encoded_hv_bytes,
        "result_bytes": result_bytes,
        "feram_link_s": feram_link_s,
        "movement_s": movement_s,
        "overall_s": overall_s,
        "movement_fraction_percent": 100.0 * movement_s / overall_s,
    }


def oms_projection(
    *,
    mgf_bytes: int,
    query_spectra: int,
    open_query_ratio: float,
    standard_candidates_per_query: float,
    open_candidates_per_query: float,
) -> dict[str, float | int]:
    open_queries = round(query_spectra * open_query_ratio)
    standard_candidates = round(query_spectra * standard_candidates_per_query)
    open_candidates = round(open_queries * open_candidates_per_query)
    candidates = standard_candidates + open_candidates

    candidate_hv_bytes = candidates * OMS_HV_BYTES
    query_hv_bytes = (query_spectra + open_queries) * OMS_HV_BYTES
    candidate_score_bytes = candidates * RESULT_ID_BYTES
    final_result_bytes = query_spectra * RESULT_ID_BYTES
    reference_hv_bytes = OMS_REFERENCE_SPECTRA * OMS_HV_BYTES

    compare_s = candidate_hv_bytes / FERAM_EXTERNAL_BYTES_S
    query_broadcast_s = query_hv_bytes / FERAM_EXTERNAL_BYTES_S
    candidate_score_s = candidate_score_bytes / FERAM_EXTERNAL_BYTES_S
    oms_core_s = compare_s + query_broadcast_s + candidate_score_s

    mgf_ingress_s = mgf_bytes / FENAND_DECOMPRESSED_BYTES_S
    reference_load_s = reference_hv_bytes / FERAM_EXTERNAL_BYTES_S
    final_result_s = final_result_bytes / FERAM_EXTERNAL_BYTES_S
    movement_s = (
        mgf_ingress_s
        + reference_load_s
        + query_broadcast_s
        + final_result_s
    )
    overall_s = (
        oms_core_s
        + mgf_ingress_s
        + reference_load_s
        + final_result_s
    )
    return {
        "mgf_bytes": mgf_bytes,
        "query_spectra": query_spectra,
        "open_queries": open_queries,
        "standard_candidates": standard_candidates,
        "open_candidates": open_candidates,
        "total_candidates": candidates,
        "candidate_hv_bytes": candidate_hv_bytes,
        "query_hv_bytes": query_hv_bytes,
        "candidate_score_bytes": candidate_score_bytes,
        "reference_hv_bytes": reference_hv_bytes,
        "final_result_bytes": final_result_bytes,
        "compare_s": compare_s,
        "query_broadcast_s": query_broadcast_s,
        "candidate_score_s": candidate_score_s,
        "core_s": oms_core_s,
        "mgf_ingress_s": mgf_ingress_s,
        "reference_load_s": reference_load_s,
        "final_result_s": final_result_s,
        "movement_s": movement_s,
        "overall_s": overall_s,
        "movement_fraction_percent": 100.0 * movement_s / overall_s,
    }


def table_row(workload: str, platform: str, result: dict[str, float | int]) -> str:
    return (
        f"{workload} & {platform} & "
        f"{float(result['overall_s']):,.3f} s & "
        f"{float(result['movement_s']):,.3f} s & "
        f"{float(result['movement_fraction_percent']):.3f}\\% \\\\"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    args = parser.parse_args()

    aggregate = json.loads(AGGREGATE_JSON.read_text(encoding="utf-8"))
    cluster_model = json.loads(CLUSTER_MODEL_JSON.read_text(encoding="utf-8"))
    oms_model = json.loads(OMS_MODEL_JSON.read_text(encoding="utf-8"))

    cluster_source_spectra = int(cluster_model["num_spectra"])
    cluster_source_core_s = float(
        cluster_model["result"]["feram_cluster_core_s"]
    )
    cluster_core_per_spectrum = cluster_source_core_s / cluster_source_spectra

    candidate_counts = oms_model["candidate_counts"]
    standard_query_count = int(oms_model["query_counts"]["standard_queries"])
    open_query_count = int(oms_model["query_counts"]["open_queries_modeled"])
    standard_candidates_per_query = (
        int(candidate_counts["standard_total"]) / standard_query_count
    )
    open_candidates_per_query = (
        int(candidate_counts["open_total"]) / open_query_count
    )
    open_query_ratio = open_query_count / standard_query_count

    table_cluster_spectra = count_mgf_spectra(TABLE_CLUSTER_MGF)
    table_oms_spectra = count_mgf_spectra(TABLE_OMS_MGF)
    if table_oms_spectra < TABLE_OMS_VALID_QUERIES:
        raise ValueError("Valid OMS query count exceeds MGF spectrum count")

    completed = aggregate["stages"]["raw_conversion"]
    full_mgf_bytes = int(completed["output_bytes_sum"])
    full_spectra = int(completed["spectra_sum"])
    table_valid_ratio = TABLE_OMS_VALID_QUERIES / table_oms_spectra
    full_oms_queries = round(full_spectra * table_valid_ratio)

    table_scope = {
        "clustering": clustering_projection(
            mgf_bytes=TABLE_CLUSTER_MGF.stat().st_size,
            spectra=table_cluster_spectra,
            core_seconds_per_spectrum=cluster_core_per_spectrum,
        ),
        "oms": oms_projection(
            mgf_bytes=TABLE_OMS_MGF.stat().st_size,
            query_spectra=TABLE_OMS_VALID_QUERIES,
            open_query_ratio=open_query_ratio,
            standard_candidates_per_query=standard_candidates_per_query,
            open_candidates_per_query=open_candidates_per_query,
        ),
    }
    full_scope = {
        "clustering": clustering_projection(
            mgf_bytes=full_mgf_bytes,
            spectra=full_spectra,
            core_seconds_per_spectrum=cluster_core_per_spectrum,
        ),
        "oms": oms_projection(
            mgf_bytes=full_mgf_bytes,
            query_spectra=full_oms_queries,
            open_query_ratio=open_query_ratio,
            standard_candidates_per_query=standard_candidates_per_query,
            open_candidates_per_query=open_candidates_per_query,
        ),
    }

    output = {
        "status": "analytical_projection_not_hardware_measurement",
        "dataset": "PXD024364 / MSV000086944",
        "model": {
            "fenand_decompressed_bytes_s": FENAND_DECOMPRESSED_BYTES_S,
            "feram_external_bytes_s": FERAM_EXTERNAL_BYTES_S,
            "clustering_hv_bits": CLUSTER_HV_BYTES * 8,
            "oms_hv_bits": OMS_HV_BYTES * 8,
            "oms_reference_spectra": OMS_REFERENCE_SPECTRA,
            "clustering_source_spectra": cluster_source_spectra,
            "clustering_source_core_s": cluster_source_core_s,
            "clustering_core_s_per_spectrum": cluster_core_per_spectrum,
            "oms_standard_candidates_per_query": (
                standard_candidates_per_query
            ),
            "oms_open_candidates_per_query": open_candidates_per_query,
            "oms_open_query_ratio": open_query_ratio,
        },
        "table_matched_single_file_scope": {
            "description": (
                "PIM rows matching the PXD024364 files used by the existing "
                "CPU/GPU table."
            ),
            "clustering_mgf": str(TABLE_CLUSTER_MGF),
            "oms_mgf": str(TABLE_OMS_MGF),
            "oms_raw_spectra": table_oms_spectra,
            "oms_valid_query_spectra": TABLE_OMS_VALID_QUERIES,
            **table_scope,
        },
        "full_converted_batch_scope": {
            "description": (
                "Projection for the 853 successfully converted files in the "
                "approximately 1 TB RAW selection."
            ),
            "files": int(completed["completed_files"]),
            "mgf_bytes": full_mgf_bytes,
            "spectra": full_spectra,
            "oms_valid_query_spectra": full_oms_queries,
            **full_scope,
        },
        "movement_boundary": {
            "clustering": (
                "MGF ingress, encoded 2048-bit HV transfer, and final cluster IDs."
            ),
            "oms": (
                "MGF ingress, one cold reference-HV load, query-HV broadcast, "
                "and final IDs. In-array candidate-HV reads and candidate scores "
                "are part of OMS core time and excluded from explicit package "
                "movement to match the CPU/GPU table boundary."
            ),
        },
        "limitations": [
            "The PIM values are analytical projections, not hardware measurements.",
            "Clustering core time scales linearly by spectra from the FeRAM bucket model.",
            "OMS uses uncapped candidate densities measured on the HMP2 query sample.",
            "PXD024364 precursor-mass composition can change OMS candidate density.",
            "The full scope covers converted MGF data, not RAW-to-MGF conversion.",
        ],
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.results_dir / "PXD024364_PIM_RUNTIME_PROJECTION.json"
    json_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    tc = table_scope["clustering"]
    to = table_scope["oms"]
    fc = full_scope["clustering"]
    fo = full_scope["oms"]
    lines = [
        "# PXD024364 PIM Runtime and Movement Projection",
        "",
        "These are analytical MultiRAM projections, not hardware measurements.",
        "",
        "## Rows For The Existing Table",
        "",
        "These rows use the same PXD024364 input files as the CPU/GPU rows in Table VI.",
        "",
        "| Workload | Platform | Overall | Movement | Fraction |",
        "| --- | --- | ---: | ---: | ---: |",
        f"| Clustering | MultiRAM FeRAM | **{tc['overall_s']:.6f} s** | **{tc['movement_s']:.6f} s** | **{tc['movement_fraction_percent']:.2f}%** |",
        f"| OMS | MultiRAM HyperOMS | **{to['overall_s']:.3f} s** | **{to['movement_s']:.6f} s** | **{to['movement_fraction_percent']:.4f}%** |",
        "",
        "LaTeX rows:",
        "",
        "```tex",
        table_row("Clustering", "MultiRAM FeRAM", tc),
        table_row("OMS", "MultiRAM HyperOMS", to),
        "```",
        "",
        "## Full PXD024364 Converted Batch",
        "",
        f"Scope: {int(completed['completed_files'])} converted files, "
        f"{full_mgf_bytes / 1e9:.3f} GB MGF, and {full_spectra:,} spectra.",
        "",
        "| Workload | Platform | Overall | Movement | Fraction |",
        "| --- | --- | ---: | ---: | ---: |",
        f"| Clustering | MultiRAM FeRAM | **{fc['overall_s']:.3f} s** | **{fc['movement_s']:.3f} s** | **{fc['movement_fraction_percent']:.2f}%** |",
        f"| OMS | MultiRAM HyperOMS | **{fo['overall_s']:,.3f} s ({fo['overall_s'] / 3600:.2f} h)** | **{fo['movement_s']:.3f} s** | **{fo['movement_fraction_percent']:.4f}%** |",
        "",
        "## Interpretation",
        "",
        f"- Clustering core time is {fc['core_s']:.3f} s; the "
        f"{fc['mgf_ingress_s']:.3f} s MGF stream dominates overall time.",
        f"- Full-batch OMS models {int(fo['total_candidates']):,} candidate "
        f"comparisons. Its {fo['core_s'] / 3600:.2f} h core time dominates the "
        f"{fo['movement_s']:.3f} s explicit movement time.",
        "- The OMS fraction excludes in-array candidate-HV reads, matching the "
        "existing table's explicit storage/link boundary. Counting those reads "
        "as movement would make the OMS core almost entirely memory-service time.",
        "",
        "## Limits",
        "",
        "- PIM values are modeled, not measured.",
        "- OMS candidate density comes from the uncapped HMP2 model; PXD mass "
        "distributions can produce a different candidate count.",
        "- The full-batch projection starts from converted MGF and excludes "
        "Thermo RAW-to-MGF conversion.",
        "",
    ]
    report_path = args.results_dir / "PXD024364_PIM_RUNTIME_PROJECTION.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(report_path)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
