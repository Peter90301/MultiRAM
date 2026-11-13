#!/usr/bin/env python3
"""Estimate analytical PIM latency from an OMS summary and bandwidth model.

This intentionally separates simulator runtime from modeled PIM latency.  The
OMS model uses candidate-window counts and assumes one 8192-bit reference HV
read/compare per candidate.  Query HV broadcast, result writeback, and FDR are
small relative to the reference-HV scan and are reported separately.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PROTEOMIC_DIR = ROOT / "proteomic_full_pipeline"
if str(PROTEOMIC_DIR) not in sys.path:
    sys.path.insert(0, str(PROTEOMIC_DIR))
HOMSTC_DIR = ROOT / "sumukh_proteomic_test/homs-tc"
if str(HOMSTC_DIR) not in sys.path:
    sys.path.insert(0, str(HOMSTC_DIR))

from pim_hyperoms_estimator import (  # type: ignore
    _candidate_rows_for_vector_cache,
    _import_oms_symbols,
    _load_config,
    _load_ref_vector_cache,
    _prepare_query_spectra,
)
import pim_hyperoms_estimator as oms_estimator  # type: ignore



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oms-summary", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--feram-external-gbps", type=float, default=256.0)
    parser.add_argument("--dram-pnm-gbps", type=float, default=19010.0)
    parser.add_argument("--feram-num-tiles", type=int, default=32)
    parser.add_argument("--feram-columns-per-tile", type=int, default=512)
    parser.add_argument("--restore-latency-ns-per-activated-row", type=float, default=50.0)
    parser.add_argument("--restore-energy-fj-per-bit", type=float, default=39.7)
    parser.add_argument("--feram-cell-read-energy-fj-per-bit", type=float, default=2.0)
    parser.add_argument("--feram-adc-energy-fj-per-candidate", type=float, default=200.0)
    parser.add_argument("--feram-digital-energy-fj-per-candidate", type=float, default=50.0)
    parser.add_argument("--feram-dac-energy-fj-per-query-row", type=float, default=20.0)
    parser.add_argument("--fenand-array-latency-us-per-query-stage", type=float, default=0.429e6 / 46665.0)
    parser.add_argument("--fenand-filter-energy-pj-per-query-stage", type=float, default=0.491e12 / 46665.0)
    parser.add_argument("--assumed-feram-endurance-cycles", type=float, default=1e12)
    parser.add_argument("--encoding-query-batch-size", type=int, default=32)
    parser.add_argument("--encoding-cycles-per-batch", type=int, default=100)
    parser.add_argument("--encoding-clock-hz", type=float, default=500e6)
    parser.add_argument("--encoding-energy-pj-per-query", type=float, default=50.0)
    parser.add_argument(
        "--open-query-mode",
        choices=["all", "summary_count_prefix"],
        default="summary_count_prefix",
        help=(
            "For open search, exact query IDs require replaying FDR. "
            "summary_count_prefix uses the summary's open_stage_queries count as "
            "a deterministic approximation; all is an upper bound."
        ),
    )
    return parser.parse_args()


def candidate_count_for_queries(
    queries,
    vector_caches,
    tolerance_resolver,
    candidate_cap: int,
) -> tuple[int, list[int]]:
    available_charges = sorted(vector_caches)
    per_query: list[int] = []
    total = 0
    for query in queries:
        charge = query.precursor_charge
        charges = [int(charge)] if charge is not None and int(charge) in vector_caches else available_charges
        windows = _candidate_rows_for_vector_cache(
            float(query.precursor_mz),
            charges,
            vector_caches,
            float(tolerance_resolver(query)),
            int(candidate_cap),
        )
        count = sum(max(0, right - left) for _charge, left, right in windows)
        per_query.append(int(count))
        total += int(count)
    return total, per_query


def summarize_counts(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "max": None, "total": 0}
    sorted_values = sorted(values)
    p95_idx = min(len(sorted_values) - 1, int(0.95 * (len(sorted_values) - 1)))
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p95": sorted_values[p95_idx],
        "max": max(values),
        "total": sum(values),
    }


def main() -> int:
    args = parse_args()
    oms_estimator.__dict__.update(_import_oms_symbols())
    summary_path = Path(args.oms_summary)
    summary: dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8"))

    query_path = Path(summary["query"])
    ref_path = Path(summary["ref"])
    config_path = Path(summary["config"])
    candidate_cap = int(summary.get("candidate_cap", 0))
    hv_dim = int(summary.get("hv_dimensionality", 8192))
    hv_bytes = hv_dim // 8
    std_ppm = float(summary.get("std_ppm", 5.0))
    open_da = float(summary.get("open_da", 500.0))
    max_queries = int(summary.get("max_queries", 0))

    homs_cfg = _load_config(config_path)
    queries = _prepare_query_spectra(query_path, homs_cfg, 2000.0)
    if max_queries > 0:
        queries = queries[:max_queries]
    vector_caches = _load_ref_vector_cache(ref_path, homs_cfg)

    std_total, std_per_query = candidate_count_for_queries(
        queries,
        vector_caches,
        lambda q: float(q.precursor_mz) * std_ppm * 1e-6,
        candidate_cap,
    )

    open_stage_queries = int(summary.get("open_stage_queries", len(queries)))
    if args.open_query_mode == "all":
        open_queries = queries
    else:
        open_queries = queries[:open_stage_queries]
    open_total, open_per_query = candidate_count_for_queries(
        open_queries,
        vector_caches,
        lambda _q: open_da,
        candidate_cap,
    )

    total_candidates = std_total + open_total
    compare_bytes = total_candidates * hv_bytes
    query_broadcast_bytes = (len(queries) + len(open_queries)) * hv_bytes
    result_bytes = total_candidates * 4

    feram_compare_sec = compare_bytes / (float(args.feram_external_gbps) * 1e9)
    feram_query_broadcast_sec = query_broadcast_bytes / (float(args.feram_external_gbps) * 1e9)
    feram_result_sec = result_bytes / (float(args.feram_external_gbps) * 1e9)
    # A 3D-DRAM PNM equivalent lower-bound view for the same HV scan.
    dram_pnm_compare_sec = compare_bytes / (float(args.dram_pnm_gbps) * 1e9)
    encoding = summary.get("encoding", {})
    encoding_batch_size = int(encoding.get("query_batch_size", args.encoding_query_batch_size))
    encoding_sec = float(
        encoding.get(
            "latency_sec",
            math.ceil(len(queries) / max(1, encoding_batch_size))
            * max(1, int(args.encoding_cycles_per_batch))
            / max(1.0, float(args.encoding_clock_hz)),
        )
    )
    encoding_energy_mj = float(
        encoding.get(
            "energy_mj",
            len(queries) * max(0.0, float(args.encoding_energy_pj_per_query)) * 1e-9,
        )
    )
    resident_columns = max(1, int(args.feram_num_tiles) * int(args.feram_columns_per_tile))
    restore_passes = sum(math.ceil(count / resident_columns) for count in std_per_query if count > 0)
    restore_passes += sum(math.ceil(count / resident_columns) for count in open_per_query if count > 0)
    restore_sec = restore_passes * max(0.0, float(args.restore_latency_ns_per_activated_row)) * 1e-9
    restore_energy_mj = (
        total_candidates
        * hv_dim
        * max(0.0, float(args.restore_energy_fj_per_bit))
        * 1e-12
    )
    query_stage_count = len(queries) + len(open_queries)
    feram_array_read_energy_mj = (
        total_candidates * hv_dim * max(0.0, float(args.feram_cell_read_energy_fj_per_bit)) * 1e-12
    )
    feram_adc_digital_energy_mj = (
        total_candidates
        * (max(0.0, float(args.feram_adc_energy_fj_per_candidate))
           + max(0.0, float(args.feram_digital_energy_fj_per_candidate)))
        * 1e-12
    )
    feram_dac_energy_mj = (
        query_stage_count * hv_dim * max(0.0, float(args.feram_dac_energy_fj_per_query_row)) * 1e-12
    )
    fenand_array_latency_sec = (
        query_stage_count * max(0.0, float(args.fenand_array_latency_us_per_query_stage)) * 1e-6
    )
    fenand_array_energy_mj = (
        query_stage_count * max(0.0, float(args.fenand_filter_energy_pj_per_query_stage)) * 1e-9
    )
    feram_array_total_energy_mj = (
        encoding_energy_mj + feram_array_read_energy_mj + feram_adc_digital_energy_mj
        + feram_dac_energy_mj + restore_energy_mj
    )

    payload = {
        "source_summary": str(summary_path),
        "model": (
            "OMS analytical PIM latency includes query encoding on the FeRAM logic die "
            "plus candidate-HV reads, destructive-read restore, query broadcast, and result writeback."
        ),
        "assumptions": {
            "hv_dim_bits": hv_dim,
            "hv_bytes_per_candidate": hv_bytes,
            "candidate_cap": candidate_cap,
            "feram_external_GBps": float(args.feram_external_gbps),
            "dram_internal_pnm_GBps": float(args.dram_pnm_gbps),
            "open_query_mode": args.open_query_mode,
            "query_batch_size": encoding_batch_size,
            "encoding_placement": encoding.get("placement", "FeRAM logic die"),
            "read_scheme": "destructive read plus restore",
            "feram_resident_columns": resident_columns,
            "restore_latency_ns_per_activated_row": float(args.restore_latency_ns_per_activated_row),
            "restore_energy_fj_per_bit": float(args.restore_energy_fj_per_bit),
            "feram_cell_read_energy_fj_per_bit": float(args.feram_cell_read_energy_fj_per_bit),
            "feram_adc_energy_fj_per_candidate": float(args.feram_adc_energy_fj_per_candidate),
            "feram_digital_energy_fj_per_candidate": float(args.feram_digital_energy_fj_per_candidate),
            "feram_dac_energy_fj_per_query_row": float(args.feram_dac_energy_fj_per_query_row),
            "fenand_array_latency_us_per_query_stage": float(args.fenand_array_latency_us_per_query_stage),
            "fenand_filter_energy_pj_per_query_stage": float(args.fenand_filter_energy_pj_per_query_stage),
            "fenand_model_provenance": "Hsu et al. HEK293 workload calibration: 0.429 s and 0.491 J for 46,665 queries",
            "assumed_feram_endurance_cycles": float(args.assumed_feram_endurance_cycles),
        },
        "query_counts": {
            "standard_queries": len(queries),
            "open_queries_modeled": len(open_queries),
            "open_stage_queries_from_summary": open_stage_queries,
        },
        "candidate_counts": {
            "standard_total": std_total,
            "open_total": open_total,
            "total": total_candidates,
            "standard_per_query": summarize_counts(std_per_query),
            "open_per_query": summarize_counts(open_per_query),
        },
        "modeled_bytes": {
            "candidate_hv_read_bytes": compare_bytes,
            "query_hv_broadcast_bytes": query_broadcast_bytes,
            "score_result_bytes": result_bytes,
        },
        "latency_sec": {
            "feram_query_encoding_sec": encoding_sec,
            "feram_restore_sec": restore_sec,
            "feram_oms_compare_sec": feram_compare_sec,
            "feram_query_broadcast_sec": feram_query_broadcast_sec,
            "feram_score_result_sec": feram_result_sec,
            "fenand_array_filter_sec": fenand_array_latency_sec,
            "feram_oms_core_sec": (
                encoding_sec
                + restore_sec
                + feram_compare_sec
                + feram_query_broadcast_sec
                + feram_result_sec
            ),
            "feram_oms_total_sec": (
                encoding_sec
                + restore_sec
                + feram_compare_sec
                + feram_query_broadcast_sec
                + feram_result_sec
                + fenand_array_latency_sec
            ),
            "dram_pnm_equivalent_compare_sec": dram_pnm_compare_sec,
            "simulator_runtime_sec": float(summary.get("simulator_wall_time_sec", 0.0)),
        },
        "energy_mj": {
            "feram_query_encoding_mj": encoding_energy_mj,
            "feram_array_read_mj": feram_array_read_energy_mj,
            "feram_adc_digital_mj": feram_adc_digital_energy_mj,
            "feram_dac_mj": feram_dac_energy_mj,
            "feram_restore_mj": restore_energy_mj,
            "feram_array_total_mj": feram_array_total_energy_mj,
            "fenand_array_filter_mj": fenand_array_energy_mj,
            "feram_plus_fenand_array_total_mj": feram_array_total_energy_mj + fenand_array_energy_mj,
            "feram_encoding_plus_restore_mj": encoding_energy_mj + restore_energy_mj,
            "encoding_included": bool(encoding.get("included_in_modeled_energy", True)),
        },
        "endurance": {
            "restore_passes": restore_passes,
            "assumed_cycles": float(args.assumed_feram_endurance_cycles),
            "fraction_of_endurance_for_most_stressed_sequential_path": (
                restore_passes / float(args.assumed_feram_endurance_cycles)
            ),
        },
    }

    output = Path(args.output) if args.output else summary_path.with_name("pim_model_latency_estimate.json")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nWrote: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
