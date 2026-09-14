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
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PROTEOMIC_DIR = ROOT / "proteomic_full_pipeline"
if str(PROTEOMIC_DIR) not in sys.path:
    sys.path.insert(0, str(PROTEOMIC_DIR))

from pim_hyperoms_estimator import (  # type: ignore
    _candidate_rows_for_vector_cache,
    _import_oms_symbols,
    _load_config,
    _load_ref_vector_cache,
    _prepare_query_spectra,
)
import pim_hyperoms_estimator as oms_estimator  # type: ignore

oms_estimator.__dict__.update(_import_oms_symbols())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oms-summary", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--feram-external-gbps", type=float, default=256.0)
    parser.add_argument("--dram-pnm-gbps", type=float, default=19010.0)
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

    payload = {
        "source_summary": str(summary_path),
        "model": (
            "OMS analytical PIM latency = candidate_hv_bytes / FeRAM_external_bandwidth; "
            "one 8192-bit reference HV read per candidate compare."
        ),
        "assumptions": {
            "hv_dim_bits": hv_dim,
            "hv_bytes_per_candidate": hv_bytes,
            "candidate_cap": candidate_cap,
            "feram_external_GBps": float(args.feram_external_gbps),
            "dram_internal_pnm_GBps": float(args.dram_pnm_gbps),
            "open_query_mode": args.open_query_mode,
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
            "feram_oms_compare_sec": feram_compare_sec,
            "feram_query_broadcast_sec": feram_query_broadcast_sec,
            "feram_score_result_sec": feram_result_sec,
            "feram_oms_total_sec": feram_compare_sec + feram_query_broadcast_sec + feram_result_sec,
            "dram_pnm_equivalent_compare_sec": dram_pnm_compare_sec,
            "simulator_runtime_sec": float(summary.get("simulator_wall_time_sec", 0.0)),
        },
    }

    output = Path(args.output) if args.output else summary_path.with_name("pim_model_latency_estimate.json")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nWrote: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
