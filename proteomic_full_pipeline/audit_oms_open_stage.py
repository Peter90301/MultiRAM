#!/usr/bin/env python3
"""Replay the existing PIM OMS policy and save the actual open-stage queries.

This is an audit helper, not a performance shortcut.  It deliberately uses the
same 8192-bit estimator, score mode, candidate cap, and FDR policy as the
recorded b1926 run.  The saved query identifiers are then used by the
uncapped 2048-bit open-search sensitivity experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEPENDENCY_ROOTS = [
    ROOT,
    ROOT / "proteomic_full_pipeline",
    ROOT / "sumukh_proteomic_test/homs-tc",
]
for dependency_root in reversed(DEPENDENCY_ROOTS):
    if str(dependency_root) not in sys.path:
        sys.path.insert(0, str(dependency_root))

import pim_hyperoms_estimator as oms  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query",
        required=True,
    )
    parser.add_argument(
        "--ref",
        required=True,
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "sumukh_proteomic_test/homs-tc/configs/hek293.ini"),
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "fenand_oms_open_sensitivity" / "open_stage_queries_b1926.npz"),
    )
    parser.add_argument("--candidate-cap", type=int, default=512)
    parser.add_argument("--gpu-chunk-size", type=int, default=8192)
    parser.add_argument("--hv-cache-dir", default=None)
    parser.add_argument("--fdr-threshold", type=float, default=0.01)
    parser.add_argument("--std-ppm", type=float, default=5.0)
    parser.add_argument("--open-da", type=float, default=500.0)
    parser.add_argument("--hv-seed", type=int, default=13)
    parser.add_argument("--hv-flip-bits", type=int, default=64)
    parser.add_argument("--score-mode", choices=["raw_hamming", "normalized", "zscore", "zscore_margin"], default="normalized")
    parser.add_argument("--margin-weight", type=float, default=1.0)
    parser.add_argument("--max-queries", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    oms.__dict__.update(oms._import_oms_symbols())

    query_path = Path(args.query)
    ref_path = Path(args.ref)
    config_path = Path(args.config)
    config = oms._load_config(config_path)
    queries = oms._prepare_query_spectra(query_path, config, 2000.0)
    if args.max_queries > 0:
        queries = queries[: args.max_queries]
    if not queries:
        raise RuntimeError("No valid query spectra were parsed")

    encoder = oms.HdcEncoder(
        spectrum_dim=config.spectrum_vector_dim,
        hv_dim=config.hv_dimensionality,
        quant_levels=config.hv_quantize_level,
        flip_bits=args.hv_flip_bits,
        seed=args.hv_seed,
        codebook="random",
    )
    encoded = [oms.EncodedSpectrum(q, encoder.encode(q, config)) for q in queries]
    vector_caches = oms._load_ref_vector_cache(ref_path, config)
    cache_dir = Path(args.hv_cache_dir) if args.hv_cache_dir else ref_path.parent
    hv_caches = oms._ensure_gpu_hv_cache(
        ref_path=ref_path,
        cache_dir=cache_dir,
        config=config,
        encoder=encoder,
        vector_caches=vector_caches,
        chunk_size=args.gpu_chunk_size,
        seed=args.hv_seed,
        flip_bits=args.hv_flip_bits,
    )
    resident_hvs = oms._load_gpu_resident_hvs(vector_caches, hv_caches)
    hit_refs = oms._search_stage_gpu(
        encoded,
        vector_caches,
        hv_caches,
        encoder,
        candidate_cap=args.candidate_cap,
        tolerance_resolver=lambda q: float(q.precursor_mz) * args.std_ppm * 1e-6,
        stage="standard",
        chunk_size=args.gpu_chunk_size,
        score_mode=args.score_mode,
        margin_weight=args.margin_weight,
        progress_interval=5000,
        resident_hvs=resident_hvs,
    )
    with oms.SpectralLibraryReader(str(ref_path), config) as reader:
        hits = oms._materialize_gpu_hits(hit_refs, reader, config)
        accepted, counts = oms._strict_fdr_merge(hits, [], args.fdr_threshold, config)

    identified = {ssm.query_identifier for ssm in accepted}
    open_indices = [i for i, q in enumerate(encoded) if q.spectrum.identifier not in identified]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np

    np.savez_compressed(
        output,
        query_indices=np.asarray(open_indices, dtype=np.int64),
        query_identifiers=np.asarray([encoded[i].spectrum.identifier for i in open_indices], dtype=str),
        all_query_identifiers=np.asarray([q.spectrum.identifier for q in encoded], dtype=str),
    )
    metadata = {
        "query": str(query_path),
        "reference": str(ref_path),
        "config": str(config_path),
        "num_queries": len(encoded),
        "standard_candidate_cap_for_policy_replay": args.candidate_cap,
        "standard_accepted": len(accepted),
        "open_stage_queries": len(open_indices),
        "counts": counts,
        "policy": {
            "std_ppm": args.std_ppm,
            "open_da": args.open_da,
            "fdr_threshold": args.fdr_threshold,
            "hv_dimensionality": config.hv_dimensionality,
            "hv_seed": args.hv_seed,
            "hv_flip_bits": args.hv_flip_bits,
            "score_mode": args.score_mode,
            "margin_weight": args.margin_weight,
        },
        "note": "candidate_cap is used only to reproduce the existing open-stage membership policy; it is not used in the uncapped exact ground truth or FeNAND sensitivity experiment.",
    }
    metadata_path = output.with_suffix(output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"Wrote: {output}")
    print(f"Wrote: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
