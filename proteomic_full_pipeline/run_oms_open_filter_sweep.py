#!/usr/bin/env python3
"""Exact, uncapped FeNAND sensitivity for the b1926 500-Da retry stage.

The standard-stage replay determines which queries enter the open retry.  For
those queries this script scans every legal precursor/charge-matched reference
row in streaming chunks.  It never materializes the open candidate matrix and
never applies the estimator's ``candidate_cap``.  Full Hamming Top-8 and all
partial-score threshold counts are accumulated from the same scan.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OMS_ROOT = ROOT / "proteomic_full_pipeline"
HOMSTC_ROOT = ROOT / "sumukh_proteomic_test/homs-tc"
for item in (ROOT, OMS_ROOT, HOMSTC_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

# Import the validated standard-stage implementation without duplicating its
# Hyper-Spec encoder and reference-vector preparation.
import run_oms_filter_sweep as standard  # type: ignore


HV_DIM = 2048
SEGMENT_BITS = 64
TOP_K = 8
DEFAULT_THRESHOLDS = (1031, 1074, 1096, 1116, 1134, 1149, 1160, 1185)
ROW_STRIDE = standard.ROW_STRIDE


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
        "--open-stage-ids",
        default=str(ROOT / "fenand_oms_open_sensitivity/open_stage_queries_b1926.npz"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "fenand_oms_open_sensitivity"),
    )
    parser.add_argument("--open-da", type=float, default=500.0)
    parser.add_argument("--chunk-size", type=int, default=262144)
    parser.add_argument("--reference-cache-dir", default=str(ROOT / "fenand_oms_sensitivity/reference_hv_cache"))
    parser.add_argument("--segments-examined", type=int, default=28)
    parser.add_argument("--thresholds", nargs="+", type=int, default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_sorted_reference_hvs(vector_files: dict[int, dict], cache_dir: Path) -> dict[int, dict]:
    result = {}
    for charge, info in sorted(vector_files.items()):
        path = cache_dir / f"reference_charge{charge}.uint8.mmap"
        expected = int(info["rows"]) * (HV_DIM // 8)
        if not path.exists() or path.stat().st_size != expected:
            raise FileNotFoundError(
                f"Missing 2048-bit reference cache {path}; run the standard sensitivity experiment first"
            )
        raw = np.memmap(path, dtype=np.uint8, mode="r", shape=(int(info["rows"]), HV_DIM // 8))
        order = np.asarray(info["sorted_rows"], dtype=np.int64)
        result[charge] = {
            "hvs": np.asarray(raw[order]),
            "sorted_mz": np.asarray(info["sorted_mz"], dtype=np.float64),
            "rows": int(info["rows"]),
        }
    return result


def query_windows(queries, refs: dict[int, dict], open_indices: np.ndarray, open_da: float):
    windows = []
    counts = np.zeros(len(open_indices), dtype=np.int64)
    available = sorted(refs)
    for out_idx, query_index in enumerate(open_indices.tolist()):
        query = queries[int(query_index)]
        charge = int(query.precursor_charge) if query.precursor_charge is not None else -1
        charges = [charge] if charge in refs else available
        query_windows = []
        total = 0
        mz = float(query.precursor_mz)
        for current_charge in charges:
            info = refs[current_charge]
            left = int(np.searchsorted(info["sorted_mz"], mz - open_da, side="left"))
            right = int(np.searchsorted(info["sorted_mz"], mz + open_da, side="right"))
            if right > left:
                query_windows.append((current_charge, left, right))
                total += right - left
        windows.append(query_windows)
        counts[out_idx] = total
    return windows, counts


def system_metadata() -> dict:
    data = {"platform": platform.platform(), "python": sys.version, "cwd": str(Path.cwd())}
    try:
        data["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True
        ).strip()
    except Exception:
        data["nvidia_smi"] = "unavailable"
    try:
        data["git_commit"] = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        data["git_commit"] = "unavailable"
    return data


def exact_streaming_sweep(query_hvs, open_queries, windows, refs, thresholds, segments, chunk_size):
    cp = standard.HyperSpec2048Encoder(standard.CODEBOOK_PATH).cp
    lut = cp.asarray(standard.oms.BYTE_POPCOUNT if hasattr(standard, "oms") else np.unpackbits(
        np.arange(256, dtype=np.uint8)[:, None], axis=1
    ).sum(axis=1).astype(np.uint8))
    # Import the estimator's packed-byte popcount table when available.
    try:
        import pim_hyperoms_estimator as oms  # type: ignore
        lut = cp.asarray(oms.BYTE_POPCOUNT)
    except Exception:
        pass

    threshold_values = [int(x) for x in thresholds]
    threshold_count = len(threshold_values)
    survivors = np.zeros((len(open_queries), threshold_count), dtype=np.int64)
    top_ids = np.full((len(open_queries), TOP_K), -1, dtype=np.int64)
    top_scores = np.full((len(open_queries), TOP_K), -1, dtype=np.int32)
    top8_retained = np.zeros((len(open_queries), threshold_count), dtype=np.int8)
    top1_retained = np.zeros((len(open_queries), threshold_count), dtype=np.int8)
    start_time = time.time()
    max_bytes = segments * SEGMENT_BITS // 8

    for qi, (query_index, query_windows) in enumerate(zip(open_queries, windows)):
        qgpu = cp.asarray(query_hvs[int(query_index)])
        best_ids = np.empty(0, dtype=np.int64)
        best_scores = np.empty(0, dtype=np.int32)
        best_flags = np.empty((0, threshold_count), dtype=bool)
        for charge, left, right in query_windows:
            hv_source = refs[charge]["hvs"]
            for start in range(left, right, chunk_size):
                stop = min(start + chunk_size, right)
                ref_gpu = cp.asarray(hv_source[start:stop])
                xor = cp.bitwise_xor(ref_gpu, qgpu)
                byte_matches = lut[xor]
                partial = byte_matches[:, :max_bytes].sum(axis=1, dtype=cp.int32)
                full = byte_matches.sum(axis=1, dtype=cp.int32)
                for ti, threshold in enumerate(threshold_values):
                    survivors[qi, ti] += int(cp.count_nonzero(partial + (HV_DIM - max_bytes * 8) >= threshold).get())

                k = min(TOP_K, int(full.size))
                if k:
                    indices = cp.argpartition(-full, k - 1)[:k]
                    local_scores = cp.asnumpy(full[indices]).astype(np.int32, copy=False)
                    base = charge * ROW_STRIDE
                    local_ids = base + (start + cp.asnumpy(indices).astype(np.int64, copy=False))
                    local_partial = cp.asnumpy(partial[indices]).astype(np.int32, copy=False)
                    local_flags = np.asarray(
                        [local_partial + (HV_DIM - max_bytes * 8) >= threshold for threshold in threshold_values],
                        dtype=bool,
                    ).T
                    if best_ids.size:
                        local_ids = np.concatenate((best_ids, local_ids))
                        local_scores = np.concatenate((best_scores, local_scores))
                        local_flags = np.concatenate((best_flags, local_flags), axis=0)
                    order = np.lexsort((local_ids, -local_scores))[:TOP_K]
                    best_ids = local_ids[order]
                    best_scores = local_scores[order]
                    best_flags = local_flags[order]
                del ref_gpu, xor, byte_matches, partial, full
        if best_ids.size:
            top_ids[qi, :best_ids.size] = best_ids
            top_scores[qi, :best_scores.size] = best_scores
            top8_retained[qi, :] = best_flags[:best_ids.size].sum(axis=0)
            top1_retained[qi, :] = best_flags[0].astype(np.int8)
        if (qi + 1) % 100 == 0 or qi + 1 == len(open_queries):
            print(
                f"[open-exact] {qi + 1}/{len(open_queries)} queries, "
                f"pairs={int(survivors[:qi + 1, 0].sum()):,} threshold-counted, "
                f"elapsed={time.time() - start_time:.1f}s",
                flush=True,
            )
    cp.cuda.Stream.null.synchronize()
    return survivors, top_ids, top_scores, top8_retained, top1_retained


def filter_stats_streaming(query_hvs, open_queries, windows, refs, top_ids, thresholds, segments, chunk_size):
    """Recover threshold statistics from an existing exact Top-8 artifact.

    This pass only reads the examined prefix and evaluates the eight saved
    exact IDs directly.  It avoids recomputing full-vector scores after an
    interrupted run has already produced the exact ground truth.
    """
    cp = standard.HyperSpec2048Encoder(standard.CODEBOOK_PATH).cp
    import pim_hyperoms_estimator as oms  # type: ignore
    lut = cp.asarray(oms.BYTE_POPCOUNT)
    threshold_values = [int(x) for x in thresholds]
    survivors = np.zeros((len(open_queries), len(threshold_values)), dtype=np.int64)
    top8_retained = np.zeros((len(open_queries), len(threshold_values)), dtype=np.int8)
    top1_retained = np.zeros((len(open_queries), len(threshold_values)), dtype=np.int8)
    max_bytes = segments * SEGMENT_BITS // 8
    start_time = time.time()
    for qi, (query_index, query_windows) in enumerate(zip(open_queries, windows)):
        qgpu = cp.asarray(query_hvs[int(query_index)])
        exact_ids = top_ids[qi][top_ids[qi] >= 0]
        retained = np.zeros((len(threshold_values),), dtype=np.int8)
        for charge, left, right in query_windows:
            hv_source = refs[charge]["hvs"]
            base = charge * ROW_STRIDE
            exact_rows = exact_ids[(exact_ids >= base + left) & (exact_ids < base + right)] - base
            for start in range(left, right, chunk_size):
                stop = min(start + chunk_size, right)
                ref_gpu = cp.asarray(hv_source[start:stop])
                xor = cp.bitwise_xor(ref_gpu, qgpu)
                partial = lut[xor[:, :max_bytes]].sum(axis=1, dtype=cp.int32)
                remaining = HV_DIM - max_bytes * 8
                for ti, threshold in enumerate(threshold_values):
                    survivors[qi, ti] += int(cp.count_nonzero(partial + remaining >= threshold).get())
                in_chunk = exact_rows[(exact_rows >= start) & (exact_rows < stop)]
                if in_chunk.size:
                    exact_partial = cp.asnumpy(partial[in_chunk - start])
                    for score in exact_partial.tolist():
                        retained += np.asarray(
                            [int(score) + remaining >= threshold for threshold in threshold_values],
                            dtype=np.int8,
                        )
                del ref_gpu, xor, partial
        top8_retained[qi] = retained
        top1_retained[qi] = (retained > 0).astype(np.int8)
        if (qi + 1) % 1000 == 0 or qi + 1 == len(open_queries):
            print(f"[open-filter-stats] {qi + 1}/{len(open_queries)} queries, elapsed={time.time() - start_time:.1f}s", flush=True)
    return survivors, top8_retained, top1_retained


def make_figure(rows: list[dict], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    x = [row["candidate_reduction_pct"] for row in rows]
    y = [row["top8_recall_pct"] for row in rows]
    default = [i for i, row in enumerate(rows) if row.get("is_default")]
    ax.plot(x, y, "o-", color="#2c6e9e", linewidth=1.0, markersize=3.5)
    if default:
        i = default[0]
        ax.scatter([x[i]], [y[i]], s=38, color="#c53d2f", zorder=3)
    ax.set_xlabel("Candidate reduction (%)", fontsize=8)
    ax.set_ylabel("Exact Top-8 recall (%)", fontsize=8)
    ax.set_ylim(max(0, min(y) - 1), 100.1)
    fig.tight_layout()
    fig.savefig(output_dir / "fenand_oms_open_sensitivity.pdf")
    fig.savefig(output_dir / "fenand_oms_open_sensitivity.png", dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    open_path = Path(args.open_stage_ids)
    if not open_path.exists():
        raise FileNotFoundError(f"Run audit_oms_open_stage.py first: {open_path}")
    import pim_hyperoms_estimator as oms  # type: ignore
    oms.__dict__.update(oms._import_oms_symbols())
    config = oms._load_config(Path(args.config))
    queries = oms._prepare_query_spectra(Path(args.query), config, 2000.0)
    open_indices = np.asarray(np.load(open_path, allow_pickle=False)["query_indices"], dtype=np.int64)
    if np.any(open_indices < 0) or np.any(open_indices >= len(queries)):
        raise ValueError("Open-stage query indices do not match the query MGF")

    encoder = standard.HyperSpec2048Encoder(standard.CODEBOOK_PATH)
    query_idx, query_vals = standard.query_feature_arrays(queries, oms, config)
    query_hvs = encoder.encode(query_idx, query_vals)
    vector_files = standard.load_reference_vector_files(Path(args.ref), config, range(1, 9))
    cache_dir = Path(args.reference_cache_dir)
    missing_cache = [
        charge for charge, info in vector_files.items()
        if not (cache_dir / f"reference_charge{charge}.uint8.mmap").exists()
    ]
    if missing_cache:
        print(f"[cache] building missing 2048-bit reference charges={missing_cache}", flush=True)
        standard.build_reference_hv_cache(
            {charge: vector_files[charge] for charge in missing_cache},
            cache_dir.parent,
            encoder,
            batch_size=32768,
            force=False,
        )
    refs = load_sorted_reference_hvs(vector_files, Path(args.reference_cache_dir))
    windows, candidate_counts = query_windows(queries, refs, open_indices, args.open_da)
    print(json.dumps({
        "total_queries": len(queries),
        "open_stage_queries": len(open_indices),
        "legal_open_candidate_pairs": int(candidate_counts.sum()),
        "mean_candidates_per_query": float(candidate_counts.mean()),
        "median_candidates_per_query": float(np.median(candidate_counts)),
        "max_candidates_per_query": int(candidate_counts.max()),
    }, indent=2), flush=True)

    exact_path = output_dir / "exact_top8_open_b1926.npz"
    counts_path = output_dir / "open_candidate_counts_b1926.npy"
    stats_path = output_dir / "open_sweep_stats.npz"
    if not args.force and exact_path.exists() and counts_path.exists() and stats_path.exists():
        saved = np.load(exact_path, allow_pickle=False)
        top_ids = saved["top8_ids"]
        top_scores = saved["top8_scores"]
        stats = np.load(stats_path, allow_pickle=False)
        survivors = stats["survivors"]
        top8_retained = stats["top8_retained"]
        top1_retained = stats["top1_retained"]
    elif not args.force and exact_path.exists() and counts_path.exists():
        saved = np.load(exact_path, allow_pickle=False)
        top_ids = saved["top8_ids"]
        top_scores = saved["top8_scores"]
        candidate_counts = np.load(counts_path, allow_pickle=False)
        survivors, top8_retained, top1_retained = filter_stats_streaming(
            query_hvs, open_indices, windows, refs, top_ids, args.thresholds, args.segments_examined, args.chunk_size
        )
        np.savez_compressed(
            stats_path,
            survivors=survivors,
            top8_retained=top8_retained,
            top1_retained=top1_retained,
        )
    else:
        survivors, top_ids, top_scores, top8_retained, top1_retained = exact_streaming_sweep(
            query_hvs, open_indices, windows, refs, args.thresholds, args.segments_examined, args.chunk_size
        )
        np.savez_compressed(
            stats_path,
            survivors=survivors,
            top8_retained=top8_retained,
            top1_retained=top1_retained,
        )
        np.savez_compressed(
            exact_path,
            query_indices=open_indices,
            top8_ids=top_ids,
            top8_scores=top_scores,
            hdc_dimension=np.asarray(HV_DIM),
            deterministic_tie_break=np.asarray("descending score, ascending charge*10000000+row"),
        )
        np.save(counts_path, candidate_counts)

    if survivors is None:
        # Existing exact artifacts from an interrupted/older run do not contain
        # threshold statistics, so regenerate the same single streaming pass.
        survivors, _unused_ids, _unused_scores, top8_retained, top1_retained = exact_streaming_sweep(
            query_hvs, open_indices, windows, refs, args.thresholds, args.segments_examined, args.chunk_size
        )
        np.savez_compressed(
            stats_path,
            survivors=survivors,
            top8_retained=top8_retained,
            top1_retained=top1_retained,
        )

    total_pairs = int(candidate_counts.sum())
    total_exact = int((top_ids >= 0).sum())
    full_denominator = int(np.all(top_ids >= 0, axis=1).sum())
    rows = []
    for ti, threshold in enumerate(args.thresholds):
        surviving = int(survivors[:, ti].sum())
        retained = int(top8_retained[:, ti].sum())
        full = int((top8_retained[:, ti] == TOP_K).sum())
        top1 = int(top1_retained[:, ti].sum())
        rows.append({
            "stage": "open_500Da",
            "segments_examined": int(args.segments_examined),
            "score_threshold_bits": int(threshold),
            "unfiltered_candidate_pairs": total_pairs,
            "surviving_candidate_pairs": surviving,
            "candidate_reduction_pct": 100.0 * (1.0 - surviving / total_pairs),
            "total_exact_top8": total_exact,
            "retained_exact_top8": retained,
            "top8_recall_pct": 100.0 * retained / total_exact,
            "full_top8_query_recall_pct": 100.0 * full / full_denominator,
            "top1_recall_pct": 100.0 * top1 / int((top_ids[:, 0] >= 0).sum()),
            "mean_survivors_per_query": float(survivors[:, ti].mean()),
            "median_survivors_per_query": float(np.median(survivors[:, ti])),
            "max_survivors_per_query": int(survivors[:, ti].max()),
        })
    eligible = [row for row in rows if row["top8_recall_pct"] >= 99.0]
    if not eligible:
        raise RuntimeError("No open-search threshold reaches aggregate Top-8 recall >= 99%")
    selected = max(eligible, key=lambda row: row["candidate_reduction_pct"])
    selected["is_default"] = True
    for row in rows:
        row.setdefault("is_default", False)

    import csv
    csv_path = output_dir / "fenand_oms_open_sensitivity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    make_figure(rows, output_dir)

    metadata = {
        "stage": "open_500Da_retry",
        "query": args.query,
        "reference": args.ref,
        "config": args.config,
        "total_b1926_queries": len(queries),
        "open_stage_queries": len(open_indices),
        "candidate_cap_used_for_open_membership_replay": 512,
        "candidate_cap_used_for_exact_ground_truth": 0,
        "candidate_cap_used_for_sweep": 0,
        "open_window_da": args.open_da,
        "legal_candidate_pairs": total_pairs,
        "mean_candidates_per_query": float(candidate_counts.mean()),
        "median_candidates_per_query": float(np.median(candidate_counts)),
        "max_candidates_per_query": int(candidate_counts.max()),
        "hdc": {"dimension_bits": HV_DIM, "segment_bits": SEGMENT_BITS, "segments_examined": args.segments_examined},
        "exact_ground_truth": str(exact_path),
        "deterministic_tie_break": "descending full Hamming score, ascending charge*10000000+sorted-row ID",
        "selected_default": selected,
        "selection_rule": "maximum candidate reduction among points with aggregate exact Top-8 recall >= 99%",
        "system": system_metadata(),
        "rows": rows,
    }
    (output_dir / "open_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output_dir / "open_paper_ready_text.md").write_text(
        "# FeNAND Open-Search Sensitivity\n\n"
        f"The 500-Da retry stage contains **{total_pairs:,}** exact legal comparisons across "
        f"**{len(open_indices):,}** queries (mean {candidate_counts.mean():,.1f}, median "
        f"{np.median(candidate_counts):,.0f}, maximum {candidate_counts.max():,}). "
        f"The selected operating point examines {selected['segments_examined']}/32 64-bit segments "
        f"at a {selected['score_threshold_bits']}-bit threshold, eliminating "
        f"**{selected['candidate_reduction_pct']:.4f}%** of legal comparisons while retaining "
        f"**{selected['top8_recall_pct']:.4f}%** of exact Top-8 candidate memberships and "
        f"all eight candidates for **{selected['full_top8_query_recall_pct']:.4f}%** of retry queries.\n\n"
        "The exact reference was produced by streaming every legal 500-Da candidate; no "
        "candidate cap, approximate ANN pruning, or subsampling was used before ranking.\n",
        encoding="utf-8",
    )
    print(json.dumps({"selected_default": selected, "csv": str(csv_path), "metadata": str(output_dir / 'open_metadata.json')}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
