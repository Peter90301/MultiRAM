#!/usr/bin/env python3
"""Reproducible FeNAND coarse-filter sensitivity experiment for OMS.

The experiment uses the b1926 query MGF, the homs-tc precursor/charge rules,
and the 2048-bit Hyper-Spec level-ID codebook.  It builds a reference-HV
cache from the existing sparse spectral-library vectors, computes an
unfiltered exact Top-8 reference, and then evaluates a partial-HV upper-bound
filter without candidate truncation.
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
from typing import Dict, Iterable, List, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OMS_PIPELINE = ROOT / "proteomic_full_pipeline"
HOMSTC = ROOT / "sumukh_proteomic_test/homs-tc"
GPU_BENCH = ROOT / "sumukh_proteomic_test"
HYPERSPEC = ROOT / "external/Hyper-Spec"
for item in (str(OMS_PIPELINE), str(GPU_BENCH), str(HOMSTC)):
    if item not in sys.path:
        sys.path.insert(0, item)

from fenand_coarse_filter import (  # noqa: E402
    CoarseFilterConfig,
    candidate_reduction,
    exact_topk,
    filter_candidates,
)


ROW_STRIDE = 10_000_000
TOP_K = 8
HV_DIM = 2048
SEGMENT_BITS = 64
MAX_PEAKS = 50
CODEBOOK_PATH = Path(os.environ.get("MULTIRAM_HYPERSPEC_CODEBOOK", str(HYPERSPEC / "lv_id_hvs_D_2048_Q_16_bin_27981_flip_2.0.npz")))


def import_oms_symbols():
    import pim_hyperoms_estimator as oms  # type: ignore

    oms.__dict__.update(oms._import_oms_symbols())
    return oms


def _compile_encoder_kernel(cp):
    source = r'''
    __device__ char get2d_bin(
        const unsigned int* p, const int i, const int DIM, const int d) {
        unsigned int v = ((*(p + i * ((DIM + 32 - 1) / 32) + d / 32)
                          >> ((32 - 1) - d % 32)) & 0x01);
        return v == 0 ? -1 : 1;
    }
    extern "C" __global__ void encode_hv(
        const unsigned int* id_hvs,
        const unsigned int* level_hvs,
        const int* feature_indices,
        const float* feature_values,
        int max_peaks,
        unsigned int* output,
        int n,
        int q_levels,
        int dim,
        int packed_dim) {
        const int d = threadIdx.x + blockIdx.x * blockDim.x;
        if (d >= dim) return;
        for (int sample = blockIdx.y; sample < n;
             sample += gridDim.y * blockDim.y) {
            float accumulator = 0.0f;
            const int start = sample * max_peaks;
            const int stop = (sample + 1) * max_peaks;
            for (int f = start; f < stop; ++f) {
                if (feature_values[f] >= 0.0f) {
                    const int level = (int)(feature_values[f] * q_levels);
                    accumulator += get2d_bin(level_hvs, level, dim, d)
                                 * get2d_bin(id_hvs, feature_indices[f], dim, d);
                }
            }
            const int lane = threadIdx.x % warpSize;
            const unsigned int pattern = __ballot_sync(0xffffffff,
                                                        accumulator > 0.0f);
            if (lane == 0) {
                output[sample * packed_dim + d / warpSize] = pattern;
            }
        }
    }
    '''
    return cp.RawKernel(source, "encode_hv")


class HyperSpec2048Encoder:
    def __init__(self, codebook_path: Path):
        try:
            import cupy as cp  # type: ignore
        except ImportError as exc:
            raise RuntimeError("The full experiment requires the homstc CuPy environment") from exc
        self.cp = cp
        # CuPy 12 / NVRTC does not know the Blackwell compute capability.
        os.environ.setdefault("CUPY_COMPILE_WITH_PTX", "1")
        try:
            import pim_hyperoms_estimator as oms  # type: ignore

            oms._prepare_cupy()
        except Exception:
            pass
        codebook = np.load(codebook_path, mmap_mode="r")
        if codebook["id_hvs"].shape != (27981, 64) or codebook["lv_hvs"].shape != (17, 64):
            raise ValueError(f"Unexpected Hyper-Spec codebook shape: {codebook.files}")
        self.id_hvs = cp.asarray(codebook["id_hvs"], dtype=cp.uint32)
        self.level_hvs = cp.asarray(codebook["lv_hvs"], dtype=cp.uint32)
        self.kernel = _compile_encoder_kernel(cp)
        self.hv_bytes = HV_DIM // 8

    def encode(self, feature_indices: np.ndarray, feature_values: np.ndarray) -> np.ndarray:
        n, peak_count = feature_indices.shape
        if feature_values.shape != (n, peak_count):
            raise ValueError("feature index/value arrays have different shapes")
        cp = self.cp
        output = cp.zeros((n, HV_DIM // 32), dtype=cp.uint32)
        max_grid_y = int(cp.cuda.runtime.getDeviceProperties(0)["maxGridSize"][1])
        self.kernel(
            ((HV_DIM + 1023) // 1024, min(n, max_grid_y)),
            (1024,),
            (
                self.id_hvs,
                self.level_hvs,
                cp.asarray(feature_indices.reshape(-1), dtype=cp.int32),
                cp.asarray(feature_values.reshape(-1), dtype=cp.float32),
                peak_count,
                output,
                n,
                16,
                HV_DIM,
                HV_DIM // 32,
            ),
        )
        cp.cuda.Stream.null.synchronize()
        # The packed representation is only used for XOR/popcount; preserving
        # each uint32 word byte-for-byte keeps query/reference encoding aligned.
        return output.get().view(np.uint8).reshape(n, self.hv_bytes)


def _normalised_sparse(sparse: Iterable[Tuple[int, float]], max_peaks: int = MAX_PEAKS):
    values = [(int(idx), float(value)) for idx, value in sparse if float(value) > 0]
    values = values[:max_peaks]
    if not values:
        return [], []
    scale = max(value for _, value in values)
    if scale <= 0:
        return [], []
    return [idx for idx, _ in values], [value / scale for _, value in values]


def sparse_batch_from_rows(csr_info, indices, intensities, start: int, stop: int):
    count = stop - start
    feature_indices = np.full((count, MAX_PEAKS), -1, dtype=np.int32)
    feature_values = np.full((count, MAX_PEAKS), -1.0, dtype=np.float32)
    for local, row in enumerate(range(start, stop)):
        left = int(csr_info[row])
        right = int(csr_info[row + 1])
        length = min(right - left, MAX_PEAKS)
        if length:
            vals = np.asarray(intensities[left:left + length], dtype=np.float32)
            scale = float(np.max(vals))
            if scale > 0:
                feature_indices[local, :length] = np.asarray(indices[left:left + length], dtype=np.int32)
                feature_values[local, :length] = vals / scale
    return feature_indices, feature_values


def load_reference_vector_files(ref_path: Path, config, charges: Iterable[int]):
    result = {}
    for charge in sorted(set(int(x) for x in charges)):
        vector_path = ref_path.with_name(
            f"{ref_path.stem}_vec_{config.spectrum_vector_dim}.charge{charge}.npz"
        )
        if not vector_path.exists():
            continue
        data = np.load(vector_path, mmap_mode="r")
        precursor = np.asarray(data["pr_mzs"], dtype=np.float64)
        order = np.argsort(precursor, kind="stable")
        result[charge] = {
            "data": data,
            "precursor_mz": precursor,
            "sorted_mz": precursor[order],
            "sorted_rows": order.astype(np.int64, copy=False),
            "rows": int(precursor.size),
        }
    return result


def build_reference_hv_cache(
    vector_files: Dict[int, dict],
    output_dir: Path,
    encoder: HyperSpec2048Encoder,
    batch_size: int,
    force: bool,
) -> Dict[int, dict]:
    cache_dir = output_dir / "reference_hv_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for charge, info in sorted(vector_files.items()):
        path = cache_dir / f"reference_charge{charge}.uint8.mmap"
        metadata_path = path.with_suffix(path.suffix + ".json")
        expected_bytes = info["rows"] * encoder.hv_bytes
        ready = path.exists() and path.stat().st_size == expected_bytes and metadata_path.exists()
        if force or not ready:
            print(f"[build] charge={charge} rows={info['rows']}", flush=True)
            out = np.memmap(path, dtype=np.uint8, mode="w+", shape=(info["rows"], encoder.hv_bytes))
            data = info["data"]
            for start in range(0, info["rows"], batch_size):
                stop = min(start + batch_size, info["rows"])
                idx, vals = sparse_batch_from_rows(
                    data["csr_info"], data["spectra_idx"], data["spectra_intensities"], start, stop
                )
                out[start:stop] = encoder.encode(idx, vals)
                if start == 0 or stop == info["rows"] or stop % (batch_size * 10) == 0:
                    print(f"[build] charge={charge} {stop}/{info['rows']}", flush=True)
            out.flush()
            del out
            metadata_path.write_text(
                json.dumps(
                    {
                        "charge": charge,
                        "rows": info["rows"],
                        "hv_dimensionality": HV_DIM,
                        "hv_bytes": encoder.hv_bytes,
                        "codebook": str(CODEBOOK_PATH),
                        "normalization": "per-spectrum max intensity",
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        result[charge] = {**info, "hvs": np.memmap(path, dtype=np.uint8, mode="r", shape=(info["rows"], encoder.hv_bytes))}
    return result


def query_feature_arrays(queries, oms, config):
    idx = np.full((len(queries), MAX_PEAKS), -1, dtype=np.int32)
    vals = np.full((len(queries), MAX_PEAKS), -1.0, dtype=np.float32)
    for row, query in enumerate(queries):
        sparse = oms.spectrum_to_vector(
            query, config.min_mz, config.max_mz, config.bin_size,
            config.spectrum_vector_dim, config.min_bound,
        )
        feature_idx, feature_vals = _normalised_sparse(sparse)
        length = min(len(feature_idx), MAX_PEAKS)
        if length:
            idx[row, :length] = feature_idx[:length]
            vals[row, :length] = feature_vals[:length]
    return idx, vals


def build_candidates(queries, refs: Dict[int, dict], std_ppm: float):
    offsets = np.zeros(len(queries) + 1, dtype=np.int64)
    chunks = []
    mzs = np.empty(len(queries), dtype=np.float64)
    charges = np.empty(len(queries), dtype=np.int16)
    for qidx, query in enumerate(queries):
        mz = float(query.precursor_mz)
        charge = int(query.precursor_charge) if query.precursor_charge is not None else -1
        mzs[qidx] = mz
        charges[qidx] = charge
        info = refs.get(charge)
        if info is not None:
            tolerance = mz * float(std_ppm) * 1e-6
            left = int(np.searchsorted(info["sorted_mz"], mz - tolerance, side="left"))
            right = int(np.searchsorted(info["sorted_mz"], mz + tolerance, side="right"))
            rows = info["sorted_rows"][left:right]
            if rows.size:
                chunks.append(charge * ROW_STRIDE + rows.astype(np.int64))
                offsets[qidx + 1] = rows.size
        offsets[qidx + 1] += offsets[qidx]
    candidate_ids = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)
    return candidate_ids, offsets, mzs, charges


def gather_reference_hvs(candidate_ids: np.ndarray, refs: Dict[int, dict], start: int, stop: int):
    ids = candidate_ids[start:stop]
    out = np.empty((ids.size, HV_DIM // 8), dtype=np.uint8)
    cursor = 0
    for charge in np.unique(ids // ROW_STRIDE):
        charge = int(charge)
        mask = (ids // ROW_STRIDE) == charge
        rows = (ids[mask] % ROW_STRIDE).astype(np.int64)
        count = rows.size
        if charge not in refs:
            raise KeyError(f"Missing reference cache for charge {charge}")
        out[np.flatnonzero(mask)] = refs[charge]["hvs"][rows]
        cursor += count
    return out


def materialize_candidate_hvs(candidate_ids, offsets, refs, output_dir: Path, force: bool):
    path = output_dir / "candidate_hvs_b1926.uint8.mmap"
    meta = path.with_suffix(path.suffix + ".json")
    pairs = int(candidate_ids.size)
    expected = pairs * (HV_DIM // 8)
    if force or not path.exists() or path.stat().st_size != expected:
        out = np.memmap(path, dtype=np.uint8, mode="w+", shape=(pairs, HV_DIM // 8))
        for start in range(0, len(offsets) - 1):
            left, right = int(offsets[start]), int(offsets[start + 1])
            if right > left:
                out[left:right] = gather_reference_hvs(candidate_ids, refs, left, right)
        out.flush()
        del out
        meta.write_text(json.dumps({"pairs": pairs, "hv_dimensionality": HV_DIM, "hv_bytes": HV_DIM // 8}, indent=2) + "\n", encoding="utf-8")
    return np.memmap(path, dtype=np.uint8, mode="r", shape=(pairs, HV_DIM // 8))


def evaluate_config(query_hvs, candidate_ids, offsets, candidate_hvs, exact_ids, segments, threshold, save_ids=False):
    retained_counts = []
    retained_exact = 0
    total_exact = 0
    full_queries = 0
    top1_hits = 0
    full_top8_denominator = 0
    top1_denominator = 0
    saved = []
    saved_offsets = [0]
    for qidx in range(len(query_hvs)):
        left, right = int(offsets[qidx]), int(offsets[qidx + 1])
        ids = candidate_ids[left:right]
        refs = candidate_hvs[left:right]
        cfg = CoarseFilterConfig(HV_DIM, SEGMENT_BITS, segments, threshold)
        survivors = filter_candidates(query_hvs[qidx], refs, ids, cfg)
        retained_counts.append(int(survivors.size))
        exact = exact_ids[qidx]
        valid_exact = exact[exact >= 0]
        survivor_set = set(survivors.tolist())
        kept = sum(int(x) in survivor_set for x in valid_exact)
        retained_exact += kept
        total_exact += int(valid_exact.size)
        if valid_exact.size == TOP_K:
            full_top8_denominator += 1
        if valid_exact.size == TOP_K and kept == valid_exact.size:
            full_queries += 1
        if valid_exact.size:
            top1_denominator += 1
        if valid_exact.size and int(valid_exact[0]) in survivor_set:
            top1_hits += 1
        if save_ids:
            saved.append(survivors)
            saved_offsets.append(saved_offsets[-1] + int(survivors.size))
    surviving_pairs = int(sum(retained_counts))
    total_pairs = int(candidate_ids.size)
    query_count = len(query_hvs)
    result = {
        "segments_examined": int(segments),
        "score_threshold_bits": int(threshold),
        "unfiltered_candidate_pairs": total_pairs,
        "surviving_candidate_pairs": surviving_pairs,
        "candidate_reduction_pct": 100.0 * candidate_reduction(total_pairs, surviving_pairs),
        "total_exact_top8": int(total_exact),
        "retained_exact_top8": int(retained_exact),
        "top8_recall_pct": 100.0 * retained_exact / total_exact if total_exact else 0.0,
        "full_top8_query_recall_pct": 100.0 * full_queries / full_top8_denominator if full_top8_denominator else 0.0,
        "mean_top8_retained_per_query": float(retained_exact / query_count) if query_count else 0.0,
        "top1_recall_pct": 100.0 * top1_hits / top1_denominator if top1_denominator else 0.0,
    }
    if save_ids:
        flat = np.concatenate(saved) if saved else np.empty(0, dtype=np.int64)
        result["filtered_ids"] = flat
        result["filtered_offsets"] = np.asarray(saved_offsets, dtype=np.int64)
    return result


def system_metadata():
    data = {"platform": platform.platform(), "python": sys.version, "cwd": str(Path.cwd())}
    try:
        data["nvidia_smi"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
    except Exception:
        data["nvidia_smi"] = "unavailable"
    try:
        data["git_commit"] = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        data["git_commit"] = "unavailable"
    return data


def make_figure(rows: List[dict], output_dir: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    x = [row["candidate_reduction_pct"] for row in rows]
    y = [row["top8_recall_pct"] for row in rows]
    defaults = [i for i, row in enumerate(rows) if row.get("is_default")]
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    ax.plot(x, y, "o-", color="#2c6e9e", linewidth=1.0, markersize=3.5)
    if defaults:
        i = defaults[0]
        ax.scatter([x[i]], [y[i]], s=38, color="#c53d2f", zorder=3)
        ax.annotate(f"default\n{x[i]:.2f}% / {y[i]:.2f}%", (x[i], y[i]), xytext=(-8, -22), textcoords="offset points", fontsize=6)
    ax.set_xlabel("Candidate reduction (%)", fontsize=8)
    ax.set_ylabel("Exact Top-8 recall (%)", fontsize=8)
    ax.set_ylim(max(0, min(y) - 1), 100.1)
    ax.grid(True, linewidth=0.35, alpha=0.45)
    fig.tight_layout()
    pdf = output_dir / "FeNAND_oms_sensitivity.pdf"
    png = output_dir / "FeNAND_oms_sensitivity.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--ref", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "sumukh_proteomic_test/homs-tc/configs/hek293.ini")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "fenand_oms_sensitivity")
    parser.add_argument("--std-ppm", type=float, default=5.0)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--ref-build-batch", type=int, default=32768)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    oms = import_oms_symbols()
    config = oms._load_config(args.config)
    queries = oms._prepare_query_spectra(args.query, config, float(config.max_mz))
    if args.max_queries > 0:
        queries = queries[:args.max_queries]
    if not queries:
        raise RuntimeError("No valid query spectra after homs-tc preprocessing")
    print(f"[query] valid spectra={len(queries)}", flush=True)
    query_idx, query_values = query_feature_arrays(queries, oms, config)
    encoder = HyperSpec2048Encoder(CODEBOOK_PATH)
    query_hvs = encoder.encode(query_idx, query_values)
    np.save(args.output_dir / "query_hvs_b1926.npy", query_hvs)

    query_charges = [int(q.precursor_charge) for q in queries if q.precursor_charge is not None and 1 <= int(q.precursor_charge) <= 8]
    vector_files = load_reference_vector_files(args.ref, config, query_charges)
    if not vector_files:
        raise RuntimeError("No matching homs-tc reference vector files")
    refs = build_reference_hv_cache(vector_files, args.output_dir, encoder, args.ref_build_batch, args.force_rebuild)
    candidate_ids, offsets, query_mz, query_charge = build_candidates(queries, refs, args.std_ppm)
    np.savez_compressed(
        args.output_dir / "legal_candidates_b1926.npz",
        candidate_ids=candidate_ids,
        offsets=offsets,
        query_precursor_mz=query_mz,
        query_charge=query_charge,
    )
    print(f"[candidates] queries={len(queries)} pairs={candidate_ids.size} avg={candidate_ids.size / len(queries):.3f} median={np.median(np.diff(offsets)):.3f} max={np.diff(offsets).max(initial=0)}", flush=True)
    candidate_hvs = materialize_candidate_hvs(candidate_ids, offsets, refs, args.output_dir, args.force_rebuild)

    exact_ids = np.full((len(queries), TOP_K), -1, dtype=np.int64)
    exact_scores = np.full((len(queries), TOP_K), -1, dtype=np.int16)
    for qidx in range(len(queries)):
        left, right = int(offsets[qidx]), int(offsets[qidx + 1])
        if right <= left:
            continue
        ids, scores = exact_topk(query_hvs[qidx], candidate_hvs[left:right], candidate_ids[left:right], TOP_K)
        exact_ids[qidx, :ids.size] = ids
        exact_scores[qidx, :scores.size] = scores
        if qidx == 0 or (qidx + 1) % 5000 == 0 or qidx + 1 == len(queries):
            print(f"[exact] {qidx + 1}/{len(queries)}", flush=True)
    np.savez_compressed(
        args.output_dir / "exact_top8_b1926.npz",
        top8_ids=exact_ids,
        top8_scores=exact_scores,
        query_count=np.asarray([len(queries)], dtype=np.int64),
        top_k=np.asarray([TOP_K], dtype=np.int64),
        tie_break=np.asarray(["ascending candidate ID"]),
    )

    valid_kth = exact_scores[:, TOP_K - 1]
    valid_kth = valid_kth[valid_kth >= 0]
    calibration_count = max(1, int(len(valid_kth) * args.calibration_fraction))
    calibration_scores = valid_kth[:calibration_count]
    threshold = int(np.floor(np.percentile(calibration_scores, 1.0)))
    threshold = max(0, min(HV_DIM, threshold))
    print(f"[filter] calibrated threshold={threshold} bits from first {calibration_count} queries (1st percentile of exact Top-8 kth score)", flush=True)
    # Keep the FeNAND pass partial (28 of 32 64-bit segments) and sweep the
    # fixed acceptance threshold.  This exposes the actual reduction/recall
    # trade-off instead of hiding it behind a candidate-list cap.
    segment_sweep = [28] * 8
    threshold_sweep = sorted(set(
        int(np.floor(np.percentile(calibration_scores, percentile)))
        for percentile in (1, 10, 25, 50, 75, 90, 95, 99)
    ))
    rows = []
    for segments, threshold_value in zip(segment_sweep, threshold_sweep):
        result = evaluate_config(query_hvs, candidate_ids, offsets, candidate_hvs, exact_ids, segments, threshold_value)
        result["filter_parameter"] = "score_threshold_bits"
        result["filter_value"] = threshold_value
        result["threshold_sweep_percentile"] = next(
            percentile for percentile in (1, 10, 25, 50, 75, 90, 95, 99)
            if int(np.floor(np.percentile(calibration_scores, percentile))) == threshold_value
        )
        rows.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "filtered_ids"}, sort_keys=True), flush=True)
    eligible = [r for r in rows if r["top8_recall_pct"] >= 99.0]
    if eligible:
        default = max(eligible, key=lambda r: (r["candidate_reduction_pct"], r["top8_recall_pct"]))
    else:
        default = max(rows, key=lambda r: (r["top8_recall_pct"], r["candidate_reduction_pct"]))
    for row in rows:
        row["is_default"] = row is default
    selected = evaluate_config(query_hvs, candidate_ids, offsets, candidate_hvs, exact_ids, default["segments_examined"], default["score_threshold_bits"], save_ids=True)
    np.savez_compressed(
        args.output_dir / "filtered_candidate_ids_default.npz",
        filtered_candidate_ids=selected.pop("filtered_ids"),
        filtered_offsets=selected.pop("filtered_offsets"),
        segments_examined=np.asarray([default["segments_examined"]]),
        score_threshold_bits=np.asarray([threshold]),
    )
    with (args.output_dir / "fenand_oms_sensitivity.csv").open("w", encoding="utf-8") as fh:
        fields = list(rows[0].keys())
        fh.write(",".join(fields) + "\n")
        for row in rows:
            fh.write(",".join(str(row[field]) for field in fields) + "\n")

    # Marginal trade-off is reported in increasing segment order.  A knee is
    # only called when the measured recall loss per added reduction changes by
    # at least 2x; otherwise the result is intentionally described as smooth.
    marginal = []
    for previous, current in zip(rows, rows[1:]):
        delta_reduction = current["candidate_reduction_pct"] - previous["candidate_reduction_pct"]
        delta_recall = current["top8_recall_pct"] - previous["top8_recall_pct"]
        marginal.append({
            "from_segments": previous["segments_examined"],
            "to_segments": current["segments_examined"],
            "delta_reduction_pct_points": delta_reduction,
            "delta_recall_pct_points": delta_recall,
            "recall_loss_per_additional_reduction": (-delta_recall / delta_reduction) if delta_reduction > 0 else None,
        })
    positive_rates = [x["recall_loss_per_additional_reduction"] for x in marginal if x["recall_loss_per_additional_reduction"] is not None and x["recall_loss_per_additional_reduction"] > 0]
    default_index = next(i for i, row in enumerate(rows) if row is default)
    local_rates = []
    if default_index > 0:
        local_rates.append(marginal[default_index - 1]["recall_loss_per_additional_reduction"])
    if default_index < len(marginal):
        local_rates.append(marginal[default_index]["recall_loss_per_additional_reduction"])
    baseline_rate = float(np.median(positive_rates)) if positive_rates else 0.0
    knee_classification = "B: Pareto-efficient default but no clear knee"
    if local_rates and baseline_rate > 0 and max(local_rates) >= 2.0 * baseline_rate:
        knee_classification = "A: clear knee near default"

    figure_paths = make_figure(rows, args.output_dir)
    old_reduction = 99.9536
    old_recall = 99.7070
    runtime = time.perf_counter() - started
    metadata = {
        "dataset": {"query": str(args.query), "reference": str(args.ref), "workload": "b1926"},
        "query_count": len(queries),
        "library_rows_by_charge": {str(c): int(v["rows"]) for c, v in refs.items()},
        "legal_candidate_pairs": int(candidate_ids.size),
        "average_candidate_count": float(candidate_ids.size / len(queries)),
        "median_candidate_count": float(np.median(np.diff(offsets))),
        "max_candidate_count": int(np.diff(offsets).max(initial=0)),
        "hdc": {
            "dimension_bits": HV_DIM,
            "segment_bits": SEGMENT_BITS,
            "codebook": str(CODEBOOK_PATH),
            "level_count": 16,
            "normalization": "per-spectrum max intensity",
            "note": "The current OMS estimator cache is 8192-bit; no OMS 2048-bit cache was present. This experiment rebuilds 2048-bit Hyper-Spec reference/query HVs from the same sparse spectra.",
        },
        "precursor_rules": {"stage": "standard precursor stage", "window": "symmetric", "tolerance_ppm": args.std_ppm, "charge": "query charge matched to reference charge"},
        "exact_top_k": TOP_K,
        "tie_break": "descending full Hamming similarity, then ascending stable candidate ID",
        "filter_definition": "partial matching bits + unexamined bits upper bound >= fixed threshold; no candidate cap or list truncation",
        "scope_note": "This exact ground truth covers the standard 5-ppm precursor stage. The existing estimator's open-search stage uses a separate 500-Da policy and candidate cap; it is not silently merged into this result.",
        "threshold_calibration": {"calibration_fraction": args.calibration_fraction, "threshold_bits": threshold, "statistic": "1st percentile of calibration-query exact Top-8 kth score"},
        "default": default,
        "sweep": rows,
        "marginal_tradeoff": marginal,
        "knee_classification": knee_classification,
        "old_unsupported_result": {"candidate_reduction_pct": old_reduction, "top8_recall_pct": old_recall},
        "difference_from_old_percentage_points": {"reduction": default["candidate_reduction_pct"] - old_reduction, "recall": default["top8_recall_pct"] - old_recall},
        "runtime_seconds": runtime,
        "system": system_metadata(),
        "command": " ".join([str(Path(sys.executable)), str(Path(__file__).resolve())] + sys.argv[1:]),
        "figure_paths": [str(x) for x in figure_paths] if figure_paths else [],
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    readme = render_readme(metadata)
    readme = readme.replace(
        "stable candidate ID.\n\n## Selected operating point",
        "stable candidate ID.\n\nThis result covers the standard 5-ppm precursor stage. The existing estimator has a separate 500-Da open-search stage with a different policy; it is not silently presented as part of this exact ground truth.\n\n## Selected operating point",
    )
    query_recall_line = (
        f"- Query-level all-Top-{metadata['exact_top_k']} recall: "
        f"**{default['full_top8_query_recall_pct']:.4f}%**\n"
    )
    readme = readme.replace(
        query_recall_line,
        query_recall_line + f"- Knee classification: **{metadata['knee_classification']}**\n",
    )
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"default": default, "knee": knee_classification, "runtime_seconds": runtime, "output_dir": str(args.output_dir)}, indent=2), flush=True)


def render_readme(metadata: dict) -> str:
    default = metadata["default"]
    return f'''# Reproducible FeNAND OMS coarse-filter sensitivity\n\nThis experiment evaluates the b1926 OMS workload against an unfiltered exact Hamming Top-{metadata["exact_top_k"]} reference. Precursor/charge matching defines the legal candidate set; FeNAND then evaluates a configurable prefix of 64-bit HDC segments and retains a candidate when\n\n```text\npartial_matches + unexamined_bits >= {default["score_threshold_bits"]}\n```\n\nThe filter is an upper-bound test, not `candidate_cap` and not list truncation. Ties in the exact search are deterministic: descending full-vector similarity followed by ascending stable candidate ID.\n\n## Selected operating point\n\n- HDC dimension: **{metadata["hdc"]["dimension_bits"]} bits**\n- Segments examined: **{default["segments_examined"]}/{metadata["hdc"]["dimension_bits"] // metadata["hdc"]["segment_bits"]}**\n- Fixed acceptance threshold: **{default["score_threshold_bits"]} matching bits**\n- Legal unfiltered comparisons: **{default["unfiltered_candidate_pairs"]:,}**\n- Surviving comparisons: **{default["surviving_candidate_pairs"]:,}**\n- Candidate reduction: **{default["candidate_reduction_pct"]:.4f}%**\n- Exact Top-{metadata["exact_top_k"]} recall: **{default["top8_recall_pct"]:.4f}%**\n- Query-level all-Top-{metadata["exact_top_k"]} recall: **{default["full_top8_query_recall_pct"]:.4f}%**\n\n## Reproducibility artifacts\n\n- `exact_top8_b1926.npz`: exact unfiltered Top-{metadata["exact_top_k"]} IDs and scores.\n- `legal_candidates_b1926.npz`: every legal query/reference pair before filtering.\n- `filtered_candidate_ids_default.npz`: retained IDs and query offsets at the selected point.\n- `fenand_oms_sensitivity.csv`: every measured sweep point.\n- `metadata.json`: parameters, timing, hardware, and command.\n\nThe current OMS estimator's existing precomputed cache is 8192-bit; this experiment does not relabel it as 2048-bit. It uses the repository's 2048-bit Hyper-Spec level-ID codebook and rebuilds reference/query HVs from the same sparse homs-tc vectors.\n\n## Rerun\n\n```bash\nCUDA_VISIBLE_DEVICES=0 python3 \\\n  proteomic_full_pipeline/run_oms_filter_sweep.py\n```\n\n## Old unsupported values\n\nThe previous 99.9536% reduction / 99.7070% Top-{metadata["exact_top_k"]} recall pair has no reproducible provenance in the repository and is not used for threshold selection. The new result differs by {metadata["difference_from_old_percentage_points"]["reduction"]:.4f} percentage points in reduction and {metadata["difference_from_old_percentage_points"]["recall"]:.4f} percentage points in recall.\n'''


if __name__ == "__main__":
    main()
