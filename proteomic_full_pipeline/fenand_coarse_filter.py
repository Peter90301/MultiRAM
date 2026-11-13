"""FeNAND coarse HDC filtering for the MultiRAM OMS hierarchy.

This module deliberately contains only the coarse stage.  Precursor-window
selection is performed by the caller, and full-vector Hamming ranking remains
the FeRAM/exact-search stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


BYTE_POPCOUNT = np.unpackbits(
    np.arange(256, dtype=np.uint8)[:, None], axis=1
).sum(axis=1).astype(np.uint8)


@dataclass(frozen=True)
class CoarseFilterConfig:
    """Hardware-level coarse-filter parameters.

    ``score_threshold_bits`` is a fixed minimum full-vector similarity.  A
    candidate is retained whenever its partial score plus the maximum possible
    contribution of unexamined bits reaches that threshold.
    """

    hv_dim_bits: int
    segment_bits: int = 64
    segments_examined: int = 1
    score_threshold_bits: int = 0
    chunk_size: int = 131072

    def __post_init__(self) -> None:
        if self.hv_dim_bits <= 0 or self.hv_dim_bits % 8:
            raise ValueError("hv_dim_bits must be a positive multiple of 8")
        if self.segment_bits <= 0 or self.segment_bits % 8:
            raise ValueError("segment_bits must be a positive multiple of 8")
        if self.hv_dim_bits % self.segment_bits:
            raise ValueError("hv_dim_bits must be divisible by segment_bits")
        if not 0 <= self.segments_examined <= self.hv_dim_bits // self.segment_bits:
            raise ValueError("segments_examined is outside the HV")
        if not 0 <= self.score_threshold_bits <= self.hv_dim_bits:
            raise ValueError("score_threshold_bits is outside the HV score range")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")


def _packed_hamming_matches(query_hv: np.ndarray, refs: np.ndarray) -> np.ndarray:
    query = np.asarray(query_hv, dtype=np.uint8).reshape(1, -1)
    reference = np.asarray(refs, dtype=np.uint8)
    if reference.ndim != 2 or reference.shape[1] != query.shape[1]:
        raise ValueError("reference_hvs must have shape (N, hv_dim / 8)")
    xor = np.bitwise_xor(reference, query)
    return (reference.shape[1] * 8 - BYTE_POPCOUNT[xor].sum(axis=1)).astype(np.int32)


def filter_candidates(
    query_hv: np.ndarray,
    reference_hvs: np.ndarray,
    candidate_ids: Iterable[int] | np.ndarray,
    config: CoarseFilterConfig,
) -> np.ndarray:
    """Return candidate IDs that pass the FeNAND upper-bound test.

    For each candidate, only the first ``segments_examined`` segments are
    read.  If ``p`` matching bits are observed, the full-vector score cannot
    exceed ``p + (D - examined_bits)``.  Candidates whose bound is below the
    fixed acceptance threshold are pruned.  This is not list truncation and
    cannot discard a candidate that actually reaches the configured threshold.
    """

    ids = np.asarray(candidate_ids, dtype=np.int64)
    refs = np.asarray(reference_hvs, dtype=np.uint8)
    if refs.ndim != 2 or refs.shape[0] != ids.size:
        raise ValueError("candidate_ids and reference_hvs must have equal rows")
    expected_bytes = config.hv_dim_bits // 8
    if refs.shape[1] != expected_bytes:
        raise ValueError("reference_hvs does not match config.hv_dim_bits")
    query = np.asarray(query_hv, dtype=np.uint8).reshape(-1)
    if query.size != expected_bytes:
        raise ValueError("query_hv does not match config.hv_dim_bits")

    examined_bytes = config.segments_examined * config.segment_bits // 8
    remaining_bits = config.hv_dim_bits - examined_bytes * 8
    keep = np.empty(ids.size, dtype=bool)
    for start in range(0, ids.size, config.chunk_size):
        stop = min(start + config.chunk_size, ids.size)
        partial = _packed_hamming_matches(query[:examined_bytes], refs[start:stop, :examined_bytes])
        keep[start:stop] = partial + remaining_bits >= config.score_threshold_bits
    return ids[keep]


def exact_topk(
    query_hv: np.ndarray,
    reference_hvs: np.ndarray,
    candidate_ids: Iterable[int] | np.ndarray,
    top_k: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute deterministic full-vector Hamming Top-K.

    Ties are resolved by ascending candidate ID, so repeated runs and all
    sensitivity points use the same ordering.
    """

    ids = np.asarray(candidate_ids, dtype=np.int64)
    scores = _packed_hamming_matches(query_hv, reference_hvs)
    if ids.size != scores.size:
        raise ValueError("candidate_ids and reference_hvs must have equal rows")
    k = min(max(int(top_k), 0), ids.size)
    if k == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int32)
    order = np.lexsort((ids, -scores))[:k]
    return ids[order], scores[order]


def candidate_reduction(unfiltered_pairs: int, surviving_pairs: int) -> float:
    """Return the fraction of candidate comparisons eliminated."""

    if unfiltered_pairs <= 0:
        return 0.0
    return 1.0 - float(surviving_pairs) / float(unfiltered_pairs)


def top8_recall(exact_top8_ids: np.ndarray, retained_ids: np.ndarray) -> tuple[int, int]:
    """Return retained exact-Top8 count and total valid Top8 count."""

    exact = np.asarray(exact_top8_ids, dtype=np.int64).reshape(-1)
    retained = set(np.asarray(retained_ids, dtype=np.int64).tolist())
    valid = exact[exact >= 0]
    return sum(int(value) in retained for value in valid), int(valid.size)
