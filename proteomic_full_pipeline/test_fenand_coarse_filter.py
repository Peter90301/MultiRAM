import numpy as np

from fenand_coarse_filter import (
    CoarseFilterConfig,
    candidate_reduction,
    exact_topk,
    filter_candidates,
    top8_recall,
)


def test_loose_filter_retains_all_candidates():
    query = np.zeros(32, dtype=np.uint8)
    refs = np.arange(5 * 32, dtype=np.uint8).reshape(5, 32)
    ids = np.arange(5)
    cfg = CoarseFilterConfig(256, segments_examined=0, score_threshold_bits=0)
    np.testing.assert_array_equal(filter_candidates(query, refs, ids, cfg), ids)


def test_more_aggressive_filter_never_increases_candidate_count():
    rng = np.random.default_rng(7)
    query = rng.integers(0, 256, 32, dtype=np.uint8)
    refs = rng.integers(0, 256, (100, 32), dtype=np.uint8)
    ids = np.arange(100)
    loose = filter_candidates(query, refs, ids, CoarseFilterConfig(256, 64, 1, 0))
    strict = filter_candidates(query, refs, ids, CoarseFilterConfig(256, 64, 1, 180))
    assert strict.size <= loose.size


def test_exact_topk_matches_bruteforce_and_tie_breaks_by_id():
    query = np.zeros(1, dtype=np.uint8)
    refs = np.asarray([[0], [255], [0]], dtype=np.uint8)
    ids = np.asarray([20, 30, 10])
    top_ids, scores = exact_topk(query, refs, ids, top_k=3)
    np.testing.assert_array_equal(top_ids, [10, 20, 30])
    np.testing.assert_array_equal(scores, [8, 8, 0])


def test_metric_formulas():
    assert candidate_reduction(100, 25) == 0.75
    kept, total = top8_recall(np.asarray([1, 2, 3, -1]), np.asarray([2, 4]))
    assert (kept, total) == (1, 3)
