#!/usr/bin/env python3
"""Focused checks for the FeNAND coarse-filter analytical model."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "sumukh_proteomic_test"
sys.path.insert(0, str(BENCHMARK_DIR))

from gpu_benchmark import estimate_fenand_coarse_filter  # noqa: E402


class FeNANDCoarseFilterTest(unittest.TestCase):
    def test_pair_reduction_uses_observed_bucket_occupancy(self) -> None:
        spectra = [(f"s{i}", {}) for i in range(4)]
        buckets = {0: [0, 1, 2], 1: [3]}

        result = estimate_fenand_coarse_filter(
            spectra,
            buckets,
            metadata_bytes_per_spectrum=16,
            output_bytes_per_spectrum=256,
            decompressed_stream_gbps=8.1,
            package_link_gbps=256,
            setup_overhead_us=50,
            filter_energy_pj_per_spectrum=20,
            package_link_energy_pj_per_bit=0.8,
        )

        self.assertEqual(result["unfiltered_candidate_pairs"], 6)
        self.assertEqual(result["filtered_candidate_pairs"], 3)
        self.assertAlmostEqual(result["candidate_pair_survival_ratio"], 0.5)
        self.assertAlmostEqual(result["candidate_pair_reduction_percent"], 50.0)
        self.assertEqual(result["fenand_metadata_bytes"], 64)
        self.assertEqual(result["fenand_to_feram_bytes"], 1024)
        self.assertEqual(result["fenand_effective_output_GBps"], 8.1)
        self.assertGreater(result["fenand_filter_total_s"], 50e-6)
        self.assertEqual(result["spectrum_survival_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
