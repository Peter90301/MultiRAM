#!/usr/bin/env python3
"""Focused checks for the FeNAND coarse-filter analytical model."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "sumukh_proteomic_test"
sys.path.insert(0, str(BENCHMARK_DIR))

from gpu_benchmark import (  # noqa: E402
    bucket_spectra_by_precursor,
    estimate_fenand_coarse_filter,
    nn_chain_hac_from_distance,
    parse_mgf_file,
)


class FeNANDCoarseFilterTest(unittest.TestCase):
    def test_mgf_parser_preserves_precursor_charge(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "query.mgf"
            path.write_text(
                "BEGIN IONS\nTITLE=test\nPEPMASS=501.2\nCHARGE=3+\n100 10\nEND IONS\n",
                encoding="utf-8",
            )

            spectra = parse_mgf_file(path)

        self.assertEqual(spectra[0][1]["precursor_charge"], 3)

    def test_bucketing_is_charge_aware(self) -> None:
        spectra = [
            ("a", {"precursor_mz": 501.0, "precursor_charge": 2}),
            ("b", {"precursor_mz": 502.0, "precursor_charge": 2}),
            ("c", {"precursor_mz": 501.0, "precursor_charge": 3}),
        ]

        buckets = bucket_spectra_by_precursor(spectra, bucket_width=5.0)

        self.assertEqual(sorted(map(len, buckets.values())), [1, 2])
        self.assertIn((2, 100), buckets)
        self.assertIn((3, 100), buckets)

    def test_hamming_threshold_uses_fixed_hv_dimension(self) -> None:
        distances = np.array([[0.0, 900.0], [900.0, 0.0]], dtype=np.float32)

        clusters = nn_chain_hac_from_distance(
            distances,
            threshold_ratio=0.455,
            hv_dimension=2048,
        )

        self.assertEqual(clusters, [[0, 1]])

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
