#!/usr/bin/env python3
"""PIM-side FeRAM/HDC OMS runner used by the benchmark pipeline.

Modes:
- clustering: keep the existing NeuroSIM-style clustering estimate.
- oms: run a strict 1% FDR OMS flow using 8192-d binary hypervectors.

The OMS implementation intentionally mirrors the homs-tc preprocessing and
target-decoy post-processing stages, while replacing the GPU tensor-core
similarity engine with a hardware-aligned HDC encoder plus Hamming-style
similarity search.
"""

from __future__ import annotations

import argparse
import configparser
import copy
import json
import math
import os
import time
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


os.environ.setdefault("CUDA_PATH", "/usr")
os.environ.setdefault("CCCL_IGNORE_DEPRECATED_CUDA_BELOW_12", "1")
os.environ.setdefault("CUPY_CACHE_IN_MEMORY", "0")


REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_BENCHMARK_DIR = str(REPO_ROOT / "sumukh_proteomic_test")
if GPU_BENCHMARK_DIR not in sys.path:
    sys.path.insert(0, GPU_BENCHMARK_DIR)

HOMSTC_ROOT = str(REPO_ROOT / "sumukh_proteomic_test/homs-tc")
if HOMSTC_ROOT not in sys.path:
    sys.path.insert(0, HOMSTC_ROOT)


def _patch_pandas_for_pyteomics() -> None:
    """Compatibility shim for old pyteomics expecting pandas.version.version."""
    try:
        import pandas as pd  # type: ignore

        if not hasattr(pd, "version"):
            class _VersionShim:
                version = getattr(pd, "__version__", "0.0.0")

            pd.version = _VersionShim()  # type: ignore[attr-defined]
    except Exception:
        pass


def _import_gpu_benchmark_symbols():
    try:
        from gpu_benchmark import estimate_feram_clustering_from_spectra, parse_mgf_file  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "gpu_benchmark dependencies are unavailable in this environment. "
            "Use an environment with torch installed for --mode clustering."
        ) from exc
    return estimate_feram_clustering_from_spectra, parse_mgf_file


_patch_pandas_for_pyteomics()

from spectrum_utils.spectrum import MsmsSpectrum


BYTE_POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1).astype(np.uint8)
NS = {"pep": "http://regis-web.systemsbiology.net/pepXML"}


def _import_oms_symbols():
    from homs_tc.definition import Config, SpectrumSpectrumMatch  # type: ignore
    from homs_tc.filter import filter_fdr, filter_group_fdr  # type: ignore
    from homs_tc.preprocess_utils import process_spectrum, spectrum_to_vector  # type: ignore
    from homs_tc.reader import SpectralLibraryReader, read_mgf  # type: ignore
    from homs_tc.writer import write_mztab  # type: ignore

    return {
        "Config": Config,
        "SpectrumSpectrumMatch": SpectrumSpectrumMatch,
        "filter_fdr": filter_fdr,
        "filter_group_fdr": filter_group_fdr,
        "process_spectrum": process_spectrum,
        "spectrum_to_vector": spectrum_to_vector,
        "SpectralLibraryReader": SpectralLibraryReader,
        "read_mgf": read_mgf,
        "write_mztab": write_mztab,
    }


@dataclass
class EncodedSpectrum:
    spectrum: MsmsSpectrum
    packed_hv: np.ndarray


@dataclass
class SearchHit:
    query_spectrum: MsmsSpectrum
    library_spectrum: MsmsSpectrum
    score: float
    stage: str


class HdcEncoder:
    """8192-d binary HDC encoder with HyperOMS-compatible codebooks."""

    def __init__(
        self,
        spectrum_dim: int,
        hv_dim: int,
        quant_levels: int,
        flip_bits: int,
        seed: int,
        codebook: str = "random",
    ) -> None:
        self.spectrum_dim = int(spectrum_dim)
        self.hv_dim = int(hv_dim)
        self.quant_levels = max(2, int(quant_levels))
        self.flip_bits = max(1, min(int(flip_bits), self.hv_dim))
        self.seed = int(seed)
        self.codebook = str(codebook)
        self._position_packed = self._build_position_table()
        self._level_bits = self._build_level_table()

    def encode(self, spectrum: MsmsSpectrum, config: Config) -> np.ndarray:
        sparse = spectrum_to_vector(
            spectrum,
            config.min_mz,
            config.max_mz,
            config.bin_size,
            config.spectrum_vector_dim,
            config.min_bound,
        )
        if not sparse:
            return np.zeros(self.hv_dim // 8, dtype=np.uint8)

        bins = np.asarray([idx for idx, _ in sparse], dtype=np.int32)
        intensities = np.asarray([float(value) for _, value in sparse], dtype=np.float32)
        max_intensity = float(np.max(intensities)) if intensities.size else 0.0
        if max_intensity <= 0:
            return np.zeros(self.hv_dim // 8, dtype=np.uint8)

        levels = np.clip(
            np.rint((intensities / max_intensity) * (self.quant_levels - 1)),
            0,
            self.quant_levels - 1,
        ).astype(np.int32)

        accumulator = np.zeros(self.hv_dim, dtype=np.int32)
        for bin_idx, level_idx in zip(bins, levels):
            if bin_idx < 0 or bin_idx >= self.spectrum_dim:
                continue
            pos_bits = np.unpackbits(self._position_packed[bin_idx], bitorder="little")
            xor_bits = np.bitwise_xor(pos_bits, self._level_bits[level_idx])
            sign = 1 - (xor_bits.astype(np.int32) << 1)
            accumulator += sign * max(1, int(level_idx) + 1)

        return np.packbits((accumulator >= 0).astype(np.uint8), bitorder="little")

    def hamming_similarity(self, query_packed: np.ndarray, refs_packed: np.ndarray) -> np.ndarray:
        xor = np.bitwise_xor(refs_packed, query_packed)
        distances = BYTE_POPCOUNT[xor].sum(axis=1, dtype=np.int32)
        return (self.hv_dim - distances).astype(np.float32)

    def _build_position_table(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        if self.codebook == "random":
            bits = rng.integers(0, 2, size=(self.spectrum_dim, self.hv_dim), dtype=np.uint8)
            return np.packbits(bits, axis=1, bitorder="little")

        permutation = rng.permutation(self.hv_dim)
        table = np.empty((self.spectrum_dim, self.hv_dim // 8), dtype=np.uint8)
        bits = rng.integers(0, 2, size=self.hv_dim, dtype=np.uint8)
        table[0] = np.packbits(bits, bitorder="little")
        for idx in range(1, self.spectrum_dim):
            start = ((idx - 1) * self.flip_bits) % self.hv_dim
            flip_idx = permutation.take(np.arange(start, start + self.flip_bits), mode="wrap")
            bits[flip_idx] ^= 1
            table[idx] = np.packbits(bits, bitorder="little")
        return table

    def _build_level_table(self) -> np.ndarray:
        table = np.empty((self.quant_levels, self.hv_dim), dtype=np.uint8)
        if self.codebook == "random":
            bits = np.concatenate(
                [
                    np.zeros(self.hv_dim // 2, dtype=np.uint8),
                    np.ones(self.hv_dim - self.hv_dim // 2, dtype=np.uint8),
                ]
            )
            for level in range(self.quant_levels):
                level_bits = bits.copy()
                flip = int((level / float(max(self.quant_levels - 1, 1))) * self.hv_dim / 2.0)
                if flip > 0:
                    level_bits[:flip] ^= 1
                table[level] = level_bits
            return table

        rng = np.random.default_rng(self.seed + 1)
        permutation = rng.permutation(self.hv_dim)
        bits = rng.integers(0, 2, size=self.hv_dim, dtype=np.uint8)
        table[0] = bits.copy()
        flips_per_level = max(1, self.hv_dim // self.quant_levels)
        for level in range(1, self.quant_levels):
            start = ((level - 1) * flips_per_level) % self.hv_dim
            flip_idx = permutation.take(np.arange(start, start + flips_per_level), mode="wrap")
            bits[flip_idx] ^= 1
            table[level] = bits.copy()
        return table


class LibraryHvCache:
    """Lazy cache to avoid re-encoding the entire library up front."""

    def __init__(self, lib_reader: SpectralLibraryReader, config: Config, encoder: HdcEncoder) -> None:
        self.lib_reader = lib_reader
        self.config = config
        self.encoder = encoder
        self._cache: Dict[int, Optional[EncodedSpectrum]] = {}

    def get(self, spec_id: int) -> Optional[EncodedSpectrum]:
        if spec_id in self._cache:
            return self._cache[spec_id]

        spectrum = self.lib_reader.get_spectrum(spec_id, process_peaks=False)
        processed = process_spectrum(spectrum, self.config, True)
        if not processed.is_valid:
            self._cache[spec_id] = None
            return None

        packed_hv = self.encoder.encode(processed, self.config)
        encoded = EncodedSpectrum(processed, packed_hv)
        self._cache[spec_id] = encoded
        return encoded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PIM clustering/OMS in benchmark-compatible format.")
    parser.add_argument("--mode", choices=["clustering", "oms"], required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--ref", default=None, help="Reference SPLIB file path (required for --mode oms).")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "sumukh_proteomic_test/homs-tc/configs/hek293.ini"),
        help="INI config file used to read OMS/FDR settings.",
    )
    parser.add_argument("--output", default=None, help="Output mzTab path for --mode oms.")
    parser.add_argument("--fdr-threshold", type=float, default=0.01, help="Final strict PSM-level FDR threshold.")
    parser.add_argument(
        "--std-ppm",
        type=float,
        default=None,
        help="Narrow precursor ppm window. Default from config precursor_tolerance_mass_ppm.",
    )
    parser.add_argument(
        "--open-da",
        type=float,
        default=None,
        help="Wide precursor Da window. Default from config precursor_tolerance_mass_open_da.",
    )
    parser.add_argument(
        "--candidate-cap",
        type=int,
        default=0,
        help="Optional cap on candidate references per query/window; 0 means evaluate all precursor-window candidates.",
    )
    parser.add_argument(
        "--hv-flip-bits",
        type=int,
        default=None,
        help="Number of bit flips between adjacent position hypervectors; default is hv_dimensionality // 2.",
    )
    parser.add_argument("--hv-seed", type=int, default=13, help="Deterministic seed used to build HDC codebooks.")
    parser.add_argument(
        "--hdc-codebook",
        choices=["random", "locality"],
        default="random",
        help="Position/level codebook. random matches the HyperOMS/homs-tc item-memory style; locality keeps the older adjacent-bin flip table.",
    )
    parser.add_argument(
        "--backend",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Functional simulator backend for OMS similarity search.",
    )
    parser.add_argument(
        "--gpu-chunk-size",
        type=int,
        default=8192,
        help="Number of library spectra per GPU similarity/encoding chunk.",
    )
    parser.add_argument(
        "--hv-cache-dir",
        default=None,
        help="Directory for GPU backend encoded library HV memmaps. Defaults to the reference directory.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=1000,
        help="Print progress every N query spectra in GPU backend; 0 disables progress.",
    )
    parser.add_argument(
        "--no-gpu-resident-library",
        action="store_true",
        help="Do not preload sorted encoded library HVs into GPU memory.",
    )
    parser.add_argument(
        "--score-mode",
        choices=["raw_hamming", "normalized", "zscore", "zscore_margin"],
        default="zscore_margin",
        help=(
            "HDC score calibration. raw_hamming is D-HammingDistance; normalized is "
            "1 - 2*distance/D; zscore and zscore_margin normalize the best hit against "
            "the per-query candidate score distribution before FDR."
        ),
    )
    parser.add_argument(
        "--margin-weight",
        type=float,
        default=1.0,
        help="Weight for best-minus-second-best normalized similarity in --score-mode zscore_margin.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="Optional cap on processed query spectra for calibration/debug runs; 0 means all queries.",
    )
    parser.add_argument("--max-peaks", type=int, default=50)
    parser.add_argument("--mz-max", type=float, default=2000.0)
    parser.add_argument("--cluster-bucket-width-da", type=float, default=10.0)
    parser.add_argument(
        "--disable-fenand-coarse-filter",
        action="store_true",
        help="Keep precursor bucketing but model it as free host preprocessing.",
    )
    parser.add_argument("--fenand-metadata-bytes-per-spectrum", type=float, default=16.0)
    parser.add_argument("--fenand-output-bytes-per-spectrum", type=float, default=256.0)
    parser.add_argument("--fenand-decompressed-gbps", type=float, default=8.1)
    parser.add_argument("--fenand-package-link-gbps", type=float, default=256.0)
    parser.add_argument("--fenand-filter-setup-us", type=float, default=50.0)
    parser.add_argument(
        "--fenand-filter-energy-pj-per-spectrum", type=float, default=20.0
    )
    parser.add_argument("--package-link-energy-pj-per-bit", type=float, default=0.8)
    return parser.parse_args()


def _load_config(config_path: Path) -> Config:
    cp = configparser.ConfigParser()
    with config_path.open("r", encoding="utf-8") as fh:
        cp.read_file(fh)
    return Config(cp)


def _read_query_spectra(query_path: Path) -> List[MsmsSpectrum]:
    try:
        return list(read_mgf(str(query_path)))
    except Exception:
        spectra: List[MsmsSpectrum] = []
        in_ions = False
        params: Dict[str, str] = {}
        mzs: List[float] = []
        intensities: List[float] = []
        index = 0

        with query_path.open("r", encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                upper = line.upper()
                if upper == "BEGIN IONS":
                    in_ions = True
                    params = {}
                    mzs = []
                    intensities = []
                    continue
                if upper == "END IONS":
                    if in_ions and "PEPMASS" in params and mzs:
                        identifier = params.get("TITLE", f"query_{index}")
                        precursor_mz = float(params["PEPMASS"].split()[0].split(",")[0])
                        retention_time = float(params.get("RTINSECONDS", 0.0) or 0.0)
                        charge = None
                        if "CHARGE" in params:
                            token = params["CHARGE"].rstrip("+-")
                            try:
                                charge = int(token)
                            except ValueError:
                                charge = None
                        spectrum = MsmsSpectrum(
                            identifier,
                            precursor_mz,
                            charge,
                            np.asarray(mzs, dtype=np.float32),
                            np.asarray(intensities, dtype=np.float32),
                            retention_time=retention_time,
                        )
                        spectrum.index = index
                        spectrum.is_processed = False
                        spectra.append(spectrum)
                        index += 1
                    in_ions = False
                    continue
                if not in_ions:
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    params[key.strip().upper()] = value.strip()
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        mzs.append(float(parts[0]))
                        intensities.append(float(parts[1]))
                    except ValueError:
                        continue
        return spectra


def _prepare_query_spectra(query_path: Path, config: Config, mz_max: float) -> List[MsmsSpectrum]:
    processed_queries: List[MsmsSpectrum] = []
    for query_spectrum in _read_query_spectra(query_path):
        if float(query_spectrum.precursor_mz) > float(mz_max):
            continue
        if query_spectrum.precursor_charge is not None:
            candidates = [query_spectrum]
        else:
            candidates = []
            for charge in (2, 3):
                duplicated = copy.copy(query_spectrum)
                duplicated.precursor_charge = charge
                duplicated.is_processed = False
                candidates.append(duplicated)
        for candidate in candidates:
            processed = process_spectrum(candidate, config, False)
            if processed.is_valid:
                processed_queries.append(processed)
    return processed_queries


def _build_precursor_index(spec_info_charge: Dict[int, Dict[str, np.ndarray]]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    index: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for charge, info in spec_info_charge.items():
        ids = np.asarray(info["id"], dtype=np.int64)
        mzs = np.asarray(info["precursor_mz"], dtype=np.float64)
        if mzs.size == 0:
            continue
        order = np.argsort(mzs)
        index[int(charge)] = (mzs[order], ids[order])
    return index


def _candidate_ids_for_window(
    query_mz: float,
    charges: Sequence[int],
    precursor_index: Dict[int, Tuple[np.ndarray, np.ndarray]],
    tolerance_da: float,
    candidate_cap: int,
) -> np.ndarray:
    hits: List[np.ndarray] = []
    for charge in charges:
        if charge not in precursor_index:
            continue
        arr_mz, arr_ids = precursor_index[charge]
        left = int(np.searchsorted(arr_mz, query_mz - tolerance_da, side="left"))
        right = int(np.searchsorted(arr_mz, query_mz + tolerance_da, side="right"))
        if right > left:
            hits.append(arr_ids[left:right])

    if not hits:
        return np.empty(0, dtype=np.int64)

    merged = np.unique(np.concatenate(hits).astype(np.int64, copy=False))
    if candidate_cap > 0 and merged.size > candidate_cap:
        merged = merged[:candidate_cap]
    return merged


def _candidate_charges(query_spectrum: MsmsSpectrum, precursor_index: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> List[int]:
    charge = query_spectrum.precursor_charge
    if charge is not None and int(charge) in precursor_index:
        return [int(charge)]
    return list(precursor_index.keys())


def _ref_vector_file(ref_path: Path, config: Config, charge: int) -> Path:
    return ref_path.with_name(f"{ref_path.stem}_vec_{config.spectrum_vector_dim}.charge{charge}.npz")


def _hv_cache_file(
    cache_dir: Path,
    ref_path: Path,
    config: Config,
    seed: int,
    flip_bits: int,
    codebook: str,
    charge: int,
) -> Path:
    name = (
        f"{ref_path.stem}_pim_hv_D{config.hv_dimensionality}_Q{config.hv_quantize_level}"
        f"_seed{seed}_flip{flip_bits}_{codebook}.charge{charge}.uint8.mmap"
    )
    return cache_dir / name


def _load_ref_vector_cache(ref_path: Path, config: Config) -> Dict[int, Dict[str, np.ndarray]]:
    caches: Dict[int, Dict[str, np.ndarray]] = {}
    for charge in range(1, 9):
        vector_file = _ref_vector_file(ref_path, config, charge)
        if not vector_file.exists():
            continue
        data = np.load(vector_file, mmap_mode="r")
        pr_mzs = data["pr_mzs"]
        row_order = np.argsort(pr_mzs)
        caches[charge] = {
            "pr_mzs": pr_mzs,
            "sorted_pr_mzs": pr_mzs[row_order],
            "sorted_rows": row_order.astype(np.int64, copy=False),
            "spectra_idx": data["spectra_idx"],
            "spectra_intensities": data["spectra_intensities"],
            "spectra_identifier": data["spectra_identifier"],
            "csr_info": data["csr_info"],
        }
    if not caches:
        raise FileNotFoundError(
            f"No precomputed homs-tc vector files found next to {ref_path} "
            f"for spectrum_vector_dim={config.spectrum_vector_dim}."
        )
    return caches


def _prepare_cupy():
    import cupy as cp  # type: ignore
    import cupy.cuda.compiler as cp_compiler  # type: ignore
    from cupy_backends.cuda.libs import nvrtc as cp_nvrtc  # type: ignore

    try:
        nvrtc_major, _nvrtc_minor = cp_nvrtc.getVersion()
        cc = int(cp.cuda.Device().compute_capability)
    except Exception:
        return cp
    if nvrtc_major < 13 and cc >= 100:
        os.environ.setdefault("CUPY_COMPILE_WITH_PTX", "1")
        cp_compiler._use_ptx = True
        cp_compiler._get_arch = lambda: "90"
    return cp


def _rows_to_dense_sparse(cache: Dict[str, np.ndarray], start: int, stop: int) -> Tuple[np.ndarray, np.ndarray]:
    csr = cache["csr_info"]
    idx = cache["spectra_idx"]
    intensities = cache["spectra_intensities"]
    row_count = int(stop - start)
    lengths = (csr[start + 1 : stop + 1] - csr[start:stop]).astype(np.int32)
    max_len = int(lengths.max(initial=0))
    bins = np.full((row_count, max_len), -1, dtype=np.int32)
    vals = np.zeros((row_count, max_len), dtype=np.float32)
    for local_row, length in enumerate(lengths.tolist()):
        if length <= 0:
            continue
        left = int(csr[start + local_row])
        right = left + int(length)
        bins[local_row, :length] = idx[left:right]
        vals[local_row, :length] = intensities[left:right]
    return bins, vals


def _encode_sparse_batch_gpu(
    bins_np: np.ndarray,
    vals_np: np.ndarray,
    position_bits_gpu,
    level_bits_gpu,
    quant_levels: int,
    hv_dim: int,
):
    cp = _prepare_cupy()

    if bins_np.size == 0:
        return cp.empty((bins_np.shape[0], hv_dim // 8), dtype=cp.uint8)

    bins = cp.asarray(bins_np)
    vals = cp.asarray(vals_np)
    n_rows, n_peaks = bins.shape
    max_vals = cp.maximum(vals.max(axis=1, keepdims=True), cp.float32(1e-12))
    levels = cp.rint((vals / max_vals) * (quant_levels - 1)).astype(cp.int32)
    levels = cp.clip(levels, 0, quant_levels - 1)
    acc = cp.zeros((n_rows, hv_dim), dtype=cp.int16)

    for peak_idx in range(n_peaks):
        peak_bins = bins[:, peak_idx]
        valid = peak_bins >= 0
        if not bool(valid.any()):
            continue
        row_idx = cp.nonzero(valid)[0]
        pos_bits = position_bits_gpu[peak_bins[row_idx]]
        lev_bits = level_bits_gpu[levels[row_idx, peak_idx]]
        xor_bits = cp.bitwise_xor(pos_bits, lev_bits)
        weight = (levels[row_idx, peak_idx] + 1).astype(cp.int16)[:, None]
        sign = (1 - (xor_bits.astype(cp.int16) << 1)) * weight
        acc[row_idx] += sign

    bits = (acc >= 0).astype(cp.uint8).reshape(n_rows, hv_dim // 8, 8)
    weights = cp.asarray([1, 2, 4, 8, 16, 32, 64, 128], dtype=cp.uint8)
    return (bits * weights).sum(axis=2).astype(cp.uint8)


def _ensure_gpu_hv_cache(
    ref_path: Path,
    cache_dir: Path,
    config: Config,
    encoder: HdcEncoder,
    vector_caches: Dict[int, Dict[str, np.ndarray]],
    chunk_size: int,
    seed: int,
    flip_bits: int,
) -> Dict[int, np.memmap]:
    cp = _prepare_cupy()

    cache_dir.mkdir(parents=True, exist_ok=True)
    hv_bytes = config.hv_dimensionality // 8
    position_bits_gpu = cp.asarray(np.unpackbits(encoder._position_packed, axis=1, bitorder="little"))
    level_bits_gpu = cp.asarray(encoder._level_bits)
    hv_caches: Dict[int, np.memmap] = {}

    for charge, cache in sorted(vector_caches.items()):
        n_rows = int(cache["pr_mzs"].shape[0])
        cache_flip_bits = 0 if encoder.codebook == "random" else flip_bits
        hv_file = _hv_cache_file(cache_dir, ref_path, config, seed, cache_flip_bits, encoder.codebook, charge)
        meta_file = hv_file.with_suffix(hv_file.suffix + ".json")
        expected = n_rows * hv_bytes
        if not hv_file.exists() or hv_file.stat().st_size != expected:
            print(f"[PIM-GPU] Building HV cache charge={charge} rows={n_rows} -> {hv_file}", flush=True)
            mmap = np.memmap(hv_file, dtype=np.uint8, mode="w+", shape=(n_rows, hv_bytes))
            start_time = time.time()
            for start in range(0, n_rows, chunk_size):
                stop = min(start + chunk_size, n_rows)
                bins_np, vals_np = _rows_to_dense_sparse(cache, start, stop)
                packed = _encode_sparse_batch_gpu(
                    bins_np,
                    vals_np,
                    position_bits_gpu,
                    level_bits_gpu,
                    config.hv_quantize_level,
                    config.hv_dimensionality,
                )
                mmap[start:stop] = cp.asnumpy(packed)
                if start and (start // chunk_size) % 50 == 0:
                    print(
                        f"[PIM-GPU] charge={charge} encoded {start}/{n_rows} rows "
                        f"elapsed={time.time() - start_time:.1f}s",
                        flush=True,
                    )
            mmap.flush()
            meta_file.write_text(
                json.dumps(
                    {
                        "ref": str(ref_path),
                        "charge": charge,
                        "rows": n_rows,
                        "hv_bytes": hv_bytes,
                        "hv_dimensionality": config.hv_dimensionality,
                        "hv_quantize_level": config.hv_quantize_level,
                        "seed": seed,
                        "flip_bits": flip_bits,
                        "codebook": encoder.codebook,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        hv_caches[charge] = np.memmap(hv_file, dtype=np.uint8, mode="r", shape=(n_rows, hv_bytes))
    return hv_caches


def _candidate_rows_for_vector_cache(
    query_mz: float,
    charges: Sequence[int],
    vector_caches: Dict[int, Dict[str, np.ndarray]],
    tolerance_da: float,
    candidate_cap: int,
) -> List[Tuple[int, int, int]]:
    windows: List[Tuple[int, int, int]] = []
    remaining = int(candidate_cap) if candidate_cap > 0 else None
    for charge in charges:
        cache = vector_caches.get(int(charge))
        if cache is None:
            continue
        mzs = cache["sorted_pr_mzs"]
        left = int(np.searchsorted(mzs, query_mz - tolerance_da, side="left"))
        right = int(np.searchsorted(mzs, query_mz + tolerance_da, side="right"))
        if right <= left:
            continue
        if remaining is not None:
            take = min(right - left, remaining)
            right = left + take
            remaining -= take
        windows.append((int(charge), left, right))
        if remaining == 0:
            break
    return windows


def _load_gpu_resident_hvs(
    vector_caches: Dict[int, Dict[str, np.ndarray]],
    hv_caches: Dict[int, np.memmap],
):
    cp = _prepare_cupy()
    resident = {}
    for charge, hv_cache in sorted(hv_caches.items()):
        rows = vector_caches[charge]["sorted_rows"]
        print(f"[PIM-GPU] Loading sorted resident HV cache charge={charge} rows={len(rows)}", flush=True)
        resident[charge] = {
            "hvs": cp.asarray(hv_cache[rows]),
            "ids": np.asarray(vector_caches[charge]["spectra_identifier"][rows], dtype=np.int64),
        }
    return resident


def _find_best_hit_gpu(
    query: EncodedSpectrum,
    windows: Sequence[Tuple[int, int, int]],
    vector_caches: Dict[int, Dict[str, np.ndarray]],
    hv_caches: Dict[int, np.memmap],
    encoder: HdcEncoder,
    chunk_size: int,
    score_mode: str,
    margin_weight: float,
    resident_hvs=None,
):
    cp = _prepare_cupy()

    lut = cp.asarray(BYTE_POPCOUNT)
    query_gpu = cp.asarray(query.packed_hv)
    best_raw = -1.0
    second_raw = -1.0
    best_id: Optional[int] = None
    count = 0
    score_sum = 0.0
    score_sumsq = 0.0

    for charge, left, right in windows:
        if right <= left:
            continue
        if resident_hvs is not None and charge in resident_hvs:
            hv_source = resident_hvs[charge]["hvs"]
            ids = resident_hvs[charge]["ids"]
            sorted_rows = None
        else:
            hv_source = hv_caches[charge]
            ids = vector_caches[charge]["spectra_identifier"]
            sorted_rows = vector_caches[charge]["sorted_rows"]

        for start in range(left, right, chunk_size):
            stop = min(start + chunk_size, right)
            if sorted_rows is None:
                refs = hv_source[start:stop]
            else:
                chunk_rows = sorted_rows[start:stop]
                refs = cp.asarray(hv_source[chunk_rows])
            distances = lut[cp.bitwise_xor(refs, query_gpu)].sum(axis=1)
            scores = (1.0 - (2.0 * distances.astype(cp.float32) / float(encoder.hv_dim))).astype(cp.float32)
            chunk_count = int(scores.size)
            if chunk_count <= 0:
                continue
            count += chunk_count
            score_sum += float(scores.sum().get())
            score_sumsq += float((scores * scores).sum().get())
            if chunk_count >= 2:
                top2 = cp.sort(scores)[-2:]
                second_candidate = float(top2[0].get())
                best_candidate = float(top2[1].get())
                idx = int(cp.argmax(scores).get())
            else:
                best_candidate = float(scores[0].get())
                second_candidate = -1.0
                idx = 0
            if best_candidate > best_raw:
                second_raw = max(second_raw, best_raw, second_candidate)
                best_raw = best_candidate
                if sorted_rows is None:
                    best_id = int(ids[start + idx])
                else:
                    best_id = int(ids[int(sorted_rows[start + idx])])
            elif best_candidate > second_raw:
                second_raw = best_candidate

    if best_id is None:
        return None
    if score_mode == "raw_hamming":
        final_score = (best_raw + 1.0) * 0.5 * float(encoder.hv_dim)
    elif score_mode == "normalized":
        final_score = best_raw
    else:
        mean = score_sum / max(count, 1)
        variance = max((score_sumsq / max(count, 1)) - mean * mean, 1e-12)
        zscore = (best_raw - mean) / math.sqrt(variance)
        if score_mode == "zscore_margin":
            margin = best_raw - second_raw if second_raw > -1.0 else 0.0
            final_score = zscore + float(margin_weight) * margin
        else:
            final_score = zscore
    return best_id, float(final_score)


def _search_stage_gpu(
    queries: Sequence[EncodedSpectrum],
    vector_caches: Dict[int, Dict[str, np.ndarray]],
    hv_caches: Dict[int, np.memmap],
    encoder: HdcEncoder,
    candidate_cap: int,
    tolerance_resolver,
    stage: str,
    chunk_size: int,
    score_mode: str,
    margin_weight: float,
    progress_interval: int,
    resident_hvs=None,
) -> List[Tuple[EncodedSpectrum, int, float, str]]:
    hits: List[Tuple[EncodedSpectrum, int, float, str]] = []
    start_time = time.time()
    available_charges = sorted(vector_caches)
    for query_idx, query in enumerate(queries, start=1):
        charge = query.spectrum.precursor_charge
        charges = [int(charge)] if charge is not None and int(charge) in vector_caches else available_charges
        windows = _candidate_rows_for_vector_cache(
            float(query.spectrum.precursor_mz),
            charges,
            vector_caches,
            float(tolerance_resolver(query.spectrum)),
            candidate_cap,
        )
        if windows:
            best = _find_best_hit_gpu(
                query,
                windows,
                vector_caches,
                hv_caches,
                encoder,
                chunk_size,
                score_mode,
                margin_weight,
                resident_hvs,
            )
            if best is not None:
                best_id, score = best
                hits.append((query, best_id, score, stage))
        if progress_interval > 0 and query_idx % progress_interval == 0:
            print(
                f"[PIM-GPU] {stage}: searched {query_idx}/{len(queries)} queries, "
                f"hits={len(hits)}, elapsed={time.time() - start_time:.1f}s",
                flush=True,
            )
    return hits


def _materialize_gpu_hits(
    hit_refs: Sequence[Tuple[EncodedSpectrum, int, float, str]],
    lib_reader: SpectralLibraryReader,
    config: Config,
) -> List[SearchHit]:
    spectrum_cache: Dict[int, Optional[MsmsSpectrum]] = {}
    hits: List[SearchHit] = []
    for query, spec_id, score, stage in hit_refs:
        if spec_id not in spectrum_cache:
            spectrum = lib_reader.get_spectrum(int(spec_id), process_peaks=False)
            processed = process_spectrum(spectrum, config, True)
            spectrum_cache[spec_id] = processed if processed.is_valid else None
        library_spectrum = spectrum_cache[spec_id]
        if library_spectrum is None:
            continue
        hits.append(
            SearchHit(
                query_spectrum=query.spectrum,
                library_spectrum=library_spectrum,
                score=float(score),
                stage=stage,
            )
        )
    return hits


def _find_best_hit(
    query: EncodedSpectrum,
    candidate_ids: np.ndarray,
    cache: LibraryHvCache,
    encoder: HdcEncoder,
    stage: str,
    score_mode: str,
    margin_weight: float,
) -> Optional[SearchHit]:
    ref_specs: List[MsmsSpectrum] = []
    ref_hvs: List[np.ndarray] = []
    for spec_id in candidate_ids.tolist():
        encoded = cache.get(int(spec_id))
        if encoded is None:
            continue
        ref_specs.append(encoded.spectrum)
        ref_hvs.append(encoded.packed_hv)

    if not ref_specs:
        return None

    refs_matrix = np.stack(ref_hvs, axis=0)
    distances = encoder.hv_dim - encoder.hamming_similarity(query.packed_hv, refs_matrix)
    norm_scores = 1.0 - (2.0 * distances / float(encoder.hv_dim))
    if score_mode == "raw_hamming":
        scores = encoder.hamming_similarity(query.packed_hv, refs_matrix)
    elif score_mode == "normalized":
        scores = norm_scores
    else:
        mean = float(np.mean(norm_scores))
        std = float(np.std(norm_scores))
        std = max(std, 1e-6)
        best_raw = float(np.max(norm_scores))
        if norm_scores.size >= 2:
            second_raw = float(np.partition(norm_scores, -2)[-2])
        else:
            second_raw = best_raw
        zscore = (norm_scores - mean) / std
        if score_mode == "zscore_margin":
            margin = norm_scores - second_raw
            scores = zscore + float(margin_weight) * margin
        else:
            scores = zscore
    best_idx = int(np.argmax(scores))
    return SearchHit(
        query_spectrum=query.spectrum,
        library_spectrum=ref_specs[best_idx],
        score=float(scores[best_idx]),
        stage=stage,
    )


def _search_stage(
    queries: Sequence[EncodedSpectrum],
    precursor_index: Dict[int, Tuple[np.ndarray, np.ndarray]],
    cache: LibraryHvCache,
    encoder: HdcEncoder,
    candidate_cap: int,
    tolerance_resolver,
    stage: str,
    score_mode: str,
    margin_weight: float,
) -> List[SearchHit]:
    hits: List[SearchHit] = []
    for query in queries:
        charges = _candidate_charges(query.spectrum, precursor_index)
        if not charges:
            continue
        tolerance_da = float(tolerance_resolver(query.spectrum))
        candidate_ids = _candidate_ids_for_window(
            float(query.spectrum.precursor_mz),
            charges,
            precursor_index,
            tolerance_da=tolerance_da,
            candidate_cap=candidate_cap,
        )
        if candidate_ids.size == 0:
            continue
        best_hit = _find_best_hit(
            query,
            candidate_ids,
            cache,
            encoder,
            stage=stage,
            score_mode=score_mode,
            margin_weight=margin_weight,
        )
        if best_hit is not None:
            hits.append(best_hit)
    return hits


def _search_hits_to_ssms(hits: Sequence[SearchHit]) -> List[SpectrumSpectrumMatch]:
    return [
        SpectrumSpectrumMatch(
            hit.query_spectrum,
            hit.library_spectrum,
            search_engine_score=hit.score,
            q=math.nan,
        )
        for hit in hits
    ]


def _strict_fdr_merge(
    std_hits: Sequence[SearchHit],
    open_hits: Sequence[SearchHit],
    fdr_threshold: float,
    config: Config,
) -> Tuple[List[SpectrumSpectrumMatch], Dict[str, int]]:
    std_ssms = _search_hits_to_ssms(std_hits)
    open_ssms = _search_hits_to_ssms(open_hits)

    accepted_std = list(filter_fdr(std_ssms, fdr_threshold))
    identified_std = {ssm.query_identifier for ssm in accepted_std}
    accepted_open = [
        ssm
        for ssm in filter_group_fdr(
            open_ssms,
            fdr_threshold,
            config.fdr_tolerance_mass,
            config.fdr_tolerance_mode,
            config.fdr_min_group_size,
        )
        if ssm.query_identifier not in identified_std
    ]
    identifications = accepted_std + accepted_open
    counts = {
        "standard_raw": len(std_ssms),
        "standard_accepted": len(accepted_std),
        "open_raw": len(open_ssms),
        "open_accepted": len(accepted_open),
        "final_identifications": len(identifications),
    }
    return identifications, counts


def _summarize_hit_scores(hits: Sequence[SearchHit]) -> Dict[str, object]:
    target_scores = [float(hit.score) for hit in hits if not hit.library_spectrum.is_decoy]
    decoy_scores = [float(hit.score) for hit in hits if hit.library_spectrum.is_decoy]

    def _stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
        if not values:
            return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
        arr = np.asarray(values, dtype=np.float64)
        return {
            "count": int(arr.size),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(np.max(arr)),
        }

    return {
        "total": len(hits),
        "target": _stats(target_scores),
        "decoy": _stats(decoy_scores),
    }


def run_pim_oms(args: argparse.Namespace) -> dict:
    globals().update(_import_oms_symbols())

    if not args.ref:
        raise ValueError("--ref is required when --mode oms")

    query_path = Path(args.query)
    ref_path = Path(args.ref)
    config_path = Path(args.config)

    if not ref_path.exists():
        raise FileNotFoundError(f"Reference file does not exist: {ref_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    homs_cfg = _load_config(config_path)
    homs_cfg.fdr_threshold = float(args.fdr_threshold)
    std_ppm = float(args.std_ppm if args.std_ppm is not None else homs_cfg.precursor_tolerance_mass_ppm)
    open_da = float(args.open_da if args.open_da is not None else homs_cfg.precursor_tolerance_mass_open_da)
    flip_bits = int(args.hv_flip_bits if args.hv_flip_bits is not None else max(1, homs_cfg.hv_dimensionality // 2))

    query_spectra = _prepare_query_spectra(query_path, homs_cfg, mz_max=float(args.mz_max))
    if int(args.max_queries) > 0:
        query_spectra = query_spectra[: int(args.max_queries)]
    if not query_spectra:
        raise RuntimeError("No valid query spectra parsed after preprocessing.")

    encoder = HdcEncoder(
        spectrum_dim=homs_cfg.spectrum_vector_dim,
        hv_dim=homs_cfg.hv_dimensionality,
        quant_levels=homs_cfg.hv_quantize_level,
        flip_bits=flip_bits,
        seed=int(args.hv_seed),
        codebook=str(args.hdc_codebook),
    )
    encoded_queries = [EncodedSpectrum(q, encoder.encode(q, homs_cfg)) for q in query_spectra]

    simulator_start = time.time()

    if args.backend == "gpu":
        vector_caches = _load_ref_vector_cache(ref_path, homs_cfg)
        cache_dir = Path(args.hv_cache_dir) if args.hv_cache_dir else ref_path.parent
        hv_caches = _ensure_gpu_hv_cache(
            ref_path=ref_path,
            cache_dir=cache_dir,
            config=homs_cfg,
            encoder=encoder,
            vector_caches=vector_caches,
            chunk_size=int(args.gpu_chunk_size),
            seed=int(args.hv_seed),
            flip_bits=flip_bits,
        )
        resident_hvs = None if args.no_gpu_resident_library else _load_gpu_resident_hvs(vector_caches, hv_caches)

        std_hit_refs = _search_stage_gpu(
            encoded_queries,
            vector_caches,
            hv_caches,
            encoder,
            candidate_cap=int(args.candidate_cap),
            tolerance_resolver=lambda q: float(q.precursor_mz) * std_ppm * 1e-6,
            stage="standard",
            chunk_size=int(args.gpu_chunk_size),
            score_mode=str(args.score_mode),
            margin_weight=float(args.margin_weight),
            progress_interval=int(args.progress_interval),
            resident_hvs=resident_hvs,
        )

        with SpectralLibraryReader(str(ref_path), homs_cfg) as lib_reader:
            std_hits = _materialize_gpu_hits(std_hit_refs, lib_reader, homs_cfg)
            accepted_std, std_counts = _strict_fdr_merge(std_hits, [], float(args.fdr_threshold), homs_cfg)
            identified_std = {ssm.query_identifier for ssm in accepted_std}
            open_queries = [q for q in encoded_queries if q.spectrum.identifier not in identified_std]

            open_hit_refs = _search_stage_gpu(
                open_queries,
                vector_caches,
                hv_caches,
                encoder,
                candidate_cap=int(args.candidate_cap),
                tolerance_resolver=lambda _q: open_da,
                stage="open",
                chunk_size=int(args.gpu_chunk_size),
                score_mode=str(args.score_mode),
                margin_weight=float(args.margin_weight),
                progress_interval=int(args.progress_interval),
                resident_hvs=resident_hvs,
            )
            open_hits = _materialize_gpu_hits(open_hit_refs, lib_reader, homs_cfg)
            final_identifications, counts = _strict_fdr_merge(std_hits, open_hits, float(args.fdr_threshold), homs_cfg)

        out_mztab = Path(args.output) if args.output else query_path.with_suffix(".pim_oms.mztab")
        writer_args = SimpleNamespace(output=str(out_mztab), ref=str(ref_path), query=str(query_path))
        write_mztab(final_identifications, writer_args, homs_cfg)

        summary = {
            "mode": "oms",
            "backend": "gpu",
            "query": str(query_path),
            "ref": str(ref_path),
            "config": str(config_path),
            "std_ppm": std_ppm,
            "open_da": open_da,
            "fdr_threshold": float(args.fdr_threshold),
            "candidate_cap": int(args.candidate_cap),
            "gpu_chunk_size": int(args.gpu_chunk_size),
            "hv_cache_dir": str(cache_dir),
            "hv_dimensionality": int(homs_cfg.hv_dimensionality),
            "hv_quantize_level": int(homs_cfg.hv_quantize_level),
            "hv_flip_bits": flip_bits,
            "hv_seed": int(args.hv_seed),
            "hdc_codebook": str(args.hdc_codebook),
            "score_mode": str(args.score_mode),
            "margin_weight": float(args.margin_weight),
            "max_queries": int(args.max_queries),
            "num_query_spectra": len(encoded_queries),
            "counts": counts,
            "diagnostics": {
                "standard_scores": _summarize_hit_scores(std_hits),
                "open_scores": _summarize_hit_scores(open_hits),
            },
            "standard_stage_queries_after_fdr": std_counts["final_identifications"],
            "open_stage_queries": len(open_queries),
            "simulator_wall_time_sec": time.time() - simulator_start,
            "output_mztab": str(out_mztab),
        }

        summary_path = out_mztab.with_suffix(out_mztab.suffix + ".summary.json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary["summary_json"] = str(summary_path)
        return summary

    with SpectralLibraryReader(str(ref_path), homs_cfg) as lib_reader:
        precursor_index = _build_precursor_index(lib_reader.spec_info["charge"])
        cache = LibraryHvCache(lib_reader, homs_cfg, encoder)

        std_hits = _search_stage(
            encoded_queries,
            precursor_index,
            cache,
            encoder,
            candidate_cap=int(args.candidate_cap),
            tolerance_resolver=lambda q: float(q.precursor_mz) * std_ppm * 1e-6,
            stage="standard",
            score_mode=str(args.score_mode),
            margin_weight=float(args.margin_weight),
        )
        accepted_std, std_counts = _strict_fdr_merge(std_hits, [], float(args.fdr_threshold), homs_cfg)
        identified_std = {ssm.query_identifier for ssm in accepted_std}

        open_queries = [q for q in encoded_queries if q.spectrum.identifier not in identified_std]
        open_hits = _search_stage(
            open_queries,
            precursor_index,
            cache,
            encoder,
            candidate_cap=int(args.candidate_cap),
            tolerance_resolver=lambda _q: open_da,
            stage="open",
            score_mode=str(args.score_mode),
            margin_weight=float(args.margin_weight),
        )

        final_identifications, counts = _strict_fdr_merge(std_hits, open_hits, float(args.fdr_threshold), homs_cfg)

    out_mztab = Path(args.output) if args.output else query_path.with_suffix(".pim_oms.mztab")
    writer_args = SimpleNamespace(output=str(out_mztab), ref=str(ref_path), query=str(query_path))
    write_mztab(final_identifications, writer_args, homs_cfg)

    summary = {
        "mode": "oms",
        "backend": "cpu",
        "query": str(query_path),
        "ref": str(ref_path),
        "config": str(config_path),
        "std_ppm": std_ppm,
        "open_da": open_da,
        "fdr_threshold": float(args.fdr_threshold),
        "candidate_cap": int(args.candidate_cap),
        "hv_dimensionality": int(homs_cfg.hv_dimensionality),
        "hv_quantize_level": int(homs_cfg.hv_quantize_level),
        "hv_flip_bits": flip_bits,
        "hv_seed": int(args.hv_seed),
        "hdc_codebook": str(args.hdc_codebook),
        "score_mode": str(args.score_mode),
        "margin_weight": float(args.margin_weight),
        "max_queries": int(args.max_queries),
        "num_query_spectra": len(encoded_queries),
        "counts": counts,
        "diagnostics": {
            "standard_scores": _summarize_hit_scores(std_hits),
            "open_scores": _summarize_hit_scores(open_hits),
        },
        "standard_stage_queries_after_fdr": std_counts["final_identifications"],
        "open_stage_queries": len(open_queries),
        "simulator_wall_time_sec": time.time() - simulator_start,
        "output_mztab": str(out_mztab),
    }

    summary_path = out_mztab.with_suffix(out_mztab.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary


def main() -> int:
    args = parse_args()
    query_path = Path(args.query)
    if not query_path.exists():
        raise FileNotFoundError(f"Query file does not exist: {query_path}")

    if args.mode == "oms":
        oms_summary = run_pim_oms(args)
        print(json.dumps(oms_summary, indent=2))
        return 0

    estimate_feram_clustering_from_spectra, parse_mgf_file = _import_gpu_benchmark_symbols()
    spectra = parse_mgf_file(str(query_path), max_peaks=args.max_peaks, mz_max=args.mz_max)
    n = len(spectra)
    if n == 0:
        raise RuntimeError("No spectra parsed from query file.")

    if args.mode == "clustering":
        result = estimate_feram_clustering_from_spectra(
            spectra,
            bucket_width=args.cluster_bucket_width_da,
            max_items_per_bucket=2000,
            d_dim=2048,
            num_tiles=32,
            tile_cols=512,
            row_parallelism=2048,
            neurosim_dac_ns=2.0,
            neurosim_wl_driver_ns=1.0,
            neurosim_cell_read_ns=4.0,
            neurosim_sense_amp_ns=1.0,
            neurosim_adc_ns=8.0,
            neurosim_adder_tree_ns=3.0,
            neurosim_local_topk_ns=6.0,
            neurosim_global_interconnect_ns=5.0,
            neurosim_global_topk_ns=7.0,
            neurosim_dac_energy_fj_per_row=20.0,
            neurosim_cell_read_energy_fj=2.0,
            neurosim_adc_energy_fj_per_col=200.0,
            neurosim_digital_energy_fj_per_col=50.0,
            fenand_coarse_filter=not args.disable_fenand_coarse_filter,
            fenand_metadata_bytes_per_spectrum=args.fenand_metadata_bytes_per_spectrum,
            fenand_output_bytes_per_spectrum=args.fenand_output_bytes_per_spectrum,
            fenand_decompressed_stream_gbps=args.fenand_decompressed_gbps,
            fenand_package_link_gbps=args.fenand_package_link_gbps,
            fenand_setup_overhead_us=args.fenand_filter_setup_us,
            fenand_filter_energy_pj_per_spectrum=(
                args.fenand_filter_energy_pj_per_spectrum
            ),
            package_link_energy_pj_per_bit=args.package_link_energy_pj_per_bit,
        )
        print(json.dumps({"num_spectra": n, "mode": args.mode, "result": result}, indent=2))
        return 0

    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
