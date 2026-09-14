#!/usr/bin/env python3
"""Run Hyper-Spec GPU clustering on a single MGF query file."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _patch_pandas_for_pyteomics() -> None:
    try:
        import pandas as pd  # type: ignore

        if not hasattr(pd, "version"):
            class _VersionShim:
                version = getattr(pd, "__version__", "0.0.0")

            pd.version = _VersionShim()  # type: ignore[attr-defined]
    except Exception:
        pass


_patch_pandas_for_pyteomics()

from pyteomics import mgf


HYPERSPEC_SRC = Path("/home/tsl012/multiomic/genomic_proteomic_CPU_test/Hyper-Spec-linux/src")
HYPERSPEC_PYTHON = Path("/home/tsl012/miniconda3/envs/multiomic/bin/python")


def _detect_cuda_path() -> str:
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        return cuda_path
    nvcc = shutil.which("nvcc")
    if not nvcc:
        return "/usr"
    return str(Path(nvcc).resolve().parent.parent)


def _prepare_cuda_env(env: dict[str, str]) -> dict[str, str]:
    cuda_path = _detect_cuda_path()
    env["CUDA_PATH"] = cuda_path
    env.setdefault("CCCL_IGNORE_DEPRECATED_CUDA_BELOW_12", "1")
    env.setdefault("CUPY_CACHE_IN_MEMORY", "0")

    cuda_include = Path(cuda_path) / "include"
    if cuda_include.is_dir():
        existing = env.get("CPLUS_INCLUDE_PATH", "")
        include_parts = [str(cuda_include)]
        if existing:
            include_parts.append(existing)
        env["CPLUS_INCLUDE_PATH"] = ":".join(include_parts)

    return env


def _detect_charges(query_path: Path) -> list[int]:
    charges: set[int] = set()
    with mgf.read(str(query_path)) as reader:
        for spectrum in reader:
            charge = spectrum["params"].get("charge")
            if not charge:
                continue
            value = charge[0] if isinstance(charge, (list, tuple)) else charge
            if isinstance(value, str):
                value = value.rstrip("+-")
            try:
                charges.add(int(value))
            except (TypeError, ValueError):
                continue
    return sorted(charges) or [2, 3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input MGF query file.")
    parser.add_argument("--output-prefix", required=True, help="Output prefix for Hyper-Spec artifacts.")
    parser.add_argument("--cpu-core-preprocess", type=int, default=8)
    parser.add_argument("--cpu-core-cluster", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--min-peaks", type=int, default=5)
    parser.add_argument("--min-mz-range", type=float, default=250.0)
    parser.add_argument("--min-mz", type=float, default=101.0)
    parser.add_argument("--max-mz", type=float, default=1500.0)
    parser.add_argument("--remove-precursor-tol", type=float, default=1.5)
    parser.add_argument("--min-intensity", type=float, default=0.01)
    parser.add_argument("--max-peaks-used", type=int, default=50)
    parser.add_argument("--scaling", default="off", choices=["off", "root", "log", "rank"])
    parser.add_argument("--hd-dim", type=int, default=2048)
    parser.add_argument("--hd-q", type=int, default=16)
    parser.add_argument("--hd-id-flip-factor", type=float, default=2.0)
    parser.add_argument("--precursor-tol", type=float, default=20.0)
    parser.add_argument("--precursor-tol-mode", default="ppm", choices=["ppm", "Da"])
    parser.add_argument("--fragment-tol", type=float, default=0.05)
    parser.add_argument("--eps", type=float, default=0.2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    query_path = Path(args.input).resolve()
    output_prefix = Path(args.output_prefix).resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    input_dir = output_prefix.parent / f"{output_prefix.stem}_hyperspec_input"
    input_dir.mkdir(parents=True, exist_ok=True)
    staged_query = input_dir / query_path.name
    if staged_query.exists() or staged_query.is_symlink():
        staged_query.unlink()
    staged_query.symlink_to(query_path)

    charges = _detect_charges(query_path)

    cmd = [
        str(HYPERSPEC_PYTHON),
        "main.py",
        str(input_dir),
        str(output_prefix),
        "--cpu_core_preprocess",
        str(args.cpu_core_preprocess),
        "--cpu_core_cluster",
        str(args.cpu_core_cluster),
        "--batch_size",
        str(args.batch_size),
        "--use_gpu_cluster",
        "--min_peaks",
        str(args.min_peaks),
        "--min_mz_range",
        str(args.min_mz_range),
        "--min_mz",
        str(args.min_mz),
        "--max_mz",
        str(args.max_mz),
        "--remove_precursor_tol",
        str(args.remove_precursor_tol),
        "--min_intensity",
        str(args.min_intensity),
        "--max_peaks_used",
        str(args.max_peaks_used),
        "--scaling",
        args.scaling,
        "--hd_dim",
        str(args.hd_dim),
        "--hd_Q",
        str(args.hd_q),
        "--hd_id_flip_factor",
        str(args.hd_id_flip_factor),
        "--precursor_tol",
        str(args.precursor_tol),
        args.precursor_tol_mode,
        "--fragment_tol",
        str(args.fragment_tol),
        "--cluster_alg",
        "dbscan",
        "--eps",
        str(args.eps),
        "--cluster_charges",
        *[str(charge) for charge in charges],
    ]

    env = _prepare_cuda_env(os.environ.copy())

    process = subprocess.run(
        cmd,
        cwd=str(HYPERSPEC_SRC),
        env=env,
        text=True,
    )
    return int(process.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
