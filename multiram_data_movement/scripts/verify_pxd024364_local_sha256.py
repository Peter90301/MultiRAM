#!/usr/bin/env python3
"""Create a resumable local SHA-256 manifest after size-verifying PXD024364."""

from __future__ import annotations

import csv
import hashlib
import os
import time
from pathlib import Path


ROOT = Path("/mnt/hdd/tsunghan/raw-ms-dataset/proteomic_large_dataset")
SOURCE_MANIFEST = ROOT / "manifests/selected_raw_manifest.tsv"
OUTPUT_MANIFEST = ROOT / "manifests/local_sha256.tsv"
RAW_ROOT = ROOT / "raw"


def main() -> int:
    with SOURCE_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    completed: set[str] = set()
    if OUTPUT_MANIFEST.exists():
        with OUTPUT_MANIFEST.open(newline="", encoding="utf-8") as handle:
            completed = {row["relative_path"] for row in csv.DictReader(handle, delimiter="\t")}

    write_header = not OUTPUT_MANIFEST.exists() or OUTPUT_MANIFEST.stat().st_size == 0
    with OUTPUT_MANIFEST.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "size_bytes", "sha256", "hash_wall_s"],
            delimiter="\t",
        )
        if write_header:
            writer.writeheader()
        for row in rows:
            relative = row["relative_path"]
            if relative in completed:
                continue
            path = RAW_ROOT / relative
            expected = int(row["size_bytes"])
            if not path.exists() or path.stat().st_size != expected:
                raise RuntimeError(f"Missing or wrong-size file: {path}")
            digest = hashlib.sha256()
            start = time.perf_counter()
            with path.open("rb", buffering=8 * 1024 * 1024) as source:
                while chunk := source.read(8 * 1024 * 1024):
                    digest.update(chunk)
                if hasattr(os, "posix_fadvise"):
                    os.posix_fadvise(source.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            writer.writerow(
                {
                    "relative_path": relative,
                    "size_bytes": expected,
                    "sha256": digest.hexdigest(),
                    "hash_wall_s": f"{time.perf_counter() - start:.6f}",
                }
            )
            handle.flush()
            os.fsync(handle.fileno())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
