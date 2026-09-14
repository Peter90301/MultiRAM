#!/usr/bin/env python3
"""Build a deterministic ~1 TB raw-file subset for PXD024364/MSV000086944."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


REMOTE = ":ftp:/v03/MSV000086944/raw/raw/"
FTP_ARGS = [
    "--ftp-host",
    "massive-ftp.ucsd.edu",
    "--ftp-user",
    "anonymous",
    "--ftp-explicit-tls",
    "--ftp-no-check-certificate",
    "--contimeout",
    "20s",
    "--timeout",
    "1m",
]


@dataclass(frozen=True)
class RemoteFile:
    size_bytes: int
    relative_path: str
    path_hash: str


def obscure_password() -> str:
    return subprocess.check_output(
        ["rclone", "obscure", "anonymous"], text=True
    ).strip()


def fetch_remote_manifest() -> list[RemoteFile]:
    command = [
        "rclone",
        "lsf",
        "-R",
        "--files-only",
        "--format",
        "sp",
        "--separator",
        "\t",
        *FTP_ARGS,
        "--ftp-pass",
        obscure_password(),
        REMOTE,
    ]
    output = subprocess.check_output(command, text=True)
    files: list[RemoteFile] = []
    for line in output.splitlines():
        size_text, relative_path = line.split("\t", 1)
        files.append(
            RemoteFile(
                size_bytes=int(size_text),
                relative_path=relative_path,
                path_hash=hashlib.sha256(relative_path.encode("utf-8")).hexdigest(),
            )
        )
    if not files:
        raise RuntimeError("MassIVE raw manifest was empty")
    return files


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-bytes", type=int, default=1_000_000_000_000)
    parser.add_argument(
        "--host-dram-bytes", type=int, default=810_904_592_384
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = fetch_remote_manifest()

    write_tsv(
        args.output_dir / "official_raw_manifest.tsv",
        ["size_bytes", "relative_path", "path_sha256"],
        [
            {
                "size_bytes": item.size_bytes,
                "relative_path": item.relative_path,
                "path_sha256": item.path_hash,
            }
            for item in sorted(files, key=lambda item: item.relative_path)
        ],
    )

    selected: list[RemoteFile] = []
    selected_bytes = 0
    for item in sorted(files, key=lambda item: (item.path_hash, item.relative_path)):
        if selected_bytes >= args.target_bytes:
            break
        selected.append(item)
        selected_bytes += item.size_bytes

    write_tsv(
        args.output_dir / "selected_raw_manifest.tsv",
        ["selection_rank", "size_bytes", "relative_path", "path_sha256"],
        [
            {
                "selection_rank": rank,
                "size_bytes": item.size_bytes,
                "relative_path": item.relative_path,
                "path_sha256": item.path_hash,
            }
            for rank, item in enumerate(selected, start=1)
        ],
    )
    (args.output_dir / "selected_paths.txt").write_text(
        "".join(f"{item.relative_path}\n" for item in selected), encoding="utf-8"
    )

    summary = {
        "dataset": "PXD024364 / MSV000086944",
        "source": "ftp://massive-ftp.ucsd.edu/v03/MSV000086944/raw/raw/",
        "selection_method": (
            "Sort independent raw files by SHA-256(relative_path), then include files "
            "until cumulative bytes first reach target_bytes."
        ),
        "target_bytes": args.target_bytes,
        "selected_files": len(selected),
        "selected_bytes": selected_bytes,
        "selected_TB_decimal": selected_bytes / 1e12,
        "selected_TiB": selected_bytes / 2**40,
        "host_dram_bytes": args.host_dram_bytes,
        "selected_to_host_dram_ratio": selected_bytes / args.host_dram_bytes,
        "remote_raw_files": len(files),
        "remote_raw_bytes": sum(item.size_bytes for item in files),
    }
    (args.output_dir / "selection_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
