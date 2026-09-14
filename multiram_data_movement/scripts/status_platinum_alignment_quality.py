#!/usr/bin/env python3
"""Show progress for the Platinum NA12878 alignment-quality run."""

from __future__ import annotations

import json
from pathlib import Path


RESULT_DIR = Path(
    "/home/tsl012/multiomic/multiram_data_movement/"
    "results_platinum_na12878_hg38/alignment_quality"
)


def main() -> None:
    pid_path = RESULT_DIR / "run.pid"
    summary_path = RESULT_DIR / "alignment_quality_summary.json"
    result: dict[str, object] = {}
    if pid_path.exists():
        pid = int(pid_path.read_text(encoding="utf-8").strip())
        result["pid"] = pid
        result["running"] = Path(f"/proc/{pid}").exists()
    else:
        result["running"] = False
    if summary_path.exists():
        result.update(json.loads(summary_path.read_text(encoding="utf-8")))
    else:
        result["status"] = "waiting_for_first_pair"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
