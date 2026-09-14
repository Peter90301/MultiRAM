#!/usr/bin/env bash
set -euo pipefail

FRAMEWORK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${FRAMEWORK_ROOT}/configs/platinum_na12878_hg38_storage_config.json"
RAW="${FRAMEWORK_ROOT}/results_platinum_na12878_hg38/raw_logs/genomics_alignment/cpu/storage_bound/full/run_0"
OUT_DIR="${FRAMEWORK_ROOT}/results_platinum_na12878_hg38/workload_outputs/full"

mkdir -p "${RAW}" "${OUT_DIR}"
printf '%q ' "${FRAMEWORK_ROOT}/workloads/genomics/genomics_minimap2_pairs.sh" "${CONFIG}" > "${RAW}/command.sh"
echo >> "${RAW}/command.sh"

START="$(date +%s.%N)"
set +e
MULTIRAM_PAIR_TIMING_TSV="${RAW}/pair_timing.tsv" \
MULTIRAM_OUTPUT_DIR="${OUT_DIR}" \
/usr/bin/time -v -o "${RAW}/time_v.log" \
  "${FRAMEWORK_ROOT}/workloads/genomics/genomics_minimap2_pairs.sh" "${CONFIG}" \
  > "${RAW}/stdout.log" 2> "${RAW}/stderr.log"
RC=$?
set -e
END="$(date +%s.%N)"

python3 - "${RAW}/metadata.json" "${START}" "${END}" "${RC}" <<'PY'
import json
import platform
import socket
import sys

out, start, end, rc = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
meta = {
  "workload": "genomics_alignment",
  "stage": "genomics_alignment",
  "platform": "cpu",
  "dataset": "Platinum_Genomes_NA12878_PRJEB3246_36FASTQ_vs_hg38",
  "regime": "storage_bound_full",
  "run_id": "0",
  "run_mode": "warm_time_only",
  "total_runtime_s": max(0.0, end - start),
  "return_code": rc,
  "hostname": socket.gethostname(),
  "kernel": platform.platform(),
  "config": "/home/tsl012/multiomic/multiram_data_movement/configs/platinum_na12878_hg38_storage_config.json",
  "note": "full 18 paired FASTQ Platinum NA12878 run; output discarded; perf unavailable"
}
with open(out, "w", encoding="utf-8") as fh:
    json.dump(meta, fh, indent=2)
PY

exit "${RC}"
