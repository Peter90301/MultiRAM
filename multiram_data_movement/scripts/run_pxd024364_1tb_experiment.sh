#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/home/tsl012/multiomic/multiram_data_movement/scripts"
ROOT="/mnt/hdd/tsunghan/raw-ms-dataset/proteomic_large_dataset"
LOG_DIR="${ROOT}/results/orchestrator"
DOWNLOAD_UNIT="multiram-pxd024364-download.service"
GENOMICS_UNIT="multiram-platinum-alignment-quality.service"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/experiment.log") 2>&1

echo "$(date --iso-8601=seconds) waiting for download"
while [[ "$(systemctl --user is-active "${DOWNLOAD_UNIT}" 2>/dev/null || true)" == "active" ]]; do
  sleep 60
done

download_result="$(systemctl --user show "${DOWNLOAD_UNIT}" --property=Result --value 2>/dev/null || true)"
if [[ -n "${download_result}" && "${download_result}" != "success" ]]; then
  echo "Download unit result is ${download_result}; refusing to profile incomplete input" >&2
  exit 1
fi

python3 "${SCRIPT_DIR}/status_pxd024364_download.py" | tee "${LOG_DIR}/download_final_status.json"
python3 - "${LOG_DIR}/download_final_status.json" <<'PY'
import json
import sys

status = json.load(open(sys.argv[1], encoding="utf-8"))
if status["completed_files"] != status["selected_files"] or status["mismatched_files"]:
    raise SystemExit("Download size verification is incomplete")
PY

echo "$(date --iso-8601=seconds) waiting for unrelated genomic I/O job"
while [[ "$(systemctl --user is-active "${GENOMICS_UNIT}" 2>/dev/null || true)" == "active" ]]; do
  sleep 60
done

echo "$(date --iso-8601=seconds) generating local SHA-256 provenance manifest"
python3 "${SCRIPT_DIR}/verify_pxd024364_local_sha256.py"

echo "$(date --iso-8601=seconds) evicting RAW page-cache entries"
python3 "${SCRIPT_DIR}/evict_pxd024364_page_cache.py" raw
echo "$(date --iso-8601=seconds) starting RAW-to-MS2-MGF conversion"
python3 "${SCRIPT_DIR}/run_pxd024364_raw_conversion.py"
python3 "${SCRIPT_DIR}/summarize_pxd024364_experiment.py"

for stage in cpu_clustering gpu_clustering cpu_oms gpu_oms; do
  echo "$(date --iso-8601=seconds) evicting MGF page-cache entries before ${stage}"
  python3 "${SCRIPT_DIR}/evict_pxd024364_page_cache.py" mgf
  echo "$(date --iso-8601=seconds) starting ${stage}"
  if [[ "${stage}" == gpu_* ]]; then
    gpu_index="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2,2n | head -n 1 | cut -d, -f1 | tr -d ' ')"
    echo "$(date --iso-8601=seconds) selected physical GPU ${gpu_index} for ${stage}"
    python3 "${SCRIPT_DIR}/run_pxd024364_baseline_stage.py" "${stage}" --gpu-index "${gpu_index}"
  else
    python3 "${SCRIPT_DIR}/run_pxd024364_baseline_stage.py" "${stage}"
  fi
  python3 "${SCRIPT_DIR}/summarize_pxd024364_experiment.py"
done

echo "$(date --iso-8601=seconds) all stages complete"
