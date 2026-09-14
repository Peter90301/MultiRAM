#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/home/tsl012/multiomic/multiram_data_movement/scripts"
ROOT="/mnt/hdd/tsunghan/raw-ms-dataset/proteomic_large_dataset"
LOG_DIR="${ROOT}/results/orchestrator"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/gpu_oms.log") 2>&1

if [[ -z "${GPU_INDEX:-}" ]]; then
  GPU_INDEX="$(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | sort -t, -k2,2n \
      | head -n 1 \
      | cut -d, -f1 \
      | tr -d ' '
  )"
fi

echo "$(date --iso-8601=seconds) evicting MGF page-cache entries before gpu_oms"
python3 "${SCRIPT_DIR}/evict_pxd024364_page_cache.py" mgf
echo "$(date --iso-8601=seconds) starting gpu_oms on physical GPU ${GPU_INDEX}"
python3 "${SCRIPT_DIR}/run_pxd024364_baseline_stage.py" \
  gpu_oms --gpu-index "${GPU_INDEX}"
python3 "${SCRIPT_DIR}/summarize_pxd024364_experiment.py"
echo "$(date --iso-8601=seconds) gpu_oms complete"
