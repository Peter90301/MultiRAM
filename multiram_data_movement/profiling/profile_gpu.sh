#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${FRAMEWORK_ROOT}/scripts/common.sh"

usage() {
  cat <<'EOF'
Usage:
  profile_gpu.sh --config CONFIG --workload NAME --stage STAGE --platform PLATFORM \
    --dataset DATASET --regime storage_bound|dram_bound --run-id ID --run-mode cold|warm [--gpu-index N] [--with-ncu] -- COMMAND...

Metrics:
  - wall-clock runtime
  - Nsight Systems CUDA memcpy timeline when nsys is available
  - optional Nsight Compute memory metrics when ncu is available
  - nvidia-smi power samples when available
EOF
}

CONFIG=""
WORKLOAD=""
STAGE=""
PLATFORM="gpu"
DATASET=""
REGIME=""
RUN_ID="0"
RUN_MODE="warm"
GPU_INDEX="0"
WITH_NCU="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --workload) WORKLOAD="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --platform) PLATFORM="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --regime) REGIME="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --run-mode) RUN_MODE="$2"; shift 2 ;;
    --gpu-index) GPU_INDEX="$2"; shift 2 ;;
    --with-ncu) WITH_NCU="1"; shift ;;
    --help) usage; exit 0 ;;
    --) shift; break ;;
    *) die "Unknown argument: $1" ;;
  esac
done

[[ $# -gt 0 ]] || die "Missing workload command after --"
require_config "${CONFIG}"

RESULTS_DIR="$(absolute_or_rooted "$(json_get "${CONFIG}" results_dir)")"
RAW_DIR="${RESULTS_DIR}/raw_logs/${STAGE}/${PLATFORM}/${REGIME}/${RUN_MODE}/run_${RUN_ID}"
ensure_dir "${RAW_DIR}"

NSYS_TOOL="$(json_get "${CONFIG}" tools.nsys)"
NCU_TOOL="$(json_get "${CONFIG}" tools.ncu)"
NVIDIA_SMI="$(json_get "${CONFIG}" tools.nvidia_smi)"
[[ -n "${NSYS_TOOL}" ]] || NSYS_TOOL="nsys"
[[ -n "${NCU_TOOL}" ]] || NCU_TOOL="ncu"
[[ -n "${NVIDIA_SMI}" ]] || NVIDIA_SMI="nvidia-smi"

STDOUT_LOG="${RAW_DIR}/stdout.log"
STDERR_LOG="${RAW_DIR}/stderr.log"
POWER_CSV="${RAW_DIR}/power_samples.csv"
META_JSON="${RAW_DIR}/metadata.json"
STATUS_FILE="${RAW_DIR}/status.txt"
NSYS_BASE="${RAW_DIR}/nsys_profile"
NSYS_STATS="${RAW_DIR}/nsys_stats.txt"
NCU_CSV="${RAW_DIR}/ncu_metrics.csv"

CACHE_DROP_STATUS="not_requested"
if [[ "${RUN_MODE}" == "cold" ]]; then
  sync || true
  if [[ "${EUID}" -eq 0 && -w /proc/sys/vm/drop_caches ]]; then
    echo 3 > /proc/sys/vm/drop_caches
    CACHE_DROP_STATUS="dropped"
  else
    warn "Cold-cache requested, but dropping Linux page cache requires root. Continuing without dropping cache."
    CACHE_DROP_STATUS="not_permitted"
  fi
fi

COMMAND=("$@")
printf '%q ' "${COMMAND[@]}" > "${RAW_DIR}/command.sh"
echo >> "${RAW_DIR}/command.sh"

POWER_INTERVAL="$(json_get "${CONFIG}" profiling.power_sample_interval_s)"
[[ -n "${POWER_INTERVAL}" ]] || POWER_INTERVAL="1"
POWER_PID=""
if command -v "${NVIDIA_SMI}" >/dev/null 2>&1; then
  (
    echo "timestamp,power_w"
    while true; do
      value="$("${NVIDIA_SMI}" --query-gpu=timestamp,power.draw --format=csv,noheader,nounits -i "${GPU_INDEX}" 2>/dev/null | head -n 1 || true)"
      if [[ -n "${value}" ]]; then
        echo "${value}"
      fi
      sleep "${POWER_INTERVAL}"
    done
  ) > "${POWER_CSV}" &
  POWER_PID="$!"
else
  warn "nvidia-smi not found; GPU power logging disabled"
  : > "${POWER_CSV}"
fi

cleanup() {
  if [[ -n "${POWER_PID}" ]]; then
    kill "${POWER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

HOSTNAME_VALUE="$(hostname)"
KERNEL_VALUE="$(uname -a)"
GPU_NAME="$("${NVIDIA_SMI}" --query-gpu=name --format=csv,noheader -i "${GPU_INDEX}" 2>/dev/null | head -n 1 || true)"
GIT_COMMIT="$(git -C "${FRAMEWORK_ROOT}" rev-parse --short HEAD 2>/dev/null || true)"
START_EPOCH="$(date +%s.%N)"

set +e
if command -v "${NSYS_TOOL}" >/dev/null 2>&1; then
  "${NSYS_TOOL}" profile --force-overwrite=true --trace=cuda,nvtx,osrt \
    --output="${NSYS_BASE}" --stats=true -- \
    "${COMMAND[@]}" >"${STDOUT_LOG}" 2>"${STDERR_LOG}"
  RC=$?
  "${NSYS_TOOL}" stats "${NSYS_BASE}.nsys-rep" > "${NSYS_STATS}" 2>>"${STDERR_LOG}" || true
else
  warn "nsys not found; running command without Nsight Systems"
  "${COMMAND[@]}" >"${STDOUT_LOG}" 2>"${STDERR_LOG}"
  RC=$?
  : > "${NSYS_STATS}"
fi

if [[ "${WITH_NCU}" == "1" ]]; then
  if command -v "${NCU_TOOL}" >/dev/null 2>&1; then
    "${NCU_TOOL}" --csv --metrics dram__bytes_read.sum,dram__bytes_write.sum \
      --target-processes all "${COMMAND[@]}" > "${NCU_CSV}" 2>>"${STDERR_LOG}" || true
  else
    warn "ncu requested but not found"
    : > "${NCU_CSV}"
  fi
else
  : > "${NCU_CSV}"
fi
set -e

END_EPOCH="$(date +%s.%N)"
TOTAL_RUNTIME_S="$(python3 - <<PY
print(max(0.0, float("${END_EPOCH}") - float("${START_EPOCH}")))
PY
)"
echo "${RC}" > "${STATUS_FILE}"

python3 - "${META_JSON}" <<PY
import json
meta = {
  "workload": "${WORKLOAD}",
  "stage": "${STAGE}",
  "platform": "${PLATFORM}",
  "dataset": "${DATASET}",
  "regime": "${REGIME}",
  "run_id": "${RUN_ID}",
  "run_mode": "${RUN_MODE}",
  "total_runtime_s": float("${TOTAL_RUNTIME_S}"),
  "cache_drop_status": "${CACHE_DROP_STATUS}",
  "raw_log_dir": "${RAW_DIR}",
  "command": open("${RAW_DIR}/command.sh", "r", encoding="utf-8").read().strip(),
  "return_code": int("${RC}"),
  "hostname": "${HOSTNAME_VALUE}",
  "kernel": "${KERNEL_VALUE}",
  "gpu_index": "${GPU_INDEX}",
  "gpu_name": "${GPU_NAME}",
  "git_commit": "${GIT_COMMIT}",
  "config": "${CONFIG}"
}
with open("${META_JSON}", "w", encoding="utf-8") as fh:
    json.dump(meta, fh, indent=2)
PY

info "GPU profiling completed with rc=${RC}; logs: ${RAW_DIR}"
exit "${RC}"

