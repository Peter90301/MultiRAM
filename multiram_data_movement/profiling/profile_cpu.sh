#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${FRAMEWORK_ROOT}/scripts/common.sh"

usage() {
  cat <<'EOF'
Usage:
  profile_cpu.sh --config CONFIG --workload NAME --stage STAGE --platform PLATFORM \
    --dataset DATASET --regime storage_bound|dram_bound --run-id ID --run-mode cold|warm -- COMMAND...

Metrics:
  - wall-clock runtime
  - /usr/bin/time -v metrics
  - perf stat events
  - cache drop status for cold-cache runs
  - metadata JSON for analysis
EOF
}

CONFIG=""
WORKLOAD=""
STAGE=""
PLATFORM="cpu"
DATASET=""
REGIME=""
RUN_ID="0"
RUN_MODE="warm"

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
    --help) usage; exit 0 ;;
    --) shift; break ;;
    *) die "Unknown argument: $1" ;;
  esac
done

[[ $# -gt 0 ]] || die "Missing workload command after --"
require_config "${CONFIG}"
[[ -n "${WORKLOAD}" && -n "${STAGE}" && -n "${DATASET}" && -n "${REGIME}" ]] || die "Missing required metadata"

RESULTS_DIR="$(absolute_or_rooted "$(json_get "${CONFIG}" results_dir)")"
RAW_DIR="${RESULTS_DIR}/raw_logs/${STAGE}/${PLATFORM}/${REGIME}/${RUN_MODE}/run_${RUN_ID}"
ensure_dir "${RAW_DIR}"

TIME_LOG="${RAW_DIR}/time_v.log"
PERF_LOG="${RAW_DIR}/perf_stat.log"
STDOUT_LOG="${RAW_DIR}/stdout.log"
STDERR_LOG="${RAW_DIR}/stderr.log"
META_JSON="${RAW_DIR}/metadata.json"
STATUS_FILE="${RAW_DIR}/status.txt"

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

PERF_TOOL="$(json_get "${CONFIG}" tools.perf)"
if [[ -z "${PERF_TOOL}" ]]; then
  PERF_TOOL="perf"
fi
PERF_EVENTS="$(python3 - "${CONFIG}" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as fh:
    cfg = json.load(fh)
print(",".join(cfg.get("profiling", {}).get("perf_events", [])))
PY
)"
[[ -n "${PERF_EVENTS}" ]] || PERF_EVENTS="cycles,instructions,cache-references,cache-misses,LLC-loads,LLC-load-misses,LLC-stores,major-faults,minor-faults"

COMMAND=("$@")
printf '%q ' "${COMMAND[@]}" > "${RAW_DIR}/command.sh"
echo >> "${RAW_DIR}/command.sh"

HOSTNAME_VALUE="$(hostname)"
KERNEL_VALUE="$(uname -a)"
CPU_MODEL="$(lscpu 2>/dev/null | awk -F: '/Model name/ {gsub(/^[ \t]+/, "", $2); print $2; exit}')"
GIT_COMMIT="$(git -C "${FRAMEWORK_ROOT}" rev-parse --short HEAD 2>/dev/null || true)"
START_EPOCH="$(date +%s.%N)"

set +e
if command -v "${PERF_TOOL}" >/dev/null 2>&1; then
  /usr/bin/time -v -o "${TIME_LOG}" \
    "${PERF_TOOL}" stat -x, -o "${PERF_LOG}" -e "${PERF_EVENTS}" -- \
    "${COMMAND[@]}" >"${STDOUT_LOG}" 2>"${STDERR_LOG}"
  RC=$?
else
  warn "perf not found; running only with /usr/bin/time -v"
  /usr/bin/time -v -o "${TIME_LOG}" \
    "${COMMAND[@]}" >"${STDOUT_LOG}" 2>"${STDERR_LOG}"
  RC=$?
  : > "${PERF_LOG}"
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
  "cpu_model": "${CPU_MODEL}",
  "git_commit": "${GIT_COMMIT}",
  "config": "${CONFIG}"
}
with open("${META_JSON}", "w", encoding="utf-8") as fh:
    json.dump(meta, fh, indent=2)
PY

info "CPU profiling completed with rc=${RC}; logs: ${RAW_DIR}"
exit "${RC}"

