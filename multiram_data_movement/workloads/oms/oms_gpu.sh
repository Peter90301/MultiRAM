#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${FRAMEWORK_ROOT}/scripts/common.sh"

CONFIG="${1:-}"
[[ -n "${CONFIG}" ]] || die "Usage: oms_gpu.sh CONFIG"
require_config "${CONFIG}"

ANN_SOLO="$(resolve_tool "${CONFIG}" ann_solo)"
LIBRARY="$(json_get_required "${CONFIG}" workloads.oms.library)"
QUERY="$(json_get_required "${CONFIG}" workloads.oms.query)"
OUTPUT="$(absolute_or_rooted "$(json_get_required "${CONFIG}" workloads.oms.output)")"
EXTRA_ARGS="$(json_get "${CONFIG}" commands.oms_gpu_extra_args)"
ensure_dir "$(dirname "${OUTPUT}")"
require_file "${LIBRARY}"
require_file "${QUERY}"

# ANN-SoLo CLI options can vary by version; put version-specific options in config.
exec "${ANN_SOLO}" ${EXTRA_ARGS} --library "${LIBRARY}" --query "${QUERY}" --out "${OUTPUT}"

