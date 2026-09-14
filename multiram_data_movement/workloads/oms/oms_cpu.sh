#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${FRAMEWORK_ROOT}/scripts/common.sh"

CONFIG="${1:-}"
[[ -n "${CONFIG}" ]] || die "Usage: oms_cpu.sh CONFIG"
require_config "${CONFIG}"

SPECTRAST="$(resolve_tool "${CONFIG}" spectrast)"
LIBRARY="$(json_get_required "${CONFIG}" workloads.oms.library)"
QUERY="$(json_get_required "${CONFIG}" workloads.oms.query)"
OUTPUT="$(absolute_or_rooted "$(json_get_required "${CONFIG}" workloads.oms.output)")"
EXTRA_ARGS="$(json_get "${CONFIG}" commands.oms_cpu_extra_args)"
ensure_dir "$(dirname "${OUTPUT}")"
require_file "${LIBRARY}"
require_file "${QUERY}"

# SpectraST installations differ. The default command is intentionally simple;
# pass installation-specific flags through commands.oms_cpu_extra_args.
exec "${SPECTRAST}" ${EXTRA_ARGS} -sL"${LIBRARY}" -sQ"${QUERY}" -sO"${OUTPUT}"

