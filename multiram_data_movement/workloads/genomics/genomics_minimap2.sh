#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${FRAMEWORK_ROOT}/scripts/common.sh"

CONFIG="${1:-}"
[[ -n "${CONFIG}" ]] || die "Usage: genomics_minimap2.sh CONFIG"
require_config "${CONFIG}"

TOOL_NAME="$(json_get "${CONFIG}" workloads.genomics.tool)"
[[ -n "${TOOL_NAME}" ]] || TOOL_NAME="minimap2"
REFERENCE="$(json_get_required "${CONFIG}" workloads.genomics.reference)"
READS="$(json_get_required "${CONFIG}" workloads.genomics.reads)"
OUTPUT="$(absolute_or_rooted "$(json_get_required "${CONFIG}" workloads.genomics.output)")"
THREADS="$(json_get "${CONFIG}" threads)"
[[ -n "${THREADS}" ]] || THREADS="1"

ensure_dir "$(dirname "${OUTPUT}")"
require_file "${REFERENCE}"
require_file "${READS}"

case "${TOOL_NAME}" in
  minimap2)
    TOOL="$(resolve_tool "${CONFIG}" minimap2)"
    PRESET="$(json_get "${CONFIG}" workloads.genomics.minimap2_preset)"
    [[ -n "${PRESET}" ]] || PRESET="map-ont"
    exec "${TOOL}" -t "${THREADS}" -x "${PRESET}" "${REFERENCE}" "${READS}" > "${OUTPUT}"
    ;;
  minigraph)
    TOOL="$(resolve_tool "${CONFIG}" minigraph)"
    exec "${TOOL}" -t "${THREADS}" "${REFERENCE}" "${READS}" > "${OUTPUT}"
    ;;
  *)
    die "Unsupported genomics tool '${TOOL_NAME}'. Use minimap2 or minigraph."
    ;;
esac

