#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${FRAMEWORK_ROOT}/scripts/common.sh"

CONFIG="${1:-}"
[[ -n "${CONFIG}" ]] || die "Usage: proteomics_clustering_gpu.sh CONFIG"
require_config "${CONFIG}"

QUERY="$(json_get_required "${CONFIG}" workloads.proteomics_clustering.query)"
OUTPUT="$(absolute_or_rooted "$(json_get_required "${CONFIG}" workloads.proteomics_clustering.output)")"
COMMAND_TEMPLATE="$(json_get "${CONFIG}" commands.proteomics_clustering_gpu)"
ensure_dir "$(dirname "${OUTPUT}")"
require_file "${QUERY}"

[[ -n "${COMMAND_TEMPLATE}" ]] || die "Configure commands.proteomics_clustering_gpu with a RAPIDS/Python clustering entry point."

export QUERY OUTPUT
exec bash -lc "${COMMAND_TEMPLATE}"

