#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

warn() {
  echo "[WARN] $*" >&2
}

info() {
  echo "[INFO] $*" >&2
}

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || die "Required file not found: ${path}"
}

require_config() {
  local config="$1"
  require_file "${config}"
}

json_get() {
  local config="$1"
  local key="$2"
  python3 - "$config" "$key" <<'PY'
import json
import sys

path, dotted = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as fh:
    data = json.load(fh)
cur = data
for part in dotted.split("."):
    if isinstance(cur, dict) and part in cur:
        cur = cur[part]
    else:
        print("")
        sys.exit(0)
if isinstance(cur, (dict, list)):
    print(json.dumps(cur))
elif cur is None:
    print("")
else:
    print(cur)
PY
}

json_get_required() {
  local config="$1"
  local key="$2"
  local value
  value="$(json_get "${config}" "${key}")"
  [[ -n "${value}" ]] || die "Missing required config key: ${key}"
  echo "${value}"
}

resolve_tool() {
  local config="$1"
  local key="$2"
  local tool
  tool="$(json_get "${config}" "tools.${key}")"
  [[ -n "${tool}" ]] || die "Missing tools.${key} in ${config}"
  if [[ "${tool}" == */* ]]; then
    [[ -x "${tool}" ]] || die "Configured tool is not executable: ${tool}"
    echo "${tool}"
    return
  fi
  command -v "${tool}" >/dev/null 2>&1 || die "Tool '${tool}' not found. Configure tools.${key} in ${config}."
  echo "${tool}"
}

ensure_dir() {
  mkdir -p "$1"
}

absolute_or_rooted() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    echo "${path}"
  else
    echo "${FRAMEWORK_ROOT}/${path}"
  fi
}

