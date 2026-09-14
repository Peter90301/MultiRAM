#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${FRAMEWORK_ROOT}/scripts/common.sh"

CONFIG="${1:-}"
[[ -n "${CONFIG}" ]] || die "Usage: genomics_minimap2_pairs.sh CONFIG"
require_config "${CONFIG}"

TOOL="$(resolve_tool "${CONFIG}" minimap2)"
REFERENCE="$(json_get_required "${CONFIG}" workloads.genomics.reference)"
PAIR_MANIFEST="$(json_get_required "${CONFIG}" workloads.genomics.pair_manifest)"
READS_DIR="$(json_get_required "${CONFIG}" workloads.genomics.reads_dir)"
OUTPUT_DIR="$(absolute_or_rooted "$(json_get_required "${CONFIG}" workloads.genomics.output_dir)")"
TIMING_TSV="$(absolute_or_rooted "$(json_get_required "${CONFIG}" workloads.genomics.pair_timing_tsv)")"
THREADS="$(json_get "${CONFIG}" threads)"
PRESET="$(json_get "${CONFIG}" workloads.genomics.minimap2_preset)"
OUTPUT_MODE="$(json_get "${CONFIG}" workloads.genomics.output_mode)"
MAX_PAIRS="$(json_get "${CONFIG}" workloads.genomics.max_pairs)"
EXTRA_ARGS_JSON="$(json_get "${CONFIG}" workloads.genomics.extra_args)"

[[ -n "${THREADS}" ]] || THREADS="1"
[[ -n "${PRESET}" ]] || PRESET="sr"
[[ -n "${OUTPUT_MODE}" ]] || OUTPUT_MODE="discard"
[[ -n "${MAX_PAIRS}" ]] || MAX_PAIRS="0"
if [[ -n "${MULTIRAM_OUTPUT_DIR:-}" ]]; then
  OUTPUT_DIR="${MULTIRAM_OUTPUT_DIR}"
fi
if [[ -n "${MULTIRAM_PAIR_TIMING_TSV:-}" ]]; then
  TIMING_TSV="${MULTIRAM_PAIR_TIMING_TSV}"
fi
if [[ -n "${MULTIRAM_MAX_PAIRS:-}" ]]; then
  MAX_PAIRS="${MULTIRAM_MAX_PAIRS}"
fi

require_file "${REFERENCE}"
require_file "${PAIR_MANIFEST}"
[[ -d "${READS_DIR}" ]] || die "Reads directory not found: ${READS_DIR}"
ensure_dir "${OUTPUT_DIR}"
ensure_dir "$(dirname "${TIMING_TSV}")"

mapfile -t EXTRA_ARGS < <(python3 - "${EXTRA_ARGS_JSON}" <<'PY'
import json
import sys

raw = sys.argv[1]
if not raw:
    sys.exit(0)
value = json.loads(raw)
if not isinstance(value, list):
    raise SystemExit("workloads.genomics.extra_args must be a JSON list")
for item in value:
    print(str(item))
PY
)

printf "run_accession\tr1_fastq\tr2_fastq\tinput_bytes\toutput_path\toutput_bytes\truntime_s\treturn_code\n" > "${TIMING_TSV}"

count=0
failed=0
while IFS=$'\t' read -r run_accession sample_title r1_fastq r2_fastq r1_bytes r2_bytes read_count base_count; do
  [[ -n "${run_accession}" ]] || continue
  count=$((count + 1))
  if [[ "${MAX_PAIRS}" != "0" && "${count}" -gt "${MAX_PAIRS}" ]]; then
    break
  fi

  R1="${READS_DIR}/${r1_fastq}"
  R2="${READS_DIR}/${r2_fastq}"
  require_file "${R1}"
  require_file "${R2}"
  INPUT_BYTES=$(( $(stat -c '%s' "${R1}") + $(stat -c '%s' "${R2}") ))

  case "${OUTPUT_MODE}" in
    discard)
      OUT="/dev/null"
      ;;
    per_pair)
      OUT="${OUTPUT_DIR}/${run_accession}.paf"
      ;;
    *)
      die "Unsupported output_mode '${OUTPUT_MODE}'. Use discard or per_pair."
      ;;
  esac

  info "minimap2 ${run_accession}: ${r1_fastq} ${r2_fastq}"
  START="$(date +%s.%N)"
  set +e
  "${TOOL}" -t "${THREADS}" -x "${PRESET}" "${EXTRA_ARGS[@]}" "${REFERENCE}" "${R1}" "${R2}" > "${OUT}"
  RC=$?
  set -e
  END="$(date +%s.%N)"
  RUNTIME="$(python3 - <<PY
print(max(0.0, float("${END}") - float("${START}")))
PY
)"
  if [[ "${OUT}" == "/dev/null" ]]; then
    OUTPUT_BYTES=0
  else
    OUTPUT_BYTES="$(stat -c '%s' "${OUT}")"
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${run_accession}" "${r1_fastq}" "${r2_fastq}" "${INPUT_BYTES}" "${OUT}" "${OUTPUT_BYTES}" "${RUNTIME}" "${RC}" >> "${TIMING_TSV}"

  if [[ "${RC}" -ne 0 ]]; then
    failed=$((failed + 1))
    warn "minimap2 failed for ${run_accession} with rc=${RC}"
  fi
done < <(tail -n +2 "${PAIR_MANIFEST}")

if [[ "${failed}" -ne 0 ]]; then
  die "${failed} minimap2 pair run(s) failed"
fi
