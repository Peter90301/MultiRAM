#!/usr/bin/env bash
set -euo pipefail

DEST_ROOT="/mnt/hdd/tsunghan/raw-ms-dataset/proteomic_large_dataset"
MANIFEST_DIR="${DEST_ROOT}/manifests"
RAW_DIR="${DEST_ROOT}/raw"
LOG_DIR="${DEST_ROOT}/logs"
SELECTOR="/home/tsl012/multiomic/multiram_data_movement/scripts/prepare_pxd024364_1tb_manifest.py"
REMOTE=":ftp:/v03/MSV000086944/raw/raw/"

mkdir -p "${MANIFEST_DIR}" "${RAW_DIR}" "${LOG_DIR}"

if [[ ! -s "${MANIFEST_DIR}/selected_paths.txt" ]]; then
  python3 "${SELECTOR}" --output-dir "${MANIFEST_DIR}"
fi

FTP_PASS="$(rclone obscure anonymous)"
exec rclone copy "${REMOTE}" "${RAW_DIR}" \
  --files-from-raw "${MANIFEST_DIR}/selected_paths.txt" \
  --ftp-host massive-ftp.ucsd.edu \
  --ftp-user anonymous \
  --ftp-pass "${FTP_PASS}" \
  --ftp-explicit-tls \
  --ftp-no-check-certificate \
  --transfers 4 \
  --checkers 8 \
  --multi-thread-streams 1 \
  --contimeout 30s \
  --timeout 10m \
  --retries 20 \
  --low-level-retries 20 \
  --retries-sleep 30s \
  --partial-suffix .partial \
  --stats 60s \
  --stats-one-line-date \
  --log-level INFO \
  --log-file "${LOG_DIR}/rclone_download.log"
