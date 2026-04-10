#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/webshop_smoke.yaml}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
TASK_ID="${TASK_ID:-0}"

export PYTHONUNBUFFERED=1
export WEBSHOP_DATA_DIR="${WEBSHOP_DATA_DIR:-$HOME/webshop_data}"
export WEBSHOP_SEARCH_DIR="${WEBSHOP_SEARCH_DIR:-$HOME/webshop_data}"
export JAVA_HOME="${JAVA_HOME:-${CONDA_PREFIX:-}}"
export JVM_PATH="${JVM_PATH:-${CONDA_PREFIX:-}/lib/jvm/lib/server/libjvm.so}"

echo "============================================================"
echo "Miyabi Smoke Test"
echo "  REPO_ROOT:      ${REPO_ROOT}"
echo "  PYTHON:         ${PYTHON}"
echo "  CONFIG:         ${CONFIG_PATH}"
echo "  MODEL_PATH:     ${MODEL_PATH}"
echo "  TASK_ID:        ${TASK_ID}"
echo "  WEBSHOP_DATA:   ${WEBSHOP_DATA_DIR}"
echo "  WEBSHOP_SEARCH: ${WEBSHOP_SEARCH_DIR}"
echo "============================================================"

"${PYTHON}" "${REPO_ROOT}/scripts/smoke_test_webshop.py" \
  --config "${CONFIG_PATH}" \
  --task_id "${TASK_ID}" \
  "$@"
