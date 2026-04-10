#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_DIR="${DATA_DIR:-$HOME/webshop_data}"
SEARCH_DIR="${SEARCH_DIR:-$HOME/webshop_data}"

echo "============================================================"
echo "Miyabi Readiness Check"
echo "  REPO_ROOT=${REPO_ROOT}"
echo "  DATA_DIR=${DATA_DIR}"
echo "  SEARCH_DIR=${SEARCH_DIR}"
echo "============================================================"

missing=0

check_path() {
  local path="$1"
  local label="$2"
  if [[ -e "${path}" ]]; then
    echo "[OK]    ${label}: ${path}"
  else
    echo "[MISS]  ${label}: ${path}"
    missing=1
  fi
}

echo
echo "== Repo files =="
check_path "${REPO_ROOT}/configs/webshop_gigpo_aligned.yaml" "formal config"
check_path "${REPO_ROOT}/configs/webshop_smoke.yaml" "smoke config"
check_path "${REPO_ROOT}/run_full_train.sh" "train launcher"
check_path "${REPO_ROOT}/scripts/train_grpo_webshop.py" "train entry"
check_path "${REPO_ROOT}/scripts/smoke_test_webshop.py" "smoke entry"
check_path "${REPO_ROOT}/scripts/miyabi_smoke.sh" "smoke shell"
check_path "${REPO_ROOT}/scripts/miyabi_qsub_grpo.sh" "grpo qsub"
check_path "${REPO_ROOT}/scripts/miyabi_qsub_full.sh" "full qsub"
check_path "${REPO_ROOT}/vendor/webshop_env" "vendored webshop env"

echo
echo "== Smoke data =="
check_path "${DATA_DIR}/items_shuffle_1000.json" "items_shuffle_1000.json"
check_path "${DATA_DIR}/items_ins_v2_1000.json" "items_ins_v2_1000.json"
check_path "${DATA_DIR}/items_human_ins.json" "items_human_ins.json"
check_path "${SEARCH_DIR}/indexes_1k" "indexes_1k"

echo
echo "== Full training data =="
check_path "${DATA_DIR}/items_shuffle.json" "items_shuffle.json"
check_path "${DATA_DIR}/items_ins_v2.json" "items_ins_v2.json"
check_path "${SEARCH_DIR}/indexes" "indexes"

echo
if [[ -d "${SEARCH_DIR}/indexes_1k" && ! -d "${SEARCH_DIR}/indexes" ]]; then
  echo "[STATE] smoke-ready only"
elif [[ -d "${SEARCH_DIR}/indexes_1k" && -d "${SEARCH_DIR}/indexes" ]]; then
  echo "[STATE] full-train-ready"
else
  echo "[STATE] not smoke-ready"
fi

echo
if command -v wandb >/dev/null 2>&1; then
  echo "[OK]    wandb command found"
else
  echo "[WARN]  wandb command not found"
fi

if [[ -n "${JAVA_HOME:-}" ]]; then
  echo "[INFO]  JAVA_HOME=${JAVA_HOME}"
else
  echo "[WARN]  JAVA_HOME is not set"
fi

if [[ -n "${JVM_PATH:-}" ]]; then
  echo "[INFO]  JVM_PATH=${JVM_PATH}"
else
  echo "[WARN]  JVM_PATH is not set"
fi

echo
if [[ "${missing}" -ne 0 ]]; then
  echo "Readiness check finished with missing items."
  exit 1
fi

echo "Readiness check finished."
