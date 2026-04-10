#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/webshop_gigpo_aligned.yaml}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-full_train_seed${SEED}}"
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/runs/${RUN_NAME}}"

WANDB_ENABLE="${WANDB_ENABLE:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-webshop_on_miyabi_ai_ready}"
WANDB_NAME="${WANDB_NAME:-${RUN_NAME}}"
WANDB_GROUP="${WANDB_GROUP:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1
export WEBSHOP_DATA_DIR="${WEBSHOP_DATA_DIR:-$HOME/webshop_data}"
export WEBSHOP_SEARCH_DIR="${WEBSHOP_SEARCH_DIR:-$HOME/webshop_data}"
export JAVA_HOME="${JAVA_HOME:-${CONDA_PREFIX:-}}"
export JVM_PATH="${JVM_PATH:-${CONDA_PREFIX:-}/lib/jvm/lib/server/libjvm.so}"

mkdir -p "${SAVE_DIR}"

ARGS=(
  --config "${CONFIG_PATH}"
  --model_name "${MODEL_PATH}"
  --save_dir "${SAVE_DIR}"
  --seed "${SEED}"
)

if [[ "${WANDB_ENABLE}" == "1" ]]; then
  ARGS+=(--wandb --wandb_project "${WANDB_PROJECT}" --wandb_name "${WANDB_NAME}" --wandb_mode "${WANDB_MODE}")
  if [[ -n "${WANDB_GROUP}" ]]; then
    ARGS+=(--wandb_group "${WANDB_GROUP}")
  fi
  if [[ -n "${WANDB_ENTITY}" ]]; then
    ARGS+=(--wandb_entity "${WANDB_ENTITY}")
  fi
fi

echo "============================================================"
echo "webshop_on_miyabi_ai_ready Full Training"
echo "  GPU_ID:        ${GPU_ID}"
echo "  SEED:          ${SEED}"
echo "  PYTHON:        ${PYTHON}"
echo "  CONFIG:        ${CONFIG_PATH}"
echo "  MODEL_PATH:    ${MODEL_PATH}"
echo "  SAVE_DIR:      ${SAVE_DIR}"
echo "  WANDB_ENABLE:  ${WANDB_ENABLE}"
echo "  START:         $(date)"
echo "============================================================"

"${PYTHON}" -u "${REPO_ROOT}/scripts/train_grpo_webshop.py" "${ARGS[@]}" "$@" 2>&1 | tee "${SAVE_DIR}/console.log"
