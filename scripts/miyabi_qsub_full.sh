#!/bin/bash
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=48:00:00
#PBS -N bd_gigpo_full
#PBS -j oe
#PBS -W group_list=gq50

set -euo pipefail

cd "${PBS_O_WORKDIR}"

source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/Modules/init/bash 2>/dev/null || true
module load miniforge3/24.11.0-0 2>/dev/null || module load miniforge3 2>/dev/null || module load miniforge 2>/dev/null || true

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
        source "${CONDA_BASE}/etc/profile.d/conda.sh"
    fi
fi

if [[ -z "${CONDA_EXE:-}" && -f "/work/opt/local/aarch64/cores/miniforge3/24.11.0-0/etc/profile.d/conda.sh" ]]; then
    source "/work/opt/local/aarch64/cores/miniforge3/24.11.0-0/etc/profile.d/conda.sh"
fi

conda activate bd

REPO_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
SEED="${SEED:-42}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
SAVE_DIR="${REPO_ROOT}/runs/miyabi_full_seed${SEED}"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export WEBSHOP_DATA_DIR="${WEBSHOP_DATA_DIR:-$HOME/webshop_data}"
export WEBSHOP_SEARCH_DIR="${WEBSHOP_SEARCH_DIR:-$HOME/webshop_data}"
export WANDB_PROJECT="${WANDB_PROJECT:-webshop_on_miyabi_ai_ready}"
export JAVA_HOME="${JAVA_HOME:-${CONDA_PREFIX}}"
export JVM_PATH="${JVM_PATH:-${CONDA_PREFIX}/lib/jvm/lib/server/libjvm.so}"

mkdir -p "${SAVE_DIR}"

"${REPO_ROOT}/run_full_train.sh" \
  --seed "${SEED}" \
  --model_name "${MODEL_PATH}" \
  --save_dir "${SAVE_DIR}" \
  --B 4 \
  --K 2 \
  --w_br 1.0 \
  --w_dpo 1.0
