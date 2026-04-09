#!/bin/bash
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=48:00:00
#PBS -N grpo_job
#PBS -j oe
#PBS -m abe
#PBS -W group_list=gq50

# ============================================================
# GRPO Baseline Training (Experiment A)
# 用法: qsub -v SEED=42 tools/qsub_grpo.sh
# ============================================================

cd ${PBS_O_WORKDIR}

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

SEED="${SEED:-42}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-$(python -c 'from transformers.utils import TRANSFORMERS_CACHE; print(TRANSFORMERS_CACHE)')/../models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/$(ls ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/ | head -1)}"
SAVE_DIR="${REPO_ROOT}/runs/grpo_seed${SEED}"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="${WANDB_PROJECT:-webshop-branching-dueling}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-grpo-baseline}"
export WANDB_NAME="${WANDB_NAME:-grpo_seed${SEED}}"
export WANDB_DIR="${WANDB_DIR:-${REPO_ROOT}/wandb}"

mkdir -p "${SAVE_DIR}"
mkdir -p "${WANDB_DIR}"

echo "============================================================"
echo "GRPO Baseline | Seed=${SEED} | $(date)"
echo "Model: ${MODEL_PATH}"
echo "Save:  ${SAVE_DIR}"
echo "W&B:   ${WANDB_PROJECT} / ${WANDB_RUN_GROUP} / ${WANDB_NAME}"
echo "============================================================"

python -u "${REPO_ROOT}/scripts/train_branching_dueling_webshop.py" \
    "${REPO_ROOT}/configs/webshop_config.yaml" \
    --model_name "${MODEL_PATH}" \
    --save_dir "${SAVE_DIR}" \
    --seed "${SEED}" \
    --iters 150 \
    --eval_every 10 \
    --eval_games 150 \
    --N 16 \
    --B 0 \
    --K 0 \
    --queries_per_step 4 \
    --w_base 1.0 \
    --w_br 0.0 \
    --w_dpo 0.0 \
    --beta_kl 0.0 \
    --lr 1e-6 \
    2>&1 | tee "${SAVE_DIR}/console.log"
