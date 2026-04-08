#!/bin/bash
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=48:00:00
#PBS -N grpo_s${SEED}
#PBS -j oe
#PBS -m abe

# ============================================================
# GRPO Baseline Training (Experiment A)
# 用法: qsub -v SEED=42 tools/qsub_grpo.sh
# ============================================================

cd ${PBS_O_WORKDIR}

module load miniforge 2>/dev/null || true
conda activate bd

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="${WANDB_PROJECT:-webshop-branching-dueling}"

SEED="${SEED:-42}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-$(python -c 'from transformers.utils import TRANSFORMERS_CACHE; print(TRANSFORMERS_CACHE)')/../models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/$(ls ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/ | head -1)}"
SAVE_DIR="${REPO_ROOT}/runs/grpo_seed${SEED}"

mkdir -p "${SAVE_DIR}"

echo "============================================================"
echo "GRPO Baseline | Seed=${SEED} | $(date)"
echo "Model: ${MODEL_PATH}"
echo "Save:  ${SAVE_DIR}"
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
