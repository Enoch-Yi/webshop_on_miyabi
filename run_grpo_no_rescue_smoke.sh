#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/u-yinuo/miniconda3/envs/branching_dueling/bin/python"
MODEL_PATH="/home/u-yinuo/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
GPU_ID="${GPU_ID:-2}"
SAVE_DIR="${REPO_ROOT}/runs/grpo_no_rescue_smoke"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1

mkdir -p "${SAVE_DIR}"

echo "============================================================"
echo "GRPO No-Rescue Smoke Test"
echo "  GPU: ${GPU_ID}"
echo "  --no_rescue enabled"
echo "  Iters: 2"
echo "============================================================"

${PYTHON} -u "${REPO_ROOT}/scripts/train_branching_dueling_webshop.py" \
    "${REPO_ROOT}/configs/webshop_config.yaml" \
    --model_name "${MODEL_PATH}" \
    --save_dir "${SAVE_DIR}" \
    --seed 42 \
    --iters 2 \
    --eval_every 1 \
    --eval_games 20 \
    --N 8 \
    --B 0 \
    --K 0 \
    --queries_per_step 4 \
    --w_base 1.0 \
    --w_br 0.0 \
    --w_dpo 0.0 \
    --beta_kl 0.0 \
    --lr 1e-6 \
    --no_rescue \
    2>&1 | tee "${SAVE_DIR}/console.log"

echo "Done."
