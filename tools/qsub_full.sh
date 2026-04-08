#!/bin/bash
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=48:00:00
#PBS -N bd_s${SEED}
#PBS -j oe
#PBS -m abe

# ============================================================
# Full Branching Dueling Training (Experiment D)
# 用法: qsub -v SEED=42 tools/qsub_full.sh
#
# 消融实验可通过环境变量控制:
#   qsub -v SEED=42,W_BR=0.0,W_DPO=0.0       → Experiment A (GRPO only)
#   qsub -v SEED=42,W_DPO=0.0                  → Experiment B (+ Branch PG)
#   qsub -v SEED=42                             → Experiment C/D (Full)
#   qsub -v SEED=42,STATE_SEL=random            → Experiment E
#   qsub -v SEED=42,ACTION_PAIR=random           → Experiment F
#   qsub -v SEED=42,STATE_SEL=random,ACTION_PAIR=random → Experiment G
# ============================================================

cd ${PBS_O_WORKDIR}

module load miniforge 2>/dev/null || true
conda activate bd

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="${WANDB_PROJECT:-webshop-branching-dueling}"

SEED="${SEED:-42}"
W_BASE="${W_BASE:-1.0}"
W_BR="${W_BR:-1.0}"
W_DPO="${W_DPO:-1.0}"
N_ROLLOUTS="${N_ROLLOUTS:-8}"
B_STATES="${B_STATES:-4}"
K_DUELS="${K_DUELS:-2}"
Q_PER_STEP="${Q_PER_STEP:-4}"
STATE_SEL="${STATE_SEL:-top_k}"
ACTION_PAIR="${ACTION_PAIR:-cdb}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-$(python -c 'import os; print(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/"))')$(ls ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/ | head -1)}"

TAG="bd_w${W_BASE}_${W_BR}_${W_DPO}_ss${STATE_SEL}_ap${ACTION_PAIR}_seed${SEED}"
SAVE_DIR="${REPO_ROOT}/runs/${TAG}"

mkdir -p "${SAVE_DIR}"

echo "============================================================"
echo "Branching Dueling | ${TAG} | $(date)"
echo "  w_base=${W_BASE} w_br=${W_BR} w_dpo=${W_DPO}"
echo "  N=${N_ROLLOUTS} B=${B_STATES} K=${K_DUELS} Q=${Q_PER_STEP}"
echo "  state_sel=${STATE_SEL} action_pair=${ACTION_PAIR}"
echo "============================================================"

python -u "${REPO_ROOT}/scripts/train_branching_dueling_webshop.py" \
    "${REPO_ROOT}/configs/webshop_config.yaml" \
    --model_name "${MODEL_PATH}" \
    --save_dir "${SAVE_DIR}" \
    --seed "${SEED}" \
    --iters 150 \
    --eval_every 10 \
    --eval_games 150 \
    --N "${N_ROLLOUTS}" \
    --B "${B_STATES}" \
    --K "${K_DUELS}" \
    --queries_per_step "${Q_PER_STEP}" \
    --w_base "${W_BASE}" \
    --w_br "${W_BR}" \
    --w_dpo "${W_DPO}" \
    --state_selection_mode "${STATE_SEL}" \
    --action_pair_mode "${ACTION_PAIR}" \
    --beta_kl 0.0 \
    --lr 1e-6 \
    2>&1 | tee "${SAVE_DIR}/console.log"
