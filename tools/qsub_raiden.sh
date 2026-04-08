#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -jc gtn-container_g1.24h
#$ -ac d=nvcr-pytorch-2310

# ============================================================
# Branching Dueling No-Rescue on RAIDEN A100
#
# 用法（在登录节点）:
#   SEED=44 EXP=grpo    qsub tools/qsub_raiden.sh   → 实验 A
#   SEED=44 EXP=branch  qsub tools/qsub_raiden.sh   → 实验 B
#   SEED=44 EXP=full    qsub tools/qsub_raiden.sh   → 实验 C
#   SEED=44 EXP=rand_ss qsub tools/qsub_raiden.sh   → 实验 E
#   SEED=44 EXP=rand_ap qsub tools/qsub_raiden.sh   → 实验 F
# ============================================================

# --- Container environment ---
. /fefs/opt/dgx/env_set/nvcr-pytorch-2310-py3.sh

# --- Proxy ---
export MY_PROXY_URL="http://10.1.10.1:8080/"
export HTTP_PROXY=$MY_PROXY_URL HTTPS_PROXY=$MY_PROXY_URL
export http_proxy=$MY_PROXY_URL https_proxy=$MY_PROXY_URL ftp_proxy=$MY_PROXY_URL

# --- User Python prefix ---
mkdir -p ~/.raiden/nvcr-pytorch-2310-py3
export PATH="${HOME}/.raiden/nvcr-pytorch-2310-py3/bin:$PATH"
export LD_LIBRARY_PATH="${HOME}/.raiden/nvcr-pytorch-2310-py3/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUSERBASE="${HOME}/.raiden/nvcr-pytorch-2310-py3"
PYTHON_SITE="$(python -c 'import site; print(site.getusersitepackages())')"
export PYTHONPATH="${PYTHON_SITE}:${PYTHONPATH:-}"

# --- JVM for pyserini ---
export JAVA_HOME=/home/zhangzy/.jdk/jdk-21.0.10+7
export JVM_PATH=$JAVA_HOME/lib/server/libjvm.so

# --- Training config ---
export PYTHONUNBUFFERED=1
export WANDB_PROJECT=webshop-branching-dueling
export WANDB_INIT_TIMEOUT=300

SEED="${SEED:-44}"
EXP="${EXP:-grpo}"
REPO_ROOT="${HOME}/yinuo/webshop_no_rescue_raiden"
MODEL="/home/zhangzy/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

# --- Experiment-specific parameters ---
case "${EXP}" in
    grpo)
        W_BR=0.0; W_DPO=0.0; B=0; K=0; SS=top_k; AP=cdb; N=16; Q=8
        TAG="grpo_seed${SEED}"
        ;;
    branch)
        W_BR=1.0; W_DPO=0.0; B=4; K=2; SS=top_k; AP=cdb; N=8; Q=4
        TAG="branch_pg_seed${SEED}"
        ;;
    full)
        W_BR=1.0; W_DPO=1.0; B=4; K=2; SS=top_k; AP=cdb; N=8; Q=4
        TAG="full_seed${SEED}"
        ;;
    rand_ss)
        W_BR=1.0; W_DPO=0.0; B=4; K=2; SS=random; AP=cdb; N=8; Q=4
        TAG="rand_state_seed${SEED}"
        ;;
    rand_ap)
        W_BR=1.0; W_DPO=0.0; B=4; K=2; SS=top_k; AP=random; N=8; Q=4
        TAG="rand_pair_seed${SEED}"
        ;;
    *)
        echo "Unknown EXP=${EXP}. Use: grpo, branch, full, rand_ss, rand_ap"
        exit 1
        ;;
esac

SAVE_DIR="${REPO_ROOT}/runs/${TAG}"
cd "${REPO_ROOT}"
mkdir -p "${SAVE_DIR}"

echo "============================================================"
echo "Experiment: ${EXP} | Seed: ${SEED} | Tag: ${TAG}"
echo "  w_br=${W_BR} w_dpo=${W_DPO} B=${B} K=${K} N=${N} Q=${Q}"
echo "  state_sel=${SS} action_pair=${AP} --no_rescue"
echo "  Start: $(date)"
echo "============================================================"

python -u scripts/train_branching_dueling_webshop.py \
    configs/webshop_config.yaml \
    --model_name "${MODEL}" \
    --save_dir "${SAVE_DIR}" \
    --seed "${SEED}" \
    --iters 150 \
    --eval_every 10 \
    --eval_games 150 \
    --N "${N}" \
    --B "${B}" \
    --K "${K}" \
    --queries_per_step "${Q}" \
    --w_base 1.0 \
    --w_br "${W_BR}" \
    --w_dpo "${W_DPO}" \
    --state_selection_mode "${SS}" \
    --action_pair_mode "${AP}" \
    --beta_kl 0.01 \
    --lr 1e-6 \
    --no_rescue \
    2>&1 | tee "${SAVE_DIR}/console.log"

echo "Done: ${TAG} $(date)"
