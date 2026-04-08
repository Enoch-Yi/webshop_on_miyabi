#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -jc gtn-container_g1.168h
#$ -ac d=nvcr-pytorch-2310
#$ -N grpo_s${SEED:-44}

# ============================================================
# GRPO Full Training on RAIDEN A100
# 用法（在登录节点）:
#   SEED=44 qsub tools/qsub_grpo_raiden.sh
# ============================================================

# --- 容器环境 ---
. /fefs/opt/dgx/env_set/nvcr-pytorch-2310-py3.sh

# --- 代理 ---
export MY_PROXY_URL="http://10.1.10.1:8080/"
export HTTP_PROXY=$MY_PROXY_URL HTTPS_PROXY=$MY_PROXY_URL
export http_proxy=$MY_PROXY_URL https_proxy=$MY_PROXY_URL ftp_proxy=$MY_PROXY_URL

# --- 用户 Python 前缀 ---
mkdir -p ~/.raiden/nvcr-pytorch-2310-py3
export PATH="${HOME}/.raiden/nvcr-pytorch-2310-py3/bin:$PATH"
export LD_LIBRARY_PATH="${HOME}/.raiden/nvcr-pytorch-2310-py3/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUSERBASE="${HOME}/.raiden/nvcr-pytorch-2310-py3"
PYTHON_SITE="$(python -c 'import site; print(site.getusersitepackages())')"
export PYTHONPATH="${PYTHON_SITE}:${PYTHONPATH:-}"

# --- JVM ---
export JAVA_HOME=/home/zhangzy/.jdk/jdk-21.0.10+7
export JVM_PATH=$JAVA_HOME/lib/server/libjvm.so

# --- 训练参数 ---
export PYTHONUNBUFFERED=1
SEED="${SEED:-44}"
REPO_ROOT="${HOME}/yinuo/webshop_first_on_miyabi/webshop_on_raiden"
MODEL_PATH="/home/zhangzy/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
SAVE_DIR="${REPO_ROOT}/runs/grpo_seed${SEED}"

cd "${REPO_ROOT}"
mkdir -p "${SAVE_DIR}"

echo "============================================================"
echo "GRPO Full Training on RAIDEN"
echo "  Seed:  ${SEED}"
echo "  Model: ${MODEL_PATH}"
echo "  Save:  ${SAVE_DIR}"
echo "  Start: $(date)"
echo "============================================================"

python -u scripts/train_branching_dueling_webshop.py \
    configs/webshop_config.yaml \
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

echo "============================================================"
echo "Training complete. $(date)"
echo "============================================================"
