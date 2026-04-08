#!/bin/bash
set -euo pipefail

# ============================================================
# Smoke Test — 验证环境和代码能正常工作
# 在交互式作业中运行:
#   qsub -I -q debug-g -l select=1 -l walltime=01:00:00
#   bash tools/run_smoke_test.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

module load miniforge 2>/dev/null || true
conda activate bd

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Auto-detect model path
MODEL_PATH="${MODEL_PATH:-$(python -c "
import os, glob
base = os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots')
snaps = glob.glob(os.path.join(base, '*'))
print(snaps[0]) if snaps else exit(1)
")}"

SAVE_DIR="${REPO_ROOT}/runs/smoke_test"
mkdir -p "${SAVE_DIR}"

echo "============================================================"
echo "Smoke Test | $(date)"
echo "  Model: ${MODEL_PATH}"
echo "  Output: ${SAVE_DIR}"
echo "============================================================"

python -u "${REPO_ROOT}/scripts/train_branching_dueling_webshop.py" \
    "${REPO_ROOT}/configs/webshop_config.yaml" \
    --model_name "${MODEL_PATH}" \
    --save_dir "${SAVE_DIR}" \
    --seed 42 \
    --iters 2 \
    --eval_every 1 \
    --eval_games 5 \
    --N 4 \
    --B 0 \
    --K 0 \
    --queries_per_step 2 \
    --w_base 1.0 \
    --w_br 0.0 \
    --w_dpo 0.0 \
    --lr 1e-6 \
    2>&1 | tee "${SAVE_DIR}/console.log"

echo ""
echo "=== Smoke test 结果 ==="
cat "${SAVE_DIR}/log.jsonl" | python -c "
import json, sys
for line in sys.stdin:
    e = json.loads(line)
    print(f\"iter={e['iter']}  train_score={e.get('train_score',0):.3f}  \"
          f\"eval_succ={e.get('eval_succ','--')}  l_base={e.get('l_base',0):.5f}\")
"
