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

SMOKE_NUM_PRODUCTS="${SMOKE_NUM_PRODUCTS:-1000}"
SMOKE_CONFIG="${SAVE_DIR}/smoke_webshop_config.yaml"

python - <<'PY' "${REPO_ROOT}/configs/webshop_config.yaml" "${SMOKE_CONFIG}" "${SMOKE_NUM_PRODUCTS}"
import sys
import yaml

src, dst, num_products = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg.setdefault("env", {})["num_products"] = num_products
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

echo "============================================================"
echo "Smoke Test | $(date)"
echo "  Model: ${MODEL_PATH}"
echo "  Output: ${SAVE_DIR}"
echo "  Config: ${SMOKE_CONFIG} (num_products=${SMOKE_NUM_PRODUCTS})"
echo "============================================================"

python -u "${REPO_ROOT}/scripts/train_branching_dueling_webshop.py" \
    "${SMOKE_CONFIG}" \
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
