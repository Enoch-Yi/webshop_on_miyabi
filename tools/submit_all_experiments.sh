#!/bin/bash
set -euo pipefail

# ============================================================
# 一键提交全部实验
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

echo "=== 提交 GRPO Baseline (Experiment A) ==="
qsub -v SEED=42 tools/qsub_grpo.sh
qsub -v SEED=43 tools/qsub_grpo.sh
qsub -v SEED=44 tools/qsub_grpo.sh

echo "=== 提交 Full Method (Experiment D) ==="
qsub -v SEED=42 tools/qsub_full.sh
qsub -v SEED=43 tools/qsub_full.sh
qsub -v SEED=44 tools/qsub_full.sh

echo "=== 提交消融: + Branch PG only (Experiment B) ==="
qsub -v SEED=42,W_DPO=0.0 tools/qsub_full.sh
qsub -v SEED=43,W_DPO=0.0 tools/qsub_full.sh
qsub -v SEED=44,W_DPO=0.0 tools/qsub_full.sh

echo "=== 提交消融: Random State Selection (Experiment E) ==="
qsub -v SEED=42,STATE_SEL=random tools/qsub_full.sh

echo "=== 提交消融: Random Action Pair (Experiment F) ==="
qsub -v SEED=42,ACTION_PAIR=random tools/qsub_full.sh

echo "=== 提交消融: All Random (Experiment G) ==="
qsub -v SEED=42,STATE_SEL=random,ACTION_PAIR=random tools/qsub_full.sh

echo ""
echo "=== 所有实验已提交。用 qstat 查看状态 ==="
qstat
