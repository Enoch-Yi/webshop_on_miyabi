#!/bin/bash
# 一键提交全部实验到 RAIDEN
# 在 RAIDEN 登录节点执行

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=== Submitting experiments to RAIDEN ==="

SEED=44 EXP=grpo    qsub tools/qsub_raiden.sh
SEED=44 EXP=branch  qsub tools/qsub_raiden.sh
SEED=44 EXP=full    qsub tools/qsub_raiden.sh
SEED=44 EXP=rand_ss qsub tools/qsub_raiden.sh
SEED=44 EXP=rand_ap qsub tools/qsub_raiden.sh

echo ""
echo "=== All submitted. Check with: qstat ==="
qstat
