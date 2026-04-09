#!/bin/bash
set -euo pipefail

# ============================================================
# Miyabi 环境一键配置脚本
# 在 Miyabi-G 登录节点上运行
# ============================================================

echo "=== Step 1: 创建 conda 环境 ==="
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

conda create -n bd python=3.10 -y
conda activate bd

echo "=== Step 2: 安装 Python 依赖 ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
pip install -r "${REPO_ROOT}/requirements.txt"

echo "=== Step 3: 下载 Qwen2.5-1.5B-Instruct ==="
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
print('Downloading tokenizer...')
AutoTokenizer.from_pretrained(model_name)
print('Downloading model...')
AutoModelForCausalLM.from_pretrained(model_name)
print('Done. Model cached.')
"

echo "=== Step 4: 验证 ==="
python -c "
import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')
import transformers; print(f'transformers {transformers.__version__}')
import yaml; print('yaml OK')
import flask; print('Flask OK')
import bs4; print('bs4 OK')
"

echo ""
echo "=== 环境配置完成 ==="
echo "下一步: 准备 WebShop 数据（见 README.md Step 3）"
