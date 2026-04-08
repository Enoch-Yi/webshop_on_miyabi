#!/bin/bash
# RAIDEN 环境配置（在 qrsh 交互式节点上运行一次）
set -euo pipefail

echo "=== Step 1: Container env ==="
. /fefs/opt/dgx/env_set/nvcr-pytorch-2310-py3.sh

echo "=== Step 2: Proxy ==="
export MY_PROXY_URL="http://10.1.10.1:8080/"
export HTTP_PROXY=$MY_PROXY_URL HTTPS_PROXY=$MY_PROXY_URL
export http_proxy=$MY_PROXY_URL https_proxy=$MY_PROXY_URL ftp_proxy=$MY_PROXY_URL

echo "=== Step 3: User Python prefix ==="
mkdir -p ~/.raiden/nvcr-pytorch-2310-py3
export PATH="${HOME}/.raiden/nvcr-pytorch-2310-py3/bin:$PATH"
export LD_LIBRARY_PATH="${HOME}/.raiden/nvcr-pytorch-2310-py3/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUSERBASE="${HOME}/.raiden/nvcr-pytorch-2310-py3"
PYTHON_SITE="$(python -c 'import site; print(site.getusersitepackages())')"
export PYTHONPATH="${PYTHON_SITE}:${PYTHONPATH:-}"

echo "=== Step 4: Install deps (skip torch, use container's) ==="
pip install --user transformers accelerate pyyaml numpy tqdm rich beautifulsoup4 cleantext Flask rank_bm25 spacy thefuzz wandb safetensors "gym==0.24.0" "werkzeug<3.0"
python -m spacy download en_core_web_sm

echo "=== Step 5: Download model ==="
pip install --user huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct'); print('Model downloaded.')"

echo "=== Step 6: Data symlinks ==="
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ln -sf ~/branching_dueling_webshop_alfworld_on_RIKEN/WebShop/data "${REPO}/data"
ln -sf ~/branching_dueling_webshop_alfworld_on_RIKEN/WebShop/data "${REPO}/webshop_data"
ln -sf ~/branching_dueling_webshop_alfworld_on_RIKEN/WebShop/search_engine "${REPO}/search_engine"

echo "=== Step 7: Verify ==="
export JAVA_HOME=/home/zhangzy/.jdk/jdk-21.0.10+7
export JVM_PATH=$JAVA_HOME/lib/server/libjvm.so
python -c "import torch; print(f'torch={torch.__version__}, CUDA={torch.cuda.is_available()}')"
cd "${REPO}" && python -c "import sys; sys.path.insert(0,'.'); from webshop_env.envs.web_agent_text_env import WebAgentTextEnv; print('WebShop OK')"

echo "=== Setup complete! ==="
