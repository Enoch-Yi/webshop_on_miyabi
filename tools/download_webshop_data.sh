#!/bin/bash
set -euo pipefail

# ============================================================
# WebShop 数据下载脚本
# 在 Miyabi 登录节点或交互式作业中运行
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${REPO_ROOT}/webshop_data"

mkdir -p "${DATA_DIR}"

echo "=== 下载 WebShop 数据 ==="
echo "目标目录: ${DATA_DIR}"
echo ""
echo "WebShop 数据需要手动准备。有以下几种方式："
echo ""
echo "方式 A: 从本地 scp 上传（推荐）"
echo "  在本地机器上执行："
echo "  scp items_shuffle.json <user>@miyabi-g.jcahpc.jp:${DATA_DIR}/"
echo "  scp items_shuffle_1000.json <user>@miyabi-g.jcahpc.jp:${DATA_DIR}/"
echo "  scp items_ins_v2.json <user>@miyabi-g.jcahpc.jp:${DATA_DIR}/"
echo "  scp items_human_ins.json <user>@miyabi-g.jcahpc.jp:${DATA_DIR}/"
echo "  scp -r search_engine/indexes <user>@miyabi-g.jcahpc.jp:${DATA_DIR}/"
echo "  scp -r search_engine/indexes_1k <user>@miyabi-g.jcahpc.jp:${DATA_DIR}/"
echo ""
echo "方式 B: 从 WebShop 官方仓库下载"
echo "  git clone https://github.com/princeton-nlp/WebShop.git /tmp/WebShop"
echo "  cd /tmp/WebShop && bash setup.sh"
echo "  cp /tmp/WebShop/data/* ${DATA_DIR}/"
echo ""
echo "所需文件清单："
echo "  ${DATA_DIR}/items_shuffle.json       (全量 1.18M 商品, ~3GB)"
echo "  ${DATA_DIR}/items_shuffle_1000.json   (1K 子集, smoke test 用)"
echo "  ${DATA_DIR}/items_ins_v2.json"
echo "  ${DATA_DIR}/items_human_ins.json"
echo "  ${DATA_DIR}/indexes/                  (Lucene 全量索引, ~5GB)"
echo "  ${DATA_DIR}/indexes_1k/               (1K 索引, smoke test 用)"

# Check what's already there
echo ""
echo "=== 当前已有文件 ==="
ls -lh "${DATA_DIR}/" 2>/dev/null || echo "(目录为空)"
