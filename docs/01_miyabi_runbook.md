# Miyabi Runbook

本文件给出从前期准备到正式训练的最短可执行路径。

## 1. 登录

```bash
ssh -l <username> miyabi-g.jcahpc.jp
```

登录时通常需要：

- SSH key
- OTP

## 2. 进入工作目录

推荐把真正的实验仓库放在 `/work`，不要依赖 `/home` 进行大数据准备。

建议路径：

```bash
cd /work/<group>/<username>
```

如果当前已经在家目录下有可用仓库，也至少要确认数据目录是否真的完整。

## 3. 克隆仓库并准备 Python 环境

```bash
cd /work/<group>/<username>
git clone <your_github_repo_url> webshop_on_miyabi_ai_ready
cd webshop_on_miyabi_ai_ready
```

然后手动准备环境：

```bash
module load miniforge
conda create -n bd python=3.10 -y
conda activate bd
pip install -r requirements.txt
```

## 4. 准备模型

```bash
python - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
PY
```

## 5. 准备 WebShop 数据

最少先确认：

```bash
ls ~/webshop_data
```

### smoke-ready

至少应当看到：

- `items_shuffle_1000.json`
- `items_ins_v2_1000.json`
- `items_human_ins.json`
- `indexes_1k`

### full-train-ready

至少应当看到：

- `items_shuffle.json`
- `items_ins_v2.json`
- `items_human_ins.json`
- `indexes`

## 6. 运行基础检查

在这个仓库里执行：

```bash
cd /path/to/webshop_on_miyabi_ai_ready
bash scripts/check_miyabi_readiness.sh
```

这个检查脚本会告诉你：

- 当前仓库路径是否存在
- 当前数据目录是否存在
- 是否只满足 smoke
- 是否满足 full training
- 还缺哪些关键文件

## 7. Smoke Test

申请交互式节点：

```bash
qsub -I -q debug-g -l select=1 -l walltime=00:30:00
```

然后在节点上：

```bash
cd /work/<group>/<username>/webshop_on_miyabi_ai_ready
conda activate bd
export WEBSHOP_DATA_DIR="${WEBSHOP_DATA_DIR:-$HOME/webshop_data}"
export JAVA_HOME="${JAVA_HOME:-$CONDA_PREFIX}"
export JVM_PATH="${JVM_PATH:-$CONDA_PREFIX/lib/jvm/lib/server/libjvm.so}"
bash scripts/miyabi_smoke.sh
```

## 8. 正式训练

### GRPO baseline

```bash
cd /work/<group>/<username>/webshop_on_miyabi_ai_ready
qsub -q regular-g -W group_list=gq50 -v SEED=42 scripts/miyabi_qsub_grpo.sh
```

### Full method

```bash
cd /work/<group>/<username>/webshop_on_miyabi_ai_ready
qsub -q regular-g -W group_list=gq50 -v SEED=42 scripts/miyabi_qsub_full.sh
```

## 9. 监控

```bash
qstat
qstat -v
qdel <JOB_ID>
```

如果作业已经开始跑，再检查：

```bash
ls runs
tail -f runs/<run_name>/console.log
```

## 10. wandb

如果要开 `wandb`：

```bash
wandb login
export WANDB_ENTITY="<your_entity>"
```

检查：

```bash
wandb status
echo "$WANDB_PROJECT"
echo "$WANDB_ENTITY"
```

## 11. 最常见的三类阻塞

### 数据不完整

最典型现象：

- 能跑 smoke
- 不能跑全量正式训练

### Java / pyserini 问题

最典型现象：

- `LuceneSearcher`
- `libjvm`
- `jnius`

优先检查：

```bash
echo "$JAVA_HOME"
echo "$JVM_PATH"
```

### wandb 不工作

优先检查：

```bash
wandb status
echo "$WANDB_ENTITY"
echo "$WANDB_PROJECT"
```
