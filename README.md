# Branching Dueling on WebShop

在线 RL 方法 "Branching Dueling" 在 WebShop 电商购物环境上的实验代码。

支持在 Miyabi 超级计算机（H100 96GB）、RAIDEN（A100）和本地 GPU 上运行。

## 目录结构

```
webshop_on_miyabi/
├── README.md                   # 本文件
├── requirements.txt            # Python 依赖
├── .gitignore
├── configs/
│   └── webshop_config.yaml     # WebShop 环境配置
├── scripts/
│   └── train_branching_dueling_webshop.py  # 训练主脚本
├── agents/                     # 核心算法模块
│   ├── fgts_cdb.py             # FGTS.CDB（Contextual Dueling Bandit）
│   ├── qwen_policy.py          # Qwen 生成式策略
│   └── trajectory.py           # 轨迹/偏好数据结构
├── webshop_env/                # WebShop 环境代码（不含数据）
│   ├── envs/                   # Gym 环境
│   ├── engine/                 # 搜索/奖励引擎
│   ├── templates/              # HTML 模板
│   └── static/                 # 静态资源
├── tools/                      # 工具脚本
│   ├── setup_miyabi.sh         # Miyabi 环境一键配置
│   ├── download_webshop_data.sh # 数据准备说明
│   ├── run_smoke_test.sh       # 冒烟测试
│   ├── qsub_grpo.sh           # PBS: GRPO baseline
│   ├── qsub_full.sh           # PBS: 完整方法 + 全部消融
│   └── submit_all_experiments.sh # 一键提交全部实验
└── webshop_data/               # WebShop 数据（git 忽略，需单独准备）
    ├── items_shuffle.json      # 全量 1.18M 商品
    ├── items_shuffle_1000.json # 1K 子集（smoke test）
    ├── items_ins_v2.json
    ├── items_human_ins.json
    ├── indexes/                # Lucene 全量索引
    └── indexes_1k/             # 1K 索引
```

---

## 方法概述

### 算法：Branching Dueling (Credit Learning for LLM Agents)

总损失函数：**L = L_base + λ·L_br + γ·L_DPO**

| 组件 | 公式 | 作用 |
|------|------|------|
| **L_base** | Group-normalized PG on base rollouts (Eq.3) | 轨迹级 GRPO，和标准 GRPO 一致 |
| **L_br** | Group-normalized PG on branched suffixes (Eq.4) | 在分支点的 suffix 轨迹上做更精细的 PG |
| **L_DPO** | Local DPO at branching states (Eq.5) | 在分支点直接比较两个 action 的偏好 |

辅助模块：
- **State Selection**（Eq.9-12）：用 CDB uncertainty + trajectory importance 选择在哪些 state 做分支
- **FGTS.CDB**：在选定 state 用 Thompson Sampling 选择 action pair 做对决

### 实验设计

目标：证明 Branching Dueling > GRPO。和 HCAPO 论文 (2603.08754) 在同一起跑线。

| 编号 | 实验名 | L_base | L_br | L_DPO | State Sel. | Action Pair | 证明什么 |
|:----:|--------|:------:|:----:|:-----:|:----------:|:-----------:|---------|
| A | GRPO baseline | ✓ | ✗ | ✗ | — | — | 起跑线 |
| B | + Branch PG | ✓ | ✓ | ✗ | CDB | CDB | L_br 的增量价值 |
| C | + Local DPO | ✓ | ✓ | ✓ | CDB | CDB | L_DPO 的增量价值 |
| D | Full method | ✓ | ✓ | ✓ | CDB | CDB | 完整方法 |
| E | Random state | ✓ | ✓ | ✓ | Random | CDB | State selection 价值 |
| F | Random pair | ✓ | ✓ | ✓ | CDB | Random | CDB action selection 价值 |
| G | All random | ✓ | ✓ | ✓ | Random | Random | 两者都随机 |

参考数据（引用自 HCAPO Table 1，Qwen2.5-1.5B-Instruct）：

| 方法 | WebShop Score | WebShop Succ% |
|------|:---:|:---:|
| GRPO | 75.8 | 56.8% |
| GiGPO | 83.1 | 65.0% |
| HCAPO | 83.8 | 68.5% |

---

## Step 1: 登录 Miyabi-G

```bash
ssh -l <username> miyabi-g.jcahpc.jp
# 输入 SSH passphrase
# 输入 OTP 验证码（Google Authenticator / Microsoft Authenticator）
```

首次登录需要：
1. 在 https://miyabi-www.jcahpc.jp/ 注册 SSH 公钥
2. 配置 OTP（一次性密码）

详细步骤见 Miyabi User's Guide Section 2.3-2.4。

## Step 2: 克隆代码 & 配置环境

```bash
# 克隆到 /work 目录（比 /home 大，5TB 配额）
cd /work/<groupname>/<username>
git clone <repo_url> webshop_on_miyabi
cd webshop_on_miyabi

# 一键配置 Python 环境
bash tools/setup_miyabi.sh
```

`setup_miyabi.sh` 会：
1. 创建 conda 环境 `bd`（Python 3.10）
2. 安装全部 Python 依赖
3. 下载 Qwen2.5-1.5B-Instruct 模型
4. 验证环境

如果自动脚本失败，手动执行：

```bash
module load miniforge
conda create -n bd python=3.10 -y
conda activate bd
pip install -r requirements.txt

# 下载模型（登录节点可直接联网，无需 proxy）
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
"
```

### Miyabi 网络说明

- **登录节点**：可直接访问外网（pip install、git clone、wandb login 均可）
- **计算节点**：支持 HTTP/HTTPS/SSH 外网访问（通过 NAT），**不需要设置 proxy**
- **wandb**：登录节点和计算节点都能正常使用

```bash
# wandb 登录
wandb login
# 输入你的 API key
# 或者在环境变量/作业脚本中设置:
export WANDB_API_KEY="your_key_here"
```

## Step 3: 准备 WebShop 数据

WebShop 数据太大（~11GB），不包含在 git 仓库中。需要手动准备。

### 方式 A：从本地上传（推荐）

在你的本地机器上执行：

```bash
# 上传数据文件
scp items_shuffle.json <user>@miyabi-g.jcahpc.jp:/work/<group>/<user>/webshop_on_miyabi/webshop_data/
scp items_shuffle_1000.json <user>@miyabi-g.jcahpc.jp:/work/<group>/<user>/webshop_on_miyabi/webshop_data/
scp items_ins_v2.json <user>@miyabi-g.jcahpc.jp:/work/<group>/<user>/webshop_on_miyabi/webshop_data/
scp items_human_ins.json <user>@miyabi-g.jcahpc.jp:/work/<group>/<user>/webshop_on_miyabi/webshop_data/

# 上传搜索索引
scp -r indexes/ <user>@miyabi-g.jcahpc.jp:/work/<group>/<user>/webshop_on_miyabi/webshop_data/
scp -r indexes_1k/ <user>@miyabi-g.jcahpc.jp:/work/<group>/<user>/webshop_on_miyabi/webshop_data/
```

### 方式 B：从 WebShop 官方下载

```bash
cd /tmp && git clone https://github.com/princeton-nlp/WebShop.git
cd WebShop && bash setup.sh
cp data/* /work/<group>/<user>/webshop_on_miyabi/webshop_data/
```

### 验证数据

```bash
ls -lh webshop_data/
# 应该看到:
# items_shuffle.json       (~3GB)
# items_shuffle_1000.json  (~3MB)
# items_ins_v2.json
# items_human_ins.json
# indexes/                 (~5GB)
# indexes_1k/              (~5MB)
```

## Step 4: Smoke Test

在交互式作业中验证一切正常：

```bash
# 申请 1 个 H100 节点，最多 30 分钟
qsub -I -q debug-g -l select=1 -l walltime=00:30:00

# 在计算节点上
conda activate bd
cd /work/<group>/<user>/webshop_on_miyabi
bash tools/run_smoke_test.sh
```

成功标准：
- 2 个 iteration 正常完成，没有 crash
- `log.jsonl` 中 `l_base` 不为 0
- `eval_succ` 或 `train_score` > 0

## Step 5: 提交正式训练

### 单个实验

```bash
cd /work/<group>/<user>/webshop_on_miyabi

# GRPO baseline, seed 42
qsub -v SEED=42 tools/qsub_grpo.sh

# Full method, seed 42
qsub -v SEED=42 tools/qsub_full.sh

# 消融: Branch PG only (w_dpo=0)
qsub -v SEED=42,W_DPO=0.0 tools/qsub_full.sh

# 消融: Random state selection
qsub -v SEED=42,STATE_SEL=random tools/qsub_full.sh

# 消融: Random action pair
qsub -v SEED=42,ACTION_PAIR=random tools/qsub_full.sh
```

### 一键提交全部实验

```bash
bash tools/submit_all_experiments.sh
```

这会提交 15 个作业：
- 3× GRPO baseline (seeds 42/43/44)
- 3× Full method (seeds 42/43/44)
- 3× Branch PG only (seeds 42/43/44)
- 1× Random state selection
- 1× Random action pair
- 1× All random

### 监控作业

```bash
qstat                    # 查看所有作业状态
qstat -v                 # 详细信息（含等待原因）
qstat --nodeuse          # 查看节点使用率
qdel <JOB_ID>           # 取消作业
```

## Step 6: 查看结果

### 日志文件

每个实验在 `runs/<experiment_name>/` 下生成：

| 文件 | 内容 |
|------|------|
| `console.log` | 完整控制台输出 |
| `log.jsonl` | 每个 iteration 一行 JSON，含所有 metrics |
| `training_log_*.txt` | 详细的 prompt/response 日志（每步模型输出） |
| `best_model/` | 最佳 eval score 时的 checkpoint |
| `final_model/` | 训练结束时的 checkpoint |

### 查看训练曲线

```bash
# 快速查看
python -c "
import json
for line in open('runs/grpo_seed42/log.jsonl'):
    e = json.loads(line)
    es = e.get('eval_succ', '--')
    print(f\"iter={e['iter']:3d}  score={e.get('train_score',0):.3f}  \"
          f\"eval_succ={es}  l_base={e.get('l_base',0):.5f}\")
"
```

### wandb Dashboard

如果配置了 wandb，所有 metrics 会自动上传到 wandb dashboard。

### 关键指标

| 指标 | 含义 | 期望趋势 |
|------|------|---------|
| `train_score` | base rollout 平均分 | 逐步上升 |
| `eval_succ` | 评估成功率 | **核心指标**，逐步上升 |
| `eval_score` | 评估平均分 | 逐步上升 |
| `l_base` | base PG loss | 非零 |
| `l_br` | branch PG loss | 非零（实验 B/C/D） |
| `l_dpo` | local DPO loss | < 0.693（实验 C/D） |
| `pref_N` | 有效偏好对数量 | > 0（实验 C/D） |
| `action_rescue_rate` | 无效 action 被替换的比例 | 逐步下降 |
| `invalid_action_rate` | 无效 action 比例 | 逐步下降 |

---

## 在其他机器上运行

### 本地 GPU（A6000 / 4090）

不需要 PBS，直接运行：

```bash
conda activate bd  # 或你的环境名
cd webshop_on_miyabi

# 设置数据目录（如果不在默认位置）
export WEBSHOP_DATA_DIR=/path/to/webshop/data

# GRPO smoke test
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_branching_dueling_webshop.py \
    configs/webshop_config.yaml \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --save_dir runs/grpo_smoke \
    --seed 42 --iters 2 --eval_every 1 --eval_games 5 \
    --N 8 --B 0 --K 0 --queries_per_step 4 \
    --w_base 1.0 --w_br 0.0 --w_dpo 0.0

# Full method
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_branching_dueling_webshop.py \
    configs/webshop_config.yaml \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --save_dir runs/bd_full_seed42 \
    --seed 42 --iters 150 --eval_every 10 --eval_games 150 \
    --N 8 --B 4 --K 2 --queries_per_step 4 \
    --w_base 1.0 --w_br 1.0 --w_dpo 1.0
```

### RAIDEN（A100）

RAIDEN 的计算节点需要设置 proxy：

```bash
export MY_PROXY_URL="http://10.1.10.1:8080/"
export HTTP_PROXY=$MY_PROXY_URL
export HTTPS_PROXY=$MY_PROXY_URL
export http_proxy=$MY_PROXY_URL
export https_proxy=$MY_PROXY_URL
```

作业提交使用 qsub（Altair Grid Engine），语法和 Miyabi（PBS）不同。

---

## 命令行参数参考

```
python scripts/train_branching_dueling_webshop.py <config.yaml> [options]

必需参数:
  config_path           WebShop 环境配置 YAML

模型参数:
  --model_name STR      HuggingFace 模型名或本地路径
  --lr FLOAT            学习率 (default: 1e-6)
  --seed INT            随机种子

训练参数:
  --iters INT           训练迭代数 (default: 150)
  --eval_every INT      评估间隔 (default: 10)
  --eval_games INT      评估任务数 (default: 20)
  --save_dir STR        输出目录

采样参数:
  --N INT               每个 query 的 base rollout 数 (default: 8)
  --B INT               branching state 数量，设 0 = GRPO only (default: 4)
  --K INT               每个 state 的 duel 次数 (default: 2)
  --queries_per_step INT  每步采样的不同 task 数 (default: 1)

损失权重:
  --w_base FLOAT        L_base 权重 (default: 1.0)
  --w_br FLOAT          L_br 权重，设 0 = 无 branch PG (default: 1.0)
  --w_dpo FLOAT         L_DPO 权重，设 0 = 无 local DPO (default: 1.0)
  --beta_kl FLOAT       KL 惩罚系数 (default: 0.0)

消融控制:
  --state_selection_mode {top_k,tau_s,random}
                        State selection 模式 (default: top_k)
  --action_pair_mode {cdb,random}
                        Action pair 选择模式 (default: cdb)

Rescue 策略:
  --allow_syntax_training {0,1}   (default: 1)
  --syntax_rescue_weight FLOAT    (default: 0.25)
  --allow_semantic_training {0,1} (default: 0)
  --semantic_rescue_weight FLOAT  (default: 0.0)
```

## 常见问题

### Q: 训练很慢怎么办？
A: 减少 `--queries_per_step`（减少每步的 task 数）或减少 `--N`（减少每个 task 的 rollout 数）。或者用 `--eval_games 20` 减少评估时间。

### Q: JVM/pyserini crash 怎么办？
A: WebShop 的搜索引擎依赖 Java。如果 JVM 和 PyTorch 冲突：
1. 设置 `export JVM_PATH=/path/to/libjvm.so`
2. 或者只用 1K 子集（在 config.yaml 中设 `num_products: 1000`）

### Q: eval_succ 始终为 0？
A: 增大 `--queries_per_step`（推荐 4-8）和 `--N`（推荐 16），确保每步有足够的成功轨迹产生 advantage 信号。

### Q: l_dpo ≈ 0.693 不下降？
A: 这是 DPO loss 的随机基线。说明偏好对信号太弱（两条 branch suffix 的 reward 差异太小）。考虑先关掉 DPO（`--w_dpo 0.0`），确认 L_base + L_br 能工作后再打开。
