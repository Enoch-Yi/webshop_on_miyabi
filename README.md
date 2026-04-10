# webshop_on_miyabi_ai_ready

这是一个面向 `Miyabi` 的**单仓可运行**仓库。

它不是单纯的 runbook，也不是只放说明文档的骨架，而是同时包含：

- 一份可运行的 `BD` 训练代码副本
- Miyabi 环境准备说明
- 数据核验脚本
- `smoke test` 启动脚本
- 正式训练 `qsub` 脚本
- 推荐实验顺序
- 常见故障排查

设计目标就是：

1. 把这个仓库上传到 GitHub
2. 在 `Miyabi` 上直接 `git clone`
3. 配环境
4. 跑一次 `smoke`
5. 通过后直接开始正式训练

默认外部数据目录仍然是：

- `~/webshop_data`

如果你的实际路径不同，只需要改环境变量，不需要再依赖第二个训练仓库。

## 仓库不包含什么

为了适合直接发 GitHub，这个仓库不会直接提交下面这些大文件：

- WebShop 数据集本体
- 搜索索引
- Hugging Face 模型权重
- 训练产生的 `runs/`

也就是说，GitHub 上传的是代码、配置、脚本和文档；真正到 `Miyabi` 上后，数据默认仍从 `~/webshop_data` 读取。

## 适用对象

- 在 `Miyabi` 上手工运行实验的人
- 需要接手或继续推进 `Miyabi` 实验的 AI
- 需要快速判断“现在能不能开跑”的排障脚本

## 建议阅读顺序

1. `AGENTS.md`
2. `docs/00_current_state.md`
3. `docs/01_miyabi_runbook.md`
4. `docs/02_experiment_plan.md`

## 目录

```text
webshop_on_miyabi_ai_ready/
  AGENTS.md
  README.md
  configs/
  src/
  vendor/
  run_full_train.sh
  docs/
    00_current_state.md
    01_miyabi_runbook.md
    02_experiment_plan.md
  scripts/
    bootstrap_miyabi_env.sh
    check_miyabi_readiness.sh
    setup_webshop_assets.py
    smoke_test_webshop.py
    train_grpo_webshop.py
    miyabi_smoke.sh
    miyabi_qsub_grpo.sh
    miyabi_qsub_full.sh
```

## 当前定位

这个仓库就是 `Miyabi` 上要直接使用的仓库。

- 数据目录：`~/webshop_data`
- 调度系统：PBS / `qsub`
- 目标硬件：`Miyabi-G`（GH200 / 96GB GPU）

## 快速开始

### 0. 克隆并准备环境

```bash
git clone <your_github_repo_url> webshop_on_miyabi_ai_ready
cd webshop_on_miyabi_ai_ready
bash scripts/bootstrap_miyabi_env.sh
conda activate bd
```

### 1. 基础检查

```bash
bash scripts/check_miyabi_readiness.sh
```

### 2. Smoke

```bash
bash scripts/miyabi_smoke.sh
```

### 3. GRPO baseline

```bash
qsub -q regular-g -W group_list=gq50 scripts/miyabi_qsub_grpo.sh
```

### 4. Full method

```bash
qsub -q regular-g -W group_list=gq50 scripts/miyabi_qsub_full.sh
```

## 与现有资料的关系

这个仓库内容是从这些现有资料中抽取、重组和显式化出来的：

- `webshop_on_miyabi/README.md`
- `RAIDEN_MIYABI/miyabi_guide.md`
- `RAIDEN_MIYABI/How to login Raiden`
- `branching_dueling_webshop_alfworld_on_RIKEN/EXPERIMENT_PLAN.md`
- `branching_dueling_webshop_alfworld_on_RIKEN/QSUB_TEMPLATES.md`

重点不是“信息更多”，而是：

- 给 AI 明确先后顺序
- 给人明确检查点
- 把“现在缺什么才能正式训练”说清楚
- 让仓库本身就能直接跑起来
