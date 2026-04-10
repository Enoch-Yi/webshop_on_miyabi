# Experiment Plan

本文件给出在 `Miyabi` 上的推荐实验顺序。目标不是一次性把全部实验提满，而是减少空耗作业。

## 总原则

1. 先确认环境
2. 先 smoke
3. 先单 seed
4. 先 baseline
5. 再做 full method
6. 最后再放大多 seed 和消融

## 阶段 0：环境确认

目标：

- 确认当前仓库可运行
- 确认 `~/webshop_data` 数据齐全
- 确认 `wandb` 正常
- 确认 `JAVA_HOME` / `JVM_PATH` 正常

成功标准：

- `scripts/check_miyabi_readiness.sh` 报告没有关键缺失

## 阶段 1：Smoke Test

目标：

- 验证最小训练链路通
- 验证日志能写出
- 验证 `wandb` 和 `console.log` 正常

推荐命令：

```bash
cd ~/webshop_on_miyabi_ai_ready
qsub -I -q debug-g -l select=1 -l walltime=00:30:00
conda activate bd
export WEBSHOP_DATA_DIR="${WEBSHOP_DATA_DIR:-$HOME/webshop_data}"
export JAVA_HOME="${JAVA_HOME:-$CONDA_PREFIX}"
export JVM_PATH="${JVM_PATH:-$CONDA_PREFIX/lib/jvm/lib/server/libjvm.so}"
bash scripts/miyabi_smoke.sh
```

## 阶段 2：GRPO Baseline

目标：

- 确认正式训练参数下至少能稳定跑前几个 iteration
- 确认 `log.jsonl` 和 `console.log` 持续更新

推荐只先跑：

- `seed=42`

命令：

```bash
cd ~/webshop_on_miyabi_ai_ready
qsub -q regular-g -W group_list=gq50 -v SEED=42 scripts/miyabi_qsub_grpo.sh
```

## 阶段 3：Branch PG Only

目标：

- 单独看 `L_base + L_br` 是否有增益

命令：

```bash
cd ~/webshop_on_miyabi_ai_ready
qsub -q regular-g -W group_list=gq50 -v SEED=42,W_DPO=0.0 scripts/miyabi_qsub_full.sh
```

## 阶段 4：Full Method

目标：

- 观察 local DPO 加入后是否优于 branch PG only

命令：

```bash
cd ~/webshop_on_miyabi_ai_ready
qsub -q regular-g -W group_list=gq50 -v SEED=42 scripts/miyabi_qsub_full.sh
```

## 阶段 5：多 seed

只有在单 seed 结果合理时再做：

- `42`
- `43`
- `44`

## 阶段 6：随机消融

建议最后再做：

- `STATE_SEL=random`
- `ACTION_PAIR=random`
- `STATE_SEL=random,ACTION_PAIR=random`

## 必看指标

至少检查：

- `train_score`
- `eval_score`
- `eval_succ`
- `l_base`
- `l_br`
- `l_dpo`
- `pref_N`
- `invalid_action_rate`

## 当前已知风险

### 风险 1：数据可能只有 1K 索引

如果只有 `indexes_1k` 没有 `indexes`，则：

- smoke 可以做
- full training 不能贸然开

### 风险 2：日志不一定足够细

如果控制台里看不到 rollout 进度，要同时检查：

- `console.log`
- `log.jsonl`
- `training_log_*.txt`

### 风险 3：先前已有失败作业

目录里已有很多旧 `*.o*` 调度器输出时，不要默认最近一次就代表当前环境正常。应重新做 readiness check 和 smoke。
