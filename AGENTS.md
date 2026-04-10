# AGENTS

本文件写给在 `Miyabi` 上接手实验的 AI。

## 目标

你的目标不是重新设计实验，而是安全、可核验地推进现有 WebShop 实验。

优先级顺序必须是：

1. 确认当前仓库和数据路径
2. 确认 Python / Java / wandb 环境
3. 先做 smoke test
4. 再做单 seed 正式训练
5. 最后才扩大到多 seed / 消融

## 绝对不要跳过的检查

在启动任何正式训练前，必须确认：

- 当前仓库根目录存在
- `run_full_train.sh` 存在
- `scripts/train_grpo_webshop.py` 存在
- `~/webshop_data` 存在
- `~/webshop_data/items_shuffle.json` 存在
- `~/webshop_data/items_shuffle_1000.json` 存在
- `~/webshop_data/items_ins_v2.json` 存在
- `~/webshop_data/items_human_ins.json` 存在
- `~/webshop_data/indexes_1k` 存在
- 如果要跑全量正式训练，`~/webshop_data/indexes` 也必须存在
- `wandb status` 正常
- `JAVA_HOME` 和 `JVM_PATH` 可用

如果缺的是 `indexes` 而不是 `indexes_1k`，结论应当是：

- 可以先做 1K smoke
- 不能贸然开始全量正式训练

## 推荐动作顺序

### 情况 A：第一次接手 Miyabi 环境

1. 读 `docs/00_current_state.md`
2. 跑 `scripts/check_miyabi_readiness.sh`
3. 根据输出补齐缺失项
4. 跑 `scripts/miyabi_smoke.sh`
5. smoke 成功后，再提交 GRPO baseline

### 情况 B：已经有环境，但不确定现在是否能开跑

1. 先重新跑 `scripts/check_miyabi_readiness.sh`
2. 只检查最近 1-2 次作业输出，不要一次性翻全部旧日志
3. 如果数据不完整，先停
4. 如果环境完整，先提交单 seed baseline

### 情况 C：AI 只被要求“继续实验”

不要直接提交全套实验。默认顺序：

1. `GRPO seed42`
2. `Branch PG only seed42`
3. `Full method seed42`
4. 单 seed 合理后，再补 `seed43/44`

## 不要做的事

- 不要一上来提交全部作业
- 不要假设 `indexes_1k` 能替代全量 `indexes`
- 不要把 `smoke test` 的成功误判成“全量训练一定可跑”
- 不要在 `Miyabi` 上随意下载无关内容
- 不要忽略 `wandb`、`JAVA_HOME`、`JVM_PATH`

## 你应该产出的结论格式

当你检查环境后，应明确给出下面四类结论：

1. 当前路径是否正确
2. 当前数据是否够跑 smoke
3. 当前数据是否够跑正式训练
4. 下一条最合理的命令是什么

如果条件不足，直接说明阻塞点，不要继续推测。
