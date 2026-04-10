# WebShop on Miyabi 实验方案

> 本文件定义了在 Miyabi-G (GH200 96GB) 上的完整实验矩阵。
> 所有实验基于 `Qwen2.5-1.5B-Instruct`，config 基线为 `configs/webshop_gigpo_aligned.yaml`。

## 时间估算

基于当前 GRPO seed42 (TDS=8) 实测数据：平均每 epoch ≈ 44 min。

| 实验类型 | per-epoch 估算 | 150 epochs 总计 | 48h 能跑 epochs | 需要提交次数 |
|----------|---------------|----------------|-----------------|-------------|
| GRPO (TDS=8) | ~44 min | ~111h | ~63 | 2-3 次 |
| GRPO (TDS=16) | ~89 min | ~222h | ~31 | 5 次 |
| Full Method (TDS=16, B=4 K=2) | ~160 min | ~399h | ~17 | 9 次 |

**重要**：已实现 **resume from checkpoint** 机制。每次 eval 后自动保存 `resume_ckpt/`，
重复提交同一命令即可从上次中断处继续训练。只需多次 qsub 同一脚本即可。

## 通用参数（所有实验共享，与 GiGPO 论文 Appendix E.1 对齐）

| 参数 | 值 | 说明 |
|------|----|------|
| model | Qwen/Qwen2.5-1.5B-Instruct | 1.5B 参数 |
| train_data_size | 16 | 每轮采样任务数 |
| group_size | 8 | 每任务 rollout 数（共 128 条轨迹/epoch） |
| total_epochs | 150 | 训练轮数 |
| eval_every | 10 | 每 10 epoch 评估一次 |
| eval_games | 150 | 评估任务数 |
| lr | 1e-6 | 学习率 |
| beta_kl | 0.01 | KL 惩罚系数 |
| max_prompt_length | 4096 | 最大 prompt 长度 |
| max_response_tokens | 512 | 最大生成长度 |

---

## 实验分组

### Group 0：Baseline

GRPO 基线，无 branching/DPO 组件。

| ID | 名称 | w_br | w_dpo | B | K | state_sel | action_pair | seed |
|----|------|------|-------|---|---|-----------|-------------|------|
| E0-s42 | grpo_baseline | 0.0 | 0.0 | 0 | 0 | — | — | 42 |
| E0-s43 | grpo_baseline | 0.0 | 0.0 | 0 | 0 | — | — | 43 |
| E0-s44 | grpo_baseline | 0.0 | 0.0 | 0 | 0 | — | — | 44 |

### Group 1：Full Method（论文默认设定）

完整 GiGPO：L_base + L_br + L_dpo，state_selection=top_k, action_pair=cdb。

| ID | 名称 | w_br | w_dpo | B | K | state_sel | action_pair | seed |
|----|------|------|-------|---|---|-----------|-------------|------|
| E1-s42 | full_topk_cdb | 1.0 | 1.0 | 4 | 2 | top_k | cdb | 42 |
| E1-s43 | full_topk_cdb | 1.0 | 1.0 | 4 | 2 | top_k | cdb | 43 |
| E1-s44 | full_topk_cdb | 1.0 | 1.0 | 4 | 2 | top_k | cdb | 44 |

### Group 2：Branch PG Only（消融 DPO）

仅 L_base + L_br，关闭 DPO。

| ID | 名称 | w_br | w_dpo | B | K | state_sel | action_pair | seed |
|----|------|------|-------|---|---|-----------|-------------|------|
| E2-s42 | brpg_only | 1.0 | 0.0 | 4 | 2 | top_k | cdb | 42 |

### Group 3：State Selection 消融

固定 action_pair=cdb，变化 state selection 策略。

| ID | 名称 | w_br | w_dpo | B | K | state_sel | tau_s | action_pair | seed |
|----|------|------|-------|---|---|-----------|-------|-------------|------|
| E3a-s42 | full_random_state | 1.0 | 1.0 | 4 | 2 | random | — | cdb | 42 |
| E3b-s42 | full_tau0.5 | 1.0 | 1.0 | 4 | 2 | tau_s | 0.5 | cdb | 42 |
| E3c-s42 | full_tau1.0 | 1.0 | 1.0 | 4 | 2 | tau_s | 1.0 | cdb | 42 |

### Group 4：Action Pair 消融

固定 state_selection=top_k，变化 action pair 策略。

| ID | 名称 | w_br | w_dpo | B | K | state_sel | action_pair | seed |
|----|------|------|-------|---|---|-----------|-------------|------|
| E4a-s42 | full_random_pair | 1.0 | 1.0 | 4 | 2 | top_k | random | 42 |

### Group 5：双随机消融（Double Random）

state selection 和 action pair 都用随机。

| ID | 名称 | w_br | w_dpo | B | K | state_sel | action_pair | seed |
|----|------|------|-------|---|---|-----------|-------------|------|
| E5a-s42 | full_double_random | 1.0 | 1.0 | 4 | 2 | random | random | 42 |

### Group 6：CDB 超参敏感性（低优先级）

固定 state_selection=top_k, action_pair=cdb，逐一变化 CDB 参数。
基线值：cdb_p=128, cdb_eta=1.0, cdb_mu=0.1, cdb_delta=0.01, cdb_sigma0=1.0

| ID | 名称 | 变化参数 | 值 | seed |
|----|------|----------|----|------|
| E6a-s42 | cdb_p64 | cdb_p | 64 | 42 |
| E6b-s42 | cdb_p256 | cdb_p | 256 | 42 |
| E6c-s42 | cdb_mu0.01 | cdb_mu | 0.01 | 42 |
| E6d-s42 | cdb_mu1.0 | cdb_mu | 1.0 | 42 |
| E6e-s42 | cdb_delta0.001 | cdb_delta | 0.001 | 42 |
| E6f-s42 | cdb_delta0.1 | cdb_delta | 0.1 | 42 |

---

## 推荐执行顺序

```
第一批（核心对比）:
  E0-s42  GRPO Baseline           ← 已在运行 (旧代码)，建议用新代码重跑
  E1-s42  Full Method (top_k+cdb) ← 最重要，论文方法
  E2-s42  Branch PG Only          ← 消融 DPO 的贡献

第二批（消融实验）:
  E3a-s42 random state selection
  E4a-s42 random action pair
  E5a-s42 double random

第三批（tau_s 消融）:
  E3b-s42 tau_s=0.5
  E3c-s42 tau_s=1.0

第四批（多 seed 复现，仅在单 seed 结果合理后）:
  E0-s43, E0-s44
  E1-s43, E1-s44

第五批（CDB 超参敏感性，最低优先级）:
  E6a ~ E6f
```

---

## 提交命令

工作目录：`cd ~/webshop_on_miyabi/webshop_on_miyabi`

> **Resume 说明**：所有命令可重复提交。如果 checkpoint_dir 中已有 `resume_ckpt/`，
> 训练会自动从上次保存的 epoch 继续。无需修改任何参数。

---

### 第一批：核心对比（3 个作业）

```bash
cd ~/webshop_on_miyabi/webshop_on_miyabi

# E0-s42: GRPO Baseline seed42 (~222h, 需提交约 5 次)
qsub -v SEED=42 scripts/miyabi_qsub_grpo.sh

# E1-s42: Full Method top_k+cdb seed42 (~399h, 需提交约 9 次)
qsub -v SEED=42,RUN_TAG=full_topk_cdb,STATE_SEL=top_k,ACTION_PAIR=cdb scripts/miyabi_qsub_full.sh

# E2-s42: Branch PG Only seed42 (~399h, 需提交约 9 次)
qsub -v SEED=42,RUN_TAG=brpg_only,W_DPO=0.0,STATE_SEL=top_k,ACTION_PAIR=cdb scripts/miyabi_qsub_full.sh
```

---

### 第二批：消融实验（3 个作业，第一批开始跑后即可提交）

```bash
cd ~/webshop_on_miyabi/webshop_on_miyabi

# E3a-s42: Random State Selection (~399h, 需提交约 9 次)
qsub -v SEED=42,RUN_TAG=full_random_state,STATE_SEL=random,ACTION_PAIR=cdb scripts/miyabi_qsub_full.sh

# E4a-s42: Random Action Pair (~399h, 需提交约 9 次)
qsub -v SEED=42,RUN_TAG=full_random_pair,STATE_SEL=top_k,ACTION_PAIR=random scripts/miyabi_qsub_full.sh

# E5a-s42: Double Random (~399h, 需提交约 9 次)
qsub -v SEED=42,RUN_TAG=full_double_random,STATE_SEL=random,ACTION_PAIR=random scripts/miyabi_qsub_full.sh
```

---

### 第三批：tau_s 消融（2 个作业）

```bash
cd ~/webshop_on_miyabi/webshop_on_miyabi

# E3b-s42: tau_s=0.5
qsub -v SEED=42,RUN_TAG=full_tau0.5,STATE_SEL=tau_s,TAU_S=0.5,ACTION_PAIR=cdb scripts/miyabi_qsub_full.sh

# E3c-s42: tau_s=1.0
qsub -v SEED=42,RUN_TAG=full_tau1.0,STATE_SEL=tau_s,TAU_S=1.0,ACTION_PAIR=cdb scripts/miyabi_qsub_full.sh
```

---

### 第四批：多 seed 复现（仅在单 seed 结果合理后提交）

```bash
cd ~/webshop_on_miyabi/webshop_on_miyabi

# E0-s43, E0-s44: GRPO Baseline
qsub -v SEED=43 scripts/miyabi_qsub_grpo.sh
qsub -v SEED=44 scripts/miyabi_qsub_grpo.sh

# E1-s43, E1-s44: Full Method
qsub -v SEED=43,RUN_TAG=full_topk_cdb,STATE_SEL=top_k,ACTION_PAIR=cdb scripts/miyabi_qsub_full.sh
qsub -v SEED=44,RUN_TAG=full_topk_cdb,STATE_SEL=top_k,ACTION_PAIR=cdb scripts/miyabi_qsub_full.sh
```

---

### 第五批：CDB 超参敏感性（最低优先级）

```bash
cd ~/webshop_on_miyabi/webshop_on_miyabi

# E6a-s42: cdb_p=64
qsub -v SEED=42,RUN_TAG=full_cdb_p64,CDB_P=64 scripts/miyabi_qsub_full.sh

# E6b-s42: cdb_p=256
qsub -v SEED=42,RUN_TAG=full_cdb_p256,CDB_P=256 scripts/miyabi_qsub_full.sh

# E6c-s42: cdb_mu=0.01
qsub -v SEED=42,RUN_TAG=full_cdb_mu0.01,CDB_MU=0.01 scripts/miyabi_qsub_full.sh

# E6d-s42: cdb_mu=1.0
qsub -v SEED=42,RUN_TAG=full_cdb_mu1.0,CDB_MU=1.0 scripts/miyabi_qsub_full.sh

# E6e-s42: cdb_delta=0.001
qsub -v SEED=42,RUN_TAG=full_cdb_delta0.001,CDB_DELTA=0.001 scripts/miyabi_qsub_full.sh

# E6f-s42: cdb_delta=0.1
qsub -v SEED=42,RUN_TAG=full_cdb_delta0.1,CDB_DELTA=0.1 scripts/miyabi_qsub_full.sh
```

---

### Resume 操作（作业到期后重新提交）

```bash
# 直接重复提交完全相同的命令即可，训练会自动 resume
# 例如 E0-s42 到期后：
qsub -v SEED=42 scripts/miyabi_qsub_grpo.sh
# 会自动从 /work/gq50/q50006/checkpoints/miyabi_grpo_seed42/resume_ckpt/ 恢复
```

---

## 输出目录结构

所有 run 将保存在 `runs/` 下，checkpoint 保存在 `/work/gq50/q50006/checkpoints/` 下：

```
runs/
  miyabi_grpo_seed42/           # E0-s42
  miyabi_grpo_seed43/           # E0-s43
  miyabi_full_topk_cdb_seed42/  # E1-s42
  miyabi_brpg_only_seed42/      # E2-s42
  miyabi_full_random_state_seed42/  # E3a-s42
  miyabi_full_tau0.5_seed42/    # E3b-s42
  miyabi_full_tau1.0_seed42/    # E3c-s42
  miyabi_full_random_pair_seed42/   # E4a-s42
  miyabi_full_double_random_seed42/ # E5a-s42
  miyabi_full_cdb_p64_seed42/   # E6a-s42
  ...
```

## 注意事项

1. **Miyabi 停机**：2026/04/22 09:00，剩余约 11 天，每个作业 walltime=48h
2. **48h 不够 150 epochs**：已实现 resume 机制，作业到期后重复提交同一命令即可继续
3. **队列限制**：`regular-g` 队列可能有并发作业数限制，建议分批提交
4. **当前运行中**：JOB 1617313 (GRPO seed42) 使用旧代码（train_data_size=8，无 checkpoint，无 resume），需要用新代码重新开始
5. **wandb**：所有实验默认启用 wandb，记录到 `webshop_on_miyabi_ai_ready` 项目，run name 与目录名一致
6. **checkpoint**：best_model、final_model、resume_ckpt 保存到 `/work/gq50/q50006/checkpoints/{RUN_TAG}_seed{SEED}/`
7. **Resume 原理**：每次 eval (每10 epoch) 自动保存 resume_ckpt/，包含 model + optimizer + epoch + CDB 状态 + 随机数状态。下次启动时自动检测恢复
