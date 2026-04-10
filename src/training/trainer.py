from __future__ import annotations

import copy
import os
import random
import time
from typing import Any, Dict, List

import numpy as np
import torch

from src.algorithms.branching import group_normalize_branched, select_branch_positions
from src.algorithms.cdb import EmbeddingFeatureExtractor, FGTSCDB
from src.algorithms.grpo import SameTaskGRPO
from src.algorithms.losses import compute_dpo_loss, compute_kl_penalty, compute_pg_loss
from src.algorithms.preferences import preference_from_duel
from src.envs.webshop_env import WebShopEnvConfig, WebShopTextEnvWrapper
from src.envs.webshop_group_env import SameTaskGroupSampler
from src.models.qwen_policy import QwenPolicy
from src.rollout.collector import collect_one_trajectory, replay_to_state_and_branch
from src.training.logging import (
    JsonlLogger,
    PromptResponseLogger,
    WandbLogger,
    format_console_metrics,
)
from src.utils.types import BranchingSettings, LossWeights, Preference, RolloutMetrics, Trajectory


class WebShopTrainer:
    def __init__(self, cfg: Dict[str, Any], model_name: str, save_dir: str) -> None:
        self.cfg = cfg
        self.save_dir = save_dir
        seed = int(cfg["training"]["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        env_cfg = WebShopEnvConfig(**cfg["env"])
        self.env = WebShopTextEnvWrapper(env_cfg)
        self.group_sampler = SameTaskGroupSampler(
            self.env,
            train_data_size=int(cfg["training"]["train_data_size"]),
            group_size=int(cfg["training"]["group_size"]),
            seed=seed,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = QwenPolicy(
            model_name=model_name,
            device=self.device,
            max_prompt_length=int(cfg["model"]["max_prompt_length"]),
            max_response_tokens=int(cfg["model"]["max_response_tokens"]),
            history_length=int(cfg["model"]["history_length"]),
            temperature=float(cfg["model"]["gen_temperature"]),
            eval_temperature=float(cfg["model"]["eval_temperature"]),
            top_p=float(cfg["model"]["top_p"]),
        )
        self.ref_model = None
        if float(cfg["training"]["beta_kl"]) > 0:
            self.ref_model = copy.deepcopy(self.policy.model).eval()
            for param in self.ref_model.parameters():
                param.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.policy.model.parameters(), lr=float(cfg["training"]["lr"]))
        self.logger = JsonlLogger(save_dir, cfg["logging"]["jsonl_name"])
        run_tag = os.path.basename(save_dir.rstrip("/")) or "run"
        self.console_log = bool(cfg["logging"].get("console_log", True))
        self.prompt_response_logger = (
            PromptResponseLogger(
                save_dir=save_dir,
                tag=run_tag,
                max_bytes=int(cfg["logging"].get("prompt_response_max_bytes", 1_000_000)),
            )
            if bool(cfg["logging"].get("prompt_response_log", True))
            else None
        )
        self.wandb_logger = WandbLogger(
            enabled=bool(cfg["logging"].get("wandb_enabled", False)),
            project=cfg["logging"].get("wandb_project"),
            name=cfg["logging"].get("wandb_name", run_tag),
            group=cfg["logging"].get("wandb_group"),
            entity=cfg["logging"].get("wandb_entity"),
            mode=str(cfg["logging"].get("wandb_mode", "online")),
            tags=cfg["logging"].get("wandb_tags"),
            config=cfg,
            save_dir=save_dir,
        )
        self.loss_weights = LossWeights(
            w_base=float(cfg["algorithm"]["w_base"]),
            w_br=float(cfg["algorithm"]["w_br"]),
            w_dpo=float(cfg["algorithm"]["w_dpo"]),
            beta_dpo=float(cfg["algorithm"]["beta_dpo"]),
            lambda_br=float(cfg["algorithm"]["lambda_br"]),
            gamma_dpo=float(cfg["algorithm"]["gamma_dpo"]),
        )
        self.branching_settings = BranchingSettings(
            B=int(cfg["algorithm"]["B"]),
            K=int(cfg["algorithm"]["K"]),
            state_selection_mode=str(cfg["algorithm"]["state_selection_mode"]),
            action_pair_mode=str(cfg["algorithm"]["action_pair_mode"]),
            tau_s=cfg["algorithm"]["tau_s"],
            cdb_p=int(cfg["algorithm"]["cdb_p"]),
            cdb_eta=float(cfg["algorithm"]["cdb_eta"]),
            cdb_mu=float(cfg["algorithm"]["cdb_mu"]),
            cdb_delta=float(cfg["algorithm"]["cdb_delta"]),
            cdb_sigma0=float(cfg["algorithm"]["cdb_sigma0"]),
            lambda_u=float(cfg["algorithm"]["lambda_u"]),
            lambda_i=float(cfg["algorithm"]["lambda_i"]),
        )
        self.feature_extractor = EmbeddingFeatureExtractor(
            embed_layer=self.policy.model.get_input_embeddings(),
            tokenizer=self.policy.tokenizer,
            p=self.branching_settings.cdb_p,
            seed=seed,
        )
        self.cdb = FGTSCDB(
            p=self.branching_settings.cdb_p,
            eta=self.branching_settings.cdb_eta,
            mu=self.branching_settings.cdb_mu,
            delta=self.branching_settings.cdb_delta,
            sigma0=self.branching_settings.cdb_sigma0,
        )

    def _flatten(self, grouped: List[List[Trajectory]]) -> List[Trajectory]:
        return [traj for group in grouped for traj in group]

    def train(self) -> None:
        estimator = SameTaskGRPO()
        total_epochs = int(self.cfg["training"]["total_epochs"])
        eval_every = int(self.cfg["training"]["eval_every"])
        start = time.time()
        try:
            for epoch in range(total_epochs):
                epoch_start = time.time()
                batch = self.group_sampler.sample_train_batch()
                grouped_trajectories: list[list[Trajectory]] = []
                metric_acc = RolloutMetrics()
                total_groups = len(batch.grouped_task_ids)
                group_size = int(self.cfg["training"]["group_size"])
                for group_idx, task_id_group in enumerate(batch.grouped_task_ids):
                    group_trajs: list[Trajectory] = []
                    for rollout_idx, task_id in enumerate(task_id_group):
                        trajectory, metrics = collect_one_trajectory(
                            env=self.env,
                            policy=self.policy,
                            task_id=task_id,
                            max_steps=int(self.cfg["env"]["max_steps"]),
                            history_length=int(self.cfg["model"]["history_length"]),
                            success_reward=float(self.cfg["training"]["success_reward"]),
                            invalid_action_penalty=float(self.cfg["training"]["invalid_action_penalty"]),
                            top_m_actions=int(self.cfg["training"]["top_m_actions"]),
                            pr_logger=self.prompt_response_logger,
                            iteration=epoch,
                            phase="base_rollout",
                        )
                        group_trajs.append(trajectory)
                        metric_acc.format_invalid_count += metrics.format_invalid_count
                        metric_acc.action_not_in_available_count += metrics.action_not_in_available_count
                        metric_acc.invalid_transition_count += metrics.invalid_transition_count
                        metric_acc.total_steps += metrics.total_steps
                        if self.console_log:
                            print(
                                f"  [epoch {epoch}] q{group_idx + 1}/{total_groups} "
                                f"base {rollout_idx + 1}/{group_size}  "
                                f"steps={len(trajectory)} won={trajectory.won} "
                                f"score={trajectory.score:.3f}  "
                                f"({time.time() - epoch_start:.0f}s)",
                                flush=True,
                            )
                    grouped_trajectories.append(group_trajs)

                flat_trajectories = self._flatten(grouped_trajectories)
                base_advantages = estimator.compute(grouped_trajectories)
                base_phase_s = time.time() - epoch_start

                branched_trajectories: list[Trajectory] = []
                prefs: list[Preference] = []
                num_branch_candidates = 0
                branch_group_id = 0
                if self.branching_settings.B > 0 and self.branching_settings.K > 0:
                    for task_group in grouped_trajectories:
                        task_branch_candidates = select_branch_positions(
                            task_group,
                            num_branch_states=self.branching_settings.B,
                            state_selection_mode=self.branching_settings.state_selection_mode,
                            tau_s=self.branching_settings.tau_s,
                            lambda_u=self.branching_settings.lambda_u,
                            lambda_i=self.branching_settings.lambda_i,
                            cdb=self.cdb,
                            feat_ext=self.feature_extractor,
                        )
                        num_branch_candidates += len(task_branch_candidates)
                        if self.console_log:
                            print(
                                f"  [epoch {epoch}] state selection(task={task_group[0].task_id}) "
                                f"{len(task_branch_candidates)} candidates  "
                                f"({time.time() - epoch_start:.0f}s)",
                                flush=True,
                            )
                        for candidate in task_branch_candidates:
                            base_trajectory = task_group[candidate.traj_idx]
                            task_id = base_trajectory.task_id
                            prefix_actions = (
                                base_trajectory.prefix_actions(candidate.step_idx - 1)
                                if candidate.step_idx > 0 else []
                            )
                            available_actions = candidate.available_actions
                            action_features = candidate.action_features
                            for _ in range(self.branching_settings.K):
                                if len(available_actions) < 2:
                                    continue
                                if self.branching_settings.action_pair_mode == "random":
                                    sampled = random.sample(range(len(available_actions)), k=min(2, len(available_actions)))
                                    a1_idx, a2_idx = sampled[0], sampled[-1]
                                else:
                                    self.cdb.sgld_step()
                                    a1_idx, a2_idx = self.cdb.select_actions(action_features)
                                a1 = available_actions[a1_idx]
                                a2 = available_actions[a2_idx]
                                traj1, metrics1 = replay_to_state_and_branch(
                                    env=self.env,
                                    policy=self.policy,
                                    task_id=task_id,
                                    prefix_actions=prefix_actions,
                                    branch_action=a1,
                                    branch_group_id=branch_group_id,
                                    max_steps=int(self.cfg["env"]["max_steps"]),
                                    success_reward=float(self.cfg["training"]["success_reward"]),
                                    invalid_action_penalty=float(self.cfg["training"]["invalid_action_penalty"]),
                                    top_m_actions=int(self.cfg["training"]["top_m_actions"]),
                                    pr_logger=self.prompt_response_logger,
                                    iteration=epoch,
                                    phase="branch_rollout",
                                )
                                traj2, metrics2 = replay_to_state_and_branch(
                                    env=self.env,
                                    policy=self.policy,
                                    task_id=task_id,
                                    prefix_actions=prefix_actions,
                                    branch_action=a2,
                                    branch_group_id=branch_group_id,
                                    max_steps=int(self.cfg["env"]["max_steps"]),
                                    success_reward=float(self.cfg["training"]["success_reward"]),
                                    invalid_action_penalty=float(self.cfg["training"]["invalid_action_penalty"]),
                                    top_m_actions=int(self.cfg["training"]["top_m_actions"]),
                                    pr_logger=self.prompt_response_logger,
                                    iteration=epoch,
                                    phase="branch_rollout",
                                )
                                branched_trajectories.extend([traj1, traj2])
                                for metrics in (metrics1, metrics2):
                                    metric_acc.format_invalid_count += metrics.format_invalid_count
                                    metric_acc.action_not_in_available_count += metrics.action_not_in_available_count
                                    metric_acc.invalid_transition_count += metrics.invalid_transition_count
                                    metric_acc.total_steps += metrics.total_steps

                                if traj1.terminal_reward != traj2.terminal_reward:
                                    y_tilde = 1.0 if traj1.terminal_reward > traj2.terminal_reward else -1.0
                                    self.cdb.record(action_features, a1_idx, a2_idx, y_tilde)
                                pref = preference_from_duel(
                                    prompt=candidate.prompt,
                                    action_a=a1,
                                    action_b=a2,
                                    traj_a=traj1,
                                    traj_b=traj2,
                                )
                                if pref is not None:
                                    prefs.append(pref)
                            branch_group_id += 1
                if self.console_log and not (self.branching_settings.B > 0 and self.branching_settings.K > 0):
                    print(
                        f"  [epoch {epoch}] state selection: 0 candidates  "
                        f"({time.time() - epoch_start:.0f}s)",
                        flush=True,
                    )

                branch_advantages, branch_group_sizes = group_normalize_branched(branched_trajectories)
                branch_phase_s = time.time() - epoch_start - base_phase_s
                if self.console_log:
                    print(
                        f"  [epoch {epoch}] computing losses...  "
                        f"({time.time() - epoch_start:.0f}s)",
                        flush=True,
                    )
                self.policy.model.train()
                self.optimizer.zero_grad(set_to_none=True)
                base_loss, base_stats = compute_pg_loss(
                    policy=self.policy,
                    trajectories=flat_trajectories,
                    advantages=base_advantages,
                    max_train_steps=int(self.cfg["training"]["max_train_steps"]),
                    scale=self.loss_weights.w_base,
                )
                branch_loss, branch_stats = compute_pg_loss(
                    policy=self.policy,
                    trajectories=branched_trajectories,
                    advantages=branch_advantages,
                    max_train_steps=int(self.cfg["training"]["max_train_steps"]),
                    scale=self.loss_weights.w_br * self.loss_weights.lambda_br,
                )
                dpo_loss = compute_dpo_loss(
                    self.policy,
                    prefs,
                    self.loss_weights.beta_dpo,
                    scale=self.loss_weights.w_dpo * self.loss_weights.gamma_dpo,
                )
                kl_loss = compute_kl_penalty(
                    self.policy,
                    self.ref_model,
                    flat_trajectories,
                    beta_kl=float(self.cfg["training"]["beta_kl"]),
                    max_train_steps=int(self.cfg["training"]["max_train_steps"]),
                )
                total_loss = base_loss + branch_loss + dpo_loss + kl_loss
                has_grad = any(param.grad is not None for param in self.policy.model.parameters())
                if has_grad:
                    torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), 1.0)
                    self.optimizer.step()
                loss_phase_s = time.time() - epoch_start - base_phase_s - branch_phase_s
                epoch_total_s = time.time() - epoch_start
                if self.console_log:
                    print(
                        f"  [epoch {epoch}] DONE  base={base_phase_s:.0f}s  "
                        f"branch={branch_phase_s:.0f}s  loss={loss_phase_s:.0f}s  "
                        f"total={epoch_total_s:.0f}s",
                        flush=True,
                    )

                payload = {
                    "epoch": epoch,
                    "elapsed_s": round(time.time() - start, 2),
                    "train_score": float(np.mean([t.score for t in flat_trajectories])) if flat_trajectories else 0.0,
                    "train_succ": float(np.mean([1.0 if t.won else 0.0 for t in flat_trajectories])) if flat_trajectories else 0.0,
                    "avg_steps": float(np.mean([len(t) for t in flat_trajectories])) if flat_trajectories else 0.0,
                    "base_loss": float(base_loss.detach().cpu()),
                    "branch_loss": float(branch_loss.detach().cpu()),
                    "dpo_loss": float(dpo_loss.detach().cpu()),
                    "kl_loss": float(kl_loss.detach().cpu()),
                    "total_loss": float(total_loss.detach().cpu()),
                    "group_size": int(self.cfg["training"]["group_size"]),
                    "train_data_size": int(self.cfg["training"]["train_data_size"]),
                    "top_m_actions": int(self.cfg["training"]["top_m_actions"]),
                    "max_train_steps": int(self.cfg["training"]["max_train_steps"]),
                    "w_base": self.loss_weights.w_base,
                    "w_br": self.loss_weights.w_br,
                    "w_dpo": self.loss_weights.w_dpo,
                    "num_branch_candidates": num_branch_candidates,
                    "num_branched_trajectories": len(branched_trajectories),
                    "num_preferences": len(prefs),
                    "branch_group_sizes": branch_group_sizes,
                    "branch_selection_scope": "per_task",
                    "cdb_history": len(self.cdb.history),
                    "pg_response_steps": base_stats["pg_response_steps"] + branch_stats["pg_response_steps"],
                    "pg_action_template_steps": base_stats["pg_action_template_steps"] + branch_stats["pg_action_template_steps"],
                    "pg_branch_action_steps": branch_stats["pg_branch_action_steps"],
                }
                payload.update(metric_acc.to_dict())
                self.logger.log(payload)
                self.wandb_logger.log(payload, step=epoch)
                if self.console_log:
                    print(format_console_metrics(f"[train epoch={epoch}]", payload), flush=True)

                if (epoch + 1) % eval_every == 0:
                    self.evaluate(epoch=epoch)
        finally:
            if self.prompt_response_logger is not None:
                self.prompt_response_logger.close()
            self.wandb_logger.finish()

    @torch.no_grad()
    def evaluate(self, epoch: int) -> None:
        self.policy.set_eval(True)
        eval_games = int(self.cfg["training"]["eval_games"])
        selected = self.env.eval_goal_ids[:eval_games]
        trajectories: list[Trajectory] = []
        metric_acc = RolloutMetrics()
        for task_id in selected:
            trajectory, metrics = collect_one_trajectory(
                env=self.env,
                policy=self.policy,
                task_id=task_id,
                max_steps=int(self.cfg["env"]["max_steps"]),
                history_length=int(self.cfg["model"]["history_length"]),
                success_reward=float(self.cfg["training"]["success_reward"]),
                invalid_action_penalty=float(self.cfg["training"]["invalid_action_penalty"]),
                top_m_actions=int(self.cfg["training"]["top_m_actions"]),
                pr_logger=self.prompt_response_logger,
                iteration=epoch,
                phase="eval_rollout",
            )
            trajectories.append(trajectory)
            metric_acc.format_invalid_count += metrics.format_invalid_count
            metric_acc.action_not_in_available_count += metrics.action_not_in_available_count
            metric_acc.invalid_transition_count += metrics.invalid_transition_count
            metric_acc.total_steps += metrics.total_steps
        payload = {
            "epoch": epoch,
            "eval_score": float(np.mean([t.score for t in trajectories])) if trajectories else 0.0,
            "eval_succ": float(np.mean([1.0 if t.won else 0.0 for t in trajectories])) if trajectories else 0.0,
            "eval_steps": float(np.mean([len(t) for t in trajectories])) if trajectories else 0.0,
        }
        payload.update(metric_acc.to_dict())
        self.logger.log(payload)
        self.wandb_logger.log(payload, step=epoch)
        if self.console_log:
            print(format_console_metrics(f"[eval epoch={epoch}]", payload), flush=True)
        self.policy.set_eval(False)
