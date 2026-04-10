from __future__ import annotations

import random
from typing import Sequence

import torch
import torch.nn.functional as F

from src.models.qwen_policy import QwenPolicy, teacher_forced_logp
from src.utils.types import Preference, StepRecord, Trajectory


def collect_pg_steps(
    trajectories: Sequence[Trajectory],
    advantages: Sequence[float],
    max_steps_per_traj: int,
) -> list[tuple[StepRecord, float, float]]:
    pairs: list[tuple[StepRecord, float, float]] = []
    for trajectory, advantage in zip(trajectories, advantages):
        if abs(advantage) < 1e-12:
            continue
        eligible = [
            step for step in trajectory.steps
            if (step.train_mode == "response" and step.full_response)
            or (step.train_mode == "action_template" and step.train_action_text)
        ]
        eligible = [step for step in eligible if step.train_weight > 0]
        if len(eligible) > max_steps_per_traj:
            branch_steps = [step for step in eligible if step.is_branch_action]
            other_steps = [step for step in eligible if not step.is_branch_action]
            keep = branch_steps[:max_steps_per_traj]
            remaining = max_steps_per_traj - len(keep)
            if remaining > 0 and len(other_steps) > remaining:
                other_steps = random.sample(other_steps, remaining)
            elif remaining <= 0:
                other_steps = []
            eligible = keep + other_steps
        for step in eligible:
            pairs.append((step, float(advantage), float(step.train_weight)))
    return pairs


def compute_pg_loss(
    policy: QwenPolicy,
    trajectories: Sequence[Trajectory],
    advantages: Sequence[float],
    max_train_steps: int,
    scale: float = 1.0,
) -> tuple[torch.Tensor, dict[str, int]]:
    pairs = collect_pg_steps(trajectories, advantages, max_train_steps)
    stats = {
        "pg_response_steps": 0,
        "pg_action_template_steps": 0,
        "pg_branch_action_steps": 0,
        "num_pairs": len(pairs),
    }
    if not pairs:
        return torch.tensor(0.0, device=policy.device), stats
    total_weight = sum(weight for _, _, weight in pairs)
    if total_weight <= 0:
        return torch.tensor(0.0, device=policy.device), stats
    total = 0.0
    for step, advantage, weight in pairs:
        if step.train_mode == "response":
            logp = policy.logp_of_response(step.prompt, step.full_response)
            stats["pg_response_steps"] += 1
        elif step.train_mode == "action_template":
            logp = policy.logp_of_action_template(step.prompt, step.train_action_text)
            stats["pg_action_template_steps"] += 1
            if step.is_branch_action:
                stats["pg_branch_action_steps"] += 1
        else:
            continue
        micro = -(advantage * logp) * (scale * weight / total_weight)
        micro.backward()
        total += (advantage * logp).detach().item() * weight
    return torch.tensor(-total / total_weight, device=policy.device), stats


def compute_dpo_loss(
    policy: QwenPolicy,
    prefs: Sequence[Preference],
    beta_dpo: float,
    scale: float = 1.0,
) -> torch.Tensor:
    if not prefs:
        return torch.tensor(0.0, device=policy.device)
    n = len(prefs)
    total = 0.0
    for pref in prefs:
        logp = policy.logp_of_action_templates(pref.prompt, [pref.a_pos, pref.a_neg])
        diff = logp[0] - logp[1]
        micro = -F.logsigmoid(beta_dpo * diff) * (scale / n)
        micro.backward()
        total += micro.detach().item() * n / scale if scale != 0 else 0.0
    return torch.tensor(total / n, device=policy.device)


def compute_kl_penalty(
    policy: QwenPolicy,
    ref_model,
    trajectories: Sequence[Trajectory],
    beta_kl: float,
    max_train_steps: int,
) -> torch.Tensor:
    if ref_model is None or beta_kl <= 0:
        return torch.tensor(0.0, device=policy.device)
    eligible_steps: list[StepRecord] = []
    for trajectory in trajectories:
        steps = [step for step in trajectory.steps if step.full_response and step.train_mode == "response"]
        if len(steps) > max_train_steps:
            steps = random.sample(steps, max_train_steps)
        eligible_steps.extend(steps)
    if not eligible_steps:
        return torch.tensor(0.0, device=policy.device)

    tokenizer = policy.tokenizer
    max_len = policy.max_prompt_length + policy.max_response_tokens
    n = len(eligible_steps)
    total = 0.0
    for step in eligible_steps:
        logp_cur = policy.logp_of_response(step.prompt, step.full_response)
        with torch.no_grad():
            logp_ref = teacher_forced_logp(
                model=ref_model,
                tokenizer=tokenizer,
                prompt=step.prompt,
                response=step.full_response,
                device=policy.device,
                max_length=max_len,
            )
        kl_approx = logp_cur - logp_ref.detach()
        micro = beta_kl * kl_approx / n
        micro.backward()
        total += kl_approx.detach().item()
    return torch.tensor(total / n, device=policy.device)
