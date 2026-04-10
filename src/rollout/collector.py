from __future__ import annotations

from typing import List, Sequence

from src.actions.projection import syntax_only_projection
from src.models.qwen_policy import QwenPolicy
from src.training.logging import PromptResponseLogger
from src.utils.types import RolloutMetrics, StepRecord, Trajectory


def maybe_truncate_actions(actions: list[str], top_m_actions: int) -> list[str]:
    if top_m_actions <= 0 or len(actions) <= top_m_actions:
        return list(actions)
    return list(actions)[:top_m_actions]


def collect_one_trajectory(
    *,
    env,
    policy: QwenPolicy,
    task_id: int,
    max_steps: int,
    history_length: int,
    success_reward: float,
    invalid_action_penalty: float,
    top_m_actions: int,
    pr_logger: PromptResponseLogger | None = None,
    iteration: int = 0,
    phase: str = "base_rollout",
) -> tuple[Trajectory, RolloutMetrics]:
    observation, _info = env.reset(task_id)
    task_desc = env.instruction_text
    history: list[tuple[str, str]] = []
    steps: list[StepRecord] = []
    metrics = RolloutMetrics()
    done = False
    t = 0
    final_score = 0.0
    invalid_count = 0

    while (not done) and (t < max_steps):
        available_actions_full = env.format_available_actions()
        available_actions_used = maybe_truncate_actions(available_actions_full, top_m_actions)
        prompt = policy.build_prompt(
            task_desc=task_desc,
            observation=observation,
            history=history,
            available_actions=available_actions_used,
            step_count=t,
        )
        response = policy.generate(prompt)
        projection = syntax_only_projection(response, available_actions_used)
        if pr_logger is not None:
            pr_logger.log(
                iteration=iteration,
                step=t,
                phase=phase,
                prompt=prompt,
                response=response,
                action=projection.executed_action,
            )
        obs_before = env.observation
        next_observation, reward, done, info = env.step(projection.executed_action)
        obs_after = env.observation

        invalid_transition = (obs_before == obs_after) and (not done)
        is_invalid = (
            (not projection.is_projection_valid)
            or (not projection.action_in_available)
            or invalid_transition
        )
        if not projection.is_projection_valid:
            metrics.format_invalid_count += 1
        if not projection.action_in_available:
            metrics.action_not_in_available_count += 1
        if invalid_transition:
            metrics.invalid_transition_count += 1
        metrics.total_steps += 1
        if is_invalid:
            invalid_count += 1

        steps.append(
            StepRecord(
                t=t,
                prompt=prompt,
                observation=observation,
                available_actions_full=list(available_actions_full),
                available_actions_used=list(available_actions_used),
                generated_action=projection.generated_action,
                executed_action=projection.executed_action,
                full_response=response,
                is_projection_valid=projection.is_projection_valid,
                is_action_in_available=projection.action_in_available,
                is_invalid=is_invalid,
                reward=reward,
                done=done,
                train_mode="response",
                train_action_text="",
                train_weight=1.0,
                is_branch_action=False,
                state_is_clean=all(not prev.is_invalid for prev in steps),
            )
        )
        history.append((observation, projection.executed_action))
        observation = next_observation
        if done:
            final_score = reward
        t += 1

    won = final_score >= 1.0
    terminal_reward = (success_reward if won else 0.0) + invalid_action_penalty * invalid_count
    trajectory = Trajectory(
        task_id=task_id,
        task_desc=task_desc,
        steps=steps,
        terminal_reward=terminal_reward,
        score=final_score,
        won=won,
        invalid_action_count=invalid_count,
    )
    return trajectory, metrics


def replay_to_state_and_branch(
    *,
    env,
    policy: QwenPolicy,
    task_id: int,
    prefix_actions: Sequence[str],
    branch_action: str,
    branch_group_id: int | None,
    max_steps: int,
    success_reward: float,
    invalid_action_penalty: float,
    top_m_actions: int,
    pr_logger: PromptResponseLogger | None = None,
    iteration: int = 0,
    phase: str = "branch_rollout",
) -> tuple[Trajectory, RolloutMetrics]:
    observation, _info = env.reset(task_id)
    task_desc = env.instruction_text
    history: list[tuple[str, str]] = []
    steps: list[StepRecord] = []
    metrics = RolloutMetrics()
    final_score = 0.0
    invalid_count = 0
    done = False

    for t, action in enumerate(prefix_actions):
        available_actions_full = env.format_available_actions()
        available_actions_used = maybe_truncate_actions(available_actions_full, top_m_actions)
        prompt = policy.build_prompt(
            task_desc=task_desc,
            observation=observation,
            history=history,
            available_actions=available_actions_used,
            step_count=t,
        )
        obs_before = env.observation
        next_observation, reward, done, _info = env.step(action)
        obs_after = env.observation
        invalid_transition = (obs_before == obs_after) and (not done)
        action_not_in_available = action not in available_actions_used
        if action_not_in_available:
            metrics.action_not_in_available_count += 1
        if invalid_transition:
            metrics.invalid_transition_count += 1
        if action_not_in_available or invalid_transition:
            invalid_count += 1
        metrics.total_steps += 1
        steps.append(
            StepRecord(
                t=t,
                prompt=prompt,
                observation=observation,
                available_actions_full=list(available_actions_full),
                available_actions_used=list(available_actions_used),
                generated_action=action,
                executed_action=action,
                full_response="",
                is_projection_valid=True,
                is_action_in_available=not action_not_in_available,
                is_invalid=invalid_transition or action_not_in_available,
                reward=reward,
                done=done,
                train_mode="none",
                train_action_text="",
                train_weight=0.0,
                is_branch_action=False,
                state_is_clean=all(not prev.is_invalid for prev in steps),
            )
        )
        history.append((observation, action))
        observation = next_observation
        if done:
            final_score = reward
            trajectory = Trajectory(
                task_id=task_id,
                task_desc=task_desc,
                steps=steps,
                terminal_reward=(success_reward if final_score >= 1.0 else 0.0) + invalid_action_penalty * invalid_count,
                score=final_score,
                won=final_score >= 1.0,
                invalid_action_count=invalid_count,
                branch_group_id=branch_group_id,
            )
            return trajectory, metrics

    t = len(prefix_actions)
    available_actions_full = env.format_available_actions()
    available_actions_used = maybe_truncate_actions(available_actions_full, top_m_actions)
    prompt = policy.build_prompt(
        task_desc=task_desc,
        observation=observation,
        history=history,
        available_actions=available_actions_used,
        step_count=t,
    )
    obs_before = env.observation
    next_observation, reward, done, _info = env.step(branch_action)
    obs_after = env.observation
    if pr_logger is not None:
        pr_logger.log(
            iteration=iteration,
            step=t,
            phase=phase,
            prompt=prompt,
            response="",
            action=branch_action,
        )
    invalid_transition = (obs_before == obs_after) and (not done)
    branch_invalid = invalid_transition or branch_action not in available_actions_used
    if branch_action not in available_actions_used:
        metrics.action_not_in_available_count += 1
    if invalid_transition:
        metrics.invalid_transition_count += 1
    if branch_invalid:
        invalid_count += 1
    metrics.total_steps += 1
    steps.append(
        StepRecord(
            t=t,
            prompt=prompt,
            observation=observation,
            available_actions_full=list(available_actions_full),
            available_actions_used=list(available_actions_used),
            generated_action=branch_action,
            executed_action=branch_action,
            full_response="",
            is_projection_valid=True,
            is_action_in_available=branch_action in available_actions_used,
            is_invalid=branch_invalid,
            reward=reward,
            done=done,
            train_mode="action_template",
            train_action_text=branch_action,
            train_weight=1.0 if all(not prev.is_invalid for prev in steps) else 0.0,
            is_branch_action=True,
            state_is_clean=all(not prev.is_invalid for prev in steps),
        )
    )
    history.append((observation, branch_action))
    observation = next_observation
    if done:
        final_score = reward
    t += 1

    while (not done) and (t < max_steps):
        available_actions_full = env.format_available_actions()
        available_actions_used = maybe_truncate_actions(available_actions_full, top_m_actions)
        prompt = policy.build_prompt(
            task_desc=task_desc,
            observation=observation,
            history=history,
            available_actions=available_actions_used,
            step_count=t,
        )
        response = policy.generate(prompt)
        projection = syntax_only_projection(response, available_actions_used)
        if pr_logger is not None:
            pr_logger.log(
                iteration=iteration,
                step=t,
                phase=phase,
                prompt=prompt,
                response=response,
                action=projection.executed_action,
            )
        obs_before = env.observation
        next_observation, reward, done, _info = env.step(projection.executed_action)
        obs_after = env.observation
        invalid_transition = (obs_before == obs_after) and (not done)
        is_invalid = (
            (not projection.is_projection_valid)
            or (not projection.action_in_available)
            or invalid_transition
        )
        if not projection.is_projection_valid:
            metrics.format_invalid_count += 1
        if not projection.action_in_available:
            metrics.action_not_in_available_count += 1
        if invalid_transition:
            metrics.invalid_transition_count += 1
        metrics.total_steps += 1
        if is_invalid:
            invalid_count += 1
        steps.append(
            StepRecord(
                t=t,
                prompt=prompt,
                observation=observation,
                available_actions_full=list(available_actions_full),
                available_actions_used=list(available_actions_used),
                generated_action=projection.generated_action,
                executed_action=projection.executed_action,
                full_response=response,
                is_projection_valid=projection.is_projection_valid,
                is_action_in_available=projection.action_in_available,
                is_invalid=is_invalid,
                reward=reward,
                done=done,
                train_mode="response",
                train_action_text="",
                train_weight=1.0,
                is_branch_action=False,
                state_is_clean=all(not prev.is_invalid for prev in steps),
            )
        )
        history.append((observation, projection.executed_action))
        observation = next_observation
        if done:
            final_score = reward
        t += 1

    won = final_score >= 1.0
    trajectory = Trajectory(
        task_id=task_id,
        task_desc=task_desc,
        steps=steps,
        terminal_reward=(success_reward if won else 0.0) + invalid_action_penalty * invalid_count,
        score=final_score,
        won=won,
        invalid_action_count=invalid_count,
        branch_group_id=branch_group_id,
    )
    return trajectory, metrics
