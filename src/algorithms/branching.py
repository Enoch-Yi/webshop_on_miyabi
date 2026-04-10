from __future__ import annotations

import random
from typing import Sequence

import numpy as np

from src.algorithms.cdb import EmbeddingFeatureExtractor, FGTSCDB
from src.algorithms.grpo import group_normalize
from src.utils.types import BranchCandidate, Trajectory


def select_branch_positions(
    trajectories: Sequence[Trajectory],
    *,
    num_branch_states: int,
    state_selection_mode: str,
    tau_s: float | None,
    lambda_u: float,
    lambda_i: float,
    cdb: FGTSCDB,
    feat_ext: EmbeddingFeatureExtractor,
) -> list[BranchCandidate]:
    candidates: list[BranchCandidate] = []
    for i, trajectory in enumerate(trajectories):
        T = max(len(trajectory), 1)
        for step in trajectory.steps:
            if len(step.available_actions_used) < 2:
                continue
            if not step.state_is_clean:
                continue
            importance = 1.0 - (step.t / float(T))
            ctx = feat_ext.encode_context(step.prompt)
            afeats = feat_ext.all_action_features(ctx, step.available_actions_used)
            uncertainty = cdb.uncertainty(afeats)
            score = lambda_u * uncertainty + lambda_i * importance
            candidates.append(
                BranchCandidate(
                    traj_idx=i,
                    step_idx=step.t,
                    score=float(score),
                    context=ctx,
                    action_features=afeats,
                    prompt=step.prompt,
                    available_actions=list(step.available_actions_used),
                )
            )
    if state_selection_mode == "random":
        random.shuffle(candidates)
        return candidates[:num_branch_states]
    candidates.sort(key=lambda item: item.score, reverse=True)
    if state_selection_mode == "tau_s":
        if tau_s is None:
            raise ValueError("state_selection_mode='tau_s' requires tau_s to be set.")
        return [item for item in candidates if item.score >= tau_s]
    if state_selection_mode != "top_k":
        raise ValueError(f"Unknown state_selection_mode: {state_selection_mode}")
    return candidates[:num_branch_states]


def group_normalize_branched(
    trajectories: Sequence[Trajectory],
) -> tuple[list[float], list[int]]:
    grouped: dict[int, list[tuple[int, float]]] = {}
    for idx, trajectory in enumerate(trajectories):
        gid = trajectory.branch_group_id if trajectory.branch_group_id is not None else idx
        grouped.setdefault(gid, []).append((idx, trajectory.terminal_reward))

    advantages = [0.0] * len(trajectories)
    group_sizes: list[int] = []
    for items in grouped.values():
        group_sizes.append(len(items))
        advs = group_normalize([reward for _, reward in items])
        for (idx, _reward), adv in zip(items, advs):
            advantages[idx] = adv
    return advantages, group_sizes
