from __future__ import annotations

from typing import Sequence

import numpy as np

from src.utils.types import Trajectory


def group_normalize(values: Sequence[float], eps: float = 1e-8) -> list[float]:
    v = np.asarray(list(values), dtype=np.float64)
    if len(v) == 0:
        return []
    sigma = float(v.std())
    if sigma < eps:
        order = np.argsort(v)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.linspace(-1.0, 1.0, len(v))
        return ranks.tolist()
    mu = float(v.mean())
    return [float((x - mu) / (sigma + eps)) for x in v]


class SameTaskGRPO:
    def compute(self, grouped_trajectories: Sequence[Sequence[Trajectory]]) -> list[float]:
        advantages: list[float] = []
        for group in grouped_trajectories:
            rewards = [trajectory.terminal_reward for trajectory in group]
            advantages.extend(group_normalize(rewards))
        return advantages
