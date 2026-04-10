from __future__ import annotations

from typing import Protocol, Sequence

from src.utils.types import Trajectory


class AdvantageEstimator(Protocol):
    def compute(self, grouped_trajectories: Sequence[Sequence[Trajectory]]) -> list[float]:
        ...
