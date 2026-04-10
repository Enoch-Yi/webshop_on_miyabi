from __future__ import annotations

from src.utils.types import Preference, Trajectory


def preference_from_duel(
    *,
    prompt: str,
    action_a: str,
    action_b: str,
    traj_a: Trajectory,
    traj_b: Trajectory,
) -> Preference | None:
    if not traj_a.is_clean() or not traj_b.is_clean():
        return None
    if traj_a.terminal_reward == traj_b.terminal_reward:
        return None
    if traj_a.terminal_reward > traj_b.terminal_reward:
        return Preference(prompt=prompt, a_pos=action_a, a_neg=action_b)
    return Preference(prompt=prompt, a_pos=action_b, a_neg=action_a)
