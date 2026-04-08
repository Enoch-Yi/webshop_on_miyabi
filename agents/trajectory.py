from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass
class StepRecord:
    t: int
    prompt: str
    observation: str
    admissible_actions: List[str]
    action: str
    action_index: int
    logp_action: float
    entropy: float
    done: bool
    full_response: str = ""
    train_mode: str = "none"
    train_action_text: str = ""
    is_branch_action: bool = False
    rescue_level: str = "exact"
    train_weight: float = 1.0
    state_is_clean: bool = True


@dataclass
class Trajectory:
    gamefile: Optional[str]
    task_desc: str
    steps: List[StepRecord]
    terminal_reward: float
    won: bool
    score: float = 0.0
    generated_steps: int = 0
    placeholder_searches: int = 0
    rescued_actions: int = 0
    format_errors: int = 0
    invalid_actions: int = 0
    branch_group_id: Optional[int] = None
    has_semantic_rescue: bool = False
    has_placeholder_search: bool = False
    has_invalid_transition: bool = False

    def __len__(self) -> int:
        return len(self.steps)

    def prefix_actions(self, upto_t_inclusive: int) -> List[str]:
        """Return actions a_0..a_{upto} (inclusive)."""
        return [s.action for s in self.steps[: upto_t_inclusive + 1]]

    def state_at(self, t: int) -> Tuple[str, List[str]]:
        """Return (prompt, admissible_actions) at time t."""
        s = self.steps[t]
        return s.prompt, s.admissible_actions


@dataclass(frozen=True)
class Preference:
    prompt: str
    a_pos: str
    a_neg: str

