from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StepRecord:
    t: int
    prompt: str
    observation: str
    available_actions_full: List[str]
    available_actions_used: List[str]
    generated_action: str
    executed_action: str
    full_response: str
    is_projection_valid: bool
    is_action_in_available: bool
    is_invalid: bool
    reward: float
    done: bool
    train_mode: str = "response"
    train_action_text: str = ""
    train_weight: float = 1.0
    is_branch_action: bool = False
    state_is_clean: bool = True


@dataclass
class Trajectory:
    task_id: int
    task_desc: str
    steps: List[StepRecord] = field(default_factory=list)
    terminal_reward: float = 0.0
    score: float = 0.0
    won: bool = False
    invalid_action_count: int = 0
    branch_group_id: Optional[int] = None

    def __len__(self) -> int:
        return len(self.steps)

    def prefix_actions(self, upto_t_inclusive: int) -> List[str]:
        return [step.executed_action for step in self.steps[: upto_t_inclusive + 1]]

    def state_at(self, t: int) -> tuple[str, List[str]]:
        step = self.steps[t]
        return step.prompt, step.available_actions_used

    def is_clean(self) -> bool:
        return all(step.state_is_clean and not step.is_invalid for step in self.steps)


@dataclass
class RolloutMetrics:
    format_invalid_count: int = 0
    action_not_in_available_count: int = 0
    invalid_transition_count: int = 0
    total_steps: int = 0

    def to_dict(self) -> Dict[str, Any]:
        denom = max(self.total_steps, 1)
        return {
            "format_invalid_count": self.format_invalid_count,
            "action_not_in_available_count": self.action_not_in_available_count,
            "invalid_transition_count": self.invalid_transition_count,
            "total_steps": self.total_steps,
            "format_invalid_rate": self.format_invalid_count / denom,
            "action_not_in_available_rate": self.action_not_in_available_count / denom,
            "invalid_transition_rate": self.invalid_transition_count / denom,
        }


@dataclass
class BranchingSettings:
    B: int = 0
    K: int = 0
    state_selection_mode: str = "top_k"
    action_pair_mode: str = "cdb"
    tau_s: Optional[float] = None
    cdb_p: int = 128
    cdb_eta: float = 1.0
    cdb_mu: float = 0.1
    cdb_delta: float = 0.01
    cdb_sigma0: float = 1.0
    lambda_u: float = 1.0
    lambda_i: float = 1.0


@dataclass
class LossWeights:
    w_base: float = 1.0
    w_br: float = 0.0
    w_dpo: float = 0.0
    beta_dpo: float = 0.1
    lambda_br: float = 1.0
    gamma_dpo: float = 1.0


@dataclass(frozen=True)
class Preference:
    prompt: str
    a_pos: str
    a_neg: str


@dataclass
class BranchCandidate:
    traj_idx: int
    step_idx: int
    score: float
    context: Any
    action_features: Any
    prompt: str
    available_actions: List[str]
