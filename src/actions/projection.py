from __future__ import annotations

import re
from dataclasses import dataclass


ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.IGNORECASE | re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


@dataclass
class ProjectionResult:
    generated_action: str
    executed_action: str
    is_projection_valid: bool
    has_think_block: bool
    contains_cjk: bool
    action_in_available: bool


def syntax_only_projection(response_text: str, available_actions: list[str]) -> ProjectionResult:
    original = response_text.strip()
    lowered = original.lower()

    match = ACTION_RE.search(lowered)
    executed_action = match.group(1).strip() if match else lowered[-20:].strip()
    has_think_block = THINK_RE.search(original) is not None
    contains_cjk = re.search(r"[\u4e00-\u9fff]", original) is not None
    is_projection_valid = match is not None and has_think_block and not contains_cjk
    action_in_available = executed_action in [a.lower() for a in available_actions]

    return ProjectionResult(
        generated_action=executed_action,
        executed_action=executed_action,
        is_projection_valid=is_projection_valid,
        has_think_block=has_think_block,
        contains_cjk=contains_cjk,
        action_in_available=action_in_available,
    )
