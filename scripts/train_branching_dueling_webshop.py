#!/usr/bin/env python
"""Branching-Dueling training for LLM agents on WebShop.

Migrated from ALFWorld version. Uses a **generative** policy
(HCAPO-style <think>/<action> prompting) with the WebShop e-commerce
environment.

Key differences from ALFWorld:
  - Environment: WebShop (text mode) instead of TextWorld
  - Actions: search[query] and click[element] instead of go/take/open
  - Admissible actions parsed from page HTML (not provided by env)
  - Reward: continuous score 0-1 (not binary)
  - Max steps: 15 (not 50)
  - Max prompt length: 4096 (not 2048)
  - Evaluation reports Score + Success Rate

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_branching_dueling_webshop.py \\
        configs/webshop_config.yaml --model_name Qwen/Qwen2.5-1.5B-Instruct \\
        --iters 150
"""
from __future__ import annotations

import datetime
import difflib
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml
import time as _time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from webshop_env.envs.web_agent_text_env import WebAgentTextEnv

from agents.qwen_policy import QwenGenerativePolicy, parse_action
from agents.trajectory import Preference, StepRecord, Trajectory
from agents.fgts_cdb import EmbeddingFeatureExtractor, FGTSCDB


# ---------------------------------------------------------------------------
# Prompt / Response logger  (rotates files at ~1 MB)
# ---------------------------------------------------------------------------

class PromptResponseLogger:
    MAX_BYTES = 1_000_000

    def __init__(self, save_dir: str, tag: str) -> None:
        self._dir = save_dir
        self._tag = tag
        self._seq = 0
        self._size = 0
        self._fp = None
        os.makedirs(save_dir, exist_ok=True)
        self._open_next()

    def _fname(self) -> str:
        return os.path.join(
            self._dir,
            f"training_log_{self._tag}_{self._seq:04d}.txt",
        )

    def _open_next(self) -> None:
        if self._fp is not None:
            self._fp.close()
        self._fp = open(self._fname(), "a", encoding="utf-8")
        self._size = self._fp.tell()

    def _maybe_rotate(self) -> None:
        if self._size >= self.MAX_BYTES:
            self._seq += 1
            self._open_next()

    def log(
        self,
        *,
        iteration: int,
        step: int,
        phase: str,
        prompt: str,
        response: str,
        action: str,
        won: Optional[bool] = None,
    ) -> None:
        header = (
            f"{'='*72}\n"
            f"[iter={iteration}  step={step}  phase={phase}]\n"
            f"{'='*72}\n"
        )
        body = (
            f">>> PROMPT <<<\n{prompt}\n\n"
            f">>> RESPONSE <<<\n{response}\n\n"
            f">>> ACTION <<<\n{action}\n"
        )
        if won is not None:
            body += f">>> WON={won}\n"
        entry = header + body + "\n"
        data = entry.encode("utf-8", errors="replace")
        self._fp.write(entry)
        self._fp.flush()
        self._size += len(data)
        self._maybe_rotate()

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_yaml_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def group_normalize(values: Sequence[float], eps: float = 1e-8) -> List[float]:
    """Group-normalised advantages (Eqs. 1, 2)."""
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


# ---------------------------------------------------------------------------
# WebShop environment helpers
# ---------------------------------------------------------------------------

def make_webshop_env(
    num_products: Optional[int] = None,
    human_goals: bool = True,
    observation_mode: str = "text",
) -> WebAgentTextEnv:
    """Create a WebShop text environment."""
    _data_dir = os.environ.get("WEBSHOP_DATA_DIR",
                               os.path.join(_PROJECT_ROOT, "webshop_data"))
    if num_products is not None and num_products <= 1000:
        file_path = os.path.join(_data_dir, "items_shuffle_1000.json")
    else:
        file_path = os.path.join(_data_dir, "items_shuffle.json")
    env = WebAgentTextEnv(
        observation_mode=observation_mode,
        file_path=file_path,
        num_products=num_products,
        human_goals=human_goals,
    )
    return env


def get_num_goals(env: WebAgentTextEnv) -> int:
    return len(env.server.goals)


def get_admissible_actions(env: WebAgentTextEnv) -> List[str]:
    """Extract admissible actions from the current WebShop page state."""
    avail = env.get_available_actions()
    actions = []
    if avail["has_search_bar"]:
        actions.append("search[<query>]")
    for clickable in avail["clickables"]:
        if clickable.lower() != "search":
            actions.append(f"click[{clickable}]")
    return actions


_WEBSHOP_ACTION_RE = re.compile(r"(search|click)\[(.+)\]", re.IGNORECASE)


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.strip().split())


def _canonicalize_generated_action(generated: str) -> str:
    text = _normalize_whitespace(generated)
    text = re.sub(r"^click\s*\[\s*", "click[", text, flags=re.IGNORECASE)
    text = re.sub(r"^search\s*\[\s*", "search[", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\]\s*$", "]", text)
    return text


def match_webshop_action(
    generated: str,
    admissible: Sequence[str],
    no_rescue: bool = False,
) -> Tuple[str, int, str]:
    """Match a generated WebShop action to admissible actions.

    Returns (matched_action, index_in_admissible, rescue_level).

    rescue_level:
      - "exact": exact semantic match without repair
      - "syntax": formatting/casing cleanup only
      - "semantic": fuzzy match, placeholder search, or fallback

    If no_rescue=True, only do exact/syntax matching. If no match,
    return the raw generated action as-is (the env will handle it).
    """
    raw = generated.strip()
    gen = _canonicalize_generated_action(raw)

    m = _WEBSHOP_ACTION_RE.search(gen)
    if m:
        action_type = m.group(1).lower()
        action_arg = m.group(2).strip()

        if action_type == "search" and action_arg and action_arg != "<query>":
            action_str = f"search[{action_arg}]"
            for i, a in enumerate(admissible):
                if a.startswith("search["):
                    level = "exact" if action_str == raw else "syntax"
                    return action_str, i, level
            level = "exact" if action_str == raw else "syntax"
            return action_str, -1, level

        if action_type == "click":
            click_actions = [a for a in admissible if a.startswith("click[")]
            click_labels = []
            for ca in click_actions:
                cm = re.match(r"click\[(.+)\]", ca)
                if cm:
                    click_labels.append(cm.group(1))

            arg_lower = action_arg.lower()
            for i, label in enumerate(click_labels):
                if label.lower() == arg_lower:
                    idx = admissible.index(click_actions[i])
                    level = "exact" if click_actions[i] == raw else "syntax"
                    return click_actions[i], idx, level

            if no_rescue:
                return f"click[{action_arg}]", -1, "semantic"

            matches = difflib.get_close_matches(
                arg_lower, [l.lower() for l in click_labels], n=1, cutoff=0.4
            )
            if matches:
                label_idx = [l.lower() for l in click_labels].index(matches[0])
                idx = admissible.index(click_actions[label_idx])
                return click_actions[label_idx], idx, "semantic"

            if click_actions:
                idx = admissible.index(click_actions[0])
                return click_actions[0], idx, "semantic"

    if no_rescue:
        return gen if gen else "noop", -1, "semantic"

    if admissible:
        idx = random.randrange(len(admissible))
        return admissible[idx], idx, "semantic"
    return gen, -1, "semantic"


def _step_training_policy(
    *,
    rescue_level: str,
    had_format_error: bool,
    matched_action: str,
    is_invalid_transition: bool,
    state_is_clean: bool,
    allow_syntax_training: bool,
    syntax_rescue_weight: float,
    allow_semantic_training: bool,
    semantic_rescue_weight: float,
    is_branch_action: bool = False,
) -> Tuple[str, float]:
    """DMPO-inspired data discipline for training samples.

    - exact clean steps can train all losses
    - syntax-only repairs receive a reduced action-template weight
    - semantic rescues and invalid/no-op transitions are execution-only
    """
    if is_branch_action:
        return "action_template", 1.0

    if is_invalid_transition or matched_action == "search[<query>]" or not state_is_clean:
        return "none", 0.0

    if rescue_level == "exact" and not had_format_error:
        return "response", 1.0

    if rescue_level == "syntax":
        if allow_syntax_training and syntax_rescue_weight > 0:
            return "action_template", syntax_rescue_weight
        return "none", 0.0

    if rescue_level == "semantic":
        if allow_semantic_training and semantic_rescue_weight > 0:
            return "action_template", semantic_rescue_weight
        return "none", 0.0

    return "none", 0.0


def _trajectory_is_clean(traj: Trajectory) -> bool:
    return not (
        traj.has_semantic_rescue
        or traj.has_placeholder_search
        or traj.has_invalid_transition
    )


def _is_invalid_action(obs_before: str, obs_after: str) -> bool:
    """Detect invalid actions by checking if observation changed."""
    return obs_before.strip() == obs_after.strip()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BranchConfig:
    num_base_rollouts: int = 8       # N
    num_branch_states: int = 4       # |B_q|
    num_duels_per_state: int = 2     # K
    top_m_actions: int = 0           # 0 = no truncation; closer to paper's A(s_t)
    lambda_u: float = 1.0            # Eq. 11
    lambda_i: float = 1.0            # Eq. 11
    tau_s: Optional[float] = None    # Eq. 12; None falls back to top-B
    state_selection_mode: str = "top_k"  # "top_k", "tau_s", or "random"
    action_pair_mode: str = "cdb"        # "cdb" or "random"
    beta_dpo: float = 0.1
    lambda_br: float = 1.0
    gamma_dpo: float = 1.0
    entropy_coeff: float = 0.01
    success_reward: float = 10.0
    step_penalty_coeff: float = 0.02
    invalid_action_penalty: float = -0.1
    allow_syntax_training: bool = True
    syntax_rescue_weight: float = 0.25
    allow_semantic_training: bool = False
    semantic_rescue_weight: float = 0.0
    w_base: float = 1.0
    w_br: float = 1.0
    w_dpo: float = 1.0
    queries_per_step: int = 1            # number of different tasks per training step
    no_rescue: bool = False              # disable rescue, pass raw actions to env


def truncate_actions(admissible: Sequence[str], top_m: int) -> List[str]:
    if top_m <= 0 or len(admissible) <= top_m:
        return list(admissible)
    return list(admissible)[:top_m]


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

def _shaped_reward(
    score: float, cfg: BranchConfig, num_invalid: int = 0
) -> float:
    """Use continuous WebShop score as trajectory return proxy.

    The paper leaves R(tau) abstract; on WebShop the environment already
    provides a dense terminal score in [0, 1]. Preserving that ordering keeps
    local duels and group-normalized PG faithful to the underlying task reward.
    """
    clipped = max(0.0, min(float(score), 1.0))
    base = clipped * cfg.success_reward
    return base + cfg.invalid_action_penalty * num_invalid


# ---------------------------------------------------------------------------
# Rollout (generative)
# ---------------------------------------------------------------------------

def rollout_one(
    *,
    env: WebAgentTextEnv,
    policy: QwenGenerativePolicy,
    task_id: int,
    max_steps: int,
    top_m_actions: int,
    cfg: BranchConfig,
    pr_logger: Optional[PromptResponseLogger] = None,
    iteration: int = 0,
) -> Trajectory:
    """Run one episode on a WebShop task."""
    obs, _ = env.reset(session=task_id)
    task_desc = env.instruction_text
    admissible = get_admissible_actions(env)

    history: List[Tuple[str, str]] = []
    steps: List[StepRecord] = []
    done = False
    t = 0
    final_score = 0.0
    num_invalid = 0
    generated_steps = 0
    placeholder_searches = 0
    rescued_actions = 0
    format_errors = 0
    has_semantic_rescue = False
    has_placeholder_search = False
    has_invalid_transition = False
    state_is_clean = True

    while (not done) and (t < max_steps):
        admissible_t = truncate_actions(admissible, top_m_actions)
        observation = env.observation

        prompt = policy.build_prompt(
            task_desc=task_desc,
            observation=observation,
            history=history,
            admissible_actions=admissible_t,
            step_count=t,
        )
        _, full_resp, _, logp_a = policy.generate_action(
            prompt, admissible_t,
        )

        parsed = parse_action(full_resp)
        generated_steps += 1
        had_format_error = parsed is None
        if parsed is None:
            format_errors += 1
            raw = full_resp.strip().split("\n")[-1].strip()
            parsed = raw if raw else ""
        matched, a_idx, rescue_level = match_webshop_action(
            parsed, admissible_t, no_rescue=cfg.no_rescue,
        )
        if matched == "search[<query>]":
            placeholder_searches += 1
            has_placeholder_search = True
        if rescue_level != "exact":
            rescued_actions += 1
        if rescue_level == "semantic":
            has_semantic_rescue = True

        if pr_logger is not None:
            pr_logger.log(
                iteration=iteration, step=t, phase="base_rollout",
                prompt=prompt, response=full_resp, action=matched,
            )

        obs_before = env.observation
        next_state, reward, done, info = env.step(matched)
        obs_after = env.observation

        invalid_transition = _is_invalid_action(obs_before, obs_after) and not done
        if invalid_transition:
            num_invalid += 1
            has_invalid_transition = True

        if done:
            final_score = reward

        if cfg.no_rescue:
            train_mode = "response"
            train_weight = 1.0
        else:
            train_mode, train_weight = _step_training_policy(
                rescue_level=rescue_level,
                had_format_error=had_format_error,
                matched_action=matched,
                is_invalid_transition=invalid_transition,
                state_is_clean=state_is_clean,
                allow_syntax_training=cfg.allow_syntax_training,
                syntax_rescue_weight=cfg.syntax_rescue_weight,
                allow_semantic_training=cfg.allow_semantic_training,
                semantic_rescue_weight=cfg.semantic_rescue_weight,
            )

        steps.append(
            StepRecord(
                t=t,
                prompt=prompt,
                observation=observation,
                admissible_actions=list(admissible_t),
                action=matched,
                action_index=a_idx,
                logp_action=float(logp_a),
                entropy=0.0,
                done=done,
                full_response=full_resp,
                train_mode=train_mode,
                train_action_text=matched if train_mode == "action_template" else "",
                rescue_level=rescue_level,
                train_weight=train_weight,
                state_is_clean=state_is_clean,
            )
        )
        history.append((observation, matched))
        admissible = get_admissible_actions(env)
        state_is_clean = state_is_clean and rescue_level != "semantic" and not invalid_transition and matched != "search[<query>]"
        t += 1

    won = final_score >= 1.0
    terminal_reward = _shaped_reward(final_score, cfg, num_invalid=num_invalid)

    if pr_logger is not None:
        pr_logger.log(
            iteration=iteration, step=t, phase="base_rollout_end",
            prompt="(episode ended)", response=f"score={final_score:.4f}",
            action="", won=won,
        )
    return Trajectory(
        gamefile=str(task_id),
        task_desc=task_desc,
        steps=steps,
        terminal_reward=terminal_reward,
        won=won,
        score=final_score,
        generated_steps=generated_steps,
        placeholder_searches=placeholder_searches,
        rescued_actions=rescued_actions,
        format_errors=format_errors,
        invalid_actions=num_invalid,
        has_semantic_rescue=has_semantic_rescue,
        has_placeholder_search=has_placeholder_search,
        has_invalid_transition=has_invalid_transition,
    )


# ---------------------------------------------------------------------------
# Branching rollout: replay prefix -> take branch_action -> continue gen-pi
# ---------------------------------------------------------------------------

def replay_to_state_and_branch(
    *,
    env: WebAgentTextEnv,
    task_id: int,
    prefix_actions: Sequence[str],
    branch_action: str,
    branch_group_id: Optional[int],
    policy: QwenGenerativePolicy,
    max_steps: int,
    top_m_actions: int,
    cfg: BranchConfig,
    pr_logger: Optional[PromptResponseLogger] = None,
    iteration: int = 0,
) -> Trajectory:
    """Replay prefix actions on the same task, then branch with a new action."""
    obs, _ = env.reset(session=task_id)
    task_desc = env.instruction_text
    admissible = get_admissible_actions(env)

    history: List[Tuple[str, str]] = []
    steps: List[StepRecord] = []
    final_score = 0.0
    num_invalid = 0
    done = False
    generated_steps = 0
    placeholder_searches = 0
    rescued_actions = 0
    format_errors = 0
    has_semantic_rescue = False
    has_placeholder_search = False
    has_invalid_transition = False
    state_is_clean = True

    # --- replay prefix (no generation) ---
    for t, a in enumerate(prefix_actions):
        admissible_t = truncate_actions(admissible, top_m_actions)
        observation = env.observation
        prompt = policy.build_prompt(
            task_desc=task_desc,
            observation=observation,
            history=history,
            admissible_actions=admissible_t,
            step_count=t,
        )
        a_idx = admissible_t.index(a) if a in admissible_t else -1
        steps.append(
            StepRecord(
                t=t, prompt=prompt, observation=observation,
                admissible_actions=list(admissible_t),
                action=a, action_index=a_idx,
                logp_action=0.0, entropy=0.0, done=False,
                full_response="",
                train_mode="none",
                train_action_text="",
                rescue_level="exact",
                train_weight=0.0,
                state_is_clean=state_is_clean,
            )
        )
        obs_before = env.observation
        next_state, reward, done, info = env.step(a)
        obs_after = env.observation
        invalid_transition = _is_invalid_action(obs_before, obs_after) and not done
        if invalid_transition:
            num_invalid += 1
            has_invalid_transition = True
            state_is_clean = False
        if done:
            final_score = reward
        history.append((observation, a))
        admissible = get_admissible_actions(env)
        if done:
            won = final_score >= 1.0
            rw = _shaped_reward(final_score, cfg, num_invalid=num_invalid)
            return Trajectory(
                gamefile=str(task_id), task_desc=task_desc,
                steps=steps, terminal_reward=rw, won=won,
                score=final_score,
                generated_steps=generated_steps,
                placeholder_searches=placeholder_searches,
                rescued_actions=rescued_actions,
                format_errors=format_errors,
                invalid_actions=num_invalid,
                branch_group_id=branch_group_id,
                has_semantic_rescue=has_semantic_rescue,
                has_placeholder_search=has_placeholder_search,
                has_invalid_transition=has_invalid_transition,
            )

    # --- take forced branch action ---
    t = len(prefix_actions)
    admissible_t = truncate_actions(admissible, top_m_actions)
    observation = env.observation
    prompt = policy.build_prompt(
        task_desc=task_desc, observation=observation,
        history=history, admissible_actions=admissible_t,
        step_count=t,
    )
    a_idx = admissible_t.index(branch_action) if branch_action in admissible_t else -1
    steps.append(
        StepRecord(
            t=t, prompt=prompt, observation=observation,
            admissible_actions=list(admissible_t),
            action=branch_action, action_index=a_idx,
            logp_action=0.0, entropy=0.0, done=False,
            full_response="",
            train_mode="action_template",
            train_action_text=branch_action,
            is_branch_action=True,
            rescue_level="exact",
            train_weight=1.0 if state_is_clean else 0.0,
            state_is_clean=state_is_clean,
        )
    )
    obs_before = env.observation
    next_state, reward, done, info = env.step(branch_action)
    obs_after = env.observation
    invalid_transition = _is_invalid_action(obs_before, obs_after) and not done
    if invalid_transition:
        num_invalid += 1
        has_invalid_transition = True
    if done:
        final_score = reward
    history.append((observation, branch_action))
    admissible = get_admissible_actions(env)
    state_is_clean = state_is_clean and not invalid_transition
    t += 1

    # --- continue with generative policy ---
    while (not done) and (t < max_steps):
        admissible_t = truncate_actions(admissible, top_m_actions)
        observation = env.observation

        prompt = policy.build_prompt(
            task_desc=task_desc, observation=observation,
            history=history, admissible_actions=admissible_t,
            step_count=t,
        )
        _, full_resp, _, logp_a = policy.generate_action(
            prompt, admissible_t,
        )
        parsed = parse_action(full_resp)
        generated_steps += 1
        had_format_error = parsed is None
        if parsed is None:
            format_errors += 1
            raw = full_resp.strip().split("\n")[-1].strip()
            parsed = raw if raw else ""
        matched, a_idx, rescue_level = match_webshop_action(
            parsed, admissible_t, no_rescue=cfg.no_rescue,
        )
        if matched == "search[<query>]":
            placeholder_searches += 1
            has_placeholder_search = True
        if rescue_level != "exact":
            rescued_actions += 1
        if rescue_level == "semantic":
            has_semantic_rescue = True

        if pr_logger is not None:
            pr_logger.log(
                iteration=iteration, step=t, phase="branch_rollout",
                prompt=prompt, response=full_resp, action=matched,
            )

        obs_before = env.observation
        next_state, reward, done, info = env.step(matched)
        obs_after = env.observation
        invalid_transition = _is_invalid_action(obs_before, obs_after) and not done
        if invalid_transition:
            num_invalid += 1
            has_invalid_transition = True
        if done:
            final_score = reward

        if cfg.no_rescue:
            train_mode = "response"
            train_weight = 1.0
        else:
            train_mode, train_weight = _step_training_policy(
                rescue_level=rescue_level,
                had_format_error=had_format_error,
                matched_action=matched,
                is_invalid_transition=invalid_transition,
                state_is_clean=state_is_clean,
                allow_syntax_training=cfg.allow_syntax_training,
                syntax_rescue_weight=cfg.syntax_rescue_weight,
                allow_semantic_training=cfg.allow_semantic_training,
                semantic_rescue_weight=cfg.semantic_rescue_weight,
            )

        steps.append(
            StepRecord(
                t=t, prompt=prompt, observation=observation,
                admissible_actions=list(admissible_t),
                action=matched, action_index=a_idx,
                logp_action=float(logp_a), entropy=0.0, done=done,
                full_response=full_resp,
                train_mode=train_mode,
                train_action_text=matched if train_mode == "action_template" else "",
                rescue_level=rescue_level,
                train_weight=train_weight,
                state_is_clean=state_is_clean,
            )
        )
        history.append((observation, matched))
        admissible = get_admissible_actions(env)
        state_is_clean = state_is_clean and rescue_level != "semantic" and not invalid_transition and matched != "search[<query>]"
        t += 1

    won = final_score >= 1.0
    rw = _shaped_reward(final_score, cfg, num_invalid=num_invalid)
    return Trajectory(
        gamefile=str(task_id), task_desc=task_desc,
        steps=steps, terminal_reward=rw, won=won,
        score=final_score,
        generated_steps=generated_steps,
        placeholder_searches=placeholder_searches,
        rescued_actions=rescued_actions,
        format_errors=format_errors,
        invalid_actions=num_invalid,
        branch_group_id=branch_group_id,
        has_semantic_rescue=has_semantic_rescue,
        has_placeholder_search=has_placeholder_search,
        has_invalid_transition=has_invalid_transition,
    )


# ---------------------------------------------------------------------------
# State Selection  (Eqs. 9-12)
# ---------------------------------------------------------------------------

@dataclass
class BranchCandidate:
    traj_idx: int
    step_idx: int
    score: float
    context: np.ndarray
    action_features: np.ndarray
    prompt: str
    admissible_actions: List[str]


def select_branch_positions(
    trajectories: Sequence[Trajectory],
    *,
    cfg: BranchConfig,
    cdb: FGTSCDB,
    feat_ext: EmbeddingFeatureExtractor,
) -> List[BranchCandidate]:
    """Score visited positions and return branching candidates.

    Modes:
      - top_k: choose the top-B states by score
      - tau_s: choose all states with score >= tau_s
    """
    candidates: List[BranchCandidate] = []
    for i, tr in enumerate(trajectories):
        T = max(len(tr), 1)
        for s in tr.steps:
            if len(s.admissible_actions) < 2:
                continue
            if not s.state_is_clean:
                continue
            imp = 1.0 - (s.t / float(T))
            ctx = feat_ext.encode_context(s.prompt)
            afeats = feat_ext.all_action_features(ctx, s.admissible_actions)
            unc = cdb.uncertainty(afeats)
            score = cfg.lambda_u * unc + cfg.lambda_i * imp
            candidates.append(BranchCandidate(
                traj_idx=i,
                step_idx=s.t,
                score=float(score),
                context=ctx,
                action_features=afeats,
                prompt=s.prompt,
                admissible_actions=list(s.admissible_actions),
            ))

    if cfg.state_selection_mode == "random":
        random.shuffle(candidates)
        return candidates[:cfg.num_branch_states]
    candidates.sort(key=lambda c: c.score, reverse=True)
    if cfg.state_selection_mode == "tau_s":
        if cfg.tau_s is None:
            raise ValueError("state_selection_mode='tau_s' requires tau_s to be set.")
        return [c for c in candidates if c.score >= cfg.tau_s]
    if cfg.state_selection_mode != "top_k":
        raise ValueError(
            f"Unknown state_selection_mode: {cfg.state_selection_mode}. "
            "Expected one of ['top_k', 'tau_s', 'random']."
        )
    return candidates[:cfg.num_branch_states]


# ---------------------------------------------------------------------------
# Loss computation (gradient-accumulation style)
# ---------------------------------------------------------------------------

def _collect_pg_steps(
    trajectories: Sequence[Trajectory],
    advantages: Sequence[float],
    max_steps_per_traj: int,
) -> List[Tuple[StepRecord, float, float]]:
    pairs: List[Tuple[StepRecord, float, float]] = []
    for tr, adv in zip(trajectories, advantages):
        if abs(adv) < 1e-12:
            continue
        eligible = [
            s for s in tr.steps
            if (s.train_mode == "response" and s.full_response)
            or (s.train_mode == "action_template" and s.train_action_text)
        ]
        eligible = [s for s in eligible if s.train_weight > 0]
        if len(eligible) > max_steps_per_traj:
            branch_steps = [s for s in eligible if s.is_branch_action]
            other_steps = [s for s in eligible if not s.is_branch_action]
            keep = branch_steps[:max_steps_per_traj]
            remaining = max_steps_per_traj - len(keep)
            if remaining > 0 and len(other_steps) > remaining:
                other_steps = random.sample(other_steps, remaining)
            elif remaining <= 0:
                other_steps = []
            eligible = keep + other_steps
        for s in eligible:
            pairs.append((s, float(adv), float(s.train_weight)))
    return pairs


def _group_normalize_branched(
    trajectories: Sequence[Trajectory],
) -> Tuple[List[float], List[int]]:
    grouped: Dict[int, List[Tuple[int, float]]] = {}
    for idx, tr in enumerate(trajectories):
        gid = tr.branch_group_id if tr.branch_group_id is not None else idx
        grouped.setdefault(gid, []).append((idx, tr.terminal_reward))

    advantages = [0.0] * len(trajectories)
    group_sizes: List[int] = []
    for items in grouped.values():
        group_sizes.append(len(items))
        advs = group_normalize([reward for _, reward in items])
        for (idx, _reward), adv in zip(items, advs):
            advantages[idx] = adv
    return advantages, group_sizes


def compute_l_pg_accum(
    policy: QwenGenerativePolicy,
    trajectories: Sequence[Trajectory],
    advantages: Sequence[float],
    *,
    scale: float = 1.0,
    max_steps_per_traj: int = 8,
) -> Tuple[float, Dict[str, int]]:
    """L_base / L_br  (Eqs. 3, 4)."""
    pairs = _collect_pg_steps(trajectories, advantages, max_steps_per_traj)
    stats = {
        "pg_response_steps": 0,
        "pg_action_template_steps": 0,
        "pg_branch_action_steps": 0,
        "rescued_pg_steps": 0,
        "num_pairs": len(pairs),
    }
    if not pairs:
        return 0.0, stats
    total_weight = sum(weight for _, _, weight in pairs)
    if total_weight <= 0:
        return 0.0, stats
    total = 0.0
    for step, adv, weight in pairs:
        if step.train_mode == "response":
            logp = policy.logp_of_response(step.prompt, step.full_response)
            stats["pg_response_steps"] += 1
        elif step.train_mode == "action_template":
            logp = policy.logp_of_action_template(step.prompt, step.train_action_text)
            stats["pg_action_template_steps"] += 1
            if step.is_branch_action:
                stats["pg_branch_action_steps"] += 1
            elif step.full_response:
                stats["rescued_pg_steps"] += 1
        else:
            continue
        micro = -(adv * logp) * (scale * weight / total_weight)
        micro.backward()
        total += (adv * logp).detach().item() * weight
    return -total / total_weight, stats


def compute_kl_penalty_accum(
    policy: QwenGenerativePolicy,
    ref_model,
    trajectories: Sequence[Trajectory],
    beta_kl: float,
    *,
    max_steps_per_traj: int = 8,
) -> float:
    """KL(pi_theta || pi_ref) penalty."""
    from alfworld.llm_agents.qwen_policy import _teacher_forced_logp

    steps_with_resp = []
    for tr in trajectories:
        eligible = [s for s in tr.steps if s.full_response and s.train_mode == "response"]
        if len(eligible) > max_steps_per_traj:
            eligible = random.sample(eligible, max_steps_per_traj)
        steps_with_resp.extend(eligible)

    if not steps_with_resp or beta_kl <= 0:
        return 0.0

    n = len(steps_with_resp)
    total = 0.0
    tokenizer = policy.tokenizer
    device = policy.device
    max_len = policy.max_prompt_length + policy.max_response_tokens

    for step in steps_with_resp:
        prompt_ids = tokenizer(step.prompt, add_special_tokens=False).input_ids
        resp_ids = tokenizer(step.full_response, add_special_tokens=False).input_ids
        logp_cur = _teacher_forced_logp(
            model=policy.model, tokenizer=tokenizer,
            prompt_ids=prompt_ids, response_ids=resp_ids,
            device=device, max_length=max_len, length_normalize=True,
        )
        with torch.no_grad():
            logp_ref = _teacher_forced_logp(
                model=ref_model, tokenizer=tokenizer,
                prompt_ids=prompt_ids, response_ids=resp_ids,
                device=device, max_length=max_len, length_normalize=True,
            )
        kl_approx = logp_cur - logp_ref.detach()
        micro = (beta_kl * kl_approx) / n
        micro.backward()
        total += kl_approx.detach().item()

    return total / n


def compute_l_dpo_accum(
    policy: QwenGenerativePolicy,
    prefs: Sequence[Preference],
    beta_dpo: float,
    *,
    scale: float = 1.0,
) -> float:
    """L_DPO  (Eq. 5)."""
    if not prefs:
        return 0.0
    n = len(prefs)
    total = 0.0
    for p in prefs:
        logp = policy.logp_of_action_templates(p.prompt, [p.a_pos, p.a_neg])
        diff = logp[0] - logp[1]
        micro = -torch.nn.functional.logsigmoid(beta_dpo * diff) * (scale / n)
        micro.backward()
        total += micro.detach().item() * n / scale if scale != 0 else 0.0
    return total / n


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    *,
    env: WebAgentTextEnv,
    policy: QwenGenerativePolicy,
    eval_task_ids: Sequence[int],
    max_steps: int,
    top_m_actions: int,
    cfg: BranchConfig,
) -> Dict[str, float]:
    """Evaluate on fixed set of tasks; report Score + Success Rate."""
    policy.set_eval(True)
    scores, successes, all_steps = [], [], []
    for tid in eval_task_ids:
        tr = rollout_one(
            env=env, policy=policy, task_id=tid,
            max_steps=max_steps, top_m_actions=top_m_actions, cfg=cfg,
        )
        scores.append(tr.score)
        successes.append(float(tr.won))
        all_steps.append(len(tr))
    policy.set_eval(False)
    return {
        "avg_score": float(np.mean(scores)) if scores else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "avg_steps": float(np.mean(all_steps)) if all_steps else 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Branching-Dueling on WebShop (Algorithm 1)"
    )
    parser.add_argument("config_path", type=str)
    parser.add_argument("--model_name", type=str,
                        default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--iters", type=int, default=150)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--eval_games", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="./training_bd_webshop/")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--N", type=int, default=8)
    parser.add_argument("--B", type=int, default=4)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--top_m_actions", type=int, default=0)
    parser.add_argument("--lambda_u", type=float, default=1.0)
    parser.add_argument("--lambda_i", type=float, default=1.0)
    parser.add_argument(
        "--state_selection_mode",
        type=str,
        default="top_k",
        choices=["top_k", "tau_s", "random"],
        help="State selection rule: 'top_k' uses the highest-B scored states; "
             "'tau_s' uses the paper's threshold rule score >= tau_s; "
             "'random' selects B states uniformly at random.",
    )
    parser.add_argument(
        "--action_pair_mode",
        type=str,
        default="cdb",
        choices=["cdb", "random"],
        help="Action pair selection: 'cdb' uses FGTS.CDB Thompson Sampling; "
             "'random' selects two actions uniformly at random.",
    )
    parser.add_argument("--queries_per_step", type=int, default=1,
                        help="Number of different tasks sampled per training step.")
    parser.add_argument("--no_rescue", action="store_true",
                        help="Disable rescue: pass raw actions to env, train on all outputs (standard GRPO style).")
    parser.add_argument("--tau_s", type=float, default=None)
    parser.add_argument("--beta_dpo", type=float, default=0.1)
    parser.add_argument("--lambda_br", type=float, default=1.0)
    parser.add_argument("--gamma_dpo", type=float, default=1.0)
    parser.add_argument("--w_base", type=float, default=1.0)
    parser.add_argument("--w_br", type=float, default=1.0)
    parser.add_argument("--w_dpo", type=float, default=1.0)
    parser.add_argument("--allow_syntax_training", type=int, default=1, choices=[0, 1])
    parser.add_argument("--syntax_rescue_weight", type=float, default=0.25)
    parser.add_argument("--allow_semantic_training", type=int, default=0, choices=[0, 1])
    parser.add_argument("--semantic_rescue_weight", type=float, default=0.0)
    parser.add_argument("--max_train_steps", type=int, default=8)
    parser.add_argument("--beta_kl", type=float, default=0.0)
    parser.add_argument("--max_response_tokens", type=int, default=512)
    parser.add_argument("--gen_temperature", type=float, default=1.0)
    parser.add_argument("--eval_temp", type=float, default=0.4)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--cdb_p", type=int, default=128)
    parser.add_argument("--cdb_eta", type=float, default=1.0)
    parser.add_argument("--cdb_mu", type=float, default=0.1)
    parser.add_argument("--cdb_delta", type=float, default=0.01)
    parser.add_argument("--cdb_sigma0", type=float, default=1.0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = load_yaml_config(args.config_path)
    max_steps = int(config["env"]["max_nb_steps_per_episode"])
    num_products = config["env"].get("num_products", 1000)
    human_goals = config["env"].get("human_goals", True)
    num_train_goals = config["dataset"].get("num_train_goals", 400)
    num_eval_goals = config["dataset"].get("num_eval_goals", 100)

    bc = BranchConfig(
        num_base_rollouts=args.N,
        num_branch_states=args.B,
        num_duels_per_state=args.K,
        top_m_actions=args.top_m_actions,
        lambda_u=args.lambda_u,
        lambda_i=args.lambda_i,
        tau_s=args.tau_s,
        state_selection_mode=args.state_selection_mode,
        action_pair_mode=args.action_pair_mode,
        no_rescue=args.no_rescue,
        beta_dpo=args.beta_dpo,
        lambda_br=args.lambda_br,
        gamma_dpo=args.gamma_dpo,
        allow_syntax_training=bool(args.allow_syntax_training),
        syntax_rescue_weight=args.syntax_rescue_weight,
        allow_semantic_training=bool(args.allow_semantic_training),
        semantic_rescue_weight=args.semantic_rescue_weight,
        w_base=args.w_base,
        w_br=args.w_br,
        w_dpo=args.w_dpo,
        queries_per_step=args.queries_per_step,
    )

    # --- Model ---
    use_cuda = config.get("general", {}).get("use_cuda", True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
    )
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    model.to(device)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    policy = QwenGenerativePolicy(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_prompt_length=4096,
        max_response_tokens=args.max_response_tokens,
        temperature=args.gen_temperature,
        eval_temperature=args.eval_temp,
        history_len=2,
        top_p=args.top_p,
    )

    ref_model = None
    if args.beta_kl > 0:
        import copy
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

    # --- FGTS.CDB ---
    feat_ext = EmbeddingFeatureExtractor(
        embed_layer=model.model.embed_tokens,
        tokenizer=tokenizer,
        p=args.cdb_p,
        seed=args.seed,
    )
    cdb = FGTSCDB(
        p=args.cdb_p,
        eta=args.cdb_eta,
        mu=args.cdb_mu,
        delta=args.cdb_delta,
        sigma0=args.cdb_sigma0,
    )

    # --- WebShop Environment ---
    print(f"Creating WebShop env (num_products={num_products}) ...")
    env = make_webshop_env(
        num_products=num_products,
        human_goals=human_goals,
    )
    total_goals = get_num_goals(env)
    print(f"Total goals: {total_goals}")

    train_task_ids = list(range(min(num_train_goals, total_goals)))
    eval_start = min(num_train_goals, total_goals)
    eval_end = min(eval_start + num_eval_goals, total_goals)
    eval_task_ids = list(range(eval_start, eval_end))
    print(f"Train tasks: {len(train_task_ids)}, Eval tasks: {len(eval_task_ids)}")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # --- Output ---
    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, "log.jsonl")
    best_metric = -1.0
    t0 = datetime.datetime.now()

    tag = os.path.basename(args.save_dir.rstrip("/"))
    pr_logger = PromptResponseLogger(save_dir=args.save_dir, tag=tag)

    # ===================================================================
    # Training loop  (Algorithm 1)
    # ===================================================================
    for it in range(args.iters):
        _it_t0 = _time.time()

        # ---------------------------------------------------------------
        # 1) Sample queries_per_step tasks, do N base rollouts each
        # ---------------------------------------------------------------
        query_task_ids = random.sample(
            train_task_ids, min(bc.queries_per_step, len(train_task_ids))
        )
        base_trajs: List[Trajectory] = []
        for _qi, task_id in enumerate(query_task_ids):
            for _ri in range(bc.num_base_rollouts):
                tr = rollout_one(
                    env=env, policy=policy, task_id=task_id,
                    max_steps=max_steps, top_m_actions=bc.top_m_actions, cfg=bc,
                    pr_logger=pr_logger, iteration=it,
                )
                base_trajs.append(tr)
                print(f"  [iter {it}] q{_qi+1}/{bc.queries_per_step} "
                      f"base {_ri+1}/{bc.num_base_rollouts}  "
                      f"steps={len(tr)} won={tr.won} score={tr.score:.3f}  "
                      f"({_time.time()-_it_t0:.0f}s)", flush=True)

        base_rewards = [tr.terminal_reward for tr in base_trajs]
        A_base = group_normalize(base_rewards)
        _phase1_t = _time.time() - _it_t0

        # ---------------------------------------------------------------
        # 2) State selection  (skip if B=0 for GRPO-only mode)
        # ---------------------------------------------------------------
        if bc.num_branch_states > 0 and bc.num_duels_per_state > 0:
            branch_candidates = select_branch_positions(
                base_trajs, cfg=bc, cdb=cdb, feat_ext=feat_ext,
            )
        else:
            branch_candidates = []
        print(f"  [iter {it}] state selection: {len(branch_candidates)} candidates  "
              f"({_time.time()-_it_t0:.0f}s)", flush=True)

        # ---------------------------------------------------------------
        # 3) Local dueling
        # ---------------------------------------------------------------
        branched_trajs: List[Trajectory] = []
        prefs: List[Preference] = []
        _br_count = 0

        for branch_group_id, cand in enumerate(branch_candidates):
            tr = base_trajs[cand.traj_idx]
            cand_task_id = int(tr.gamefile)
            if cand.step_idx >= len(tr):
                continue
            prefix_actions = (
                tr.prefix_actions(cand.step_idx - 1)
                if cand.step_idx > 0 else []
            )
            admissible = cand.admissible_actions
            prompt = cand.prompt
            action_feats = cand.action_features

            for _k in range(bc.num_duels_per_state):
                if bc.action_pair_mode == "random":
                    idxs = random.sample(range(len(admissible)), min(2, len(admissible)))
                    a1_idx, a2_idx = idxs[0], idxs[-1]
                else:
                    cdb.sgld_step()
                    a1_idx, a2_idx = cdb.select_actions(action_feats)
                a1 = admissible[a1_idx]
                a2 = admissible[a2_idx]

                tr1 = replay_to_state_and_branch(
                    env=env, task_id=cand_task_id,
                    prefix_actions=prefix_actions,
                    branch_action=a1,
                    branch_group_id=branch_group_id,
                    policy=policy,
                    max_steps=max_steps,
                    top_m_actions=bc.top_m_actions,
                    cfg=bc,
                    pr_logger=pr_logger, iteration=it,
                )
                tr2 = replay_to_state_and_branch(
                    env=env, task_id=cand_task_id,
                    prefix_actions=prefix_actions,
                    branch_action=a2,
                    branch_group_id=branch_group_id,
                    policy=policy,
                    max_steps=max_steps,
                    top_m_actions=bc.top_m_actions,
                    cfg=bc,
                    pr_logger=pr_logger, iteration=it,
                )
                branched_trajs.extend([tr1, tr2])
                _br_count += 2

                r1, r2 = tr1.terminal_reward, tr2.terminal_reward
                pair_is_clean = _trajectory_is_clean(tr1) and _trajectory_is_clean(tr2)
                if r1 != r2:
                    y = 1.0 if r1 > r2 else 0.0
                    y_tilde = 2.0 * y - 1.0
                    cdb.record(action_feats, a1_idx, a2_idx, y_tilde)
                    if pair_is_clean:
                        if y_tilde > 0:
                            prefs.append(Preference(prompt=prompt,
                                                    a_pos=a1, a_neg=a2))
                        else:
                            prefs.append(Preference(prompt=prompt,
                                                    a_pos=a2, a_neg=a1))

                print(f"  [iter {it}] branch duel {_br_count}  "
                      f"r1={r1:.1f} r2={r2:.1f}  "
                      f"s1={tr1.score:.3f} s2={tr2.score:.3f}  "
                      f"({_time.time()-_it_t0:.0f}s)", flush=True)

        A_br, branch_group_sizes = _group_normalize_branched(branched_trajs)
        branch_group_count = len(branch_group_sizes)
        _phase23_t = _time.time() - _it_t0 - _phase1_t

        # ---------------------------------------------------------------
        # 4) Compute losses and update
        # ---------------------------------------------------------------
        print(f"  [iter {it}] computing losses...  "
              f"({_time.time()-_it_t0:.0f}s)", flush=True)
        model.train()
        optimizer.zero_grad(set_to_none=True)

        l_base_val, l_base_stats = compute_l_pg_accum(
            policy, base_trajs, A_base,
            scale=bc.w_base, max_steps_per_traj=args.max_train_steps,
        )
        l_br_val, l_br_stats = compute_l_pg_accum(
            policy, branched_trajs, A_br,
            scale=bc.w_br * bc.lambda_br, max_steps_per_traj=args.max_train_steps,
        ) if branched_trajs else (0.0, {
            "pg_response_steps": 0,
            "pg_action_template_steps": 0,
            "pg_branch_action_steps": 0,
            "rescued_pg_steps": 0,
            "num_pairs": 0,
        })
        l_dpo_val = compute_l_dpo_accum(
            policy, prefs, bc.beta_dpo,
            scale=bc.w_dpo * bc.gamma_dpo,
        ) if prefs else 0.0

        l_kl_val = 0.0
        if ref_model is not None and args.beta_kl > 0:
            l_kl_val = compute_kl_penalty_accum(
                policy, ref_model, base_trajs, args.beta_kl,
                max_steps_per_traj=args.max_train_steps,
            )

        has_grad = any(p.grad is not None for p in model.parameters())
        if has_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss_val = (bc.w_base * l_base_val
                          + bc.w_br * bc.lambda_br * l_br_val
                          + bc.w_dpo * bc.gamma_dpo * l_dpo_val
                          + args.beta_kl * l_kl_val)
        _phase4_t = _time.time() - _it_t0 - _phase1_t - _phase23_t
        _it_total = _time.time() - _it_t0
        print(f"  [iter {it}] DONE  base={_phase1_t:.0f}s  "
              f"branch={_phase23_t:.0f}s  loss={_phase4_t:.0f}s  "
              f"total={_it_total:.0f}s", flush=True)

        # ---------------------------------------------------------------
        # 5) Logging
        # ---------------------------------------------------------------
        train_succ = float(np.mean([float(tr.won) for tr in base_trajs]))
        train_score = float(np.mean([tr.score for tr in base_trajs]))
        train_avg_steps = float(np.mean([len(tr) for tr in base_trajs]))
        diag_gen_steps = int(sum(tr.generated_steps for tr in base_trajs))
        diag_denom = max(diag_gen_steps, 1)
        placeholder_search_rate = (
            sum(tr.placeholder_searches for tr in base_trajs) / diag_denom
        )
        action_rescue_rate = (
            sum(tr.rescued_actions for tr in base_trajs) / diag_denom
        )
        format_error_rate = (
            sum(tr.format_errors for tr in base_trajs) / diag_denom
        )
        invalid_action_rate = (
            sum(tr.invalid_actions for tr in base_trajs) / diag_denom
        )
        pg_response_steps = (
            l_base_stats["pg_response_steps"] + l_br_stats["pg_response_steps"]
        )
        pg_action_template_steps = (
            l_base_stats["pg_action_template_steps"]
            + l_br_stats["pg_action_template_steps"]
        )
        pg_branch_action_steps = l_br_stats["pg_branch_action_steps"]
        rescued_pg_steps = (
            l_base_stats["rescued_pg_steps"] + l_br_stats["rescued_pg_steps"]
        )
        elapsed = str(datetime.datetime.now() - t0).split(".")[0]

        log_entry: Dict = {
            "iter": it,
            "elapsed": elapsed,
            "train_succ": round(train_succ, 4),
            "train_score": round(train_score, 4),
            "train_avg_steps": round(train_avg_steps, 2),
            "Rmean": round(float(np.mean(base_rewards)), 3),
            "l_base": round(l_base_val, 5),
            "l_br": round(l_br_val, 5),
            "l_dpo": round(l_dpo_val, 5),
            "l_kl": round(l_kl_val, 5),
            "total_loss": round(total_loss_val, 5),
            "w_base": bc.w_base,
            "w_br": bc.w_br,
            "w_dpo": bc.w_dpo,
            "queries_per_step": bc.queries_per_step,
            "state_selection_mode": bc.state_selection_mode,
            "action_pair_mode": bc.action_pair_mode,
            "no_rescue": int(bc.no_rescue),
            "allow_syntax_training": int(bc.allow_syntax_training),
            "syntax_rescue_weight": bc.syntax_rescue_weight,
            "allow_semantic_training": int(bc.allow_semantic_training),
            "semantic_rescue_weight": bc.semantic_rescue_weight,
            "base_N": len(base_trajs),
            "br_N": len(branched_trajs),
            "pref_N": len(prefs),
            "cdb_history": len(cdb.history),
            "diag_gen_steps": diag_gen_steps,
            "placeholder_search_rate": round(placeholder_search_rate, 4),
            "action_rescue_rate": round(action_rescue_rate, 4),
            "format_error_rate": round(format_error_rate, 4),
            "invalid_action_rate": round(invalid_action_rate, 4),
            "pg_response_steps": pg_response_steps,
            "pg_action_template_steps": pg_action_template_steps,
            "pg_branch_action_steps": pg_branch_action_steps,
            "rescued_pg_steps": rescued_pg_steps,
            "branch_group_count": branch_group_count,
            "branch_group_sizes": branch_group_sizes,
            "l_br_samples": l_br_stats["num_pairs"],
        }

        # ---------------------------------------------------------------
        # 6) Periodic evaluation
        # ---------------------------------------------------------------
        if (it + 1) % args.eval_every == 0 or it == args.iters - 1:
            if eval_task_ids:
                num_eval = min(args.eval_games, len(eval_task_ids))
                eval_subset = eval_task_ids[:num_eval]
                res = evaluate(
                    env=env, policy=policy,
                    eval_task_ids=eval_subset,
                    max_steps=max_steps,
                    top_m_actions=bc.top_m_actions, cfg=bc,
                )
                log_entry["eval_score"] = round(res["avg_score"], 4)
                log_entry["eval_succ"] = round(res["success_rate"], 4)
                log_entry["eval_steps"] = round(res["avg_steps"], 2)

                current_metric = res["avg_score"]
                if current_metric > best_metric:
                    best_metric = current_metric
                    ckpt_path = os.path.join(args.save_dir, "best_model")
                    model.save_pretrained(ckpt_path)
                    tokenizer.save_pretrained(ckpt_path)
                    print(f"  -> saved best checkpoint "
                          f"(score={best_metric:.4f})")

        parts = [f"{k}={v}" for k, v in log_entry.items()]
        print(" | ".join(parts))

        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    # Save final
    final_path = os.path.join(args.save_dir, "final_model")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    pr_logger.close()
    print(f"Training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()
