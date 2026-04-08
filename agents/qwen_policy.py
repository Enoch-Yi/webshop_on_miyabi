from __future__ import annotations

import difflib
import random
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import StoppingCriteria, StoppingCriteriaList


# ---------------------------------------------------------------------------
# Teacher-forced logp helpers (used by both PG and DPO losses)
# ---------------------------------------------------------------------------

def _teacher_forced_logp(
    *,
    model,
    tokenizer,
    prompt_ids: List[int],
    response_ids: List[int],
    device: torch.device,
    max_length: int = 2048,
    length_normalize: bool = True,
) -> torch.Tensor:
    """Teacher-forcing mean-token log-prob of *response_ids* given *prompt_ids*."""
    combined = prompt_ids + response_ids
    if len(combined) > max_length:
        overflow = len(combined) - max_length
        combined = combined[overflow:]
        prompt_len = len(combined) - len(response_ids)
    else:
        prompt_len = len(prompt_ids)

    input_ids = torch.tensor([combined], device=device)

    with torch.set_grad_enabled(model.training):
        out = model(input_ids=input_ids)
        logits = out.logits  # [1, L, V]

    logprobs = F.log_softmax(logits[:, :-1, :], dim=-1)
    targets = input_ids[:, 1:]
    token_logp = torch.gather(
        logprobs, dim=-1, index=targets.unsqueeze(-1)
    ).squeeze(-1)

    start_t = max(prompt_len - 1, 0)
    end_t = max(len(combined) - 1, 0)
    n_tokens = end_t - start_t
    if n_tokens <= 0:
        return torch.tensor(0.0, device=device, requires_grad=model.training)

    total = token_logp[0, start_t:end_t].sum()
    if length_normalize:
        return total / n_tokens
    return total


def _multi_response_logp(
    *,
    model,
    tokenizer,
    prompt: str,
    responses: Sequence[str],
    device: torch.device,
    max_length: int = 2048,
    length_normalize: bool = True,
) -> torch.Tensor:
    """Compute teacher-forced logp for each response given the same prompt."""
    if not responses:
        raise ValueError("responses must be non-empty")
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    results: List[torch.Tensor] = []
    for resp in responses:
        resp_ids = tokenizer(resp, add_special_tokens=False).input_ids
        lp = _teacher_forced_logp(
            model=model, tokenizer=tokenizer,
            prompt_ids=prompt_ids, response_ids=resp_ids,
            device=device, max_length=max_length,
            length_normalize=length_normalize,
        )
        results.append(lp)
    return torch.stack(results)


# ---------------------------------------------------------------------------
# Action parsing & matching
# ---------------------------------------------------------------------------

_ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL | re.IGNORECASE)
_ACTION_OPEN_RE = re.compile(r"<action>\s*(.*)", re.DOTALL | re.IGNORECASE)


def parse_action(text: str) -> Optional[str]:
    """Extract the first <action>...</action> content from generated text."""
    m = _ACTION_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _ACTION_OPEN_RE.search(text)
    if m:
        return m.group(1).strip()
    return None


def match_to_admissible(
    generated: str,
    admissible: Sequence[str],
) -> Tuple[str, int, bool]:
    """Match a generated action string to the closest admissible action.

    Returns (matched_action, index, is_exact_match).
    Falls back to random if no close match (cutoff=0.4).
    """
    gen_lower = generated.strip().lower()
    for i, a in enumerate(admissible):
        if a.strip().lower() == gen_lower:
            return a, i, True

    matches = difflib.get_close_matches(
        gen_lower, [a.lower() for a in admissible], n=1, cutoff=0.4
    )
    if matches:
        idx = [a.lower() for a in admissible].index(matches[0])
        return admissible[idx], idx, False

    idx = random.randrange(len(admissible))
    return admissible[idx], idx, False


# ---------------------------------------------------------------------------
# StoppingCriteria: stop generation when "</action>" appears in output
# ---------------------------------------------------------------------------

class _StopOnActionTag(StoppingCriteria):
    """Stop when the decoded new tokens contain '</action>'."""

    def __init__(self, tokenizer, prompt_length: int) -> None:
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> bool:
        new_ids = input_ids[0, self.prompt_length:]
        if new_ids.numel() < 3:
            return False
        tail = self.tokenizer.decode(new_ids[-10:], skip_special_tokens=True)
        return "</action>" in tail.lower()


# ---------------------------------------------------------------------------
# QwenGenerativePolicy — HCAPO-aligned generative agent
# ---------------------------------------------------------------------------

class QwenGenerativePolicy:
    """LLM policy that generates <think> reasoning + <action> output.

    Prompt template strictly follows HCAPO (Appendix C.2.1).
    Uses Qwen chat template so the Instruct model understands the instruction.
    """

    def __init__(
        self,
        *,
        model,
        tokenizer,
        device: torch.device,
        max_prompt_length: int = 4096,
        max_response_tokens: int = 512,
        temperature: float = 1.0,
        eval_temperature: float = 0.4,
        history_len: int = 2,
        top_p: float = 0.95,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_prompt_length = max_prompt_length
        self.max_response_tokens = max_response_tokens
        self.temperature = temperature
        self.eval_temperature = eval_temperature
        self.history_len = history_len
        self.top_p = top_p
        self._is_eval = False

    # -- prompt construction (HCAPO Appendix C.2.1 + Qwen chat template) ----

    def build_prompt(
        self,
        *,
        task_desc: str,
        observation: str,
        history: Sequence[Tuple[str, str]],
        admissible_actions: Optional[Sequence[str]] = None,
        step_count: int = 0,
    ) -> str:
        hist = list(history)[-self.history_len:]
        history_length = len(hist)

        hist_lines: List[str] = []
        for obs, act in hist:
            hist_lines.append(f"Obs: {obs}\nAction: {act}")
        action_history = "\n".join(hist_lines) if hist_lines else "(none)"

        admissible_str = ""
        if admissible_actions is not None:
            admissible_str = ", ".join(admissible_actions)

        user_text = (
            f"You are an expert autonomous agent operating in the WebShop "
            f"e-commerce environment.\n"
            f"Your task is to: {task_desc}.\n"
            f"Prior to this step, you have already taken {step_count} step(s). "
            f"Below are the most recent {history_length} observations "
            f"and the corresponding actions you took:\n"
            f"{action_history}\n"
            f"You are now at step {step_count} and your current observation is: "
            f"{observation}.\n"
            f"Your admissible actions for the current situation are: "
            f"[{admissible_str}].\n"
            f"Now it's your turn to take one action for the current step. "
            f"You should first reason step-by-step about the current situation, "
            f"then think carefully which admissible action best advances the "
            f"shopping goal. This reasoning process MUST be enclosed within "
            f"<think> </think> tags. Once you've finished your reasoning, you "
            f"should choose an admissible action for current step and present "
            f"it within <action> </action> tags."
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_text},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    # -- generation ----------------------------------------------------------

    @torch.no_grad()
    def generate_action(
        self,
        prompt: str,
        admissible_actions: Sequence[str],
    ) -> Tuple[str, str, int, float]:
        """Generate reasoning + action via autoregressive decoding.

        Returns (matched_action, full_response, action_index, logp_response).
        """
        self.model.eval()

        enc = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False,
        )
        input_ids = enc.input_ids[:, -self.max_prompt_length:].to(self.device)
        attention_mask = torch.ones_like(input_ids)
        prompt_len = input_ids.shape[1]

        temp = self.eval_temperature if self._is_eval else self.temperature

        stop_criteria = StoppingCriteriaList([
            _StopOnActionTag(self.tokenizer, prompt_len),
        ])

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_response_tokens,
            temperature=max(temp, 0.01),
            top_p=self.top_p,
            do_sample=True,
            stopping_criteria=stop_criteria,
            pad_token_id=self.tokenizer.pad_token_id
                         or self.tokenizer.eos_token_id,
        )

        new_tokens = outputs[0, prompt_len:]
        full_response = self.tokenizer.decode(
            new_tokens, skip_special_tokens=True,
        )

        generated_action = parse_action(full_response)
        if generated_action is None:
            raw = full_response.strip().split("\n")[-1].strip()
            generated_action = raw if raw else ""

        if not admissible_actions:
            return generated_action, full_response, -1, 0.0

        matched, idx, _exact = match_to_admissible(
            generated_action, admissible_actions
        )

        return matched, full_response, idx, 0.0

    # -- eval mode toggle ----------------------------------------------------

    def set_eval(self, is_eval: bool = True) -> None:
        self._is_eval = is_eval

    # -- training: logp of a full response (for PG loss) ---------------------

    def logp_of_response(self, prompt: str, response: str) -> torch.Tensor:
        """Differentiable teacher-forced logp of a complete response string.

        Used for policy-gradient (L_base, L_br) losses.
        Model must be in train mode.
        """
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        resp_ids = self.tokenizer(response, add_special_tokens=False).input_ids
        return _teacher_forced_logp(
            model=self.model, tokenizer=self.tokenizer,
            prompt_ids=prompt_ids, response_ids=resp_ids,
            device=self.device,
            max_length=self.max_prompt_length + self.max_response_tokens,
            length_normalize=True,
        )

    # -- training: logp of action templates (for DPO loss) -------------------

    def logp_of_action_templates(
        self, prompt: str, actions: Sequence[str],
    ) -> torch.Tensor:
        """Differentiable logp of ``<action>X</action>`` templates.

        Used for DPO preference loss (L_DPO) where we compare two specific
        admissible actions selected by FGTS.CDB.
        """
        templates = [f"<action>{a}</action>" for a in actions]
        return _multi_response_logp(
            model=self.model, tokenizer=self.tokenizer,
            prompt=prompt, responses=templates,
            device=self.device,
            max_length=self.max_prompt_length + 128,
            length_normalize=True,
        )

    def logp_of_action_template(
        self, prompt: str, action: str,
    ) -> torch.Tensor:
        """Differentiable logp of a single ``<action>X</action>`` template."""
        return self.logp_of_action_templates(prompt, [action])[0]

    def logp_of_action_template(self, prompt: str, action: str) -> torch.Tensor:
        """Convenience wrapper for a single ``<action>...</action>`` target."""
        return self.logp_of_action_templates(prompt, [action])[0]
