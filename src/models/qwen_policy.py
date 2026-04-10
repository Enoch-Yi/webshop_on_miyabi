from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from src.prompts.webshop import build_webshop_prompt


def teacher_forced_logp(
    *,
    model,
    tokenizer,
    prompt: str,
    response: str,
    device: torch.device,
    max_length: int,
) -> torch.Tensor:
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    response_ids = tokenizer(response, add_special_tokens=False).input_ids
    combined = prompt_ids + response_ids
    if len(combined) > max_length:
        overflow = len(combined) - max_length
        combined = combined[overflow:]
        prompt_len = len(combined) - len(response_ids)
    else:
        prompt_len = len(prompt_ids)
    input_ids = torch.tensor([combined], device=device)
    outputs = model(input_ids=input_ids)
    logprobs = F.log_softmax(outputs.logits[:, :-1, :], dim=-1)
    targets = input_ids[:, 1:]
    token_logp = torch.gather(logprobs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    start_t = max(prompt_len - 1, 0)
    end_t = max(len(combined) - 1, 0)
    n_tokens = end_t - start_t
    if n_tokens <= 0:
        return torch.tensor(0.0, device=device, requires_grad=model.training)
    return token_logp[0, start_t:end_t].sum() / n_tokens


class StopOnActionTag(StoppingCriteria):
    def __init__(self, tokenizer, prompt_length: int) -> None:
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> bool:
        new_ids = input_ids[0, self.prompt_length:]
        if new_ids.numel() < 3:
            return False
        tail = self.tokenizer.decode(new_ids[-12:], skip_special_tokens=True)
        return "</action>" in tail.lower()


@dataclass
class PolicyOutput:
    prompt: str
    response: str


class QwenPolicy:
    def __init__(
        self,
        *,
        model_name: str,
        device: torch.device,
        max_prompt_length: int,
        max_response_tokens: int,
        history_length: int,
        temperature: float,
        eval_temperature: float,
        top_p: float,
    ) -> None:
        self.device = device
        torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
        ).to(device)
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_prompt_length = max_prompt_length
        self.max_response_tokens = max_response_tokens
        self.history_length = history_length
        self.temperature = temperature
        self.eval_temperature = eval_temperature
        self.top_p = top_p
        self._is_eval = False

    def build_prompt(
        self,
        *,
        task_desc: str,
        observation: str,
        history: list[tuple[str, str]],
        available_actions: list[str],
        step_count: int,
    ) -> str:
        user_text = build_webshop_prompt(
            task_description=task_desc,
            current_observation=observation,
            available_actions=available_actions,
            history=history,
            history_length=self.history_length,
            step_count=step_count,
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_text},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        self.model.eval()
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = enc.input_ids[:, -self.max_prompt_length:].to(self.device)
        attention_mask = torch.ones_like(input_ids)
        prompt_len = input_ids.shape[1]
        temp = self.eval_temperature if self._is_eval else self.temperature
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_response_tokens,
            temperature=max(temp, 0.01),
            top_p=self.top_p,
            do_sample=True,
            stopping_criteria=StoppingCriteriaList([StopOnActionTag(self.tokenizer, prompt_len)]),
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_tokens = outputs[0, prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def logp_of_response(self, prompt: str, response: str) -> torch.Tensor:
        self.model.train()
        return teacher_forced_logp(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            response=response,
            device=self.device,
            max_length=self.max_prompt_length + self.max_response_tokens,
        )

    def logp_of_action_templates(
        self,
        prompt: str,
        actions: Sequence[str],
    ) -> torch.Tensor:
        templates = [f"<action>{action}</action>" for action in actions]
        vals = [
            self.logp_of_response(prompt, template)
            for template in templates
        ]
        if not vals:
            return torch.empty(0, device=self.device)
        return torch.stack(vals)

    def logp_of_action_template(self, prompt: str, action: str) -> torch.Tensor:
        return self.logp_of_action_templates(prompt, [action])[0]

    def set_eval(self, is_eval: bool) -> None:
        self._is_eval = is_eval
