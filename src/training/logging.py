from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


class JsonlLogger:
    def __init__(self, save_dir: str, filename: str = "log.jsonl") -> None:
        self.path = Path(save_dir) / filename
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


class PromptResponseLogger:
    MAX_BYTES = 1_000_000

    def __init__(self, save_dir: str, tag: str, max_bytes: int | None = None) -> None:
        self._dir = Path(save_dir)
        self._tag = tag
        self._seq = 0
        self._size = 0
        self._fp = None
        if max_bytes is not None:
            self.MAX_BYTES = max_bytes
        self._dir.mkdir(parents=True, exist_ok=True)
        self._open_next()

    def _fname(self) -> Path:
        return self._dir / f"training_log_{self._tag}_{self._seq:04d}.txt"

    def _open_next(self) -> None:
        if self._fp is not None:
            self._fp.close()
        self._fp = self._fname().open("a", encoding="utf-8")
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
        won: bool | None = None,
    ) -> None:
        if self._fp is None:
            return
        header = (
            f"{'=' * 72}\n"
            f"[iter={iteration}  step={step}  phase={phase}]\n"
            f"{'=' * 72}\n"
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


class WandbLogger:
    def __init__(
        self,
        *,
        enabled: bool,
        project: str | None = None,
        name: str | None = None,
        group: str | None = None,
        entity: str | None = None,
        mode: str = "online",
        tags: list[str] | None = None,
        config: Dict[str, Any] | None = None,
        save_dir: str | None = None,
    ) -> None:
        self.enabled = enabled
        self._wandb = None
        if not enabled:
            return
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb logging is enabled, but the `wandb` package is not installed."
            ) from exc
        self._wandb = wandb
        wandb.init(
            project=project,
            name=name,
            group=group,
            entity=entity,
            mode=mode,
            tags=tags,
            config=config,
            dir=save_dir,
        )

    def log(self, payload: Dict[str, Any], step: int | None = None) -> None:
        if self._wandb is None:
            return
        safe_payload = {}
        for key, value in payload.items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                safe_payload[key] = value
        self._wandb.log(safe_payload, step=step)

    def finish(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()


def format_console_metrics(title: str, payload: Dict[str, Any]) -> str:
    preferred_keys = [
        "train_score",
        "train_succ",
        "eval_score",
        "eval_succ",
        "avg_steps",
        "eval_steps",
        "base_loss",
        "branch_loss",
        "dpo_loss",
        "kl_loss",
        "total_loss",
        "format_invalid_rate",
        "action_not_in_available_rate",
        "invalid_transition_rate",
        "num_branch_candidates",
        "num_branched_trajectories",
        "num_preferences",
        "cdb_history",
    ]
    parts = [title]
    for key in preferred_keys:
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return " | ".join(parts)
