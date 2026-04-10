from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

def _maybe_use_jdk21() -> None:
    preferred = os.environ.get("WEBSHOP_JAVA_HOME")
    candidates = [
        preferred,
        "/usr/lib/jvm/java-21-openjdk-amd64",
        "/usr/lib/jvm/java-1.21.0-openjdk-amd64",
    ]
    current = os.environ.get("JAVA_HOME", "")
    if "21" in current:
        return
    for candidate in candidates:
        if not candidate:
            continue
        java_bin = Path(candidate) / "bin" / "java"
        if java_bin.exists():
            os.environ["JAVA_HOME"] = str(candidate)
            path_parts = os.environ.get("PATH", "").split(":")
            java_path = str(Path(candidate) / "bin")
            if java_path not in path_parts:
                os.environ["PATH"] = f"{java_path}:{os.environ.get('PATH', '')}"
            return


def _maybe_add_vendor_webshop() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    vendor_root = repo_root / "vendor"
    if vendor_root.exists():
        sys.path.insert(0, str(vendor_root))
    assets_root = repo_root / "assets" / "webshop"
    data_root = assets_root / "data"
    search_root = assets_root / "search_engine"
    if data_root.exists():
        os.environ.setdefault("WEBSHOP_DATA_DIR", str(data_root))
    if search_root.exists():
        os.environ.setdefault("WEBSHOP_SEARCH_DIR", str(search_root))
    _maybe_use_jdk21()


_maybe_add_vendor_webshop()


def _default_products_file(num_products: int | None) -> str:
    data_dir = Path(os.environ["WEBSHOP_DATA_DIR"])
    if num_products is None:
        return str(data_dir / "items_shuffle.json")
    if num_products == 1000:
        return str(data_dir / "items_shuffle_1000.json")
    return str(data_dir / "items_shuffle.json")


def _default_attr_file(num_products: int | None) -> str:
    data_dir = Path(os.environ["WEBSHOP_DATA_DIR"])
    if num_products is None:
        return str(data_dir / "items_ins_v2.json")
    if num_products == 1000:
        return str(data_dir / "items_ins_v2_1000.json")
    return str(data_dir / "items_ins_v2.json")


@dataclass
class WebShopEnvConfig:
    observation_mode: str = "text"
    num_products: int | None = None
    human_goals: bool = True
    max_steps: int = 15
    train_goal_start: int = 0
    train_goal_end: int | None = None
    eval_goal_start: int = 11587
    eval_goal_end: int = 12087
    products_file: str | None = None
    attr_file: str | None = None


class WebShopTextEnvWrapper:
    def __init__(self, cfg: WebShopEnvConfig) -> None:
        self.cfg = cfg
        os.environ["WEBSHOP_PRODUCTS_FILE"] = cfg.products_file or _default_products_file(cfg.num_products)
        os.environ["WEBSHOP_ATTR_FILE"] = cfg.attr_file or _default_attr_file(cfg.num_products)
        from webshop_env.envs import WebAgentTextEnv  # type: ignore  # noqa: E402
        self.env = WebAgentTextEnv(
            observation_mode=cfg.observation_mode,
            num_products=cfg.num_products,
            human_goals=cfg.human_goals,
            file_path=os.environ["WEBSHOP_PRODUCTS_FILE"],
        )
        self.current_step = 0

    @property
    def all_goal_count(self) -> int:
        return len(self.env.server.goals)

    @property
    def train_goal_ids(self) -> list[int]:
        start = max(0, min(self.cfg.train_goal_start, self.all_goal_count))
        end = self.all_goal_count if self.cfg.train_goal_end is None else max(start, min(self.cfg.train_goal_end, self.all_goal_count))
        return list(range(start, end))

    @property
    def eval_goal_ids(self) -> list[int]:
        start = max(0, min(self.cfg.eval_goal_start, self.all_goal_count))
        end = max(start, min(self.cfg.eval_goal_end, self.all_goal_count))
        return list(range(start, end))

    @property
    def observation(self) -> str:
        return self.env.observation

    @property
    def instruction_text(self) -> str:
        return self.env.instruction_text

    def reset(self, task_id: int) -> Tuple[str, Dict[str, Any]]:
        self.current_step = 0
        obs, info = self.env.reset(session=task_id)
        info = dict(info or {})
        info["available_actions"] = self.get_available_actions()
        return obs, info

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        self.current_step += 1
        obs, reward, done, info = self.env.step(action)
        info = dict(info or {})
        info["available_actions"] = self.get_available_actions()
        if self.current_step >= self.cfg.max_steps and not done:
            done = True
        return obs, reward, done, info

    def get_available_actions(self) -> Dict[str, Any]:
        return self.env.get_available_actions()

    def format_available_actions(self) -> List[str]:
        avail = self.get_available_actions()
        actions: list[str] = []
        if avail.get("has_search_bar", False):
            actions.append("search[<your query>]")
        for clickable in avail.get("clickables", []):
            if clickable.lower() != "search":
                actions.append(f"click[{clickable}]".lower())
        return actions

    def close(self) -> None:
        self.env.close()
