from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.actions.projection import syntax_only_projection
from src.envs.webshop_env import WebShopEnvConfig, WebShopTextEnvWrapper
from src.prompts.webshop import build_webshop_prompt
from src.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--task_id", type=int, default=0)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    env = WebShopTextEnvWrapper(WebShopEnvConfig(**cfg["env"]))
    obs, _ = env.reset(args.task_id)
    actions = env.format_available_actions()
    prompt = build_webshop_prompt(
        task_description=env.instruction_text,
        current_observation=obs,
        available_actions=actions,
        history=[],
        history_length=int(cfg["model"]["history_length"]),
        step_count=0,
    )
    demo_response = "<think>I should inspect the page and choose one legal action.</think><action>click[back to search]</action>"
    projection = syntax_only_projection(demo_response, actions)

    print("task_id:", args.task_id)
    print("task_desc:", env.instruction_text)
    print("available_actions_count:", len(actions))
    print("first_actions:", actions[:10])
    print("projection_valid:", projection.is_projection_valid)
    print("projection_action:", projection.executed_action)
    print("prompt_preview:")
    print(prompt[:1200])


if __name__ == "__main__":
    main()
