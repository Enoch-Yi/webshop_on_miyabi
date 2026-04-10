from __future__ import annotations

import random
from dataclasses import dataclass

from src.envs.webshop_env import WebShopEnvConfig, WebShopTextEnvWrapper


@dataclass
class GroupBatch:
    task_ids: list[int]
    grouped_task_ids: list[list[int]]


class SameTaskGroupSampler:
    def __init__(self, env: WebShopTextEnvWrapper, train_data_size: int, group_size: int, seed: int) -> None:
        self.env = env
        self.train_data_size = train_data_size
        self.group_size = group_size
        self.rng = random.Random(seed)

    def sample_train_batch(self) -> GroupBatch:
        task_ids = self.rng.sample(self.env.train_goal_ids, k=min(self.train_data_size, len(self.env.train_goal_ids)))
        grouped = [[task_id for _ in range(self.group_size)] for task_id in task_ids]
        return GroupBatch(task_ids=task_ids, grouped_task_ids=grouped)
