from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.trainer import WebShopTrainer
from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--top_m_actions", type=int, default=None)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--train_data_size", type=int, default=None)
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--total_epochs", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--eval_games", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--beta_kl", type=float, default=None)
    parser.add_argument("--w_base", type=float, default=None)
    parser.add_argument("--w_br", type=float, default=None)
    parser.add_argument("--w_dpo", type=float, default=None)
    parser.add_argument("--B", type=int, default=None)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--lambda_u", type=float, default=None)
    parser.add_argument("--lambda_i", type=float, default=None)
    parser.add_argument("--beta_dpo", type=float, default=None)
    parser.add_argument("--lambda_br", type=float, default=None)
    parser.add_argument("--gamma_dpo", type=float, default=None)
    parser.add_argument("--state_selection_mode", type=str, default=None)
    parser.add_argument("--action_pair_mode", type=str, default=None)
    parser.add_argument("--tau_s", type=float, default=None)
    parser.add_argument("--cdb_p", type=int, default=None)
    parser.add_argument("--cdb_eta", type=float, default=None)
    parser.add_argument("--cdb_mu", type=float, default=None)
    parser.add_argument("--cdb_delta", type=float, default=None)
    parser.add_argument("--cdb_sigma0", type=float, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default=None)
    return parser.parse_args()


def maybe_override(cfg: dict, section: str, key: str, value) -> None:
    if value is not None:
        cfg[section][key] = value


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    maybe_override(cfg, "model", "name", args.model_name)
    maybe_override(cfg, "training", "save_dir", args.save_dir)
    maybe_override(cfg, "training", "seed", args.seed)
    maybe_override(cfg, "training", "top_m_actions", args.top_m_actions)
    maybe_override(cfg, "training", "max_train_steps", args.max_train_steps)
    maybe_override(cfg, "training", "train_data_size", args.train_data_size)
    maybe_override(cfg, "training", "group_size", args.group_size)
    maybe_override(cfg, "training", "total_epochs", args.total_epochs)
    maybe_override(cfg, "training", "eval_every", args.eval_every)
    maybe_override(cfg, "training", "eval_games", args.eval_games)
    maybe_override(cfg, "training", "lr", args.lr)
    maybe_override(cfg, "training", "beta_kl", args.beta_kl)
    maybe_override(cfg, "algorithm", "w_base", args.w_base)
    maybe_override(cfg, "algorithm", "w_br", args.w_br)
    maybe_override(cfg, "algorithm", "w_dpo", args.w_dpo)
    maybe_override(cfg, "algorithm", "B", args.B)
    maybe_override(cfg, "algorithm", "K", args.K)
    maybe_override(cfg, "algorithm", "lambda_u", args.lambda_u)
    maybe_override(cfg, "algorithm", "lambda_i", args.lambda_i)
    maybe_override(cfg, "algorithm", "beta_dpo", args.beta_dpo)
    maybe_override(cfg, "algorithm", "lambda_br", args.lambda_br)
    maybe_override(cfg, "algorithm", "gamma_dpo", args.gamma_dpo)
    maybe_override(cfg, "algorithm", "state_selection_mode", args.state_selection_mode)
    maybe_override(cfg, "algorithm", "action_pair_mode", args.action_pair_mode)
    maybe_override(cfg, "algorithm", "tau_s", args.tau_s)
    maybe_override(cfg, "algorithm", "cdb_p", args.cdb_p)
    maybe_override(cfg, "algorithm", "cdb_eta", args.cdb_eta)
    maybe_override(cfg, "algorithm", "cdb_mu", args.cdb_mu)
    maybe_override(cfg, "algorithm", "cdb_delta", args.cdb_delta)
    maybe_override(cfg, "algorithm", "cdb_sigma0", args.cdb_sigma0)
    if args.wandb:
        cfg["logging"]["wandb_enabled"] = True
    maybe_override(cfg, "logging", "wandb_project", args.wandb_project)
    maybe_override(cfg, "logging", "wandb_name", args.wandb_name)
    maybe_override(cfg, "logging", "wandb_group", args.wandb_group)
    maybe_override(cfg, "logging", "wandb_entity", args.wandb_entity)
    maybe_override(cfg, "logging", "wandb_mode", args.wandb_mode)

    model_name = cfg["model"]["name"]
    save_dir = cfg["training"]["save_dir"]
    trainer = WebShopTrainer(cfg=cfg, model_name=model_name, save_dir=save_dir)
    trainer.train()


if __name__ == "__main__":
    main()
