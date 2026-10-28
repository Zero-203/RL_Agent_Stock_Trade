from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SUPPORTED_ALGORITHMS = ("ppo", "ddpg", "td3", "sac")
_REQUIRED_SECTIONS = ("data", "environment", "training", "algorithms", "output")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    missing_sections = [section for section in _REQUIRED_SECTIONS if section not in config]
    if missing_sections:
        raise ValueError(f"Missing configuration sections: {', '.join(missing_sections)}")

    for split in ("train_file", "val_file", "test_file"):
        if not config["data"].get(split):
            raise ValueError(f"data.{split} must be configured")
    if not config["data"].get("root_dir"):
        raise ValueError("data.root_dir must be configured")

    training = config["training"]
    if int(training.get("total_timesteps", 0)) <= 0:
        raise ValueError("training.total_timesteps must be positive")
    if not training.get("policy"):
        raise ValueError("training.policy must be configured")

    missing_algorithms = [
        algorithm
        for algorithm in SUPPORTED_ALGORITHMS
        if algorithm not in config["algorithms"]
    ]
    if missing_algorithms:
        raise ValueError(
            f"Missing algorithm configurations: {', '.join(missing_algorithms)}"
        )

    environment = config["environment"]
    for key in ("hmax", "initial_amount", "buy_cost_pct", "sell_cost_pct", "reward_scaling"):
        if key not in environment:
            raise ValueError(f"environment.{key} must be configured")
    if not environment.get("tech_indicator_list"):
        raise ValueError("environment.tech_indicator_list must not be empty")
    if not environment.get("risk_indicator_col"):
        raise ValueError("environment.risk_indicator_col must be configured")
