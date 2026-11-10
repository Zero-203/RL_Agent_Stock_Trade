from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SUPPORTED_ALGORITHMS = ("ppo", "ddpg", "td3", "sac", "recurrentppo", "tqc")
ON_POLICY_ALGORITHMS = ("ppo", "recurrentppo")
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
    policy = training.get("policy")
    if policy is not None and not str(policy).strip():
        raise ValueError("training.policy must not be empty")

    known = [a for a in config["algorithms"] if a in SUPPORTED_ALGORITHMS]
    unknown = sorted(set(config["algorithms"]) - set(SUPPORTED_ALGORITHMS))
    if unknown:
        raise ValueError(f"Unknown algorithm configurations: {', '.join(unknown)}")
    if not known:
        raise ValueError("algorithms must configure at least one supported algorithm")

    environment = config["environment"]
    for key in ("hmax", "initial_amount", "buy_cost_pct", "sell_cost_pct", "reward_scaling"):
        if key not in environment:
            raise ValueError(f"environment.{key} must be configured")
    if not environment.get("tech_indicator_list"):
        raise ValueError("environment.tech_indicator_list must not be empty")
    if not environment.get("risk_indicator_col"):
        raise ValueError("environment.risk_indicator_col must be configured")

    for name, section in config["algorithms"].items():
        if "policy" in section and not str(section["policy"]).strip():
            raise ValueError(f"algorithms.{name}.policy must not be empty")
        if policy is None and "policy" not in section:
            raise ValueError(f"algorithms.{name}.policy is required when training.policy is unset")


def validate_algorithm_config(config: dict[str, Any], algorithm: str) -> None:
    validate_config(config)
    normalized = algorithm.lower()
    if normalized not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    if normalized not in config["algorithms"]:
        raise ValueError(f"Missing configuration for requested algorithm: {normalized}")
