"""Typed, validated hyperparameter configuration for Soft Actor-Critic (SAC)."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any

import torch.nn as nn

ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


def parse_ent_coef_spec(spec: str | float | int) -> tuple[bool, float]:
    """Parse SB3-style ``ent_coef`` (e.g. ``'auto'``, ``'auto_0.1'``, ``0.05``).

    Returns ``(learn_alpha, initial_alpha)``.
    """
    if isinstance(spec, bool):
        raise ValueError("ent_coef cannot be a boolean")
    if isinstance(spec, (int, float)):
        value = float(spec)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("Numeric ent_coef must be finite and strictly positive")
        return False, value
    if isinstance(spec, str):
        text = spec.strip()
        if text == "auto":
            return True, 1.0
        if text.startswith("auto_"):
            try:
                init_val = float(text.split("_", 1)[1])
            except ValueError as exc:
                raise ValueError(f"Invalid auto ent_coef specification: {spec!r}") from exc
            if not math.isfinite(init_val) or init_val <= 0.0:
                raise ValueError("Initial alpha in 'auto_<val>' must be finite and > 0")
            return True, init_val
    raise ValueError(f"Unsupported ent_coef specification: {spec!r}")


@dataclass(frozen=True)
class SACConfig:
    """Validated SAC configuration interoperable with SB3 and ``TorchSACAgent``."""

    policy: str = "MlpPolicy"
    learning_rate: float = 3e-4
    buffer_size: int = 100_000
    learning_starts: int = 1_000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str | float = "auto_0.1"
    target_entropy: str | float = "auto"
    target_update_interval: int = 1
    net_arch: tuple[int, ...] = (256, 256)
    n_critics: int = 2
    activation: str = "relu"
    use_sde: bool = False

    def __post_init__(self) -> None:
        if not self.policy or not str(self.policy).strip():
            raise ValueError("policy must be a non-empty string")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and > 0")
        if int(self.buffer_size) <= 0:
            raise ValueError("buffer_size must be > 0")
        if int(self.learning_starts) < 0:
            raise ValueError("learning_starts must be >= 0")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be > 0")
        if not math.isfinite(self.tau) or not (0.0 < self.tau <= 1.0):
            raise ValueError("tau must satisfy 0 < tau <= 1")
        if not math.isfinite(self.gamma) or not (0.0 < self.gamma <= 1.0):
            raise ValueError("gamma must satisfy 0 < gamma <= 1")
        if int(self.train_freq) <= 0:
            raise ValueError("train_freq must be > 0")
        if int(self.gradient_steps) == 0 or int(self.gradient_steps) < -1:
            raise ValueError("gradient_steps must be >= 1 or -1")
        parse_ent_coef_spec(self.ent_coef)
        if self.target_entropy != "auto":
            if isinstance(self.target_entropy, bool) or not isinstance(self.target_entropy, (int, float)):
                raise ValueError("target_entropy must be 'auto' or a finite number")
            if not math.isfinite(float(self.target_entropy)):
                raise ValueError("target_entropy must be finite")
        if int(self.target_update_interval) <= 0:
            raise ValueError("target_update_interval must be > 0")
        arch = tuple(int(h) for h in self.net_arch)
        if not arch or any(h <= 0 for h in arch):
            raise ValueError("net_arch must be a non-empty sequence of positive integers")
        object.__setattr__(self, "net_arch", arch)
        if int(self.n_critics) < 1:
            raise ValueError("n_critics must be >= 1")
        act_key = str(self.activation).lower()
        if act_key not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(ACTIVATIONS)}, got {self.activation!r}")
        object.__setattr__(self, "activation", act_key)

    @property
    def activation_cls(self) -> type[nn.Module]:
        return ACTIVATIONS[self.activation]

    @property
    def auto_entropy(self) -> bool:
        return parse_ent_coef_spec(self.ent_coef)[0]

    @property
    def initial_alpha(self) -> float:
        return parse_ent_coef_spec(self.ent_coef)[1]

    def to_sb3_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs accepted by ``stable_baselines3.SAC`` / ``build_model``."""
        policy_kwargs: dict[str, Any] = {
            "net_arch": list(self.net_arch),
            "n_critics": int(self.n_critics),
        }
        if self.activation != "relu":
            policy_kwargs["activation_fn"] = self.activation_cls
        return {
            "policy": self.policy,
            "learning_rate": float(self.learning_rate),
            "buffer_size": int(self.buffer_size),
            "learning_starts": int(self.learning_starts),
            "batch_size": int(self.batch_size),
            "tau": float(self.tau),
            "gamma": float(self.gamma),
            "train_freq": int(self.train_freq),
            "gradient_steps": int(self.gradient_steps),
            "ent_coef": self.ent_coef,
            "target_entropy": self.target_entropy,
            "target_update_interval": int(self.target_update_interval),
            "use_sde": bool(self.use_sde),
            "policy_kwargs": policy_kwargs,
        }

    def to_train_template_dict(self) -> dict[str, Any]:
        """Serialize into JSON-compatible ``algorithms.sac`` section."""
        if self.activation != "relu":
            raise ValueError("JSON train_template config only supports default ReLU activation")
        payload = self.to_sb3_kwargs()
        if payload.get("policy") == "MlpPolicy":
            payload.pop("policy")
        if payload.get("target_entropy") == "auto":
            payload.pop("target_entropy")
        if payload.get("target_update_interval") == 1:
            payload.pop("target_update_interval")
        if not payload.get("use_sde"):
            payload.pop("use_sde", None)
        return payload

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, default_policy: str = "MlpPolicy") -> "SACConfig":
        """Parse an ``algorithms.sac`` dictionary from experiment JSON config."""
        data = copy.deepcopy(raw)
        policy_kwargs = dict(data.pop("policy_kwargs", None) or {})
        if "net_arch" in policy_kwargs and "net_arch" in data:
            raise ValueError("Specify net_arch either at top level or inside policy_kwargs, not both")
        net_arch = policy_kwargs.pop("net_arch", data.pop("net_arch", (256, 256)))
        if isinstance(net_arch, dict):
            pi_arch = tuple(net_arch.get("pi", (256, 256)))
            qf_arch = tuple(net_arch.get("qf", pi_arch))
            if pi_arch != qf_arch:
                raise ValueError("SACConfig requires symmetric pi/qf net_arch; pass SB3 dict directly if asymmetric")
            net_arch = pi_arch
        if "n_critics" in policy_kwargs and "n_critics" in data:
            raise ValueError("Specify n_critics either at top level or inside policy_kwargs, not both")
        n_critics = int(policy_kwargs.pop("n_critics", data.pop("n_critics", 2)))
        activation_fn = policy_kwargs.pop("activation_fn", data.pop("activation", "relu"))
        if isinstance(activation_fn, type) and issubclass(activation_fn, nn.Module):
            inv = {v: k for k, v in ACTIVATIONS.items()}
            if activation_fn not in inv:
                raise ValueError(f"Unsupported activation class: {activation_fn}")
            activation = inv[activation_fn]
        else:
            activation = str(activation_fn)
        if policy_kwargs:
            raise ValueError(f"Unsupported SAC policy_kwargs keys: {sorted(policy_kwargs)}")
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"Unknown SAC configuration fields: {unknown}")
        return cls(
            policy=str(data.pop("policy", default_policy)),
            net_arch=tuple(net_arch),
            n_critics=n_critics,
            activation=activation,
            **data,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "SACConfig":
        """Load ``SACConfig`` from either a full train_template JSON or a standalone SAC JSON."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if "algorithms" in payload and "sac" in payload["algorithms"]:
            default_policy = payload.get("training", {}).get("policy", "MlpPolicy")
            return cls.from_dict(payload["algorithms"]["sac"], default_policy=default_policy)
        return cls.from_dict(payload)
