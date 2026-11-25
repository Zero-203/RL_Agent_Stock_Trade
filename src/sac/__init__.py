"""P4 Soft Actor-Critic (SAC) module.

Provides two complementary layers with a unified interface:

1. **Production SB3 Layer** (:mod:`sac.sb3_adapter`):
   Typed configuration, model construction, 1-slot inference loading, validation
   callback, and Critic Q-value / delta inspection fully compatible with
   ``scripts/train_template.py``, ``scripts/backtest.py``, and
   ``src/thresholds/critic.py``.

2. **Reference PyTorch Layer** (:mod:`sac.scratch_agent`, :mod:`sac.networks`):
   Self-contained from-scratch PyTorch implementation of Haarnoja et al. (2018)
   Soft Actor-Critic (squashed Gaussian actor with tanh Jacobian correction,
   twin Q-critics, Polyak target networks, and automatic entropy temperature
   tuning) for algorithm analysis, teaching/report presentation, and unit-level
   numerical verification.
"""
from __future__ import annotations

from sac.config import SACConfig
from sac.networks import ReplayBuffer, SquashedGaussianActor, TwinQCritic
from sac.sb3_adapter import (
    SB3SACWrapper,
    ValidationDiagnosticCallback,
    build_sb3_sac,
    inspect_sac_critic,
    load_sb3_sac,
    scan_sac_critic_deltas,
)
from sac.scratch_agent import TorchSACAgent

__all__ = [
    "ReplayBuffer",
    "SACConfig",
    "SB3SACWrapper",
    "SquashedGaussianActor",
    "TorchSACAgent",
    "TwinQCritic",
    "ValidationDiagnosticCallback",
    "build_sb3_sac",
    "inspect_sac_critic",
    "load_sb3_sac",
    "scan_sac_critic_deltas",
]
