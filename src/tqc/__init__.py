"""P4 Truncated Quantile Critics (TQC) module.

Provides two complementary layers with a unified interface:

1. **Production sb3-contrib Layer** (:mod:`tqc.sb3_adapter`):
   Typed configuration, model construction, 1-slot inference loading, validation
   callback, and distributional Critic quantile / delta inspection fully
   compatible with ``scripts/train_template.py``, ``scripts/backtest.py``, and
   ``src/thresholds/critic.py``.

2. **Reference PyTorch Layer** (:mod:`tqc.scratch_agent`, :mod:`tqc.networks`):
   Self-contained from-scratch PyTorch implementation of Kuznetsov et al. (ICML
   2020) *Controlling Overestimation Bias with Truncated Mixture of Continuous
   Distributional Quantile Critics* (ensemble quantile networks, pooled atom
   truncation, asymmetric Quantile Huber Loss, squashed Gaussian actor, and
   automatic entropy temperature tuning).
"""
from __future__ import annotations

from tqc.config import TQCConfig
from tqc.networks import QuantileCritic, quantile_huber_loss, summarize_quantile_distribution
from tqc.sb3_adapter import (
    SB3TQCWrapper,
    TQCValidationDiagnosticCallback,
    build_sb3_tqc,
    inspect_tqc_critic,
    load_sb3_tqc,
    scan_tqc_critic_deltas,
)
from tqc.scratch_agent import TorchTQCAgent

__all__ = [
    "QuantileCritic",
    "SB3TQCWrapper",
    "TQCConfig",
    "TQCValidationDiagnosticCallback",
    "TorchTQCAgent",
    "build_sb3_tqc",
    "inspect_tqc_critic",
    "load_sb3_tqc",
    "quantile_huber_loss",
    "scan_tqc_critic_deltas",
    "summarize_quantile_distribution",
]
