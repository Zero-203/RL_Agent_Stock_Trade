"""Distributional quantile Critic networks, asymmetric Quantile Huber Loss, and truncation utilities."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from sac.networks import build_mlp


class QuantileCritic(nn.Module):
    """Ensemble of ``n_critics`` distributional networks each outputting ``n_quantiles`` atoms.

    Forward output shape is ``(batch, n_critics, n_quantiles)``, matching
    ``sb3_contrib.tqc.policies.Critic``.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        n_critics: int = 2,
        n_quantiles: int = 25,
        activation_cls: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        if n_critics < 1 or n_quantiles < 2:
            raise ValueError("Require n_critics >= 1 and n_quantiles >= 2")
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.n_critics = int(n_critics)
        self.n_quantiles = int(n_quantiles)
        self.quantiles_total = self.n_critics * self.n_quantiles
        self.qf_networks = nn.ModuleList(
            [
                build_mlp(self.obs_dim + self.action_dim, self.n_quantiles, hidden_dims, activation_cls)
                for _ in range(self.n_critics)
            ]
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        joint = torch.cat([obs, action], dim=-1)
        return torch.stack([net(joint) for net in self.qf_networks], dim=1)


def quantile_huber_loss(
    current_quantiles: torch.Tensor,
    target_quantiles: torch.Tensor,
    cum_prob: torch.Tensor | None = None,
    *,
    kappa: float = 1.0,
    sum_over_quantiles: bool = True,
) -> torch.Tensor:
    """Asymmetric Quantile Huber Loss (Dabney et al., 2018; Kuznetsov et al., 2020).

    Parameters
    ----------
    current_quantiles:
        Predicted quantiles of shape ``(batch, n_critics, n_quantiles)`` or ``(batch, n_quantiles)``.
    target_quantiles:
        Truncated Bellman target atoms of shape ``(batch, n_target_quantiles)`` or
        ``(batch, 1, n_target_quantiles)``.
    """
    if kappa <= 0.0:
        raise ValueError("kappa must be > 0")
    if current_quantiles.ndim == 2:
        current_quantiles = current_quantiles.unsqueeze(1)
    if target_quantiles.ndim == 3:
        if target_quantiles.shape[1] != 1:
            raise ValueError("3D target_quantiles must have middle dimension 1")
        target_quantiles = target_quantiles.squeeze(1)
    if current_quantiles.ndim != 3 or target_quantiles.ndim != 2:
        raise ValueError("Expected current_quantiles (B, C, M) and target_quantiles (B, K)")
    if current_quantiles.shape[0] != target_quantiles.shape[0]:
        raise ValueError("Batch size mismatch between current_quantiles and target_quantiles")

    n_quantiles = current_quantiles.shape[2]
    if cum_prob is None:
        # Midpoint quantile fractions tau_m = (m + 0.5) / M
        cum_prob = (
            torch.arange(n_quantiles, device=current_quantiles.device, dtype=current_quantiles.dtype) + 0.5
        ) / float(n_quantiles)
        cum_prob = cum_prob.view(1, 1, n_quantiles, 1)

    # Pairwise TD error: (B, 1, 1, K) - (B, C, M, 1) -> (B, C, M, K)
    pairwise_delta = target_quantiles[:, None, None, :] - current_quantiles[:, :, :, None]
    abs_delta = pairwise_delta.abs()
    huber = torch.where(
        abs_delta > kappa,
        kappa * (abs_delta - 0.5 * kappa),
        0.5 * pairwise_delta.pow(2),
    )
    indicator = (pairwise_delta.detach() < 0.0).to(current_quantiles.dtype)
    elementwise_loss = torch.abs(cum_prob - indicator) * huber / kappa

    # Average over target atoms K; sum (or average) over online quantiles M and critics C; average over batch B
    loss_per_quantile = elementwise_loss.mean(dim=-1)  # (B, C, M)
    if sum_over_quantiles:
        return loss_per_quantile.sum(dim=-1).sum(dim=-1).mean()
    return loss_per_quantile.mean()


def summarize_quantile_distribution(
    quantiles: np.ndarray,
    top_quantiles_to_drop_per_net: int,
) -> dict[str, Any]:
    """Summarize a ``(n_critics, n_quantiles)`` or ``(batch, n_critics, n_quantiles)`` quantile array."""
    arr = np.asarray(quantiles, dtype=np.float64)
    single = arr.ndim == 2
    if single:
        arr = arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D quantile array, got shape {arr.shape}")

    batch, n_critics, n_quantiles = arr.shape
    drop_total = int(top_quantiles_to_drop_per_net) * n_critics
    keep = n_critics * n_quantiles - drop_total
    if keep <= 0 or drop_total < 0:
        raise ValueError("Invalid top_quantiles_to_drop_per_net for given quantile array")

    pooled = arr.reshape(batch, n_critics * n_quantiles)
    sorted_atoms = np.sort(pooled, axis=1)
    kept_atoms = sorted_atoms[:, :keep]
    dropped_atoms = sorted_atoms[:, keep:] if drop_total > 0 else None

    trunc_mean = kept_atoms.mean(axis=1)
    full_mean = pooled.mean(axis=1)
    overestimation_gap = full_mean - trunc_mean
    iqr = np.quantile(pooled, 0.75, axis=1) - np.quantile(pooled, 0.25, axis=1)
    tail_count = max(1, int(round(0.10 * pooled.shape[1])))
    cvar10 = sorted_atoms[:, :tail_count].mean(axis=1)
    critic_means = arr.mean(axis=2)  # (batch, n_critics)
    inter_critic_std = critic_means.std(axis=1)

    if single:
        return {
            "truncated_quantile_mean": float(trunc_mean[0]),
            "quantile_mean": float(full_mean[0]),
            "overestimation_gap": float(overestimation_gap[0]),
            "dropped_top_mean": float(dropped_atoms.mean()) if dropped_atoms is not None else None,
            "iqr": float(iqr[0]),
            "cvar10": float(cvar10[0]),
            "inter_critic_std": float(inter_critic_std[0]),
            "per_critic_means": critic_means[0].tolist(),
        }
    return {
        "truncated_quantile_mean": trunc_mean,
        "quantile_mean": full_mean,
        "overestimation_gap": overestimation_gap,
        "dropped_top_mean": dropped_atoms.mean(axis=1) if dropped_atoms is not None else None,
        "iqr": iqr,
        "cvar10": cvar10,
        "inter_critic_std": inter_critic_std,
        "per_critic_means": critic_means,
    }
