"""Pure PyTorch neural networks and replay buffer for SAC (reusable by TQC)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: Sequence[int] = (256, 256),
    activation_cls: type[nn.Module] = nn.ReLU,
) -> nn.Sequential:
    """Construct a standard feedforward MLP without output activation."""
    layers: list[nn.Module] = []
    prev_dim = int(input_dim)
    for hidden in hidden_dims:
        layers.append(nn.Linear(prev_dim, int(hidden)))
        layers.append(activation_cls())
        prev_dim = int(hidden)
    layers.append(nn.Linear(prev_dim, int(output_dim)))
    return nn.Sequential(*layers)


class ActionScaler:
    """Affine bijection between policy squashed space ``[-1, 1]`` and ``Box(low, high)``."""

    def __init__(self, low: np.ndarray, high: np.ndarray) -> None:
        low_arr = np.asarray(low, dtype=np.float32).reshape(-1)
        high_arr = np.asarray(high, dtype=np.float32).reshape(-1)
        if low_arr.shape != high_arr.shape or not np.all(high_arr > low_arr):
            raise ValueError("Action space bounds must be 1D with high > low everywhere")
        self.low = low_arr
        self.high = high_arr
        self.half_range = 0.5 * (high_arr - low_arr)
        self.midpoint = 0.5 * (high_arr + low_arr)

    def unscale_to_env(self, squashed_action: np.ndarray) -> np.ndarray:
        """Map ``[-1, 1]`` policy output to environment ``[low, high]``."""
        return (self.midpoint + self.half_range * np.asarray(squashed_action, dtype=np.float32)).astype(np.float32)

    def scale_to_unit(self, env_action: np.ndarray) -> np.ndarray:
        """Map environment ``[low, high]`` action into ``[-1, 1]`` critic input coordinates."""
        return ((np.asarray(env_action, dtype=np.float32) - self.midpoint) / self.half_range).astype(np.float32)


class ReplayBuffer:
    """Pre-allocated circular experience replay buffer for off-policy continuous control."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, seed: int = 0) -> None:
        if capacity <= 0 or obs_dim <= 0 or action_dim <= 0:
            raise ValueError("ReplayBuffer capacity, obs_dim, and action_dim must be positive")
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self._rng = np.random.default_rng(int(seed))

        self.obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

        self._ptr = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition; ``action`` is stored in unit ``[-1, 1]`` coordinates."""
        self.obs[self._ptr] = np.asarray(obs, dtype=np.float32).reshape(self.obs_dim)
        self.actions[self._ptr] = np.asarray(action, dtype=np.float32).reshape(self.action_dim)
        self.rewards[self._ptr, 0] = float(reward)
        self.next_obs[self._ptr] = np.asarray(next_obs, dtype=np.float32).reshape(self.obs_dim)
        self.dones[self._ptr, 0] = 1.0 if bool(done) else 0.0
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
        if self._size == 0:
            raise RuntimeError("Cannot sample from an empty ReplayBuffer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        indices = self._rng.integers(0, self._size, size=int(batch_size))
        return {
            "obs": torch.as_tensor(self.obs[indices], dtype=torch.float32, device=device),
            "actions": torch.as_tensor(self.actions[indices], dtype=torch.float32, device=device),
            "rewards": torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=device),
            "next_obs": torch.as_tensor(self.next_obs[indices], dtype=torch.float32, device=device),
            "dones": torch.as_tensor(self.dones[indices], dtype=torch.float32, device=device),
        }


class SquashedGaussianActor(nn.Module):
    """Tanh-squashed diagonal Gaussian policy with exact log-det-Jacobian correction."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        activation_cls: type[nn.Module] = nn.ReLU,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.epsilon = float(epsilon)
        layers: list[nn.Module] = []
        prev = self.obs_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev, int(hidden)))
            layers.append(activation_cls())
            prev = int(hidden)
        self.trunk = nn.Sequential(*layers)
        self.mu_head = nn.Linear(prev, self.action_dim)
        self.log_std_head = nn.Linear(prev, self.action_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(obs)
        mu = self.mu_head(features)
        log_std = torch.clamp(self.log_std_head(features), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample_and_log_prob(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterized action ``a = tanh(u)`` in ``[-1, 1]`` and exact ``log pi(a|s)``."""
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        gaussian_u = dist.rsample()
        squashed_action = torch.tanh(gaussian_u)
        # Haarnoja et al. (2018) Appendix C: log pi(a|s) = log mu(u|s) - sum log(1 - tanh(u)^2)
        log_prob = dist.log_prob(gaussian_u) - torch.log(1.0 - squashed_action.pow(2) + self.epsilon)
        return squashed_action, log_prob.sum(dim=-1, keepdim=True)

    @torch.no_grad()
    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        mu, _ = self.forward(obs)
        return torch.tanh(mu)


class TwinQCritic(nn.Module):
    """Ensemble of ``n_critics`` independent state-action scalar Q-networks."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        n_critics: int = 2,
        activation_cls: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        if n_critics < 1:
            raise ValueError("n_critics must be >= 1")
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.n_critics = int(n_critics)
        self.q_networks = nn.ModuleList(
            [
                build_mlp(self.obs_dim + self.action_dim, 1, hidden_dims, activation_cls)
                for _ in range(self.n_critics)
            ]
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return a tuple of ``(batch, 1)`` Q-value tensors (matches SB3 ``ContinuousCritic``)."""
        joint = torch.cat([obs, action], dim=-1)
        return tuple(net(joint) for net in self.q_networks)

    def stacked(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return ``(batch, n_critics)`` tensor across all Q-networks."""
        return torch.cat(self.forward(obs, action), dim=1)
