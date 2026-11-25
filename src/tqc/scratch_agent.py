"""Standalone PyTorch implementation of TQC (Kuznetsov et al., ICML 2020).

Note on Project Protocol:
    Official group benchmark models MUST be trained via ``scripts/train_template.py``
    (backed by ``sb3-contrib==2.7.1``). This ``TorchTQCAgent`` provides a
    transparent, glass-box reference implementation of Truncated Quantile Critics
    with identical ``predict`` and Critic quantile summary semantics for report
    documentation, distributional bias analysis, and unit-level verification.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from sac.networks import ActionScaler, ReplayBuffer, SquashedGaussianActor
from sac.scratch_agent import _reset_env, _step_env, _unwrap_env_spaces
from tqc.config import TQCConfig
from tqc.networks import QuantileCritic, quantile_huber_loss, summarize_quantile_distribution


class TorchTQCAgent:
    """From-scratch PyTorch Truncated Quantile Critics agent."""

    def __init__(
        self,
        env=None,
        config: TQCConfig | dict[str, Any] | None = None,
        *,
        obs_dim: int | None = None,
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ) -> None:
        if isinstance(config, dict):
            self.config = TQCConfig.from_dict(config)
        elif isinstance(config, TQCConfig):
            self.config = config
        elif config is None:
            self.config = TQCConfig()
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")

        self.env = env
        if env is not None:
            obs_shape, low, high = _unwrap_env_spaces(env)
            self.obs_dim = int(obs_shape[0])
            self.scaler = ActionScaler(low, high)
        else:
            if obs_dim is None or action_low is None or action_high is None:
                raise ValueError("Provide either env or (obs_dim, action_low, action_high)")
            self.obs_dim = int(obs_dim)
            self.scaler = ActionScaler(action_low, action_high)

        self.action_dim = int(self.scaler.low.shape[0])
        self.top_quantiles_to_drop_per_net = int(self.config.top_quantiles_to_drop_per_net)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.actor = SquashedGaussianActor(
            self.obs_dim,
            self.action_dim,
            hidden_dims=self.config.net_arch,
            activation_cls=self.config.activation_cls,
        ).to(self.device)
        self.critic = QuantileCritic(
            self.obs_dim,
            self.action_dim,
            hidden_dims=self.config.net_arch,
            n_critics=self.config.n_critics,
            n_quantiles=self.config.n_quantiles,
            activation_cls=self.config.activation_cls,
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_target.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.learning_rate)

        self.target_entropy = (
            -float(self.action_dim)
            if self.config.target_entropy == "auto"
            else float(self.config.target_entropy)
        )
        self.auto_entropy = self.config.auto_entropy
        self.log_ent_coef = torch.tensor(
            [np.log(self.config.initial_alpha)],
            dtype=torch.float32,
            device=self.device,
            requires_grad=self.auto_entropy,
        )
        self.ent_coef_optimizer = (
            torch.optim.Adam([self.log_ent_coef], lr=self.config.learning_rate)
            if self.auto_entropy
            else None
        )

        self.replay_buffer = ReplayBuffer(
            capacity=self.config.buffer_size,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            seed=self.seed,
        )
        self.num_timesteps = 0
        self._n_updates = 0
        self.training_history: list[dict[str, float]] = []

    @property
    def ent_coef_value(self) -> float:
        return float(self.log_ent_coef.detach().exp().item())

    def predict(
        self,
        observation: np.ndarray,
        state=None,
        episode_start=None,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, None]:
        """SB3-compatible prediction returning ``(action_in_env_bounds, None)``."""
        obs_arr = np.asarray(observation, dtype=np.float32)
        batched = obs_arr.ndim == 2
        obs_2d = obs_arr if batched else obs_arr.reshape(1, self.obs_dim)
        obs_t = torch.as_tensor(obs_2d, dtype=torch.float32, device=self.device)
        self.actor.eval()
        with torch.no_grad():
            if deterministic:
                squashed = self.actor.deterministic_action(obs_t)
            else:
                squashed, _ = self.actor.sample_and_log_prob(obs_t)
        unit_action = squashed.cpu().numpy()
        env_action = self.scaler.unscale_to_env(unit_action)
        return (env_action if batched else env_action[0]), None

    @torch.no_grad()
    def evaluate_critic(self, obs: np.ndarray, env_actions: np.ndarray) -> dict[str, Any]:
        """Evaluate full ``(batch, n_critics, n_quantiles)`` distribution and summaries."""
        obs_arr = np.asarray(obs, dtype=np.float32)
        act_arr = np.asarray(env_actions, dtype=np.float32)
        if act_arr.ndim == 1:
            act_arr = act_arr[None, :]
        if obs_arr.ndim == 1:
            obs_arr = np.repeat(obs_arr[None, :], len(act_arr), axis=0)
        scaled = self.scaler.scale_to_unit(act_arr)
        obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(scaled, dtype=torch.float32, device=self.device)
        quantiles = self.critic(obs_t, act_t).cpu().numpy().astype(np.float64)
        summary = summarize_quantile_distribution(quantiles, self.top_quantiles_to_drop_per_net)
        return {"quantiles": quantiles, **summary}

    @torch.no_grad()
    def critic_delta(
        self,
        obs: np.ndarray,
        action: np.ndarray | None = None,
        summary: str = "truncated_quantile_mean",
    ) -> dict[str, float]:
        """Compute ``summary(Z(s, a)) - summary(Z(s, 0))`` matching ``critic_delta_v1``."""
        if summary not in ("truncated_quantile_mean", "quantile_mean"):
            raise ValueError(f"Unsupported TQC summary: {summary!r}")
        obs_1d = np.asarray(obs, dtype=np.float32).reshape(self.obs_dim)
        if action is None:
            policy_action, _ = self.predict(obs_1d, deterministic=True)
        else:
            policy_action = np.asarray(action, dtype=np.float32).reshape(self.action_dim)
        zero_action = np.zeros_like(policy_action, dtype=np.float32)
        evaluated = self.evaluate_critic(obs_1d, np.stack([policy_action, zero_action], axis=0))
        series = evaluated[summary]
        policy_val, zero_val = float(series[0]), float(series[1])
        return {
            "summary": summary,
            "critic_policy_value": policy_val,
            "critic_zero_value": zero_val,
            "critic_delta": policy_val - zero_val,
            "policy_overestimation_gap": float(evaluated["overestimation_gap"][0]),
            "policy_iqr": float(evaluated["iqr"][0]),
        }

    def update(self, batch_size: int | None = None) -> dict[str, float]:
        """Run one TQC gradient update step from the replay buffer."""
        self.actor.train()
        self.critic.train()
        bsz = int(batch_size or self.config.batch_size)
        batch = self.replay_buffer.sample(bsz, device=self.device)
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        sampled_actions, log_prob = self.actor.sample_and_log_prob(obs)
        if self.auto_entropy and self.ent_coef_optimizer is not None:
            alpha = self.log_ent_coef.detach().exp()
            ent_coef_loss = -(self.log_ent_coef * (log_prob.detach() + self.target_entropy)).mean()
            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()
            ent_loss_val = float(ent_coef_loss.item())
        else:
            alpha = self.log_ent_coef.detach().exp()
            ent_loss_val = 0.0

        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample_and_log_prob(next_obs)
            next_quantiles = self.critic_target(next_obs, next_actions)  # (B, C, M)
            pooled = next_quantiles.reshape(bsz, -1)
            sorted_atoms, _ = torch.sort(pooled, dim=1)
            truncated_atoms = sorted_atoms[:, : self.config.kept_quantiles]  # (B, K)
            target_quantiles = rewards + (1.0 - dones) * self.config.gamma * (
                truncated_atoms - alpha * next_log_prob
            )

        current_quantiles = self.critic(obs, actions)
        critic_loss = quantile_huber_loss(current_quantiles, target_quantiles)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Kuznetsov et al. (2020) Eq. 11 & sb3-contrib TQC: actor maximizes full quantile mean
        qf_pi_mean = self.critic(obs, sampled_actions).mean(dim=2).mean(dim=1, keepdim=True)
        actor_loss = (alpha * log_prob - qf_pi_mean).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._n_updates += 1
        if self._n_updates % self.config.target_update_interval == 0:
            tau = self.config.tau
            with torch.no_grad():
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)

        metrics = {
            "step": float(self.num_timesteps),
            "n_updates": float(self._n_updates),
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "ent_coef_loss": ent_loss_val,
            "ent_coef": float(alpha.item()),
            "qf_pi_mean": float(qf_pi_mean.mean().item()),
            "log_prob_mean": float(log_prob.mean().item()),
        }
        self.training_history.append(metrics)
        return metrics

    def learn(
        self,
        total_timesteps: int,
        callback: Callable[[dict[str, Any]], bool | None] | None = None,
    ) -> "TorchTQCAgent":
        """Interact with ``self.env`` for ``total_timesteps`` steps and perform TQC updates."""
        if self.env is None:
            raise ValueError("TorchTQCAgent.learn requires an attached environment")
        if int(total_timesteps) <= 0:
            raise ValueError("total_timesteps must be positive")

        obs = _reset_env(self.env, seed=self.seed)
        for _ in range(int(total_timesteps)):
            if self.num_timesteps < self.config.learning_starts:
                unit_action = self._rng.uniform(-1.0, 1.0, size=self.action_dim).astype(np.float32)
                env_action = self.scaler.unscale_to_env(unit_action)
            else:
                env_action, _ = self.predict(obs, deterministic=False)
                unit_action = self.scaler.scale_to_unit(env_action)

            real_next_obs, reward, episode_done, bootstrap_done, post_step_obs = _step_env(self.env, env_action)
            self.replay_buffer.add(obs, unit_action, reward, real_next_obs, bootstrap_done)
            self.num_timesteps += 1

            update_info: dict[str, float] | None = None
            if (
                self.num_timesteps >= self.config.learning_starts
                and len(self.replay_buffer) >= min(self.config.batch_size, max(1, self.config.learning_starts))
                and self.num_timesteps % self.config.train_freq == 0
            ):
                grad_steps = (
                    self.config.train_freq
                    if self.config.gradient_steps == -1
                    else self.config.gradient_steps
                )
                eff_bsz = min(self.config.batch_size, len(self.replay_buffer))
                for _ in range(grad_steps):
                    update_info = self.update(batch_size=eff_bsz)

            if callback is not None:
                cont = callback(
                    {
                        "num_timesteps": self.num_timesteps,
                        "n_updates": self._n_updates,
                        "reward": reward,
                        "episode_done": episode_done,
                        "update_info": update_info,
                    }
                )
                if cont is False:
                    break

            if episode_done and not hasattr(self.env, "num_envs"):
                obs = _reset_env(self.env)
            else:
                obs = post_step_obs
        return self

    def save(self, path: str | Path) -> Path:
        """Persist weights, optimizer states, and configuration to a ``.pt`` checkpoint."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "finrl_p4_torch_tqc_v1",
            "config": self.config.to_sb3_kwargs(),
            "obs_dim": self.obs_dim,
            "action_low": self.scaler.low,
            "action_high": self.scaler.high,
            "seed": self.seed,
            "num_timesteps": self.num_timesteps,
            "n_updates": self._n_updates,
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "log_ent_coef": self.log_ent_coef.detach().cpu(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "ent_coef_optimizer": (
                self.ent_coef_optimizer.state_dict() if self.ent_coef_optimizer is not None else None
            ),
        }
        torch.save(payload, target)
        return target

    @classmethod
    def load(
        cls,
        path: str | Path,
        env=None,
        device: str | torch.device = "cpu",
    ) -> "TorchTQCAgent":
        checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
        if checkpoint.get("format") != "finrl_p4_torch_tqc_v1":
            raise ValueError("Unrecognized TorchTQCAgent checkpoint format")
        agent = cls(
            env=env,
            config=TQCConfig.from_dict(checkpoint["config"]),
            obs_dim=int(checkpoint["obs_dim"]),
            action_low=np.asarray(checkpoint["action_low"], dtype=np.float32),
            action_high=np.asarray(checkpoint["action_high"], dtype=np.float32),
            seed=int(checkpoint["seed"]),
            device=device,
        )
        agent.actor.load_state_dict(checkpoint["actor_state_dict"])
        agent.critic.load_state_dict(checkpoint["critic_state_dict"])
        agent.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        with torch.no_grad():
            agent.log_ent_coef.copy_(checkpoint["log_ent_coef"].to(agent.device))
        agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        if agent.ent_coef_optimizer is not None and checkpoint.get("ent_coef_optimizer") is not None:
            agent.ent_coef_optimizer.load_state_dict(checkpoint["ent_coef_optimizer"])
        agent.num_timesteps = int(checkpoint["num_timesteps"])
        agent._n_updates = int(checkpoint["n_updates"])
        return agent
