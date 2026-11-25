"""Standalone PyTorch implementation of Soft Actor-Critic (Haarnoja et al., 2018).

Note on Project Protocol:
    Official group benchmark models MUST be trained via ``scripts/train_template.py``
    (backed by ``stable-baselines3==2.7.1``). This ``TorchSACAgent`` provides a
    transparent, glass-box reference implementation with identical mathematical
    semantics and ``predict`` / Critic inspection interfaces for algorithm report
    documentation, controlled ablation, and cross-checking against SB3.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from sac.config import SACConfig
from sac.networks import ActionScaler, ReplayBuffer, SquashedGaussianActor, TwinQCritic


def _unwrap_env_spaces(env) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
    obs_space = env.observation_space
    act_space = env.action_space
    if len(obs_space.shape) != 1 or len(act_space.shape) != 1:
        raise ValueError("TorchSACAgent requires 1D continuous Box observation and action spaces")
    return obs_space.shape, np.asarray(act_space.low, dtype=np.float32), np.asarray(act_space.high, dtype=np.float32)


def _step_env(env, env_action: np.ndarray):
    """Support both Gymnasium 5-tuple environments and SB3 1-env VecEnvs."""
    if hasattr(env, "num_envs"):
        if env.num_envs != 1:
            raise ValueError("TorchSACAgent only supports single-environment stepping")
        next_obs, rewards, dones, infos = env.step(env_action[None, :])
        info = infos[0] if infos else {}
        terminal_obs = info.get("terminal_observation")
        bootstrap_done = bool(dones[0]) and not bool(info.get("TimeLimit.truncated", False))
        real_next = np.asarray(terminal_obs if (dones[0] and terminal_obs is not None) else next_obs[0], dtype=np.float32)
        return real_next, float(rewards[0]), bool(dones[0]), bootstrap_done, np.asarray(next_obs[0], dtype=np.float32)
    next_obs, reward, terminated, truncated, _ = env.step(env_action)
    obs_arr = np.asarray(next_obs, dtype=np.float32)
    return obs_arr, float(reward), bool(terminated or truncated), bool(terminated), obs_arr


def _reset_env(env, seed: int | None = None) -> np.ndarray:
    if hasattr(env, "num_envs"):
        if seed is not None and hasattr(env, "seed"):
            env.seed(int(seed))
        obs = env.reset()
        return np.asarray(obs[0], dtype=np.float32)
    out = env.reset(seed=seed) if seed is not None else env.reset()
    obs = out[0] if isinstance(out, tuple) else out
    return np.asarray(obs, dtype=np.float32)


class TorchSACAgent:
    """From-scratch PyTorch Soft Actor-Critic with SB3-compatible ``predict`` and Critic APIs."""

    def __init__(
        self,
        env=None,
        config: SACConfig | dict[str, Any] | None = None,
        *,
        obs_dim: int | None = None,
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ) -> None:
        if isinstance(config, dict):
            self.config = SACConfig.from_dict(config)
        elif isinstance(config, SACConfig):
            self.config = config
        elif config is None:
            self.config = SACConfig()
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
        self.critic = TwinQCritic(
            self.obs_dim,
            self.action_dim,
            hidden_dims=self.config.net_arch,
            n_critics=self.config.n_critics,
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
    def evaluate_critic(self, obs: np.ndarray, env_actions: np.ndarray) -> dict[str, np.ndarray]:
        """Evaluate all ``n_critics`` Q-networks on environment-space actions for one or more obs."""
        obs_arr = np.asarray(obs, dtype=np.float32)
        act_arr = np.asarray(env_actions, dtype=np.float32)
        if act_arr.ndim == 1:
            act_arr = act_arr[None, :]
        if obs_arr.ndim == 1:
            obs_arr = np.repeat(obs_arr[None, :], len(act_arr), axis=0)
        scaled = self.scaler.scale_to_unit(act_arr)
        obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(scaled, dtype=torch.float32, device=self.device)
        q_all = self.critic.stacked(obs_t, act_t).cpu().numpy().astype(np.float64)
        return {
            "q_values": q_all,
            "min_q": q_all.min(axis=1),
            "mean_q": q_all.mean(axis=1),
            "std_q": q_all.std(axis=1),
        }

    @torch.no_grad()
    def critic_delta(
        self,
        obs: np.ndarray,
        action: np.ndarray | None = None,
        summary: str = "min_q",
    ) -> dict[str, float]:
        """Compute ``summary(Q(s, a)) - summary(Q(s, 0))`` matching ``critic_delta_v1``."""
        if summary not in ("min_q", "mean_q"):
            raise ValueError(f"Unsupported SAC summary: {summary!r}")
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
            "policy_critic_std": float(evaluated["std_q"][0]),
            "zero_critic_std": float(evaluated["std_q"][1]),
        }

    def update(self, batch_size: int | None = None) -> dict[str, float]:
        """Run one SAC gradient update step from the replay buffer."""
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
            next_q_target = self.critic_target.stacked(next_obs, next_actions).min(dim=1, keepdim=True).values
            soft_next_value = next_q_target - alpha * next_log_prob
            target_q = rewards + (1.0 - dones) * self.config.gamma * soft_next_value

        current_q_list = self.critic(obs, actions)
        # 0.5 factor matches SB3 SAC critic loss convention
        critic_loss = 0.5 * sum(F.mse_loss(q_pred, target_q) for q_pred in current_q_list)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        q_pi_min = self.critic.stacked(obs, sampled_actions).min(dim=1, keepdim=True).values
        actor_loss = (alpha * log_prob - q_pi_min).mean()
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
            "q_min_mean": float(q_pi_min.mean().item()),
            "log_prob_mean": float(log_prob.mean().item()),
        }
        self.training_history.append(metrics)
        return metrics

    def learn(
        self,
        total_timesteps: int,
        callback: Callable[[dict[str, Any]], bool | None] | None = None,
    ) -> "TorchSACAgent":
        """Interact with ``self.env`` for ``total_timesteps`` steps and perform off-policy updates."""
        if self.env is None:
            raise ValueError("TorchSACAgent.learn requires an attached environment")
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
            "format": "finrl_p4_torch_sac_v1",
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
    ) -> "TorchSACAgent":
        checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
        if checkpoint.get("format") != "finrl_p4_torch_sac_v1":
            raise ValueError("Unrecognized TorchSACAgent checkpoint format")
        agent = cls(
            env=env,
            config=SACConfig.from_dict(checkpoint["config"]),
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
