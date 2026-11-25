"""Production SB3 SAC adapter, validation callback, and Critic delta diagnostics.

All official P4 benchmark runs go through ``scripts/train_template.py`` and
``scripts/backtest.py``. This module wraps ``stable_baselines3.SAC`` and
``train_model.runner`` cleanly without altering any P1/P2 contracts or shared
environment code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
import torch

from sac.config import SACConfig
from thresholds.critic import RULE, apply_critic_threshold


def build_sb3_sac(
    env_or_vec_env,
    config: SACConfig | dict[str, Any] | None = None,
    *,
    seed: int = 42,
    device: str = "auto",
    tensorboard_log: str | Path | None = None,
    verbose: int = 0,
) -> SAC:
    """Construct a ``stable_baselines3.SAC`` instance from a validated ``SACConfig``."""
    from train_model.runner import build_model

    sac_cfg = (
        SACConfig.from_dict(config)
        if isinstance(config, dict)
        else (config if isinstance(config, SACConfig) else SACConfig())
    )
    vec_env = env_or_vec_env.get_sb_env()[0] if hasattr(env_or_vec_env, "get_sb_env") else env_or_vec_env
    kwargs = sac_cfg.to_sb3_kwargs()
    policy = kwargs.pop("policy", "MlpPolicy")
    if tensorboard_log is not None:
        return build_model(
            algorithm="sac",
            vector_environment=vec_env,
            model_kwargs=kwargs,
            seed=int(seed),
            tensorboard_log=tensorboard_log,
            device=device,
            policy=policy,
            verbose=int(verbose),
        )
    policy_kwargs = kwargs.pop("policy_kwargs", None)
    return SAC(
        policy=policy,
        env=vec_env,
        policy_kwargs=policy_kwargs,
        seed=int(seed),
        device=device,
        verbose=int(verbose),
        **kwargs,
    )


def load_sb3_sac(model_path: str | Path, *, device: str = "cpu") -> SAC:
    """Load a saved SB3 SAC ``.zip`` with a 1-slot replay buffer for memory-safe inference."""
    from train_model.runner import load_model_for_inference

    return load_model_for_inference("sac", model_path, device=device)


@torch.no_grad()
def inspect_sac_critic(
    model: SAC,
    obs: np.ndarray,
    action: np.ndarray | None = None,
) -> dict[str, Any]:
    """Query all Q-networks of a loaded SB3 ``SAC`` model on ``a`` vs ``0``.

    Uses the exact same observation tensorization and ``policy.scale_action`` path
    as ``src/thresholds/critic.py`` (rule ``critic_delta_v1``).
    """
    obs_1d = np.asarray(obs, dtype=np.float32).reshape(-1)
    if action is None:
        raw_action, _ = model.predict(obs_1d, deterministic=True)
        act_1d = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    else:
        act_1d = np.asarray(action, dtype=np.float32).reshape(-1)
    zero_1d = np.zeros_like(act_1d, dtype=np.float32)

    policy = model.policy
    actions_pair = np.stack([act_1d, zero_1d], axis=0)
    scaled = policy.scale_action(actions_pair)
    obs_tensor, _ = policy.obs_to_tensor(np.repeat(obs_1d[None, :], 2, axis=0))
    act_tensor = torch.as_tensor(scaled, dtype=torch.float32, device=obs_tensor.device)
    q_pair = torch.cat(model.critic(obs_tensor, act_tensor), dim=1).detach().cpu().numpy().astype(np.float64)

    q_policy = q_pair[0]
    q_zero = q_pair[1]
    min_policy, min_zero = float(q_policy.min()), float(q_zero.min())
    mean_policy, mean_zero = float(q_policy.mean()), float(q_zero.mean())
    return {
        "algorithm": "sac",
        "n_critics": int(q_pair.shape[1]),
        "policy_action": act_1d.copy(),
        "q_policy_per_critic": q_policy.tolist(),
        "q_zero_per_critic": q_zero.tolist(),
        "min_q_policy": min_policy,
        "min_q_zero": min_zero,
        "delta_min_q": min_policy - min_zero,
        "mean_q_policy": mean_policy,
        "mean_q_zero": mean_zero,
        "delta_mean_q": mean_policy - mean_zero,
        "policy_critic_spread": float(q_policy.max() - q_policy.min()),
        "zero_critic_spread": float(q_zero.max() - q_zero.min()),
    }


def scan_sac_critic_deltas(
    model: SAC,
    env,
    *,
    summary: str = "min_q",
    seed: int = 0,
    quantiles: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90),
) -> dict[str, Any]:
    """Replay ``model`` unthresholded on ``env`` (e.g. validation split) and summarize ``critic_delta``.

    Intended for P4's pilot-based candidate threshold calibration on ``val_data.csv``
    without touching ``trade_data.csv``.
    """
    if summary not in ("min_q", "mean_q"):
        raise ValueError(f"Unsupported SAC summary {summary!r}")
    obs, _ = env.reset(seed=int(seed))
    dates = env.df["date"].drop_duplicates().tolist() if hasattr(env, "df") else []
    max_steps = (len(dates) - 1) if len(dates) > 1 else None
    records: list[dict[str, Any]] = []
    step_idx = 0
    while max_steps is None or step_idx < max_steps:
        obs_arr = np.asarray(obs, dtype=np.float32)
        raw_action, _ = model.predict(obs_arr, deterministic=True)
        act_arr = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        _, info = apply_critic_threshold(
            act_arr,
            {"rule": RULE, "summary": summary, "delta_threshold": -1e12},
            model=model,
            obs=obs_arr,
            algorithm="sac",
        )
        detail = inspect_sac_critic(model, obs_arr, act_arr)
        records.append(
            {
                "step": step_idx,
                "date": dates[step_idx] if step_idx < len(dates) else str(step_idx),
                "critic_policy_value": info["critic_policy_value"],
                "critic_zero_value": info["critic_zero_value"],
                "critic_delta": info["critic_delta"],
                "policy_critic_spread": detail["policy_critic_spread"],
                "mean_abs_action": float(np.abs(act_arr).mean()),
            }
        )
        obs, _, terminated, truncated, _ = env.step(act_arr)
        step_idx += 1
        if terminated or truncated:
            break

    frame = pd.DataFrame(records)
    deltas = frame["critic_delta"].to_numpy(dtype=np.float64)
    q_map = {f"p{int(round(q * 100)):02d}": float(np.quantile(deltas, q)) for q in quantiles}
    suggested = sorted({0.0, *(round(float(np.quantile(deltas, q)), 6) for q in (0.25, 0.50, 0.75))})
    return {
        "algorithm": "sac",
        "rule": RULE,
        "summary": summary,
        "observations": int(len(frame)),
        "delta_min": float(deltas.min()),
        "delta_mean": float(deltas.mean()),
        "delta_std": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
        "delta_max": float(deltas.max()),
        "positive_delta_fraction": float((deltas > 0.0).mean()),
        "quantiles": q_map,
        "suggested_candidates": suggested,
        "daily_records": frame,
    }


class ValidationDiagnosticCallback(BaseCallback):
    """Optional SB3 callback recording validation return, Sharpe, and mean ``critic_delta``."""

    def __init__(
        self,
        eval_env_fn,
        *,
        eval_freq: int = 5_000,
        summary: str = "min_q",
        best_model_save_path: str | Path | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        if int(eval_freq) <= 0:
            raise ValueError("eval_freq must be > 0")
        self.eval_env_fn = eval_env_fn
        self.eval_freq = int(eval_freq)
        self.summary = summary
        self.best_model_save_path = Path(best_model_save_path) if best_model_save_path else None
        self.best_sharpe: float = -np.inf
        self.evaluations: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        from backtest.metrics import curve_metrics

        env = self.eval_env_fn()
        try:
            obs, _ = env.reset(seed=0)
            dates = env.df["date"].drop_duplicates().tolist() if hasattr(env, "df") else []
            max_steps = (len(dates) - 1) if len(dates) > 1 else None
            values = [float(env.initial_amount)]
            deltas: list[float] = []
            step_idx = 0
            while max_steps is None or step_idx < max_steps:
                obs_arr = np.asarray(obs, dtype=np.float32)
                raw_action, _ = self.model.predict(obs_arr, deterministic=True)
                act_arr = np.asarray(raw_action, dtype=np.float32).reshape(-1)
                detail = inspect_sac_critic(self.model, obs_arr, act_arr)
                deltas.append(detail["delta_min_q"] if self.summary == "min_q" else detail["delta_mean_q"])
                obs, _, terminated, truncated, _ = env.step(act_arr)
                n_assets = len(act_arr)
                cash = float(env.state[0])
                prices = np.asarray(env.state[1 : 1 + n_assets], dtype=np.float64)
                positions = np.asarray(env.state[1 + n_assets : 1 + 2 * n_assets], dtype=np.float64)
                values.append(cash + float(prices @ positions))
                step_idx += 1
                if terminated or truncated:
                    break
            perf = curve_metrics(pd.Series(values))
            sharpe = perf.get("sharpe")
            record = {
                "timesteps": int(self.num_timesteps),
                "total_return": perf["total_return"],
                "sharpe": sharpe,
                "max_drawdown": perf["max_drawdown"],
                "mean_critic_delta": float(np.mean(deltas)) if deltas else 0.0,
            }
            self.evaluations.append(record)
            self.logger.record("val/total_return", float(perf["total_return"]))
            self.logger.record("val/mean_critic_delta", record["mean_critic_delta"])
            if sharpe is not None and np.isfinite(sharpe):
                self.logger.record("val/sharpe", float(sharpe))
                if float(sharpe) > self.best_sharpe:
                    self.best_sharpe = float(sharpe)
                    if self.best_model_save_path is not None:
                        self.best_model_save_path.parent.mkdir(parents=True, exist_ok=True)
                        self.model.save(str(self.best_model_save_path))
        finally:
            env.close()
        return True


class SB3SACWrapper:
    """High-level object-oriented facade over ``stable_baselines3.SAC`` and P1/P2 pipelines."""

    def __init__(self, model: SAC, config: SACConfig | None = None) -> None:
        self.model = model
        self.config = config

    @classmethod
    def from_env(
        cls,
        env_or_vec_env,
        config: SACConfig | dict[str, Any] | None = None,
        *,
        seed: int = 42,
        device: str = "auto",
        tensorboard_log: str | Path | None = None,
        verbose: int = 0,
    ) -> "SB3SACWrapper":
        cfg = (
            SACConfig.from_dict(config)
            if isinstance(config, dict)
            else (config if isinstance(config, SACConfig) else SACConfig())
        )
        model = build_sb3_sac(
            env_or_vec_env,
            cfg,
            seed=seed,
            device=device,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
        )
        return cls(model=model, config=cfg)

    @classmethod
    def load(cls, model_path: str | Path, *, device: str = "cpu") -> "SB3SACWrapper":
        return cls(model=load_sb3_sac(model_path, device=device))

    @staticmethod
    def run_official_training(
        config_path: str | Path,
        dataset_name: str,
        seed: int,
        project_root: str | Path,
        *,
        timesteps: int | None = None,
        run_type: str = "main",
        device: str | None = None,
    ) -> dict[str, Any]:
        """Delegate directly to P1's official ``train_model.runner.run_training``."""
        from train_model import load_config, run_training

        cfg = load_config(config_path)
        return run_training(
            config=cfg,
            dataset_name=dataset_name,
            algorithm="sac",
            seed=int(seed),
            project_root=project_root,
            total_timesteps=timesteps,
            run_type=run_type,
            device=device,
        )

    def learn(self, total_timesteps: int, callback=None, tb_log_name: str = "sac") -> "SB3SACWrapper":
        self.model.learn(total_timesteps=int(total_timesteps), callback=callback, tb_log_name=tb_log_name)
        return self

    def predict(
        self,
        observation: np.ndarray,
        state=None,
        episode_start=None,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, Any]:
        return self.model.predict(observation, state=state, episode_start=episode_start, deterministic=deterministic)

    def predict_with_threshold(
        self,
        observation: np.ndarray,
        *,
        delta_threshold: float = 0.0,
        summary: str = "min_q",
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs_1d = np.asarray(observation, dtype=np.float32).reshape(-1)
        raw_action, _ = self.model.predict(obs_1d, deterministic=True)
        act_1d = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        return apply_critic_threshold(
            act_1d,
            {"rule": RULE, "summary": summary, "delta_threshold": float(delta_threshold)},
            model=self.model,
            obs=obs_1d,
            algorithm="sac",
        )

    def inspect_critic(self, obs: np.ndarray, action: np.ndarray | None = None) -> dict[str, Any]:
        return inspect_sac_critic(self.model, obs, action)

    def scan_deltas(self, env, *, summary: str = "min_q", seed: int = 0) -> dict[str, Any]:
        return scan_sac_critic_deltas(self.model, env, summary=summary, seed=seed)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        stem = target.with_suffix("") if target.suffix == ".zip" else target
        self.model.save(str(stem))
        return stem.with_suffix(".zip")
