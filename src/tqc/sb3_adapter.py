"""Production sb3-contrib TQC adapter, validation callback, and quantile Critic diagnostics."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import BaseCallback
import torch

from thresholds.critic import RULE, apply_critic_threshold
from tqc.config import TQCConfig
from tqc.networks import summarize_quantile_distribution


def build_sb3_tqc(
    env_or_vec_env,
    config: TQCConfig | dict[str, Any] | None = None,
    *,
    seed: int = 42,
    device: str = "auto",
    tensorboard_log: str | Path | None = None,
    verbose: int = 0,
) -> TQC:
    """Construct an ``sb3_contrib.TQC`` instance from a validated ``TQCConfig``."""
    from train_model.runner import build_model

    tqc_cfg = (
        TQCConfig.from_dict(config)
        if isinstance(config, dict)
        else (config if isinstance(config, TQCConfig) else TQCConfig())
    )
    vec_env = env_or_vec_env.get_sb_env()[0] if hasattr(env_or_vec_env, "get_sb_env") else env_or_vec_env
    kwargs = tqc_cfg.to_sb3_kwargs()
    policy = kwargs.pop("policy", "MlpPolicy")
    if tensorboard_log is not None:
        return build_model(
            algorithm="tqc",
            vector_environment=vec_env,
            model_kwargs=kwargs,
            seed=int(seed),
            tensorboard_log=tensorboard_log,
            device=device,
            policy=policy,
            verbose=int(verbose),
        )
    policy_kwargs = kwargs.pop("policy_kwargs", None)
    return TQC(
        policy=policy,
        env=vec_env,
        policy_kwargs=policy_kwargs,
        seed=int(seed),
        device=device,
        verbose=int(verbose),
        **kwargs,
    )


def load_sb3_tqc(model_path: str | Path, *, device: str = "cpu") -> TQC:
    """Load a saved sb3-contrib TQC ``.zip`` with a 1-slot replay buffer for memory-safe inference."""
    from train_model.runner import load_model_for_inference

    return load_model_for_inference("tqc", model_path, device=device)


@torch.no_grad()
def inspect_tqc_critic(
    model: TQC,
    obs: np.ndarray,
    action: np.ndarray | None = None,
) -> dict[str, Any]:
    """Query the full quantile Critic ensemble of an ``sb3_contrib.TQC`` model on ``a`` vs ``0``.

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
    quantiles_pair = model.critic(obs_tensor, act_tensor).detach().cpu().numpy().astype(np.float64)

    drop_per_net = int(model.top_quantiles_to_drop_per_net)
    pol_stats = summarize_quantile_distribution(quantiles_pair[0], drop_per_net)
    zero_stats = summarize_quantile_distribution(quantiles_pair[1], drop_per_net)

    return {
        "algorithm": "tqc",
        "n_critics": int(quantiles_pair.shape[1]),
        "n_quantiles": int(quantiles_pair.shape[2]),
        "top_quantiles_to_drop_per_net": drop_per_net,
        "policy_action": act_1d.copy(),
        "truncated_quantile_mean_policy": pol_stats["truncated_quantile_mean"],
        "truncated_quantile_mean_zero": zero_stats["truncated_quantile_mean"],
        "delta_truncated_quantile_mean": (
            pol_stats["truncated_quantile_mean"] - zero_stats["truncated_quantile_mean"]
        ),
        "quantile_mean_policy": pol_stats["quantile_mean"],
        "quantile_mean_zero": zero_stats["quantile_mean"],
        "delta_quantile_mean": pol_stats["quantile_mean"] - zero_stats["quantile_mean"],
        "overestimation_gap_policy": pol_stats["overestimation_gap"],
        "overestimation_gap_zero": zero_stats["overestimation_gap"],
        "iqr_policy": pol_stats["iqr"],
        "iqr_zero": zero_stats["iqr"],
        "cvar10_policy": pol_stats["cvar10"],
        "cvar10_zero": zero_stats["cvar10"],
        "inter_critic_std_policy": pol_stats["inter_critic_std"],
    }


def scan_tqc_critic_deltas(
    model: TQC,
    env,
    *,
    summary: str = "truncated_quantile_mean",
    seed: int = 0,
    quantiles: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90),
) -> dict[str, Any]:
    """Replay ``model`` unthresholded on ``env`` (e.g. validation split) and summarize ``critic_delta``."""
    if summary not in ("truncated_quantile_mean", "quantile_mean"):
        raise ValueError(f"Unsupported TQC summary {summary!r}")
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
            algorithm="tqc",
        )
        detail = inspect_tqc_critic(model, obs_arr, act_arr)
        records.append(
            {
                "step": step_idx,
                "date": dates[step_idx] if step_idx < len(dates) else str(step_idx),
                "critic_policy_value": info["critic_policy_value"],
                "critic_zero_value": info["critic_zero_value"],
                "critic_delta": info["critic_delta"],
                "overestimation_gap_policy": detail["overestimation_gap_policy"],
                "iqr_policy": detail["iqr_policy"],
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
        "algorithm": "tqc",
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


class TQCValidationDiagnosticCallback(BaseCallback):
    """Optional sb3-contrib TQC callback recording validation Sharpe, delta, and truncation gap."""

    def __init__(
        self,
        eval_env_fn,
        *,
        eval_freq: int = 5_000,
        summary: str = "truncated_quantile_mean",
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
            gaps: list[float] = []
            step_idx = 0
            while max_steps is None or step_idx < max_steps:
                obs_arr = np.asarray(obs, dtype=np.float32)
                raw_action, _ = self.model.predict(obs_arr, deterministic=True)
                act_arr = np.asarray(raw_action, dtype=np.float32).reshape(-1)
                detail = inspect_tqc_critic(self.model, obs_arr, act_arr)
                deltas.append(
                    detail["delta_truncated_quantile_mean"]
                    if self.summary == "truncated_quantile_mean"
                    else detail["delta_quantile_mean"]
                )
                gaps.append(detail["overestimation_gap_policy"])
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
                "mean_overestimation_gap": float(np.mean(gaps)) if gaps else 0.0,
            }
            self.evaluations.append(record)
            self.logger.record("val/total_return", float(perf["total_return"]))
            self.logger.record("val/mean_critic_delta", record["mean_critic_delta"])
            self.logger.record("val/mean_overestimation_gap", record["mean_overestimation_gap"])
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


class SB3TQCWrapper:
    """High-level object-oriented facade over ``sb3_contrib.TQC`` and P1/P2 pipelines."""

    def __init__(self, model: TQC, config: TQCConfig | None = None) -> None:
        self.model = model
        self.config = config

    @classmethod
    def from_env(
        cls,
        env_or_vec_env,
        config: TQCConfig | dict[str, Any] | None = None,
        *,
        seed: int = 42,
        device: str = "auto",
        tensorboard_log: str | Path | None = None,
        verbose: int = 0,
    ) -> "SB3TQCWrapper":
        cfg = (
            TQCConfig.from_dict(config)
            if isinstance(config, dict)
            else (config if isinstance(config, TQCConfig) else TQCConfig())
        )
        model = build_sb3_tqc(
            env_or_vec_env,
            cfg,
            seed=seed,
            device=device,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
        )
        return cls(model=model, config=cfg)

    @classmethod
    def load(cls, model_path: str | Path, *, device: str = "cpu") -> "SB3TQCWrapper":
        return cls(model=load_sb3_tqc(model_path, device=device))

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
            algorithm="tqc",
            seed=int(seed),
            project_root=project_root,
            total_timesteps=timesteps,
            run_type=run_type,
            device=device,
        )

    def learn(self, total_timesteps: int, callback=None, tb_log_name: str = "tqc") -> "SB3TQCWrapper":
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
        summary: str = "truncated_quantile_mean",
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs_1d = np.asarray(observation, dtype=np.float32).reshape(-1)
        raw_action, _ = self.model.predict(obs_1d, deterministic=True)
        act_1d = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        return apply_critic_threshold(
            act_1d,
            {"rule": RULE, "summary": summary, "delta_threshold": float(delta_threshold)},
            model=self.model,
            obs=obs_1d,
            algorithm="tqc",
        )

    def inspect_critic(self, obs: np.ndarray, action: np.ndarray | None = None) -> dict[str, Any]:
        return inspect_tqc_critic(self.model, obs, action)

    def scan_deltas(self, env, *, summary: str = "truncated_quantile_mean", seed: int = 0) -> dict[str, Any]:
        return scan_tqc_critic_deltas(self.model, env, summary=summary, seed=seed)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        stem = target.with_suffix("") if target.suffix == ".zip" else target
        self.model.save(str(stem))
        return stem.with_suffix(".zip")
