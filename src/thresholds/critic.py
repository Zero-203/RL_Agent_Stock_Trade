"""P4 inference-time Critic threshold for SAC and TQC (rule ``critic_delta_v1``).

For one observation ``s`` and the model's deterministic action ``a`` (environment
coordinates, shape ``(n,)`` in ``[-1, 1]``), compare the critic summary of ``a``
with that of the zero ("hold everything") action using the *same* observation,
action scaling and summary rule::

    delta = summary(Q(s, a)) - summary(Q(s, 0))

The whole action vector is accepted iff ``delta > delta_threshold`` (strict; an
equal value is rejected, so the critic must prefer trading). Rejection returns an
all-zero vector. Non-finite critic values raise instead of guessing a decision.

Registered summaries (the default of each matches the algorithm's own TD target):

* SAC  ``min_q``                    min over the twin Q networks.
* SAC  ``mean_q``                   mean over the twin Q networks.
* TQC  ``truncated_quantile_mean``  pool all critics' quantiles, drop the top
  ``top_quantiles_to_drop_per_net * n_critics`` and average the rest.
* TQC  ``quantile_mean``            mean of all quantiles of all critics.

The threshold never trains the model, edits weights or touches the environment;
cash, position, suspension and risk-liquidation rules are still enforced later by
the shared environment.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

RULE = "critic_delta_v1"
SUMMARIES = {
    "sac": ("min_q", "mean_q"),
    "tqc": ("truncated_quantile_mean", "quantile_mean"),
}
_REQUIRED_KEYS = ("rule", "summary", "delta_threshold")


def validate_critic_config(config: dict[str, Any], algorithm: str) -> tuple[str, float]:
    """Return ``(summary, delta_threshold)``; reject incomplete or ambiguous configs."""
    if algorithm not in SUMMARIES:
        raise ValueError(f"Critic threshold supports only {sorted(SUMMARIES)}, got {algorithm!r}")
    if not isinstance(config, dict):
        raise ValueError("Critic threshold_config must be a JSON object")
    missing = [key for key in _REQUIRED_KEYS if key not in config]
    unknown = sorted(set(config) - set(_REQUIRED_KEYS))
    if missing or unknown:
        raise ValueError(f"Critic threshold_config keys invalid; missing={missing}, unknown={unknown}")
    if config["rule"] != RULE:
        raise ValueError(f"Unsupported critic rule {config['rule']!r}; expected {RULE!r}")
    summary = config["summary"]
    if summary not in SUMMARIES[algorithm]:
        raise ValueError(f"Summary {summary!r} is not registered for {algorithm}: {SUMMARIES[algorithm]}")
    threshold = config["delta_threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
        raise ValueError("delta_threshold must be a finite number")
    return summary, float(threshold)


def _summaries(model, algorithm: str, obs: np.ndarray, actions: np.ndarray, summary: str) -> np.ndarray:
    """Critic summary for a batch of environment-space actions sharing one observation."""
    import torch

    policy = model.policy
    scaled = policy.scale_action(np.asarray(actions, dtype=np.float32))
    obs_tensor, _ = policy.obs_to_tensor(np.repeat(obs[None, :], len(actions), axis=0))
    action_tensor = torch.as_tensor(scaled, dtype=torch.float32, device=obs_tensor.device)
    with torch.no_grad():
        if algorithm == "sac":
            q_values = torch.cat(model.critic(obs_tensor, action_tensor), dim=1)  # (batch, n_critics)
            values = q_values.min(dim=1).values if summary == "min_q" else q_values.mean(dim=1)
        else:
            quantiles = model.critic(obs_tensor, action_tensor)  # (batch, n_critics, n_quantiles)
            n_critics = quantiles.shape[1]
            pooled = quantiles.reshape(len(actions), -1)
            if summary == "truncated_quantile_mean":
                keep = pooled.shape[1] - int(model.top_quantiles_to_drop_per_net) * n_critics
                if keep <= 0:
                    raise ValueError("TQC truncation drops every quantile")
                pooled = torch.sort(pooled, dim=1).values[:, :keep]
            values = pooled.mean(dim=1)
    return values.detach().cpu().numpy().astype(np.float64)


def apply_critic_threshold(raw_action, threshold_config, *, model, obs, algorithm):
    """Return ``(action, info)`` following the P2 ``apply_threshold`` contract."""
    algorithm = (algorithm or type(model).__name__).lower()
    summary, threshold = validate_critic_config(threshold_config, algorithm)
    if model is None or getattr(model, "critic", None) is None or obs is None:
        raise ValueError("Critic threshold requires the loaded model, its critic and the current observation")
    action = np.asarray(raw_action, dtype=np.float32)
    observation = np.asarray(obs, dtype=np.float32).reshape(-1)
    zero = np.zeros_like(action)
    policy_value, zero_value = _summaries(model, algorithm, observation, np.stack([action, zero]), summary)
    delta = policy_value - zero_value
    if not (math.isfinite(policy_value) and math.isfinite(zero_value) and math.isfinite(delta)):
        raise FloatingPointError("Critic produced a non-finite value; refusing to guess a threshold decision")
    accepted = bool(delta > threshold)
    return (action.copy() if accepted else zero), {
        "threshold_mode": "critic",
        "accepted": accepted,
        "critic_policy_value": float(policy_value),
        "critic_zero_value": float(zero_value),
        "critic_delta": float(delta),
    }
