"""Deterministic validation-set Critic threshold selection and P5 audit record generation.

Selection Rule ``p4_val_sharpe_v1`` (aligned with project plan Section 2.4):
1. Strictly accepts ``split == 'validation'`` completed backtest runs; any ``test``
   manifest raises ``ValueError`` immediately.
2. Enforces cross-run comparability (same validation CSV hash, environment,
   environment source hashes, training ``run_type``, annualization, risk-free rate)
   and paired model identity (same ``model_sha256`` per ``(algorithm, seed)``).
3. Evaluates ``threshold_mode == 'critic'`` candidates for a single ``algorithm``
   (``'sac'`` or ``'tqc'``):
   - Every candidate must cover the complete seed set; missing seeds disqualify it.
   - ``null`` or non-finite ``sharpe`` on *any* seed disqualifies the candidate
     (never coerced to ``0.0``).
4. Primary criterion: highest mean validation ``sharpe`` across seeds.
5. Tie-breaking (``|mean_a - mean_b| <= tie_tolerance``): prefer smaller
   ``delta_threshold`` (less aggressive intervention), then lexicographic
   ``threshold_condition_id``.
6. If no candidate is eligible, ``selection_status`` is recorded as ``'failed'``
   with ``selected = None``; never falls back to test data.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from backtest.data import write_json
from backtest.runner import threshold_identity

SELECTION_RULE_VERSION = "p4_val_sharpe_v1"


def _is_valid_sharpe(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def select_critic_threshold(
    manifest_paths: Sequence[str | Path],
    *,
    algorithm: str,
    expected_seeds: Sequence[int] | None = None,
    tie_tolerance: float = 1e-9,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Select and freeze one Critic threshold condition from validation manifests."""
    algo = algorithm.lower().strip()
    if algo not in ("sac", "tqc"):
        raise ValueError(f"Critic threshold selection supports only 'sac' or 'tqc', got {algorithm!r}")
    if not math.isfinite(tie_tolerance) or tie_tolerance < 0.0:
        raise ValueError("tie_tolerance must be finite and >= 0")
    if not manifest_paths:
        raise ValueError("At least one validation run_manifest.json is required")

    records: list[dict[str, Any]] = []
    for raw_path in manifest_paths:
        path = Path(raw_path).resolve()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            raise ValueError(f"Cannot select from incomplete backtest run: {path}")
        if manifest.get("split") != "validation":
            raise ValueError(
                f"Selection strictly requires split='validation', got {manifest.get('split')!r} in {path}"
            )
        if manifest.get("algorithm", "").lower() != algo:
            continue

        mode = manifest.get("threshold_mode") or "none"
        if mode not in ("none", "critic"):
            continue
        if mode != "none" and not manifest.get("threshold_implementation_sha256"):
            raise ValueError(f"Threshold run missing implementation hash: {path}")
        condition = threshold_identity(
            mode,
            manifest.get("threshold_config"),
            manifest.get("threshold_implementation_sha256"),
        )
        for key in ("threshold_contract_version", "threshold_condition_id", "threshold_condition_sha256"):
            if key in manifest and manifest[key] != condition[key]:
                raise ValueError(f"Threshold condition identity mismatch ({key}) in {path}")

        metrics = json.loads((path.parent / "metrics.json").read_text(encoding="utf-8"))["strategy"]
        records.append(
            {
                "seed": int(manifest["seed"]),
                "mode": mode,
                "condition_id": condition["threshold_condition_id"],
                "condition_sha256": condition["threshold_condition_sha256"],
                "threshold_config": condition["threshold_config"],
                "threshold_implementation_sha256": condition["threshold_implementation_sha256"],
                "model_sha256": manifest["input_sha256"]["model"],
                "data_sha256": manifest["input_sha256"]["data"],
                "environment": json.dumps(manifest["environment"], sort_keys=True),
                "environment_code_sha256": json.dumps(manifest.get("environment_code_sha256", {}), sort_keys=True),
                "run_type": manifest.get("model_training_run_type", "unknown"),
                "annualization": int(manifest["annualization"]),
                "risk_free_rate": float(manifest["risk_free_rate"]),
                "manifest_path": str(path),
                "sharpe": metrics.get("sharpe"),
                "total_return": float(metrics["total_return"]),
                "max_drawdown": float(metrics["max_drawdown"]),
                "trade_count": int(metrics.get("trade_count", 0)),
                "vector_rejection_rate": metrics.get("threshold_vector_rejection_rate"),
            }
        )

    if not records:
        raise ValueError(f"No validation manifests matched algorithm={algo!r}")

    # Enforce cross-run comparability
    for field in ("data_sha256", "environment", "environment_code_sha256", "run_type", "annualization", "risk_free_rate"):
        if len({r[field] for r in records}) != 1:
            raise ValueError(f"Incomparable validation runs: {field} differs across manifests")

    seen_keys: set[tuple[int, str]] = set()
    model_by_seed: dict[int, str] = {}
    for rec in records:
        dup_key = (rec["seed"], rec["condition_sha256"])
        if dup_key in seen_keys:
            raise ValueError(f"Duplicate validation run for seed={rec['seed']} condition={rec['condition_id']}")
        seen_keys.add(dup_key)
        prev_model = model_by_seed.setdefault(rec["seed"], rec["model_sha256"])
        if prev_model != rec["model_sha256"]:
            raise ValueError(f"Seed {rec['seed']} mixes different model hashes across paired runs")

    observed_seeds = sorted(model_by_seed)
    required_seeds = sorted(int(s) for s in expected_seeds) if expected_seeds is not None else observed_seeds
    if not required_seeds:
        raise ValueError("Seed set cannot be empty")

    critic_records = [r for r in records if r["mode"] == "critic"]
    if not critic_records:
        raise ValueError(f"No threshold_mode='critic' candidate runs provided for {algo!r}")

    impl_hashes = {r["threshold_implementation_sha256"] for r in critic_records}
    if len(impl_hashes) != 1:
        raise ValueError("Critic candidates were evaluated with different threshold implementations")

    none_records = [r for r in records if r["mode"] == "none"]
    baseline_summary: dict[str, Any] | None = None
    if none_records:
        by_seed = {str(r["seed"]): r["sharpe"] for r in sorted(none_records, key=lambda x: x["seed"])}
        valid_vals = [float(v) for v in by_seed.values() if _is_valid_sharpe(v)]
        baseline_summary = {
            "per_seed_sharpe": by_seed,
            "mean_sharpe": float(np.mean(valid_vals)) if len(valid_vals) == len(required_seeds) else None,
        }

    # Group critic runs by condition_sha256
    grouped: dict[str, list[dict[str, Any]]] = {}
    for rec in critic_records:
        grouped.setdefault(rec["condition_sha256"], []).append(rec)

    candidates: list[dict[str, Any]] = []
    for cond_sha, group in grouped.items():
        group_sorted = sorted(group, key=lambda x: x["seed"])
        first = group_sorted[0]
        cfg = first["threshold_config"]
        delta_threshold = float(cfg["delta_threshold"])
        seed_map = {r["seed"]: r for r in group_sorted}

        missing_seeds = [s for s in required_seeds if s not in seed_map]
        extra_seeds = [s for s in seed_map if s not in required_seeds]
        invalid_sharpe_seeds = [
            s for s in required_seeds if s in seed_map and not _is_valid_sharpe(seed_map[s]["sharpe"])
        ]

        if missing_seeds or extra_seeds:
            eligible = False
            reason = f"seed_mismatch(missing={missing_seeds}, extra={extra_seeds})"
            mean_sharpe = None
            std_sharpe = None
        elif invalid_sharpe_seeds:
            eligible = False
            reason = f"null_or_nonfinite_sharpe_on_seeds={invalid_sharpe_seeds}"
            mean_sharpe = None
            std_sharpe = None
        else:
            sharpes = np.array([float(seed_map[s]["sharpe"]) for s in required_seeds], dtype=np.float64)
            eligible = True
            reason = None
            mean_sharpe = float(sharpes.mean())
            std_sharpe = float(sharpes.std(ddof=1)) if len(sharpes) > 1 else 0.0

        rej_vals = [
            float(r["vector_rejection_rate"])
            for r in group_sorted
            if r["vector_rejection_rate"] is not None and math.isfinite(float(r["vector_rejection_rate"]))
        ]
        candidates.append(
            {
                "threshold_condition_id": first["condition_id"],
                "threshold_condition_sha256": cond_sha,
                "threshold_config": cfg,
                "summary": cfg.get("summary"),
                "delta_threshold": delta_threshold,
                "eligible": eligible,
                "ineligible_reason": reason,
                "per_seed_sharpe": {str(r["seed"]): r["sharpe"] for r in group_sorted},
                "mean_sharpe": mean_sharpe,
                "std_sharpe": std_sharpe,
                "mean_total_return": float(np.mean([r["total_return"] for r in group_sorted])),
                "mean_vector_rejection_rate": float(np.mean(rej_vals)) if rej_vals else None,
                "manifests": {str(r["seed"]): r["manifest_path"] for r in group_sorted},
            }
        )

    # Deterministic ranking with explicit tie-tolerance rule
    eligible_pool = [c for c in candidates if c["eligible"]]
    ineligible_pool = sorted(candidates, key=lambda c: (c["delta_threshold"], c["threshold_condition_id"]))
    ineligible_pool = [c for c in ineligible_pool if not c["eligible"]]

    selected: dict[str, Any] | None = None
    selection_reason = "no_eligible_candidate"
    ranked_eligible: list[dict[str, Any]] = []

    if eligible_pool:
        best_sharpe = max(float(c["mean_sharpe"]) for c in eligible_pool)
        top_tied = [
            c for c in eligible_pool if abs(float(c["mean_sharpe"]) - best_sharpe) <= tie_tolerance
        ]
        top_tied.sort(key=lambda c: (c["delta_threshold"], c["threshold_condition_id"]))
        winner = top_tied[0]
        rest = [c for c in eligible_pool if c["threshold_condition_sha256"] != winner["threshold_condition_sha256"]]
        rest.sort(key=lambda c: (-float(c["mean_sharpe"]), c["delta_threshold"], c["threshold_condition_id"]))
        ranked_eligible = [winner, *rest]

        selection_reason = (
            "tie_broken_by_lower_delta_threshold"
            if len(top_tied) > 1
            else "unique_highest_validation_mean_sharpe"
        )
        selected = {
            "threshold_condition_id": winner["threshold_condition_id"],
            "threshold_condition_sha256": winner["threshold_condition_sha256"],
            "threshold_config": winner["threshold_config"],
            "mean_sharpe": winner["mean_sharpe"],
            "std_sharpe": winner["std_sharpe"],
            "mean_vector_rejection_rate": winner["mean_vector_rejection_rate"],
            "selection_reason": selection_reason,
            "tied_candidate_ids": [c["threshold_condition_id"] for c in top_tied],
        }

    sample = records[0]
    audit_record = {
        "selection_rule_version": SELECTION_RULE_VERSION,
        "selection_status": "selected" if selected is not None else "failed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": algo,
        "split": "validation",
        "required_seeds": required_seeds,
        "tie_tolerance": float(tie_tolerance),
        "data_sha256": sample["data_sha256"],
        "environment_code_sha256": json.loads(sample["environment_code_sha256"]),
        "threshold_implementation_sha256": next(iter(impl_hashes)),
        "model_training_run_type": sample["run_type"],
        "model_sha256_by_seed": {str(k): v for k, v in sorted(model_by_seed.items())},
        "unthresholded_validation_baseline": baseline_summary,
        "selected": selected,
        "candidates": [*ranked_eligible, *ineligible_pool],
    }

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_json(out, audit_record)
    return audit_record
