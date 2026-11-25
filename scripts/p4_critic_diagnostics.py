"""P4 CLI utility: scan validation Critic deltas (pilot calibration) & cross-check SAC/TQC implementations."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)

from backtest.data import DEFAULT_INDICATORS, write_json
from backtest.runner import prepare
from sac import SACConfig, SB3SACWrapper, TorchSACAgent, scan_sac_critic_deltas
from thresholds import select_critic_threshold
from tqc import SB3TQCWrapper, TQCConfig, TorchTQCAgent, scan_tqc_critic_deltas
from trading_environment import build_environment


def _synthetic_env(periods: int = 20):
    rows = []
    for idx, dt in enumerate(pd.bdate_range("2021-01-04", periods=periods)):
        for t_idx, tic in enumerate(("600000.SS", "600519.SS", "601318.SS")):
            price = 15.0 + 0.2 * idx + 10.0 * t_idx + (0.5 if idx % 2 == 0 else -0.3)
            row: dict[str, Any] = {
                "date": dt.strftime("%Y-%m-%d"),
                "tic": tic,
                "open": price,
                "high": price + 0.5,
                "low": price - 0.5,
                "close": price,
                "volume": 50_000,
                "day": int(dt.dayofweek),
                "turbulence": 0.0,
                "tradable": True,
            }
            row.update({ind: 0.01 * (idx + 1) for ind in DEFAULT_INDICATORS})
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame["day_index"] = frame["date"].factorize()[0]
    frame = frame.set_index("day_index")
    frame.index.name = ""
    cfg = {
        "initial_amount": 100_000,
        "hmax": 50,
        "buy_cost_pct": 0.001,
        "sell_cost_pct": 0.001,
        "reward_scaling": 1e-4,
        "tech_indicator_list": list(DEFAULT_INDICATORS),
        "risk_indicator_col": "turbulence",
        "turbulence_threshold": None,
        "print_verbosity": 100_000,
        "price_basis": "adjusted_virtual_units",
        "tradability_policy": "enforce_column",
    }
    return build_environment(frame, cfg)


def cmd_scan(args: argparse.Namespace) -> int:
    # Enforce validation-only split via backtest.runner.prepare so test CSVs can never be scanned
    _, env, _, _, effective = prepare(
        args.model,
        args.data,
        split="validation",
        train_data=args.train_data,
        validation_data=args.validation_data,
        metadata_path=args.metadata,
    )
    algorithm = effective["algorithm"].lower()
    seed = int(effective.get("seed", 0))
    if algorithm == "sac":
        wrapper = SB3SACWrapper.load(args.model, device=args.device)
        summary_name = args.summary or "min_q"
        report = scan_sac_critic_deltas(wrapper.model, env, summary=summary_name, seed=seed)
    elif algorithm == "tqc":
        wrapper = SB3TQCWrapper.load(args.model, device=args.device)
        summary_name = args.summary or "truncated_quantile_mean"
        report = scan_tqc_critic_deltas(wrapper.model, env, summary=summary_name, seed=seed)
    else:
        raise ValueError(f"Critic delta scan supports only 'sac' or 'tqc', got {algorithm!r}")

    daily_df: pd.DataFrame = report.pop("daily_records")
    report["model_path"] = str(Path(args.model).resolve())
    report["data_path"] = str(Path(args.data).resolve())
    report["model_run_type"] = effective.get("run_type", "unknown")
    report["seed"] = seed

    if args.output:
        out_json = Path(args.output)
        out_csv = out_json.with_name(out_json.stem + "_daily.csv")
        out_json.parent.mkdir(parents=True, exist_ok=True)
        daily_df.to_csv(out_csv, index=False)
        report["daily_csv"] = str(out_csv.resolve())
        write_json(out_json, report)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_compare_impl(args: argparse.Namespace) -> int:
    timesteps = int(args.timesteps)
    algorithms = ("sac", "tqc") if args.algorithm == "both" else (args.algorithm,)
    results: dict[str, Any] = {}
    for algo in algorithms:
        if algo == "sac":
            cfg = SACConfig(
                learning_rate=1e-3,
                buffer_size=128,
                learning_starts=4,
                batch_size=8,
                net_arch=(16, 16),
                n_critics=2,
            )
            sb3_agent = SB3SACWrapper.from_env(_synthetic_env(), cfg, seed=args.seed, device="cpu")
            sb3_agent.learn(total_timesteps=timesteps)
            obs, _ = _synthetic_env().reset(seed=0)
            sb3_diag = sb3_agent.inspect_critic(obs)

            torch_agent = TorchSACAgent(env=_synthetic_env(), config=cfg, seed=args.seed, device="cpu")
            torch_agent.learn(total_timesteps=timesteps)
            torch_diag = torch_agent.critic_delta(obs, summary="min_q")
            results["sac"] = {
                "sb3_updates": int(getattr(sb3_agent.model, "_n_updates", 0)),
                "torch_updates": int(torch_agent._n_updates),
                "sb3_delta_min_q": sb3_diag["delta_min_q"],
                "torch_delta_min_q": torch_diag["critic_delta"],
                "action_dim": int(len(sb3_diag["policy_action"])),
            }
        else:
            cfg_tqc = TQCConfig(
                learning_rate=1e-3,
                buffer_size=128,
                learning_starts=4,
                batch_size=8,
                net_arch=(16, 16),
                n_critics=2,
                n_quantiles=10,
                top_quantiles_to_drop_per_net=2,
            )
            sb3_tqc = SB3TQCWrapper.from_env(_synthetic_env(), cfg_tqc, seed=args.seed, device="cpu")
            sb3_tqc.learn(total_timesteps=timesteps)
            obs, _ = _synthetic_env().reset(seed=0)
            sb3_diag = sb3_tqc.inspect_critic(obs)

            torch_tqc = TorchTQCAgent(env=_synthetic_env(), config=cfg_tqc, seed=args.seed, device="cpu")
            torch_tqc.learn(total_timesteps=timesteps)
            torch_diag = torch_tqc.critic_delta(obs, summary="truncated_quantile_mean")
            results["tqc"] = {
                "sb3_updates": int(getattr(sb3_tqc.model, "_n_updates", 0)),
                "torch_updates": int(torch_tqc._n_updates),
                "sb3_delta_truncated": sb3_diag["delta_truncated_quantile_mean"],
                "sb3_overestimation_gap": sb3_diag["overestimation_gap_policy"],
                "torch_delta_truncated": torch_diag["critic_delta"],
                "torch_overestimation_gap": torch_diag["policy_overestimation_gap"],
                "action_dim": int(len(sb3_diag["policy_action"])),
            }

    payload = {"status": "passed", "synthetic_only": True, "timesteps": timesteps, "results": results}
    if args.output:
        write_json(Path(args.output), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P4 SAC/TQC Critic diagnostics and reference check")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="Scan validation-split Critic deltas for a pilot SAC/TQC zip")
    scan.add_argument("--model", required=True, help="Trained SB3/sb3-contrib .zip model archive")
    scan.add_argument("--data", required=True, help="Validation CSV recorded during training")
    scan.add_argument("--summary", help="Override summary (default: min_q for SAC, truncated_quantile_mean for TQC)")
    scan.add_argument("--device", default="cpu")
    scan.add_argument("--train-data", help="Relocated training CSV if needed for legacy metadata")
    scan.add_argument("--validation-data", help="Relocated validation CSV if needed")
    scan.add_argument("--metadata", help="Optional external metadata JSON")
    scan.add_argument("--output", help="Optional output JSON path (also writes *_daily.csv)")

    cmp_parser = sub.add_parser("compare-impl", help="Smoke-compare SB3 wrapper vs pure PyTorch SAC/TQC")
    cmp_parser.add_argument("--algorithm", choices=("sac", "tqc", "both"), default="both")
    cmp_parser.add_argument("--timesteps", type=int, default=16)
    cmp_parser.add_argument("--seed", type=int, default=7)
    cmp_parser.add_argument("--output", help="Optional output JSON path")

    sel_parser = sub.add_parser(
        "select",
        help="Select and freeze Critic threshold from validation run_manifest.json files (rule p4_val_sharpe_v1)",
    )
    sel_parser.add_argument("--algorithm", choices=("sac", "tqc"), required=True)
    sel_parser.add_argument("--manifests", nargs="+", type=Path, required=True)
    sel_parser.add_argument("--seeds", nargs="+", type=int, help="Expected formal seed set (e.g. 42 2024 2026)")
    sel_parser.add_argument("--tie-tolerance", type=float, default=1e-9)
    sel_parser.add_argument("--output", type=Path, required=True, help="Output P5 audit JSON path")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "scan":
            return cmd_scan(args)
        if args.command == "select":
            record = select_critic_threshold(
                args.manifests,
                algorithm=args.algorithm,
                expected_seeds=args.seeds,
                tie_tolerance=args.tie_tolerance,
                output_path=args.output,
            )
            print(json.dumps(record, ensure_ascii=False, indent=2))
            return 0 if record["selection_status"] == "selected" else 2
        return cmd_compare_impl(args)
    except Exception as exc:
        print(f"p4_critic_diagnostics error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
