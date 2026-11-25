"""CLI entry point for the same-universe equal-weight buy-and-hold baseline (T06)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from baselines.equal_weight import EqualWeightConfig, SUSPENSION_POLICIES, UNIT_MODES, run_equal_weight_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run same-universe equal-weight buy-and-hold baseline and emit P2-compatible CSVs"
    )
    parser.add_argument("--data", required=True, type=Path, help="Market split CSV (e.g. val_data.csv or trade_data.csv)")
    parser.add_argument(
        "--train-config",
        type=Path,
        default=PROJECT_ROOT / "config" / "train_model_p4_sac_tqc.json",
        help="Experiment JSON from which environment capital/cost/indicators are read",
    )
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--benchmark", type=Path, help="Optional index CSV (e.g. val_index.csv or test_index.csv)")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results" / "baselines")
    parser.add_argument("--output-name", help="Optional deterministic subdirectory name under --output-root")
    parser.add_argument("--suspension-policy", choices=SUSPENSION_POLICIES, default="defer_reserved_cash")
    parser.add_argument("--unit-mode", choices=UNIT_MODES, default="integer_virtual_units")
    parser.add_argument("--annualization", type=int, default=252)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        raw_cfg = json.loads(args.train_config.read_text(encoding="utf-8"))
        env_section = raw_cfg.get("environment", raw_cfg)
        eq_cfg = EqualWeightConfig.from_environment_dict(
            env_section,
            suspension_policy=args.suspension_policy,
            unit_mode=args.unit_mode,
            annualization=args.annualization,
            risk_free_rate=args.risk_free_rate,
        )
        manifest = run_equal_weight_baseline(
            args.data,
            args.output_root,
            config=eq_cfg,
            split=args.split,
            benchmark_path=args.benchmark,
            output_dir_name=args.output_name,
        )
    except Exception as exc:
        print(f"Equal-weight baseline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
