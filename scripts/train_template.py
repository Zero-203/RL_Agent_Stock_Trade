from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from train_model import SUPPORTED_ALGORITHMS, load_config, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one FinRL SB3 experiment run")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "train_model.json",
        help="Path to the JSON experiment configuration",
    )
    parser.add_argument("--dataset", required=True, help="Dataset directory under datasets/")
    parser.add_argument("--algorithm", choices=SUPPORTED_ALGORITHMS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--timesteps", type=int, help="Override only for pilot/smoke runs")
    parser.add_argument("--run-type", choices=("main", "pilot", "smoke"), default="main")
    parser.add_argument("--device", help="SB3 device, for example auto, cpu, or cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        manifest = run_training(
            config=config,
            dataset_name=args.dataset,
            algorithm=args.algorithm,
            seed=args.seed,
            project_root=PROJECT_ROOT,
            total_timesteps=args.timesteps,
            run_type=args.run_type,
            device=args.device,
        )
    except Exception as error:
        print(f"Training failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
