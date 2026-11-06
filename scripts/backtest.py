from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

# Keep evaluation resource use bounded on the shared Windows workstation.
for variable in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[variable] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"src"))


def main():
    from backtest.data import check_market, write_json
    parser = argparse.ArgumentParser(description="Metadata-driven adjusted-price FinRL backtest")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check-data", help="Check input without loading or training a model")
    check.add_argument("--data", required=True, type=Path)
    check.add_argument("--risk-column", default="turbulence")
    check.add_argument("--output", type=Path)
    run = commands.add_parser("run")
    for name in ["model", "data"]:
        run.add_argument("--"+name, required=True, type=Path)
    run.add_argument("--output-root", type=Path, default=ROOT/"results/backtest")
    run.add_argument("--split", choices=["validation", "test"], default="test")
    run.add_argument("--benchmark", type=Path)
    run.add_argument("--baseline", action="append", default=[], metavar="NAME=CSV")
    run.add_argument("--train-data", type=Path)
    run.add_argument("--validation-data", type=Path)
    run.add_argument("--metadata", type=Path)
    run.add_argument("--annualization", type=int, default=252)
    run.add_argument("--risk-free-rate", type=float, default=0)
    run.add_argument("--device", default="cpu")
    batch = commands.add_parser("batch", help="Serial evaluation of an explicit job list")
    batch.add_argument("--config", type=Path, required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("--manifests", nargs="+", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "check-data":
            result = check_market(args.data, risk_column=args.risk_column)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                write_json(args.output, result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["input_compatible"] else 2
        from backtest.runner import run_backtest, summarize_runs
        if args.command == "run":
            baselines = {}
            for value in args.baseline:
                name, path = value.split("=", 1)
                if name in baselines:
                    raise ValueError("Duplicate baseline name")
                baselines[name] = path
            result = run_backtest(args.model, args.data, args.output_root, split=args.split,
                benchmark_path=args.benchmark, baselines=baselines, train_data=args.train_data,
                validation_data=args.validation_data, metadata_path=args.metadata,
                annualization=args.annualization, risk_free_rate=args.risk_free_rate, device=args.device)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "summarize":
            result = summarize_runs(args.manifests, args.output)
            print(f"Summarized {len(result)} runs to {args.output}")
        else:
            config = json.loads(args.config.read_text(encoding="utf-8"))
            if not config.get("jobs"):
                raise ValueError("Batch jobs must be a nonempty explicit list")
            outputs = []
            for job in config["jobs"]:
                combined = dict(config.get("defaults", {}))
                combined.update(job)
                result = run_backtest(**combined)
                outputs.append(Path(result["output"])/"run_manifest.json")
                print("COMPLETED", result["output"], flush=True)
            if config.get("summary_output"):
                summarize_runs(outputs, config["summary_output"])
    except Exception as error:
        print(f"Backtest failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
