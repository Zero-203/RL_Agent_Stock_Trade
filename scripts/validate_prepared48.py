"""Run short, validation-only compatibility checks for an existing 48-stock version."""
from __future__ import annotations

import os
for variable in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[variable] = "1"

import argparse
import copy
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"src"))

import numpy as np
import pandas as pd
import torch

from backtest.data import check_market, sha256, write_json
from backtest.runner import run_backtest
from train_model.runner import run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--steps", type=int, default=128)
    args = parser.parse_args()
    if args.dataset in ["", ".", ".."] or Path(args.dataset).name != args.dataset or any(x in args.dataset for x in ["/", "\\"]):
        raise ValueError("Use a simple dataset name")
    if args.steps < 64 or args.steps > 1024 or args.steps % 32:
        raise ValueError("Compatibility budget must be a multiple of 32 in [64,1024]")
    torch.set_num_threads(1)
    dataset = ROOT/"datasets"/args.dataset
    manifest = json.loads((dataset/"dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["stock_count"] == 48 and manifest["source"] == "BaoStock"
    for filename, expected in manifest["outputs_sha256"].items():
        assert sha256(dataset/filename) == expected, filename
    for filename in ["train_data.csv", "val_data.csv", "trade_data.csv"]:
        report = check_market(dataset/filename)
        assert report["input_compatible"], report
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT/"results/dataset_validation"/("r_"+stamp+"_"+uuid.uuid4().hex[:6])
    output.mkdir(parents=True)
    report = {"status": "running", "dataset": str(dataset), "dataset_name": args.dataset,
        "data_source": "BaoStock", "stock_universe_confirmed": True,
        "source_change_confirmed": True, "scope": "technical compatibility",
        "test_strategy_predictions": 0, "algorithms": {}}
    write_json(output/"validation_report.json", report)
    base = json.loads((ROOT/"config/train_model_adjusted.example.json").read_text(encoding="utf-8"))
    try:
        for algorithm in ["ppo", "ddpg", "td3", "sac"]:
            print(f"START {algorithm}: {args.steps} steps and full validation replay", flush=True)
            config = copy.deepcopy(base)
            config["training"].update({"total_timesteps": args.steps, "device": "cpu", "verbose": 0, "logger_formats": ["csv"]})
            config["environment"]["print_verbosity"] = 100000
            parameters = config["algorithms"][algorithm]
            parameters["policy_kwargs"] = {"net_arch": [32,32]}
            if algorithm == "ppo":
                parameters.update({"n_steps": 32, "batch_size": 32, "n_epochs": 2})
            else:
                parameters.update({"buffer_size": 256, "batch_size": 16, "learning_starts": 32,
                    "train_freq": 1, "gradient_steps": 1})
            config["output"].update({"trained_models_dir": str(output/"m"),
                "tensorboard_log_dir": str(output/"tb"), "results_dir": str(output/"t")})
            write_json(output/(algorithm+"_config.json"), config)
            training = run_training(config, args.dataset, algorithm, 7, ROOT, run_type="smoke", device="cpu")
            model = Path(training["artifacts"]["model_zip"])
            replay = run_backtest(model, dataset/"val_data.csv", output/"replay",
                split="validation", benchmark_path=dataset/"val_index.csv")
            expected = pd.read_csv(training["artifacts"]["validation_account_value"], float_precision="round_trip")
            actual = pd.read_csv(Path(replay["output"])/"account_value.csv", float_precision="round_trip")
            assert expected["date"].tolist() == actual["date"].tolist()
            np.testing.assert_allclose(expected["account_value"], actual["account_value"], rtol=1e-12, atol=1e-7)
            assert replay["ordered_tickers"] == manifest["ordered_tickers"]
            report["algorithms"][algorithm] = {"status": "passed", "training_steps": args.steps,
                "seed": 7, "run_type": "smoke", "validation_sessions": len(actual),
                "p1_p2_max_absolute_account_difference": float((expected["account_value"]-actual["account_value"]).abs().max()),
                "model": str(model), "validation_replay": replay["output"],
                "account_reconciliation_passed": replay["checks"]["cash_fee_reward_reconciliation"],
                "test_strategy_predictions": 0}
            write_json(output/"validation_report.json", report)
            print("PASS", algorithm, len(actual), "validation sessions", flush=True)
            del training, replay, expected, actual
            gc.collect()
        first = report["algorithms"]["ppo"]
        repeated = run_backtest(first["model"], dataset/"val_data.csv", output/"repeat",
            split="validation", benchmark_path=dataset/"val_index.csv")
        repeat_files = ["account_value.csv", "actions.csv", "holdings.csv", "trades.csv", "metrics.json", "regime_metrics.csv"]
        for filename in repeat_files:
            assert (Path(first["validation_replay"])/filename).read_bytes() == (Path(repeated["output"])/filename).read_bytes(), filename
        for filename, expected_hash in manifest["outputs_sha256"].items():
            assert sha256(dataset/filename) == expected_hash
        report.update({"status": "passed", "validation_repeatability": True,
            "repeat_run": repeated["output"], "data_hashes_verified": True,
            "formal_strategy_performance_claim": False,
            "validation_report": str(output/"validation_report.json")})
        write_json(output/"validation_report.json", report)
        write_json(ROOT/"results/dataset_validation/latest_report.json", report)
        print(json.dumps({"status": "passed", "dataset": str(dataset), "report": report["validation_report"],
            "test_strategy_predictions": 0}, ensure_ascii=False, indent=2))
    except Exception as error:
        report.update({"status": "failed", "error": {"type": type(error).__name__, "message": str(error)}})
        write_json(output/"validation_report.json", report)
        raise


if __name__ == "__main__":
    main()
