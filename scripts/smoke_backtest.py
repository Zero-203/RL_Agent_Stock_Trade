"""Synthetic-only integration check through P1 training/save/reload and P2 replay."""
import os
for variable in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[variable] = "1"
import copy
import json
from pathlib import Path
import sys
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"unit_tests"))

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)

from test_backtest import data_fixture
from train_model.runner import run_training
from backtest.runner import run_backtest, summarize_runs
from backtest.data import write_json


def main():
    name = "backtest_synthetic_"+uuid.uuid4().hex[:8]
    directory = ROOT/"datasets"/name
    directory.mkdir(parents=True)
    for filename, start, periods in [("train_data.csv", "2019-01-02", 40), ("val_data.csv", "2020-01-02", 16), ("trade_data.csv", "2021-01-04", 16)]:
        frame = data_fixture(start, periods)
        frame.loc[(frame.tic == "AAA") & frame.date.isin(frame.date.unique()[3:5]), "tradable"] = False
        frame.to_csv(directory/filename, index=False)
    config = json.loads((ROOT/"config/train_model_adjusted.example.json").read_text(encoding="utf-8"))
    config["training"].update({"total_timesteps": 64, "device": "cpu", "verbose": 0, "logger_formats": ["csv"]})
    config["algorithms"]["ppo"].update({"n_steps": 16, "batch_size": 16, "n_epochs": 2, "policy_kwargs": {"net_arch": [16, 16]}})
    config["environment"]["print_verbosity"] = 100000
    config["output"].update({"results_dir": "results/backtest_smoke_training", "tensorboard_log_dir": "tensorboard_log/backtest_smoke"})
    training = run_training(config, name, "ppo", 7, ROOT, run_type="smoke", device="cpu")
    model_path = Path(training["artifacts"]["model_zip"])
    dates = pd.read_csv(directory/"trade_data.csv")["date"].unique()
    index = pd.DataFrame({"date": dates, "close": 100+np.arange(len(dates)),
                          "regime": ["up"]*5+["sideways"]*5+["down"]*6})
    index_path = directory/"test_index.csv"
    index.to_csv(index_path, index=False)
    first = run_backtest(model_path, directory/"trade_data.csv", ROOT/"results/backtest_smoke", benchmark_path=index_path)
    second = run_backtest(model_path, directory/"trade_data.csv", ROOT/"results/backtest_smoke", benchmark_path=index_path)
    for filename in ["account_value.csv", "actions.csv", "holdings.csv", "trades.csv", "metrics.json"]:
        if (Path(first["output"])/filename).read_bytes() != (Path(second["output"])/filename).read_bytes():
            raise AssertionError(f"Non-deterministic output: {filename}")
    validation = run_backtest(model_path, directory/"val_data.csv", ROOT/"results/backtest_smoke", split="validation")
    training_validation = pd.read_csv(training["artifacts"]["validation_account_value"], float_precision="round_trip")
    replay_validation = pd.read_csv(Path(validation["output"])/"account_value.csv", float_precision="round_trip")
    assert training_validation["date"].tolist() == replay_validation["date"].tolist()
    np.testing.assert_allclose(training_validation["account_value"], replay_validation["account_value"], rtol=1e-12, atol=1e-9)
    # Exercise off-policy archive loading with a large declared training buffer
    # without ever allocating that large buffer during this synthetic test.
    from stable_baselines3 import TD3
    from trading_environment import build_environment
    from backtest.data import load_market, DEFAULT_INDICATORS
    env = build_environment(load_market(directory/"trade_data.csv", DEFAULT_INDICATORS, "turbulence"), config["environment"])
    td3 = TD3("MlpPolicy", env, buffer_size=16, policy_kwargs={"net_arch": [16,16]}, device="cpu", seed=9)
    td3.buffer_size = 1_000_000
    td3_path = directory/"synthetic_td3.zip"
    td3.save(td3_path)
    with zipfile.ZipFile(model_path) as source:
        meta = json.loads(source.read("metadata.txt").decode().split("\n\n", 1)[1])
    meta = copy.deepcopy(meta)
    meta["effective_config"]["algorithm"] = "td3"
    meta["effective_config"]["seed"] = 9
    with zipfile.ZipFile(td3_path, "a") as target:
        target.writestr("metadata.txt", "FinRL T04 model metadata\n\n"+json.dumps(meta))
    offpolicy = run_backtest(td3_path, directory/"trade_data.csv", ROOT/"results/backtest_smoke", benchmark_path=index_path)
    assert offpolicy["inference_replay_buffer_size"] == 1
    summary_path = ROOT/"results/backtest_smoke"/(name+"_summary.csv")
    summarize_runs([Path(first["output"])/"run_manifest.json", Path(offpolicy["output"])/"run_manifest.json"], summary_path)
    result = {"status": "passed", "synthetic_only": True, "training_timesteps": 64,
        "trained_model": str(model_path), "deterministic_replay_identical": True,
        "p1_validation_curve_matches_p2_replay": True,
        "validation_run": validation["output"], "test_run": first["output"],
        "off_policy_large_declared_buffer_replayed_with_one_slot": True,
        "offpolicy_run": offpolicy["output"], "summary": str(summary_path),
        "formal_stock_strategy_performance_tested": False}
    write_json(ROOT/"results/backtest_smoke/latest_smoke_report.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
