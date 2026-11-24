"""P1/P2 v1 integration on synthetic validation data; never reads the formal test set."""
import json
import os
from pathlib import Path
import sys
import uuid

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "unit_tests")]

import numpy as np
import pandas as pd
from backtest.data import sha256, write_json
from backtest.runner import run_backtest, summarize_runs
from test_backtest import data_fixture, env_config
from train_model.runner import run_training


def main():
    name = "ifc_" + uuid.uuid4().hex[:8]
    directory = ROOT / "datasets" / name
    directory.mkdir(parents=True)
    for filename, start, periods in [("train_data.csv", "2019-01-02", 24),
                                      ("val_data.csv", "2020-01-02", 12)]:
        frame = data_fixture(start, periods)
        frame.loc[(frame.tic == "AAA") & frame.date.isin(frame.date.unique()[3:5]), "tradable"] = False
        frame.to_csv(directory / filename, index=False)
    config = {
        "data": {"root_dir": "datasets", "train_file": "train_data.csv",
                 "val_file": "val_data.csv", "test_file": "trade_data.csv"},
        "environment": env_config(),
        "training": {"total_timesteps": 16, "device": "cpu", "verbose": 0,
                     "policy": "MlpPolicy", "logger_formats": ["csv"]},
        "output": {"trained_models_dir": "trained_models", "tensorboard_log_dir": "tensorboard_log",
                   "results_dir": "results/interface_train"},
        "algorithms": {
            "ppo": {"n_steps": 4, "batch_size": 4, "n_epochs": 1,
                    "policy_kwargs": {"net_arch": [16, 16]}},
            "recurrentppo": {"policy": "MlpLstmPolicy", "n_steps": 4, "batch_size": 4, "n_epochs": 1,
                             "policy_kwargs": {"net_arch": [16], "lstm_hidden_size": 16}},
            "sac": {"buffer_size": 32, "learning_starts": 4, "batch_size": 4,
                    "train_freq": 1, "gradient_steps": 1, "policy_kwargs": {"net_arch": [16, 16]}},
            "tqc": {"buffer_size": 32, "learning_starts": 4, "batch_size": 4,
                    "train_freq": 1, "gradient_steps": 1, "top_quantiles_to_drop_per_net": 1,
                    "policy_kwargs": {"net_arch": [16, 16], "n_quantiles": 5}},
        },
    }
    manifests, checks = [], {}
    output = ROOT / "results/interface_smoke" / name
    output.mkdir(parents=True)
    for algorithm in config["algorithms"]:
        trained = run_training(config, name, algorithm, 7, ROOT, run_type="smoke", device="cpu")
        effective = trained["effective_config"]
        assert effective["actual_timesteps"] >= 16
        if algorithm in ("sac", "tqc"):
            assert effective["actual_gradient_updates"] > 0
        first = run_backtest(trained["artifacts"]["model_zip"], directory / "val_data.csv", output,
                             split="validation", threshold_mode="none")
        second = run_backtest(trained["artifacts"]["model_zip"], directory / "val_data.csv", output,
                              split="validation")
        for filename in ("account_value.csv", "actions.csv", "holdings.csv", "trades.csv", "metrics.json"):
            assert sha256(Path(first["output"]) / filename) == sha256(Path(second["output"]) / filename), filename
        p1_accounts = pd.read_csv(trained["artifacts"]["validation_account_value"], float_precision="round_trip")
        p2_accounts = pd.read_csv(Path(first["output"]) / "account_value.csv", float_precision="round_trip")
        assert p1_accounts.date.tolist() == p2_accounts.date.tolist()
        np.testing.assert_allclose(p1_accounts.account_value, p2_accounts.account_value, rtol=1e-12, atol=1e-9)
        p1_actions = pd.read_csv(trained["artifacts"]["validation_actions"], float_precision="round_trip")
        p2_actions = pd.read_csv(Path(first["output"]) / "actions.csv", float_precision="round_trip")
        p2_fills = p2_actions.pivot(index="date", columns="tic", values="executed_shares").sort_index()
        # P1 currently saves its date index with index=False. Recover row dates
        # only from the already checked account curve, and report that limitation.
        if "date" in p1_actions:
            assert p1_actions.date.tolist() == p2_fills.index.tolist()
        else:
            assert len(p1_actions) == len(p1_accounts) - 1
            assert p1_accounts.date.iloc[:-1].tolist() == p2_fills.index.tolist()
        np.testing.assert_allclose(p1_actions[["AAA", "BBB"]], p2_fills[["AAA", "BBB"]], rtol=1e-12, atol=1e-9)
        checks[algorithm] = {"actual_timesteps": effective["actual_timesteps"],
            "actual_gradient_updates": effective.get("actual_gradient_updates"),
            "max_account_difference": float(np.max(np.abs(p1_accounts.account_value - p2_accounts.account_value))),
            "p1_p2_fills_match": True, "repeat_artifacts_identical": True,
            "p1_action_dates_derived_from_account_curve": "date" not in p1_actions,
            "inference_replay_buffer_size": first["inference_replay_buffer_size"],
            "training_manifest": trained["artifacts"]["manifest"], "backtest_output": first["output"]}
        manifests.append(Path(first["output"]) / "run_manifest.json")
    summarize_runs(manifests, output / "summary.csv")
    report = {"status": "passed", "synthetic_only": True, "formal_test_performance_read": False,
              "threshold_algorithms_tested": False, "algorithms": checks}
    write_json(output / "report.json", report)
    write_json(ROOT / "results/interface_smoke/latest_report.json", {**report, "output": str(output)})
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
