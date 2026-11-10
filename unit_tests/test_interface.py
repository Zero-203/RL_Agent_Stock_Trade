"""P1 interface checks for algorithm registry, config validation, loading and thresholds."""
import os
for variable in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[variable] = "1"
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "unit_tests"))

import numpy as np
import torch
torch.set_num_threads(1)

from test_backtest import data_fixture, env_config
from backtest.runner import apply_threshold
from train_model.config import (
    SUPPORTED_ALGORITHMS,
    validate_algorithm_config,
    validate_config,
)
from trading_environment import build_environment


def base_config():
    return {
        "data": {"root_dir": "datasets", "train_file": "train_data.csv",
                 "val_file": "val_data.csv", "test_file": "trade_data.csv"},
        "environment": env_config(),
        "training": {"total_timesteps": 8, "policy": "MlpPolicy"},
        "algorithms": {"ppo": {"n_steps": 4, "batch_size": 4}},
        "output": {"trained_models_dir": "trained_models",
                   "tensorboard_log_dir": "tensorboard_log",
                   "results_dir": "results/train_model"},
    }


class InterfaceTests(unittest.TestCase):
    def test_new_algorithms_registered(self):
        from finrl.agents.stablebaselines3.models import MODELS
        for name in ("recurrentppo", "tqc"):
            self.assertIn(name, SUPPORTED_ALGORITHMS)
            self.assertIn(name, MODELS)

    def test_unknown_algorithm_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported algorithm"):
            validate_algorithm_config(base_config(), "dqn")
        config = base_config()
        config["algorithms"]["dqn"] = {}
        with self.assertRaisesRegex(ValueError, "Unknown algorithm"):
            validate_config(config)

    def test_per_algorithm_config_validation(self):
        config = base_config()
        validate_algorithm_config(config, "ppo")
        with self.assertRaisesRegex(ValueError, "Missing configuration for requested algorithm"):
            validate_algorithm_config(config, "tqc")
        config["algorithms"]["tqc"] = {"buffer_size": 16, "learning_starts": 1}
        validate_algorithm_config(config, "tqc")

    def test_algorithm_level_policy_override(self):
        from train_model.data import load_processed_csv
        from train_model.runner import build_model
        cfg = base_config()
        tmp = Path(tempfile.mkdtemp(dir=ROOT / ".test_tmp"))
        csv_path = tmp / "train_data.csv"
        data_fixture(periods=8).to_csv(csv_path, index=False)
        frame = load_processed_csv(
            csv_path, cfg["environment"]["tech_indicator_list"],
            cfg["environment"]["risk_indicator_col"],
        )
        env = build_environment(frame, cfg["environment"])
        vector_env, _ = env.get_sb_env()
        model = build_model(
            "recurrentppo", vector_env,
            {"policy": "MlpLstmPolicy", "n_steps": 4, "batch_size": 4},
            seed=1, tensorboard_log=tempfile.mkdtemp(dir=ROOT / ".test_tmp"),
            device="cpu", policy="MlpPolicy", verbose=0,
        )
        self.assertEqual(type(model).__name__, "RecurrentPPO")
        self.assertTrue(hasattr(model.policy, "lstm_actor"),
                        "recurrent policy must expose an LSTM layer")

    def test_inference_loading_classification(self):
        from unittest.mock import patch
        from finrl.agents.stablebaselines3.models import MODELS
        from train_model.runner import load_model_for_inference
        with self.assertRaisesRegex(ValueError, "Unsupported algorithm"):
            load_model_for_inference("dqn", "x.zip")
        for algo in ("ppo", "recurrentppo"):
            with patch.object(MODELS[algo], "load", return_value="ok") as mocked:
                load_model_for_inference(algo, "model.zip", device="cpu")
                mocked.assert_called_once_with("model.zip", device="cpu")
        for algo in ("ddpg", "td3", "sac", "tqc"):
            with patch.object(MODELS[algo], "load", return_value="ok") as mocked:
                load_model_for_inference(algo, "model.zip", device="cpu")
                mocked.assert_called_once_with("model.zip", device="cpu", buffer_size=1)

    def test_recurrent_state_propagation_and_episode_reset(self):
        from finrl.agents.stablebaselines3.models import DRLAgent
        from backtest.runner import replay
        from train_model.data import load_processed_csv

        class FakeRecurrentPPO:
            def __init__(self):
                self.calls = []

            def predict(self, obs, state=None, episode_start=None, deterministic=True):
                ep = bool(np.asarray(episode_start).reshape(-1)[0])
                prev = 0 if (state is None or ep) else int(state)
                next_state = prev + 1
                self.calls.append((state, ep, next_state))
                return np.zeros((1, 2) if np.asarray(obs).ndim == 2 else (2,), dtype=np.float32), next_state

        FakeRecurrentPPO.__name__ = "RecurrentPPO"
        cfg = base_config()
        tmp = Path(tempfile.mkdtemp(dir=ROOT / ".test_tmp"))
        csv_path = tmp / "val_data.csv"
        data_fixture(periods=5).to_csv(csv_path, index=False)
        frame = load_processed_csv(
            csv_path, cfg["environment"]["tech_indicator_list"],
            cfg["environment"]["risk_indicator_col"],
        )

        # Path 1: DRLAgent.DRL_prediction (training validation path)
        env1 = build_environment(frame, cfg["environment"])
        m1 = FakeRecurrentPPO()
        DRLAgent.DRL_prediction(m1, env1, deterministic=True)
        self.assertGreaterEqual(len(m1.calls), 3)
        self.assertIsNone(m1.calls[0][0])
        self.assertTrue(m1.calls[0][1])
        self.assertEqual(m1.calls[1][0], 1)
        self.assertFalse(m1.calls[1][1])

        # Independent validation dataset starts fresh with state=None, episode_start=True
        env2 = build_environment(frame, cfg["environment"])
        DRLAgent.DRL_prediction(m1, env2, deterministic=True)
        second_run_start = m1.calls[len(m1.calls) // 2]
        self.assertIsNone(second_run_start[0])
        self.assertTrue(second_run_start[1])

        # Path 2: backtest.runner.replay
        env3 = build_environment(frame, cfg["environment"])
        m2 = FakeRecurrentPPO()
        replay(m2, env3, ["AAA", "BBB"], seed=0)
        self.assertIsNone(m2.calls[0][0])
        self.assertTrue(m2.calls[0][1])
        self.assertEqual(m2.calls[1][0], 1)
        self.assertFalse(m2.calls[1][1])

    def test_threshold_unavailable_and_passthrough(self):
        from train_model.runner import run_training
        action = np.array([0.5, -0.5], dtype=float)
        result, info = apply_threshold(action, None, None)
        np.testing.assert_array_equal(result, action)
        self.assertEqual(info["threshold_mode"], "none")
        result, info = apply_threshold(action, "none", None)
        np.testing.assert_array_equal(result, action)
        with self.assertRaises(ValueError):
            apply_threshold(np.array([[0.5, -0.5]]), None, None)
        with self.assertRaises(ValueError):
            apply_threshold(np.array([2.0, 0.0]), None, None)
        with self.assertRaises(NotImplementedError):
            apply_threshold(action, "simple", {"threshold": 0.1})
        with self.assertRaises(NotImplementedError):
            apply_threshold(action, "critic", {})
        cfg = base_config()
        cfg["threshold_mode"] = "simple"
        with self.assertRaises(NotImplementedError):
            run_training(cfg, "dummy", "ppo", 1, ROOT, total_timesteps=4, run_type="smoke")


if __name__ == "__main__":
    unittest.main()
