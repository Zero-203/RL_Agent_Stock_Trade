"""P4 Critic threshold (critic_delta_v1) tests with real tiny SAC/TQC models on synthetic data."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "unit_tests")]

import numpy as np
import torch
torch.set_num_threads(1)
from stable_baselines3 import SAC
from sb3_contrib import TQC

from backtest import runner
from backtest.data import DEFAULT_INDICATORS, load_market
from test_backtest import data_fixture, env_config
from thresholds.critic import RULE, apply_critic_threshold, validate_critic_config
from trading_environment import build_environment

TICKERS = ["AAA", "BBB"]


def config(summary, delta):
    return {"rule": RULE, "summary": summary, "delta_threshold": delta}


class CriticThresholdTests(unittest.TestCase):
    def setUp(self):
        (ROOT / ".test_tmp").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".test_tmp")
        self.path = Path(self.temp.name) / "data.csv"
        data_fixture(periods=6).to_csv(self.path, index=False)

    def tearDown(self):
        self.temp.cleanup()

    def env(self):
        return build_environment(load_market(self.path, DEFAULT_INDICATORS, "turbulence"), env_config())

    def model(self, algorithm):
        vec_env, _ = self.env().get_sb_env()
        if algorithm == "sac":
            return SAC("MlpPolicy", vec_env, buffer_size=16, learning_starts=1, seed=3, device="cpu",
                       policy_kwargs={"net_arch": [8, 8]})
        return TQC("MlpPolicy", vec_env, buffer_size=16, learning_starts=1, seed=3, device="cpu",
                   top_quantiles_to_drop_per_net=2, policy_kwargs={"net_arch": [8, 8], "n_quantiles": 5})

    def observation(self):
        obs, _ = self.env().reset(seed=0)
        return np.asarray(obs, dtype=np.float32)

    def manual(self, model, algorithm, summary, obs, action):
        obs_t = torch.as_tensor(obs[None, :])
        act_t = torch.as_tensor(model.policy.scale_action(action[None, :]).astype(np.float32))
        with torch.no_grad():
            if algorithm == "sac":
                q = torch.cat(model.critic(obs_t, act_t), dim=1)[0]
                return float(q.min() if summary == "min_q" else q.mean())
            quantiles = model.critic(obs_t, act_t)[0].reshape(-1)
            if summary == "quantile_mean":
                return float(quantiles.mean())
            keep = quantiles.numel() - model.top_quantiles_to_drop_per_net * model.critic.n_critics
            return float(torch.sort(quantiles).values[:keep].mean())

    def test_summary_matches_independent_critic_computation(self):
        obs, action = self.observation(), np.array([.6, -.3], dtype=np.float32)
        for algorithm, summaries in (("sac", ("min_q", "mean_q")),
                                     ("tqc", ("truncated_quantile_mean", "quantile_mean"))):
            model = self.model(algorithm)
            for summary in summaries:
                with self.subTest(algorithm=algorithm, summary=summary):
                    _, info = apply_critic_threshold(action, config(summary, -1e6), model=model,
                                                     obs=obs, algorithm=algorithm)
                    policy = self.manual(model, algorithm, summary, obs, action)
                    zero = self.manual(model, algorithm, summary, obs, np.zeros(2, dtype=np.float32))
                    self.assertAlmostEqual(info["critic_policy_value"], policy, places=5)
                    self.assertAlmostEqual(info["critic_zero_value"], zero, places=5)
                    self.assertAlmostEqual(info["critic_delta"], policy - zero, places=5)

    def test_strict_boundary_and_whole_vector_rejection(self):
        obs, action = self.observation(), np.array([.6, -.3], dtype=np.float32)
        for algorithm, summary in (("sac", "min_q"), ("tqc", "truncated_quantile_mean")):
            model = self.model(algorithm)
            _, info = apply_critic_threshold(action, config(summary, -1e6), model=model, obs=obs, algorithm=algorithm)
            delta = info["critic_delta"]
            below, at = np.nextafter(delta, -np.inf), delta
            with self.subTest(algorithm=algorithm):
                kept, info = apply_critic_threshold(action, config(summary, below), model=model, obs=obs, algorithm=algorithm)
                self.assertTrue(info["accepted"])
                np.testing.assert_array_equal(kept, action)
                self.assertEqual(kept.dtype, np.float32)
                rejected, info = apply_critic_threshold(action, config(summary, at), model=model, obs=obs, algorithm=algorithm)
                self.assertFalse(info["accepted"])
                np.testing.assert_array_equal(rejected, np.zeros(2, dtype=np.float32))

    def test_config_validation(self):
        bad = [{}, {"rule": RULE, "summary": "min_q"}, {**config("min_q", 0), "extra": 1},
               config("truncated_quantile_mean", 0), {**config("min_q", 0), "rule": "v0"},
               config("min_q", True), config("min_q", "0.1"), config("min_q", None)]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_critic_config(value, "sac")
        with self.assertRaises(ValueError):
            validate_critic_config(config("min_q", 0), "ppo")
        self.assertEqual(validate_critic_config(config("quantile_mean", 1), "tqc"), ("quantile_mean", 1.0))

    def test_nonfinite_critic_value_fails_loudly(self):
        model, obs = self.model("sac"), self.observation()
        obs[1] = np.nan
        with self.assertRaises(FloatingPointError):
            apply_critic_threshold(np.array([.5, .5], dtype=np.float32), config("min_q", 0),
                                   model=model, obs=obs, algorithm="sac")

    def test_replay_accept_all_matches_none_and_reject_all_never_requests_orders(self):
        for algorithm, summary in (("sac", "min_q"), ("tqc", "truncated_quantile_mean")):
            model = self.model(algorithm)
            with self.subTest(algorithm=algorithm):
                base, _, _, _ = runner.replay(model, self.env(), TICKERS, algorithm=algorithm)
                accept, actions, _, _ = runner.replay(model, self.env(), TICKERS, threshold_mode="critic",
                                                      threshold_config=config(summary, -1e6), algorithm=algorithm)
                np.testing.assert_array_equal(base["account_value"], accept["account_value"])
                self.assertTrue(actions["threshold_accepted"].all())
                self.assertTrue(np.isfinite(actions["critic_delta"]).all())
                _, rejected, _, trades = runner.replay(model, self.env(), TICKERS, threshold_mode="critic",
                                                       threshold_config=config(summary, 1e6), algorithm=algorithm)
                self.assertFalse(rejected["threshold_accepted"].any())
                self.assertTrue(rejected["thresholded_action"].eq(0).all())
                self.assertTrue(rejected["requested_shares"].eq(0).all())
                self.assertEqual(len(trades), 0)

    def test_implementation_fingerprint_covers_threshold_modules(self):
        names = {f"{p.parent.name}/{p.name}" for p in runner.threshold_source_files()}
        self.assertTrue({"backtest/runner.py", "thresholds/critic.py", "thresholds/selection.py"} <= names)
        original = runner.threshold_implementation_sha256()
        real = runner.sha256
        with patch.object(runner, "sha256", side_effect=lambda p: "x" if Path(p).name == "critic.py" else real(p)):
            self.assertNotEqual(original, runner.threshold_implementation_sha256())
        fake_extra = ROOT / "src/thresholds/_ephemeral_test.py"
        with patch.object(runner, "threshold_source_files", return_value=(*runner.threshold_source_files(), fake_extra)), \
                patch.object(runner, "sha256", side_effect=lambda p: "z" if Path(p) == fake_extra else real(p)):
            self.assertNotEqual(original, runner.threshold_implementation_sha256())

    def test_p4_training_config_builds_both_models(self):
        from train_model.config import validate_algorithm_config
        from train_model.runner import build_model
        cfg = json.loads((ROOT / "config/train_model_p4_sac_tqc.json").read_text(encoding="utf-8"))
        vec_env, _ = self.env().get_sb_env()
        for algorithm in ("sac", "tqc"):
            validate_algorithm_config(cfg, algorithm)
            kwargs = dict(cfg["algorithms"][algorithm], buffer_size=64)
            model = build_model(algorithm, vec_env, kwargs, seed=1, tensorboard_log=Path(self.temp.name),
                                device="cpu", policy=cfg["training"]["policy"])
            with self.subTest(algorithm=algorithm):
                self.assertEqual((model.batch_size, model.learning_starts, model.gamma), (256, 1000, 0.99))
                self.assertEqual(model.critic.n_critics, 2)
                if algorithm == "tqc":
                    self.assertEqual((model.top_quantiles_to_drop_per_net, model.critic.n_quantiles), (2, 25))


if __name__ == "__main__":
    unittest.main()
