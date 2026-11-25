"""Unit tests for the P4 SAC module (config, PyTorch scratch SAC, and SB3 SAC adapter)."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "unit_tests")]

import numpy as np
import torch
torch.set_num_threads(1)

from backtest.data import DEFAULT_INDICATORS, load_market
from sac import (
    ReplayBuffer,
    SACConfig,
    SB3SACWrapper,
    SquashedGaussianActor,
    TorchSACAgent,
    TwinQCritic,
    ValidationDiagnosticCallback,
    inspect_sac_critic,
    scan_sac_critic_deltas,
)
from test_backtest import data_fixture, env_config
from thresholds.critic import RULE, apply_critic_threshold
from trading_environment import build_environment


class SACModuleTests(unittest.TestCase):
    def setUp(self) -> None:
        (ROOT / ".test_tmp").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".test_tmp")
        self.tmp_dir = Path(self.temp.name)
        self.csv_path = self.tmp_dir / "data.csv"
        data_fixture(periods=10).to_csv(self.csv_path, index=False)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def make_env(self):
        frame = load_market(self.csv_path, DEFAULT_INDICATORS, "turbulence")
        return build_environment(frame, env_config())

    def test_sac_config_roundtrip_and_validation(self) -> None:
        cfg = SACConfig.from_json(ROOT / "config/train_model_p4_sac_tqc.json")
        self.assertEqual(cfg.batch_size, 256)
        self.assertEqual(cfg.learning_starts, 1000)
        self.assertEqual(cfg.net_arch, (256, 256))
        self.assertEqual(cfg.n_critics, 2)
        self.assertTrue(cfg.auto_entropy)
        self.assertAlmostEqual(cfg.initial_alpha, 0.1)

        roundtrip = SACConfig.from_dict(cfg.to_train_template_dict())
        self.assertEqual(cfg, roundtrip)

        for bad in (
            {"learning_rate": 0.0},
            {"buffer_size": 0},
            {"learning_starts": -1},
            {"tau": 0.0},
            {"gamma": 1.5},
            {"ent_coef": "auto_-1"},
            {"ent_coef": -0.1},
            {"ent_coef": True},
            {"net_arch": ()},
            {"n_critics": 0},
            {"unknown_field": 123},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                SACConfig.from_dict(bad)

    def test_squashed_gaussian_and_twin_critic_shapes_and_bounds(self) -> None:
        actor = SquashedGaussianActor(obs_dim=21, action_dim=2, hidden_dims=(16, 16))
        critic = TwinQCritic(obs_dim=21, action_dim=2, hidden_dims=(16, 16), n_critics=3)
        obs = torch.randn(8, 21)
        actions, log_prob = actor.sample_and_log_prob(obs)
        self.assertEqual(actions.shape, (8, 2))
        self.assertEqual(log_prob.shape, (8, 1))
        self.assertTrue(torch.isfinite(actions).all())
        self.assertTrue(torch.isfinite(log_prob).all())
        self.assertTrue((actions.abs() <= 1.0).all())

        det = actor.deterministic_action(obs)
        self.assertEqual(det.shape, (8, 2))
        self.assertTrue((det.abs() <= 1.0).all())

        q_tuple = critic(obs, actions)
        self.assertEqual(len(q_tuple), 3)
        stacked = critic.stacked(obs, actions)
        self.assertEqual(stacked.shape, (8, 3))

    def test_replay_buffer_wraparound_and_sampling(self) -> None:
        buf = ReplayBuffer(capacity=5, obs_dim=3, action_dim=2, seed=7)
        for idx in range(8):
            buf.add(np.full(3, idx), np.array([0.1, -0.1]), float(idx), np.full(3, idx + 1), idx == 7)
        self.assertEqual(len(buf), 5)
        batch = buf.sample(4)
        self.assertEqual(batch["obs"].shape, (4, 3))
        self.assertEqual(batch["actions"].shape, (4, 2))
        self.assertTrue((batch["obs"] >= 3.0).all())

    def test_torch_sac_agent_learn_predict_critic_delta_and_checkpoint(self) -> None:
        env = self.make_env()
        cfg = SACConfig(
            learning_rate=1e-3,
            buffer_size=64,
            learning_starts=4,
            batch_size=8,
            ent_coef="auto_0.2",
            net_arch=(16, 16),
            n_critics=2,
        )
        agent = TorchSACAgent(env=env, config=cfg, seed=11, device="cpu")
        agent.learn(total_timesteps=16)
        self.assertEqual(agent.num_timesteps, 16)
        self.assertGreater(agent._n_updates, 0)
        self.assertEqual(len(agent.training_history), agent._n_updates)
        for key in ("critic_loss", "actor_loss", "ent_coef_loss", "ent_coef"):
            self.assertTrue(np.isfinite(agent.training_history[-1][key]))

        obs, _ = env.reset(seed=0)
        act_single, state = agent.predict(obs, deterministic=True)
        self.assertIsNone(state)
        self.assertEqual(act_single.shape, (2,))
        self.assertTrue(np.all(np.abs(act_single) <= 1.0))

        delta_info = agent.critic_delta(obs, summary="min_q")
        self.assertAlmostEqual(
            delta_info["critic_delta"],
            delta_info["critic_policy_value"] - delta_info["critic_zero_value"],
            places=6,
        )

        ckpt = agent.save(self.tmp_dir / "torch_sac.pt")
        reloaded = TorchSACAgent.load(ckpt, device="cpu")
        act_reloaded, _ = reloaded.predict(obs, deterministic=True)
        np.testing.assert_allclose(act_single, act_reloaded, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(
            delta_info["critic_delta"],
            reloaded.critic_delta(obs, summary="min_q")["critic_delta"],
            rtol=1e-6,
            atol=1e-7,
        )

    def test_sb3_sac_wrapper_callback_and_threshold_alignment(self) -> None:
        env = self.make_env()
        cfg = SACConfig(
            learning_rate=1e-3,
            buffer_size=64,
            learning_starts=4,
            batch_size=8,
            ent_coef="auto_0.1",
            net_arch=(16, 16),
            n_critics=2,
        )
        wrapper = SB3SACWrapper.from_env(env, cfg, seed=5, device="cpu")
        callback = ValidationDiagnosticCallback(
            self.make_env,
            eval_freq=8,
            best_model_save_path=self.tmp_dir / "best_val_sac",
        )
        wrapper.learn(total_timesteps=16, callback=callback)
        self.assertEqual(len(callback.evaluations), 2)

        obs, _ = self.make_env().reset(seed=0)
        inspected = wrapper.inspect_critic(obs)
        act_thresh, thresh_info = wrapper.predict_with_threshold(obs, delta_threshold=0.0, summary="min_q")
        _, direct_info = apply_critic_threshold(
            inspected["policy_action"],
            {"rule": RULE, "summary": "min_q", "delta_threshold": 0.0},
            model=wrapper.model,
            obs=obs,
            algorithm="sac",
        )
        self.assertAlmostEqual(inspected["delta_min_q"], direct_info["critic_delta"], places=6)
        self.assertEqual(thresh_info["accepted"], direct_info["accepted"])
        self.assertEqual(act_thresh.shape, (2,))

        saved_zip = wrapper.save(self.tmp_dir / "wrapped_sac.zip")
        reloaded = SB3SACWrapper.load(saved_zip, device="cpu")
        self.assertEqual(reloaded.model.buffer_size, 1)

        scan = scan_sac_critic_deltas(reloaded.model, self.make_env(), summary="min_q")
        self.assertEqual(scan["observations"], 9)
        self.assertIn(0.0, scan["suggested_candidates"])
        self.assertEqual(len(scan["daily_records"]), 9)


if __name__ == "__main__":
    unittest.main()
