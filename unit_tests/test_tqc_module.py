"""Unit tests for the P4 TQC module and the P4 Critic diagnostic CLI."""
from __future__ import annotations

import importlib.util
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

from backtest.data import DEFAULT_INDICATORS, load_market
from test_backtest import data_fixture, env_config
from thresholds import RULE, apply_critic_threshold
from tqc import (
    QuantileCritic,
    SB3TQCWrapper,
    TQCConfig,
    TQCValidationDiagnosticCallback,
    TorchTQCAgent,
    quantile_huber_loss,
    scan_tqc_critic_deltas,
    summarize_quantile_distribution,
)
from trading_environment import build_environment


class TQCModuleTests(unittest.TestCase):
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

    def test_tqc_config_roundtrip_and_properties(self) -> None:
        cfg = TQCConfig.from_json(ROOT / "config/train_model_p4_sac_tqc.json")
        self.assertEqual(cfg.n_critics, 2)
        self.assertEqual(cfg.n_quantiles, 25)
        self.assertEqual(cfg.top_quantiles_to_drop_per_net, 2)
        self.assertEqual(cfg.total_quantiles, 50)
        self.assertEqual(cfg.dropped_quantiles, 4)
        self.assertEqual(cfg.kept_quantiles, 46)

        roundtrip = TQCConfig.from_dict(cfg.to_train_template_dict())
        self.assertEqual(cfg, roundtrip)

        for bad in (
            {"n_quantiles": 1},
            {"top_quantiles_to_drop_per_net": -1},
            {"n_quantiles": 5, "top_quantiles_to_drop_per_net": 5},
            {"policy_kwargs": {"extra_unknown": 1}},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                TQCConfig.from_dict(bad)

    def test_quantile_critic_and_asymmetric_quantile_huber_loss(self) -> None:
        critic = QuantileCritic(obs_dim=21, action_dim=2, hidden_dims=(16, 16), n_critics=2, n_quantiles=8)
        obs = torch.randn(4, 21)
        act = torch.tanh(torch.randn(4, 2))
        q_atoms = critic(obs, act)
        self.assertEqual(q_atoms.shape, (4, 2, 8))

        zeros_pred = torch.zeros(2, 1, 4)
        zeros_tgt = torch.zeros(2, 4)
        self.assertAlmostEqual(float(quantile_huber_loss(zeros_pred, zeros_tgt).item()), 0.0, places=7)

        # For tau = 0.8, under-prediction (target > current, delta > 0) weighs 0.8 vs over-prediction 0.2
        tau = torch.tensor([[[[0.8]]]])
        pred = torch.zeros(1, 1, 1)
        pos_loss = quantile_huber_loss(pred, torch.ones(1, 1), cum_prob=tau)
        neg_loss = quantile_huber_loss(pred, -torch.ones(1, 1), cum_prob=tau)
        self.assertAlmostEqual(float(pos_loss.item() / neg_loss.item()), 4.0, places=5)

    def test_summarize_quantile_distribution_truncation_math(self) -> None:
        # 2 critics x 5 quantiles = 10 atoms: 0..9; drop 1 per critic -> keep 0..7 (mean 3.5), full mean 4.5
        atoms = np.arange(10, dtype=np.float64).reshape(2, 5)
        stats = summarize_quantile_distribution(atoms, top_quantiles_to_drop_per_net=1)
        self.assertAlmostEqual(stats["truncated_quantile_mean"], 3.5, places=7)
        self.assertAlmostEqual(stats["quantile_mean"], 4.5, places=7)
        self.assertAlmostEqual(stats["overestimation_gap"], 1.0, places=7)
        self.assertAlmostEqual(stats["dropped_top_mean"], 8.5, places=7)

    def test_torch_tqc_agent_learn_predict_critic_delta_and_checkpoint(self) -> None:
        env = self.make_env()
        cfg = TQCConfig(
            learning_rate=1e-3,
            buffer_size=64,
            learning_starts=4,
            batch_size=8,
            ent_coef="auto_0.2",
            top_quantiles_to_drop_per_net=1,
            net_arch=(16, 16),
            n_critics=2,
            n_quantiles=6,
        )
        agent = TorchTQCAgent(env=env, config=cfg, seed=13, device="cpu")
        agent.learn(total_timesteps=16)
        self.assertEqual(agent.num_timesteps, 16)
        self.assertGreater(agent._n_updates, 0)
        for key in ("critic_loss", "actor_loss", "ent_coef_loss", "ent_coef", "qf_pi_mean"):
            self.assertTrue(np.isfinite(agent.training_history[-1][key]))

        obs, _ = env.reset(seed=0)
        act_single, state = agent.predict(obs, deterministic=True)
        self.assertIsNone(state)
        self.assertEqual(act_single.shape, (2,))
        self.assertTrue(np.all(np.abs(act_single) <= 1.0))

        delta_info = agent.critic_delta(obs, summary="truncated_quantile_mean")
        self.assertAlmostEqual(
            delta_info["critic_delta"],
            delta_info["critic_policy_value"] - delta_info["critic_zero_value"],
            places=6,
        )
        self.assertGreaterEqual(delta_info["policy_overestimation_gap"], 0.0)

        ckpt = agent.save(self.tmp_dir / "torch_tqc.pt")
        reloaded = TorchTQCAgent.load(ckpt, device="cpu")
        act_reloaded, _ = reloaded.predict(obs, deterministic=True)
        np.testing.assert_allclose(act_single, act_reloaded, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(
            delta_info["critic_delta"],
            reloaded.critic_delta(obs, summary="truncated_quantile_mean")["critic_delta"],
            rtol=1e-6,
            atol=1e-7,
        )

    def test_sb3_tqc_wrapper_callback_and_threshold_alignment(self) -> None:
        env = self.make_env()
        cfg = TQCConfig(
            learning_rate=1e-3,
            buffer_size=64,
            learning_starts=4,
            batch_size=8,
            ent_coef="auto_0.1",
            top_quantiles_to_drop_per_net=1,
            net_arch=(16, 16),
            n_critics=2,
            n_quantiles=6,
        )
        wrapper = SB3TQCWrapper.from_env(env, cfg, seed=9, device="cpu")
        callback = TQCValidationDiagnosticCallback(
            self.make_env,
            eval_freq=8,
            best_model_save_path=self.tmp_dir / "best_val_tqc",
        )
        wrapper.learn(total_timesteps=16, callback=callback)
        self.assertEqual(len(callback.evaluations), 2)

        obs, _ = self.make_env().reset(seed=0)
        inspected = wrapper.inspect_critic(obs)
        act_thresh, thresh_info = wrapper.predict_with_threshold(
            obs, delta_threshold=0.0, summary="truncated_quantile_mean"
        )
        _, direct_info = apply_critic_threshold(
            inspected["policy_action"],
            {"rule": RULE, "summary": "truncated_quantile_mean", "delta_threshold": 0.0},
            model=wrapper.model,
            obs=obs,
            algorithm="tqc",
        )
        self.assertAlmostEqual(
            inspected["delta_truncated_quantile_mean"], direct_info["critic_delta"], places=6
        )
        self.assertEqual(thresh_info["accepted"], direct_info["accepted"])
        self.assertEqual(act_thresh.shape, (2,))
        self.assertGreaterEqual(inspected["overestimation_gap_policy"], 0.0)

        saved_zip = wrapper.save(self.tmp_dir / "wrapped_tqc.zip")
        reloaded = SB3TQCWrapper.load(saved_zip, device="cpu")
        self.assertEqual(reloaded.model.buffer_size, 1)

        scan = scan_tqc_critic_deltas(reloaded.model, self.make_env())
        self.assertEqual(scan["observations"], 9)
        self.assertIn(0.0, scan["suggested_candidates"])

    def test_p4_critic_diagnostics_cli_scan_and_test_isolation(self) -> None:
        from train_model.runner import run_training

        ds_name = "p4_cli_test_ds"
        ds_dir = self.tmp_dir / "datasets" / ds_name
        ds_dir.mkdir(parents=True)
        data_fixture("2019-01-02", periods=10).to_csv(ds_dir / "train_data.csv", index=False)
        data_fixture("2020-01-02", periods=6).to_csv(ds_dir / "val_data.csv", index=False)
        data_fixture("2021-01-04", periods=6).to_csv(ds_dir / "trade_data.csv", index=False)

        train_cfg = {
            "data": {
                "root_dir": str(self.tmp_dir / "datasets"),
                "train_file": "train_data.csv",
                "val_file": "val_data.csv",
                "test_file": "trade_data.csv",
            },
            "environment": env_config(),
            "training": {"total_timesteps": 8, "policy": "MlpPolicy", "device": "cpu", "verbose": 0, "logger_formats": ["csv"]},
            "algorithms": {
                "tqc": {
                    "learning_rate": 1e-3,
                    "buffer_size": 32,
                    "learning_starts": 2,
                    "batch_size": 4,
                    "top_quantiles_to_drop_per_net": 1,
                    "policy_kwargs": {"net_arch": [8, 8], "n_critics": 2, "n_quantiles": 5},
                }
            },
            "output": {
                "trained_models_dir": str(self.tmp_dir / "models"),
                "tensorboard_log_dir": str(self.tmp_dir / "tb"),
                "results_dir": str(self.tmp_dir / "results"),
            },
        }
        manifest = run_training(train_cfg, ds_name, "tqc", 3, self.tmp_dir, run_type="pilot", device="cpu")
        model_zip = manifest["artifacts"]["model_zip"]

        spec = importlib.util.spec_from_file_location("p4_cli", ROOT / "scripts/p4_critic_diagnostics.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)

        out_json = self.tmp_dir / "scan_out.json"
        with patch.object(
            sys,
            "argv",
            ["p4_critic_diagnostics.py", "scan", "--model", model_zip, "--data", str(ds_dir / "val_data.csv"), "--output", str(out_json)],
        ):
            self.assertEqual(cli.main(), 0)
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        self.assertEqual(payload["algorithm"], "tqc")
        self.assertEqual(payload["observations"], 5)
        self.assertTrue(Path(payload["daily_csv"]).is_file())

        # Must strictly refuse scanning test split (trade_data.csv)!
        with patch.object(
            sys,
            "argv",
            ["p4_critic_diagnostics.py", "scan", "--model", model_zip, "--data", str(ds_dir / "trade_data.csv")],
        ):
            self.assertEqual(cli.main(), 1)

        cmp_json = self.tmp_dir / "compare.json"
        with patch.object(
            sys,
            "argv",
            ["p4_critic_diagnostics.py", "compare-impl", "--algorithm", "both", "--timesteps", "8", "--output", str(cmp_json)],
        ):
            self.assertEqual(cli.main(), 0)
        cmp_payload = json.loads(cmp_json.read_text(encoding="utf-8"))
        self.assertEqual(cmp_payload["status"], "passed")
        self.assertIn("sac", cmp_payload["results"])
        self.assertIn("tqc", cmp_payload["results"])

    def test_validation_critic_threshold_selection_rules_and_cli(self) -> None:
        from backtest import runner
        from backtest.data import write_json
        from thresholds import select_critic_threshold

        impl_sha = "impl_sha_fixed"

        def write_manifest(
            folder: str,
            seed: int,
            mode: str,
            cfg: dict,
            sharpe,
            *,
            split: str = "validation",
            model_hash: str | None = None,
        ) -> Path:
            run_dir = self.tmp_dir / folder
            run_dir.mkdir(parents=True)
            cond = runner.threshold_identity(mode, cfg, impl_sha if mode != "none" else None)
            manifest = {
                "status": "completed",
                "algorithm": "sac",
                "seed": seed,
                "split": split,
                "model_training_run_type": "main",
                "annualization": 252,
                "risk_free_rate": 0.0,
                "environment": env_config(),
                "environment_code_sha256": {"shared": "e1"},
                "input_sha256": {"model": model_hash or f"sac_model_s{seed}", "data": "val_csv_sha"},
                **cond,
            }
            write_json(run_dir / "run_manifest.json", manifest)
            write_json(
                run_dir / "metrics.json",
                {
                    "strategy": {
                        "total_return": 0.08,
                        "annual_return": 0.04,
                        "annual_volatility": 0.12,
                        "max_drawdown": -0.05,
                        "sharpe": sharpe,
                        "trade_count": 10,
                        "threshold_vector_rejection_rate": 0.25 if mode == "critic" else None,
                    }
                },
            )
            return run_dir / "run_manifest.json"

        c0 = {"rule": RULE, "summary": "min_q", "delta_threshold": 0.0}
        c1 = {"rule": RULE, "summary": "min_q", "delta_threshold": 0.002}
        c_null = {"rule": RULE, "summary": "min_q", "delta_threshold": 0.005}

        manifests = []
        for s in (42, 2024):
            manifests.append(write_manifest(f"none_s{s}", s, "none", {}, 0.9))
            # Tie mean_sharpe = 1.20 between c0 and c1 -> tie-breaker must pick lower delta_threshold (0.0)!
            manifests.append(write_manifest(f"c0_s{s}", s, "critic", c0, 1.20))
            manifests.append(write_manifest(f"c1_s{s}", s, "critic", c1, 1.20))
            # Null sharpe on one seed must disqualify c_null instead of treating null as 0
            manifests.append(write_manifest(f"cn_s{s}", s, "critic", c_null, None if s == 2024 else 9.99))

        spec = importlib.util.spec_from_file_location("p4_cli_sel", ROOT / "scripts/p4_critic_diagnostics.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)

        sel_json = self.tmp_dir / "sac_selected.json"
        with patch.object(
            sys,
            "argv",
            [
                "p4_critic_diagnostics.py",
                "select",
                "--algorithm",
                "sac",
                "--seeds",
                "42",
                "2024",
                "--output",
                str(sel_json),
                "--manifests",
                *[str(m) for m in manifests],
            ],
        ):
            self.assertEqual(cli.main(), 0)

        record = json.loads(sel_json.read_text(encoding="utf-8"))
        self.assertEqual(record["selection_status"], "selected")
        self.assertEqual(record["selected"]["threshold_config"]["delta_threshold"], 0.0)
        self.assertEqual(record["selected"]["selection_reason"], "tie_broken_by_lower_delta_threshold")
        disqualified = [c for c in record["candidates"] if not c["eligible"]]
        self.assertEqual(len(disqualified), 1)
        self.assertIn("null_or_nonfinite_sharpe", disqualified[0]["ineligible_reason"])

        # Reject test split manifests immediately
        test_m = write_manifest("leaked_test", 42, "critic", c0, 1.5, split="test")
        with self.assertRaisesRegex(ValueError, "split='validation'"):
            select_critic_threshold([test_m], algorithm="sac")

        # All-invalid candidates -> status 'failed' and CLI exit code 2
        only_bad = [
            write_manifest("bad_42", 42, "critic", c_null, None),
            write_manifest("bad_2024", 2024, "critic", c_null, 1.0),
        ]
        fail_json = self.tmp_dir / "failed_sel.json"
        with patch.object(
            sys,
            "argv",
            [
                "p4_critic_diagnostics.py",
                "select",
                "--algorithm",
                "sac",
                "--seeds",
                "42",
                "2024",
                "--output",
                str(fail_json),
                "--manifests",
                *[str(m) for m in only_bad],
            ],
        ):
            self.assertEqual(cli.main(), 2)
        self.assertEqual(json.loads(fail_json.read_text(encoding="utf-8"))["selection_status"], "failed")


if __name__ == "__main__":
    unittest.main()
