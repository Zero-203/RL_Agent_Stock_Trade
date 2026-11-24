"""P2 threshold plumbing tests; fake filters are not research implementations."""
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
import pandas as pd
from backtest import runner
from backtest.data import DEFAULT_INDICATORS, load_market, write_json
from test_backtest import ConstantModel, data_fixture, env_config
from trading_environment import build_environment


class ThresholdTests(unittest.TestCase):
    def setUp(self):
        (ROOT / ".test_tmp").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".test_tmp")
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def environment(self, data=None, cfg=None):
        path = self.directory / "data.csv"
        (data_fixture(periods=4) if data is None else data).to_csv(path, index=False)
        return build_environment(load_market(path, DEFAULT_INDICATORS, "turbulence"), cfg or env_config())

    def test_context_dtype_order_and_original_action_survive_filter_mutation(self):
        model = ConstantModel([.29, -.51])
        model.critic = object()
        env = self.environment()
        calls = []

        def simple(raw, mode, config, **context):
            self.assertEqual(raw.dtype, np.float32)
            self.assertIs(context["model"], model)
            self.assertIs(context["critic"], model.critic)
            self.assertEqual(context["ordered_tickers"], ("AAA", "BBB"))
            self.assertEqual(context["algorithm"], "sac")
            self.assertEqual(context["hmax"], env.hmax)
            self.assertEqual(context["obs"].dtype, np.float32)
            calls.append(context["obs"].copy())
            config["q_min"] = 999  # Caller-owned configuration must remain frozen.
            raw[0] = 0
            return raw, {"threshold_mode": mode, "accepted": np.array([False, True])}

        config = {"q_min": 5}
        with patch.object(runner, "apply_threshold", side_effect=simple):
            _, actions, _, _ = runner.replay(model, env, ["AAA", "BBB"], threshold_mode="simple",
                                              threshold_config=config, algorithm="sac")
        self.assertEqual(len(calls), 3)
        self.assertEqual(config, {"q_min": 5})
        self.assertTrue((actions.loc[actions.tic == "AAA", "model_action"] > 0).all())
        self.assertTrue(actions.loc[actions.tic == "AAA", "thresholded_action"].eq(0).all())
        self.assertTrue(actions.loc[actions.tic == "BBB", "threshold_accepted"].all())

    def test_threshold_output_revalidated_before_environment_step(self):
        invalid = [np.array([[0., 0.]]), np.zeros((2, 1)), np.zeros(3),
                   np.array([np.nan, 0]), np.array([np.inf, 0]), np.array([1.00000001, 0])]
        for value in invalid:
            with self.subTest(value=value):
                env = self.environment()
                with patch.object(runner, "apply_threshold", return_value=(value, {"threshold_mode": "simple"})), \
                        patch.object(env, "step", wraps=env.step) as step:
                    with self.assertRaisesRegex(ValueError, "Threshold returned an invalid"):
                        runner.replay(ConstantModel([.5, 0]), env, ["AAA", "BBB"], threshold_mode="simple")
                    step.assert_not_called()

    def test_model_only_accepts_single_vector_or_single_batch(self):
        for value in ([[.5], [0]], [.5, 0, 0], [1.01, 0]):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Model returned an invalid"):
                runner.replay(ConstantModel(value), self.environment(), ["AAA", "BBB"])
        runner.replay(ConstantModel([[.5, 0]]), self.environment(), ["AAA", "BBB"])

    def test_invalid_diagnostics_and_action_rewrites_are_rejected(self):
        for info, action in [({"threshold_mode": "critic", "accepted": False}, [.5, 0]),
                             ({"threshold_mode": "simple", "accepted": True}, [0, 0]),
                             ({"threshold_mode": "simple", "accepted": [1, 0]}, [0, 0]),
                             ({"threshold_mode": "simple", "accepted": [True, True]}, [.4, 0]),
                             ({"threshold_mode": "none"}, [.5, 0]),
                             ({"threshold_mode": "simple", "accepted": [True, True], "critic_delta": np.nan}, [.5, 0])]:
            mode = "critic" if info["threshold_mode"] == "critic" else "simple"
            with self.subTest(info=info), patch.object(runner, "apply_threshold", return_value=(action, info)):
                with self.assertRaises(ValueError):
                    runner.replay(ConstantModel([.5, 0]), self.environment(), ["AAA", "BBB"], threshold_mode=mode)

    def test_critic_rejection_does_not_override_suspension_or_risk_liquidation(self):
        data = data_fixture(periods=4)
        dates = data.date.unique()
        data.loc[data.date == dates[1], "turbulence"] = 100
        data.loc[(data.date == dates[1]) & (data.tic == "AAA"), "tradable"] = False
        cfg = env_config()
        cfg["turbulence_threshold"] = 5
        calls = []

        def critic(raw, mode, config, **context):
            accept = not calls
            calls.append(accept)
            return (raw if accept else np.zeros_like(raw)), {
                "threshold_mode": mode, "accepted": accept,
                "critic_policy_value": 2., "critic_zero_value": 1., "critic_delta": 1.}

        with patch.object(runner, "apply_threshold", side_effect=critic):
            _, actions, _, _ = runner.replay(ConstantModel([1, 1]), self.environment(data, cfg),
                                             ["AAA", "BBB"], threshold_mode="critic")
        second = actions[actions.date == dates[1]].set_index("tic")
        self.assertFalse(second.threshold_accepted.any())
        self.assertTrue(second.requested_shares.eq(0).all())
        self.assertEqual(second.loc["AAA", "executed_shares"], 0)
        self.assertLess(second.loc["BBB", "executed_shares"], 0)
        self.assertEqual(second.loc["BBB", "reason"], "risk_liquidation")
        self.assertTrue(actions.critic_delta.eq(1).all())

    def test_request_rounding_matches_float32_environment_path(self):
        cfg = env_config()
        cfg["hmax"] = 100
        raw = np.array([.29, -.29], dtype=np.float32)
        _, actions, _, _ = runner.replay(ConstantModel(raw), self.environment(cfg=cfg), ["AAA", "BBB"])
        np.testing.assert_array_equal(actions.requested_shares.iloc[:2], (raw * 100).astype(int))

    def test_condition_identity_freezes_parameters_and_implementation(self):
        a = runner.threshold_identity("simple", {"x": 1, "y": 2}, "impl1")
        self.assertEqual(a, runner.threshold_identity("simple", {"y": 2, "x": 1}, "impl1"))
        for mode, config, impl in [("simple", {"x": 2, "y": 2}, "impl1"),
                                   ("critic", {"x": 1, "y": 2}, "impl1"),
                                   ("simple", {"x": 1, "y": 2}, "impl2")]:
            self.assertNotEqual(a["threshold_condition_sha256"], runner.threshold_identity(mode, config, impl)["threshold_condition_sha256"])
        for mode, config in [("unknown", {}), ("none", {"q_min": 2}), ("simple", []),
                             ("simple", {"q_min": float("nan")})]:
            with self.subTest(mode=mode, config=config), self.assertRaises(ValueError):
                runner.threshold_identity(mode, config)

    def manifest(self, name, mode="none", config=None, seed=1, model="model", **overrides):
        directory = self.directory / name
        directory.mkdir()
        condition = runner.threshold_identity(mode, config, "impl")
        manifest = {"status": "completed", "algorithm": "sac", "seed": seed, "split": "validation",
                    "input_sha256": {"model": model, "data": "same_data"}, "environment": env_config(),
                    "model_training_run_type": "smoke", "annualization": 252, "risk_free_rate": 0,
                    **condition, **overrides}
        write_json(directory / "run_manifest.json", manifest)
        write_json(directory / "metrics.json", {"strategy": {"total_return": .1, "annual_return": .2,
                   "annual_volatility": .3, "max_drawdown": -.1, "sharpe": None if seed == 1 else 1.}})
        return directory / "run_manifest.json"

    def test_same_model_multi_condition_summary_and_seed_counts(self):
        paths = []
        for seed in (1, 2):
            for mode, config in [("none", {}), ("simple", {"q_min": 2}),
                                 ("simple", {"q_min": 5}), ("critic", {"delta": .1})]:
                paths.append(self.manifest(f"s{seed}_{mode}_{len(paths)}", mode, config, seed, f"m{seed}"))
        output = self.directory / "summary.csv"
        frame = runner.summarize_runs(paths, output)
        self.assertEqual(len(frame), 8)
        stats = pd.read_csv(self.directory / "summary_by_algorithm.csv", header=[0, 1], index_col=[0, 1, 2, 3])
        self.assertEqual(len(stats), 4)
        self.assertTrue(stats[("total_return", "count")].eq(2).all())
        self.assertTrue(stats[("sharpe", "count")].eq(1).all())
        self.assertTrue(stats[("sharpe", "std")].isna().all())

    def test_summary_rejects_duplicate_unpaired_or_incomparable_runs(self):
        base = self.manifest("base")
        for other, error in [
                (self.manifest("duplicate"), "Duplicate"),
                (self.manifest("unpaired", "simple", {"q_min": 2}, model="different"), "same model"),
                (self.manifest("env", environment_code_sha256={"file": "changed"}), "Incomparable"),
                (self.manifest("ref", input_sha256={"data": "same_data", "model": "model", "market_index": "changed"}), "Incomparable"),
                (self.manifest("failed", status="failed"), "incomplete"),
                (self.manifest("identity", threshold_condition_sha256="tampered"), "identity mismatch")]:
            with self.subTest(other=other), self.assertRaisesRegex(ValueError, error):
                runner.summarize_runs([base, other], self.directory / "bad.csv")
        with self.assertRaisesRegex(ValueError, "At least one"):
            runner.summarize_runs([], self.directory / "empty.csv")

    def test_legacy_no_threshold_manifests_still_work(self):
        path = self.manifest("legacy")
        manifest = json.loads(path.read_text())
        manifest = {k: v for k, v in manifest.items() if not k.startswith("threshold_")}
        write_json(path, manifest)
        frame = runner.summarize_runs([path], self.directory / "legacy.csv")
        self.assertEqual(frame.threshold_mode.iloc[0], "none")

    def test_cli_and_batch_pass_inference_configuration(self):
        spec = importlib.util.spec_from_file_location("backtest_cli", ROOT / "scripts/backtest.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        path = self.directory / "threshold.json"
        path.write_text(json.dumps({"q_min": 5}), encoding="utf-8-sig")
        with patch.object(sys, "argv", ["backtest.py", "run", "--model", "m.zip", "--data", "v.csv",
                                       "--threshold-mode", "simple", "--threshold-config", str(path)]), \
                patch.object(runner, "run_backtest", return_value={}) as run:
            self.assertEqual(cli.main(), 0)
            self.assertEqual(run.call_args.kwargs["threshold_mode"], "simple")
            self.assertEqual(run.call_args.kwargs["threshold_config"], {"q_min": 5})
        batch = self.directory / "batch.json"
        write_json(batch, {"defaults": {"threshold_mode": "none"}, "jobs": [
            {"model_path": "m.zip", "data_path": "v.csv", "output_root": "out"},
            {"model_path": "m.zip", "data_path": "v.csv", "output_root": "out",
             "threshold_mode": "critic", "threshold_config": {"delta": .1}}]})
        with patch.object(sys, "argv", ["backtest.py", "batch", "--config", str(batch)]), \
                patch.object(runner, "run_backtest", return_value={"output": str(self.directory)}) as run:
            self.assertEqual(cli.main(), 0)
            self.assertEqual(run.call_args_list[0].kwargs["threshold_mode"], "none")
            self.assertEqual(run.call_args_list[1].kwargs["threshold_config"], {"delta": .1})

    def test_run_backtest_explicit_override_manifest_and_failed_filter(self):
        env = self.environment()
        model = ConstantModel([.5, 0])
        model.observation_space, model.action_space = env.observation_space, env.action_space
        model_path = self.directory / "model.zip"
        model_path.write_bytes(b"hash fixture")
        effective = {"algorithm": "ppo", "seed": 1, "environment": env_config(), "run_type": "smoke",
                     "threshold_mode": "critic", "threshold_config": {"ignored_training_field": True}}
        prepared = (env.df, env, ["AAA", "BBB"], {}, effective)
        with patch.object(runner, "prepare", return_value=prepared), \
                patch("stable_baselines3.PPO.load", return_value=model), patch.object(runner, "_plots"):
            result = runner.run_backtest(model_path, self.directory / "data.csv", self.directory / "runs",
                                         split="validation", threshold_mode="none")
            self.assertEqual(result["threshold_mode"], "none")
            self.assertEqual(result["threshold_config"], {})
            self.assertEqual(result["status"], "completed")
            metrics = json.loads((Path(result["output"]) / "metrics.json").read_text())["strategy"]
            self.assertEqual(metrics["threshold_asset_rejection_rate"], 0)
            with self.assertRaises(NotImplementedError):
                runner.run_backtest(model_path, self.directory / "data.csv", self.directory / "runs",
                                    split="validation", threshold_mode="simple", threshold_config={"q_min": 5})
        failed = [json.loads(p.read_text()) for p in (self.directory / "runs").glob("*/run_manifest.json")
                  if json.loads(p.read_text())["status"] == "failed"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["threshold_config"], {"q_min": 5})
        self.assertTrue(failed[0]["threshold_implementation_sha256"])


if __name__ == "__main__":
    unittest.main()
