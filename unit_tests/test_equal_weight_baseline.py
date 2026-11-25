"""Unit and P2-integration tests for the T06 equal-weight buy-and-hold baseline."""
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
import pandas as pd

from backtest.data import DEFAULT_INDICATORS, align_reference, load_market
from backtest import runner
from baselines import EqualWeightConfig, run_equal_weight_baseline, simulate_equal_weight
from test_backtest import ConstantModel, data_fixture, env_config
from trading_environment import build_environment


class EqualWeightBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        (ROOT / ".test_tmp").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".test_tmp")
        self.tmp_dir = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_exact_accounting_integer_and_fractional_units(self) -> None:
        # 2 stocks AAA (10 -> 11 -> 12), BBB (20 -> 21 -> 22); capital 1000, buy_cost 1%
        raw = data_fixture(periods=3)
        csv_path = self.tmp_dir / "data.csv"
        raw.to_csv(csv_path, index=False)
        frame = load_market(csv_path, DEFAULT_INDICATORS, "turbulence")

        cfg = EqualWeightConfig(initial_amount=1000.0, buy_cost_pct=0.01, unit_mode="integer_virtual_units")
        sim = simulate_equal_weight(frame, cfg)
        acc = sim["account_value"]
        # Budget 500 each:
        # AAA @ 10 * 1.01 = 10.1 -> floor(500 / 10.1) = 49 shares, outlay = 494.9, fee = 4.9
        # BBB @ 20 * 1.01 = 20.2 -> floor(500 / 20.2) = 24 shares, outlay = 484.8, fee = 4.8
        # Remaining cash = 1000 - 494.9 - 484.8 = 20.3
        # Day 1 mark: 20.3 + 49*11 + 24*21 = 20.3 + 539 + 504 = 1063.3
        # Day 2 mark: 20.3 + 49*12 + 24*22 = 20.3 + 588 + 528 = 1136.3
        np.testing.assert_allclose(acc["account_value"].to_numpy(), [1000.0, 1063.3, 1136.3], rtol=1e-12)
        self.assertAlmostEqual(sim["metrics"]["cumulative_cost"], 9.7, places=9)
        self.assertEqual(sim["metrics"]["trade_count"], 2)
        self.assertEqual(sim["metrics"]["allocated_stock_count"], 2)

        frac_cfg = EqualWeightConfig(initial_amount=1000.0, buy_cost_pct=0.01, unit_mode="fractional_units")
        frac_sim = simulate_equal_weight(frame, frac_cfg)
        self.assertAlmostEqual(frac_sim["metrics"]["residual_uninvested_cash_final"], 0.0, places=9)

    def test_first_day_suspension_defer_vs_keep_cash(self) -> None:
        raw = data_fixture(periods=4)
        dates = raw["date"].unique().tolist()
        # Suspend AAA on day 0; tradable again on day 1
        raw.loc[(raw["date"] == dates[0]) & (raw["tic"] == "AAA"), "tradable"] = False
        csv_path = self.tmp_dir / "susp.csv"
        raw.to_csv(csv_path, index=False)
        frame = load_market(csv_path, DEFAULT_INDICATORS, "turbulence")

        defer_sim = simulate_equal_weight(
            frame,
            EqualWeightConfig(initial_amount=1000.0, buy_cost_pct=0.01, suspension_policy="defer_reserved_cash"),
        )
        self.assertEqual(defer_sim["metrics"]["first_day_suspended_tickers"], ["AAA"])
        self.assertEqual(defer_sim["metrics"]["allocated_stock_count"], 2)
        aaa_trades = defer_sim["trades"].loc[defer_sim["trades"]["tic"] == "AAA"]
        self.assertEqual(len(aaa_trades), 2)  # day 0 blocked_by_mask=True, day 1 filled
        self.assertTrue(bool(aaa_trades.iloc[0]["blocked_by_mask"]))
        self.assertEqual(aaa_trades.iloc[1]["date"], dates[1])
        # On day 1 AAA price is 11 -> floor(500 / (11 * 1.01)) = 45 shares
        self.assertEqual(aaa_trades.iloc[1]["filled_shares"], 45.0)

        keep_sim = simulate_equal_weight(
            frame,
            EqualWeightConfig(initial_amount=1000.0, buy_cost_pct=0.01, suspension_policy="keep_cash"),
        )
        self.assertEqual(keep_sim["metrics"]["allocated_stock_count"], 1)
        self.assertEqual(keep_sim["metrics"]["never_allocated_tickers"], ["AAA"])
        self.assertGreaterEqual(keep_sim["metrics"]["residual_uninvested_cash_final"], 500.0)

    def test_all_period_suspended_stock_keeps_budget_in_cash(self) -> None:
        raw = data_fixture(periods=3)
        raw.loc[raw["tic"] == "AAA", "tradable"] = False
        csv_path = self.tmp_dir / "all_susp.csv"
        raw.to_csv(csv_path, index=False)
        frame = load_market(csv_path, DEFAULT_INDICATORS, "turbulence")

        sim = simulate_equal_weight(frame, EqualWeightConfig(initial_amount=1000.0, buy_cost_pct=0.01))
        self.assertEqual(sim["metrics"]["allocated_stock_count"], 1)
        self.assertEqual(sim["metrics"]["never_allocated_tickers"], ["AAA"])
        self.assertGreaterEqual(sim["metrics"]["residual_uninvested_cash_final"], 500.0)

    def test_artifacts_pass_p2_align_reference_and_run_backtest_baseline_slot(self) -> None:
        raw = data_fixture(periods=5)
        csv_path = self.tmp_dir / "val_data.csv"
        raw.to_csv(csv_path, index=False)
        dates = raw["date"].drop_duplicates().tolist()
        bench_path = self.tmp_dir / "val_index.csv"
        pd.DataFrame(
            {"date": dates, "close": [100, 102, 101, 104, 106], "regime": ["up", "up", "sideways", "up", "up"]}
        ).to_csv(bench_path, index=False)

        cfg = EqualWeightConfig.from_environment_dict(env_config())
        manifest = run_equal_weight_baseline(
            csv_path,
            self.tmp_dir / "baselines",
            config=cfg,
            split="validation",
            benchmark_path=bench_path,
            output_dir_name="eq_val",
        )
        eq_csv = Path(manifest["account_value_csv"])
        self.assertTrue(eq_csv.is_file())

        aligned = align_reference(eq_csv, dates, cfg.initial_amount, "equal_weight")
        self.assertEqual(aligned["date"].tolist(), dates)
        self.assertAlmostEqual(float(aligned["account_value"].iloc[0]), cfg.initial_amount, places=9)

        # Verify P2 run_backtest directly consumes equal_weight via baselines={"equal_weight": eq_csv}
        env = build_environment(load_market(csv_path, DEFAULT_INDICATORS, "turbulence"), env_config())
        model = ConstantModel([0.0, 0.0])
        model.observation_space, model.action_space = env.observation_space, env.action_space
        model_zip = self.tmp_dir / "dummy.zip"
        model_zip.write_bytes(b"zip fixture")
        effective = {"algorithm": "sac", "seed": 42, "environment": env_config(), "run_type": "smoke"}
        with patch.object(runner, "prepare", return_value=(env.df, env, ["AAA", "BBB"], {}, effective)), \
                patch("stable_baselines3.SAC.load", return_value=model), \
                patch.object(runner, "_plots"):
            bt_manifest = runner.run_backtest(
                model_zip,
                csv_path,
                self.tmp_dir / "bt_runs",
                split="validation",
                benchmark_path=bench_path,
                baselines={"equal_weight": eq_csv},
            )
        bt_metrics = json.loads((Path(bt_manifest["output"]) / "metrics.json").read_text(encoding="utf-8"))
        self.assertIn("equal_weight", bt_metrics)
        self.assertIn("market_index", bt_metrics)
        eq_metrics = json.loads((Path(manifest["output"]) / "metrics.json").read_text(encoding="utf-8"))["equal_weight"]
        self.assertAlmostEqual(bt_metrics["equal_weight"]["total_return"], eq_metrics["total_return"], places=12)
        self.assertAlmostEqual(bt_metrics["equal_weight"]["sharpe"], eq_metrics["sharpe"], places=12)

    def test_cli_script_runs_cleanly(self) -> None:
        raw = data_fixture(periods=4)
        csv_path = self.tmp_dir / "val_data.csv"
        raw.to_csv(csv_path, index=False)
        spec = importlib.util.spec_from_file_location("eq_cli", ROOT / "scripts/run_equal_weight_baseline.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)

        out_root = self.tmp_dir / "cli_out"
        with patch.object(
            sys,
            "argv",
            [
                "run_equal_weight_baseline.py",
                "--data",
                str(csv_path),
                "--split",
                "validation",
                "--output-root",
                str(out_root),
                "--output-name",
                "val_eq",
            ],
        ):
            self.assertEqual(cli.main(), 0)
        self.assertTrue((out_root / "val_eq" / "account_value.csv").is_file())
        self.assertTrue((out_root / "val_eq" / "metrics.json").is_file())


if __name__ == "__main__":
    unittest.main()
