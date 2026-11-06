import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"src"))

import numpy as np
import pandas as pd

from backtest.data import (DEFAULT_INDICATORS, align_reference, check_market,
                           load_market, read_metadata, resolve_tickers, sha256)
from backtest.metrics import curve_metrics, regime_metrics
from backtest.runner import prepare, replay
from trading_environment import build_environment, environment_hashes


def data_fixture(start="2021-01-04", periods=3, tickers=("AAA", "BBB")):
    rows = []
    for i, date in enumerate(pd.bdate_range(start, periods=periods)):
        for j, ticker in enumerate(tickers):
            price = 10.0+i+j*10
            row = {"date": date.strftime("%Y-%m-%d"), "tic": ticker,
                "open": price, "high": price+1, "low": price-1,
                "close": price, "volume": 10000, "day": date.dayofweek,
                "turbulence": 0.0, "tradable": True}
            row.update({name: 0.0 for name in DEFAULT_INDICATORS})
            rows.append(row)
    return pd.DataFrame(rows)


def env_config():
    return {"initial_amount": 1000, "hmax": 10, "buy_cost_pct": .01,
        "sell_cost_pct": .01, "reward_scaling": 1,
        "tech_indicator_list": DEFAULT_INDICATORS, "risk_indicator_col": "turbulence",
        "turbulence_threshold": None, "print_verbosity": 100000,
        "price_basis": "adjusted_virtual_units", "tradability_policy": "enforce_column"}


class ConstantModel:
    def __init__(self, action):
        self.action = action

    def predict(self, obs, deterministic=True):
        return np.array(self.action, dtype=float), None


class BacktestTests(unittest.TestCase):
    def setUp(self):
        base = ROOT/".test_tmp"
        base.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="backtest_", dir=base)
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def market(self, frame=None):
        path = self.directory/"market.csv"
        (data_fixture() if frame is None else frame).to_csv(path, index=False)
        return path, load_market(path, DEFAULT_INDICATORS, "turbulence")

    def metadata(self, tickers=("AAA", "BBB"), **kwargs):
        effective = {"algorithm": "ppo", "environment": env_config(), "seed": 1,
            "data_paths": {"train_csv": "missing_train", "val_csv": "missing_val"},
            "data_hashes": {"train_csv": "trainhash", "val_csv": "valhash"},
            "data_contract": {"ordered_tickers": list(tickers), "ordered_indicators": DEFAULT_INDICATORS,
                "price_basis": "adjusted_virtual_units", "execution_timing": "finrl_same_close",
                "train_end": "2019-12-31", "validation_end": "2020-12-31",
                "environment_code_sha256": environment_hashes()}}
        effective.update(kwargs)
        path = self.directory/"model.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("metadata.txt", "FinRL T04 model metadata\n\n"+json.dumps({"status": "completed", "effective_config": effective}))
        return path, effective

    def test_cash_fee_holdings_reward_and_full_date_reconciliation(self):
        _, frame = self.market()
        env = build_environment(frame, env_config())
        account, actions, holdings, trades = replay(ConstantModel([1, 0]), env, ["AAA", "BBB"])
        self.assertEqual(len(account), 3)
        self.assertEqual(len(actions), 4)
        self.assertEqual(len(holdings), 6)
        self.assertEqual(len(trades), 2)
        self.assertAlmostEqual(account["account_value"].iloc[-1], 1027.9)
        self.assertAlmostEqual(account["cumulative_cost"].iloc[-1], 2.1)
        self.assertTrue(account["date"].is_monotonic_increasing)

    def test_training_wrapper_passes_policy_kwargs_only_once(self):
        from train_model.runner import build_model
        _, frame = self.market()
        env = build_environment(frame, env_config())
        vector, _ = env.get_sb_env()
        model = build_model("ppo", vector, {"n_steps": 4, "batch_size": 4,
            "policy_kwargs": {"net_arch": [16, 16]}}, 1, self.directory/"tb", device="cpu")
        self.assertEqual(model.policy.net_arch, [16, 16])
        vector.close()

    def test_mask_blocks_orders_and_keeps_all_dates(self):
        data = data_fixture()
        data.loc[(data.tic == "AAA") & (data.date == "2021-01-04"), "tradable"] = False
        _, frame = self.market(data)
        account, actions, _, trades = replay(ConstantModel([1, 0]), build_environment(frame, env_config()), ["AAA", "BBB"])
        self.assertEqual(len(account), 3)
        self.assertEqual(len(trades), 1)
        self.assertEqual(actions.iloc[0]["reason"], "not_tradable")
        self.assertEqual(actions.iloc[0]["executed_shares"], 0)
        self.assertAlmostEqual(account["account_value"].iloc[-1], 1008.9)

    def test_risk_liquidation_cannot_sell_a_suspended_asset(self):
        data = data_fixture()
        data.loc[(data.tic == "AAA") & (data.date == "2021-01-05"), "tradable"] = False
        data.loc[data.date == "2021-01-05", "turbulence"] = 100
        cfg = env_config()
        cfg["turbulence_threshold"] = 5
        _, frame = self.market(data)
        account, actions, _, trades = replay(ConstantModel([1, 0]), build_environment(frame, cfg), ["AAA", "BBB"])
        self.assertEqual(len(trades), 1)
        self.assertEqual(actions.loc[(actions.date == "2021-01-05") & (actions.tic == "AAA"), "executed_shares"].iloc[0], 0)

    def test_no_position_means_no_short_sale_or_fee(self):
        _, frame = self.market()
        account, _, _, trades = replay(ConstantModel([-1, 0]), build_environment(frame, env_config()), ["AAA", "BBB"])
        self.assertTrue(trades.empty)
        self.assertTrue(account["account_value"].eq(1000).all())
        self.assertIsNone(curve_metrics(account["account_value"])["sharpe"])

    def test_metadata_tickers_are_checked_by_identity_not_only_dimension(self):
        data_path, _ = self.market()
        model_path, _ = self.metadata(("AAA", "CCC"))
        with self.assertRaisesRegex(ValueError, "ticker identities"):
            prepare(model_path, data_path, "test")

    def test_environment_source_and_price_semantics_must_match_training(self):
        data_path, _ = self.market()
        model_path, effective = self.metadata()
        effective["data_contract"]["environment_code_sha256"] = {"changed": "hash"}
        with zipfile.ZipFile(model_path, "w") as archive:
            archive.writestr("metadata.txt", json.dumps({"effective_config": effective}))
        with self.assertRaisesRegex(ValueError, "source differs"):
            prepare(model_path, data_path, "test")

    def test_test_cannot_be_relabelled_validation_or_overlap_validation(self):
        data_path, _ = self.market()
        model_path, _ = self.metadata()
        with self.assertRaisesRegex(ValueError, "exact recorded validation"):
            prepare(model_path, data_path, "validation")
        old_path, _ = self.market(data_fixture(start="2020-01-06"))
        with self.assertRaisesRegex(ValueError, "follow validation"):
            prepare(model_path, old_path, "test")

    def test_legacy_metadata_requires_original_training_csv_hash(self):
        original = self.directory/"training.csv"
        data_fixture().to_csv(original, index=False)
        effective = {"data_contract": {}, "data_paths": {"train_csv": str(original)},
                     "data_hashes": {"train_csv": sha256(original)}}
        self.assertEqual(resolve_tickers(effective), ["AAA", "BBB"])
        effective["data_hashes"]["train_csv"] = "incorrect"
        with self.assertRaisesRegex(ValueError, "hash"):
            resolve_tickers(effective)

    def test_nonfinite_and_retired_rows_are_rejected_without_price_filling(self):
        data = data_fixture()
        data.loc[2, "close"] = np.nan
        path = self.directory/"invalid.csv"
        data.to_csv(path, index=False)
        report = check_market(path)
        self.assertFalse(report["input_compatible"])
        self.assertEqual(report["nonfinite_numeric_rows"], 1)
        data.loc[2, "close"] = np.inf
        data.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            load_market(path, DEFAULT_INDICATORS, "turbulence")

    def test_mask_policy_cannot_be_added_only_at_evaluation(self):
        data = data_fixture()
        data.loc[0, "tradable"] = False
        _, frame = self.market(data)
        cfg = env_config()
        cfg["tradability_policy"] = "legacy_finrl"
        with self.assertRaisesRegex(ValueError, "Train with enforce_column"):
            build_environment(frame, cfg)

    def test_benchmark_missing_date_is_rejected_not_inner_joined(self):
        path = self.directory/"index.csv"
        pd.DataFrame({"date": ["2021-01-04", "2021-01-06"], "close": [100, 101]}).to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "exactly match"):
            align_reference(path, ["2021-01-04", "2021-01-05", "2021-01-06"], 1000)

    def test_known_metrics_and_zero_variance(self):
        metrics = curve_metrics([100, 110, 99])
        self.assertAlmostEqual(metrics["total_return"], -.01)
        self.assertAlmostEqual(metrics["max_drawdown"], -.1)
        self.assertAlmostEqual(metrics["annual_return"], .99**126-1)
        self.assertAlmostEqual(metrics["sharpe"], 0)
        self.assertIsNone(curve_metrics([100, 100, 100])["sharpe"])
        with self.assertRaises(ValueError):
            curve_metrics([100, 0])

    def test_regime_attribution_does_not_jump_across_noncontiguous_days(self):
        curve = pd.DataFrame({"date": ["a", "b", "c", "d"], "account_value": [100, 110, 99, 108.9]})
        index = pd.DataFrame({"date": curve.date, "regime": ["up", "down", "up", "down"]})
        summary = regime_metrics(curve, index).set_index("regime")
        self.assertAlmostEqual(summary.loc["down", "conditional_compounded_return"], .21)
        self.assertAlmostEqual(summary.loc["up", "conditional_compounded_return"], -.1)


if __name__ == "__main__":
    unittest.main()
