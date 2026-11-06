"""Shared adjusted-unit environment for training and evaluation.

Keeps FinRL's same-close execution abstraction and never pays dividends or
changes share counts for corporate actions on top of adjusted prices.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import math

import pandas as pd

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv


class AdjustedTradingEnv(StockTradingEnv):
    def __init__(self, *, tradability_policy="legacy_finrl", **kwargs):
        self.tradability_policy = tradability_policy
        self.audit_orders = False
        self.order_trace = []
        super().__init__(**kwargs)

    def _permitted(self, index):
        if self.tradability_policy == "legacy_finrl":
            return True
        value = self.data["tradable"]
        return bool(value.iloc[index] if isinstance(value, pd.Series) else value)

    def _buy_stock(self, index, action):
        return self._trade(index, action, "buy")

    def _sell_stock(self, index, action):
        # Also guard the risk-triggered liquidation branch in the base class.
        return self._trade(index, action, "sell")

    def _trade(self, index, action, side):
        cash_before = float(self.state[0])
        position_before = float(self.state[1+self.stock_dim+index])
        cost_before = float(self.cost)
        permitted = self._permitted(index)
        quantity = getattr(super(), f"_{side}_stock")(index, action) if permitted else 0
        if self.audit_orders:
            self.order_trace.append({"sequence": len(self.order_trace), "ticker_index": int(index),
                "side": side, "environment_requested_shares": float(action),
                "filled_shares": float(quantity), "execution_price": float(self.state[1+index]),
                "cash_before": cash_before, "cash_after": float(self.state[0]),
                "position_before": position_before,
                "position_after": float(self.state[1+self.stock_dim+index]),
                "fee": float(self.cost)-cost_before, "blocked_by_mask": not permitted})
        return quantity


def environment_hashes():
    import inspect
    paths = {"shared_environment": Path(__file__),
             "finrl_environment": Path(inspect.getfile(StockTradingEnv))}
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


def build_environment(frame, environment_config):
    for key in ["initial_amount", "reward_scaling", "hmax"]:
        if not math.isfinite(float(environment_config[key])) or float(environment_config[key]) <= 0:
            raise ValueError(f"environment.{key} must be positive and finite")
    for key in ["buy_cost_pct", "sell_cost_pct"]:
        if not math.isfinite(float(environment_config[key])) or not 0 <= float(environment_config[key]) < 1:
            raise ValueError(f"environment.{key} must be in [0,1)")
    indicators = list(environment_config["tech_indicator_list"])
    policy = environment_config.get("tradability_policy", "legacy_finrl")
    if policy not in ["legacy_finrl", "enforce_column"]:
        raise ValueError("tradability_policy must be legacy_finrl or enforce_column")
    if environment_config.get("price_basis", "adjusted_virtual_units") != "adjusted_virtual_units":
        raise ValueError("This environment only supports adjusted_virtual_units")
    frame = frame.copy()
    if "tradable" in frame:
        values = frame["tradable"].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
        if values.isna().any():
            raise ValueError("Invalid/missing tradable values")
        frame["tradable"] = values.astype(bool)
        if policy == "legacy_finrl" and not values.all():
            raise ValueError("Data has blocked sessions but the model uses legacy_finrl. Train with enforce_column; do not change only the backtest environment.")
    elif policy == "enforce_column":
        raise ValueError("enforce_column requires a tradable column")
    dimension = int(frame["tic"].nunique())
    kwargs = {
        "df": frame, "stock_dim": dimension, "action_space": dimension,
        "state_space": 1 + (2 + len(indicators)) * dimension,
        "hmax": int(environment_config["hmax"]),
        "initial_amount": float(environment_config["initial_amount"]),
        "num_stock_shares": [0]*dimension,
        "buy_cost_pct": [float(environment_config["buy_cost_pct"])]*dimension,
        "sell_cost_pct": [float(environment_config["sell_cost_pct"])]*dimension,
        "reward_scaling": float(environment_config["reward_scaling"]),
        "tech_indicator_list": indicators,
        "risk_indicator_col": environment_config["risk_indicator_col"],
        "print_verbosity": int(environment_config.get("print_verbosity", 10)),
    }
    if environment_config.get("turbulence_threshold") is not None:
        kwargs["turbulence_threshold"] = float(environment_config["turbulence_threshold"])
    return AdjustedTradingEnv(tradability_policy=policy, **kwargs)
