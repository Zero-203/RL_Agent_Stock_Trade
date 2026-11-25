"""Equal-weight buy-and-hold baseline on the shared FinRL / P2 adjusted-price universe.

Accounting & Execution Semantics (aligned with ``src/backtest/runner.py``):
- Pre-trade initial row at ``dates[0]`` starts with ``account_value = initial_amount``,
  ``holdings_value = 0``, so output ``account_value.csv`` passes
  ``backtest.data.align_reference`` directly via ``--baseline equal_weight=<CSV>``.
- Same-close execution, next-session mark-to-market: orders placed at session ``t``
  execute at ``dates[t]`` adjusted ``close`` with ``buy_cost_pct`` and are first
  marked to market at ``dates[t + 1]``.
- Equal capital slices: each of the ``N`` ordered stocks receives an independent
  budget ``B_i = initial_amount / N``.
- Unit rounding (``unit_mode``):
  * ``'integer_virtual_units'`` (default): ``floor(B_i / (price * (1 + buy_cost_pct)))``
    matching ``StockTradingEnv`` integer simulated units.
  * ``'fractional_units'``: exact fractional virtual units ``B_i / (price * (1 + buy_cost_pct))``.
- Suspension policy (``suspension_policy`` when ``tradability_policy == 'enforce_column'``):
  * ``'defer_reserved_cash'`` (default): if stock ``i`` is suspended (``tradable=False``)
    on ``dates[0]``, reserve its ``B_i`` in cash and buy on the first session ``t <= T-2``
    where ``tradable=True``; if never tradable, ``B_i`` remains in cash.
  * ``'keep_cash'``: only buy tradable stocks on ``dates[0]``; suspended slices stay in
    uninvested cash for the whole period.
- Uninvested cash (rounding remainders + unallocated suspended slices) stays in the
  cash account at 0% return; no rebalancing or terminal liquidation fee is charged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import math
import platform
from pathlib import Path
from typing import Any
import uuid

import numpy as np
import pandas as pd

from backtest.data import DEFAULT_INDICATORS, load_market, sha256, write_json
from backtest.metrics import curve_metrics, regime_metrics

UNIT_MODES = ("integer_virtual_units", "fractional_units")
SUSPENSION_POLICIES = ("defer_reserved_cash", "keep_cash")
TRADABILITY_POLICIES = ("enforce_column", "legacy_finrl")


@dataclass(frozen=True)
class EqualWeightConfig:
    """Validated configuration for the same-universe equal-weight buy-and-hold baseline."""

    initial_amount: float = 1_000_000.0
    buy_cost_pct: float = 0.001
    tradability_policy: str = "enforce_column"
    suspension_policy: str = "defer_reserved_cash"
    unit_mode: str = "integer_virtual_units"
    price_basis: str = "adjusted_virtual_units"
    risk_indicator_col: str = "turbulence"
    tech_indicator_list: tuple[str, ...] = tuple(DEFAULT_INDICATORS)
    annualization: int = 252
    risk_free_rate: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.initial_amount) or self.initial_amount <= 0.0:
            raise ValueError("initial_amount must be finite and > 0")
        if not math.isfinite(self.buy_cost_pct) or not (0.0 <= self.buy_cost_pct < 1.0):
            raise ValueError("buy_cost_pct must satisfy 0 <= buy_cost_pct < 1")
        if self.tradability_policy not in TRADABILITY_POLICIES:
            raise ValueError(f"tradability_policy must be one of {TRADABILITY_POLICIES}")
        if self.suspension_policy not in SUSPENSION_POLICIES:
            raise ValueError(f"suspension_policy must be one of {SUSPENSION_POLICIES}")
        if self.unit_mode not in UNIT_MODES:
            raise ValueError(f"unit_mode must be one of {UNIT_MODES}")
        if self.price_basis != "adjusted_virtual_units":
            raise ValueError("Only price_basis='adjusted_virtual_units' is supported")
        if int(self.annualization) <= 0 or float(self.risk_free_rate) <= -1.0:
            raise ValueError("Invalid annualization or risk_free_rate")
        object.__setattr__(self, "tech_indicator_list", tuple(self.tech_indicator_list))

    @classmethod
    def from_environment_dict(
        cls,
        env_cfg: dict[str, Any],
        *,
        suspension_policy: str = "defer_reserved_cash",
        unit_mode: str = "integer_virtual_units",
        annualization: int = 252,
        risk_free_rate: float = 0.0,
    ) -> "EqualWeightConfig":
        return cls(
            initial_amount=float(env_cfg["initial_amount"]),
            buy_cost_pct=float(env_cfg["buy_cost_pct"]),
            tradability_policy=str(env_cfg.get("tradability_policy", "enforce_column")),
            suspension_policy=suspension_policy,
            unit_mode=unit_mode,
            price_basis=str(env_cfg.get("price_basis", "adjusted_virtual_units")),
            risk_indicator_col=str(env_cfg.get("risk_indicator_col", "turbulence")),
            tech_indicator_list=tuple(env_cfg.get("tech_indicator_list", DEFAULT_INDICATORS)),
            annualization=int(annualization),
            risk_free_rate=float(risk_free_rate),
        )


def simulate_equal_weight(
    market_frame: pd.DataFrame,
    config: EqualWeightConfig | None = None,
    *,
    expected_tickers: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Run equal-weight buy-and-hold simulation on an already validated market DataFrame."""
    cfg = config or EqualWeightConfig()
    tickers = sorted(market_frame["tic"].astype(str).unique().tolist())
    if not tickers:
        raise ValueError("Market frame contains no tickers")
    if expected_tickers is not None and list(expected_tickers) != tickers:
        raise ValueError("Market tickers/order disagree with expected_tickers")

    dates = market_frame["date"].drop_duplicates().tolist()
    if len(dates) < 2:
        raise ValueError("At least two trading dates are required for buy-and-hold")

    if "tradable" in market_frame.columns:
        tradable_vals = market_frame["tradable"].astype(bool)
        if cfg.tradability_policy == "legacy_finrl" and not tradable_vals.all():
            raise ValueError("Data has tradable=False rows; use tradability_policy='enforce_column'")
    elif cfg.tradability_policy == "enforce_column":
        raise ValueError("enforce_column requires a 'tradable' column in market_frame")

    n_assets = len(tickers)
    per_stock_budget = float(cfg.initial_amount) / float(n_assets)
    allocated = np.zeros(n_assets, dtype=bool)
    positions = np.zeros(n_assets, dtype=np.float64)
    cash = float(cfg.initial_amount)
    cumulative_cost = 0.0

    # Index daily rows cleanly by day integer index (load_processed_csv sets 0..T-1 index)
    by_date = {
        date: group.sort_values("tic").reset_index(drop=True)
        for date, group in market_frame.groupby("date", sort=False)
    }
    for date, day_df in by_date.items():
        if day_df["tic"].tolist() != tickers:
            raise ValueError(f"Incomplete or misordered ticker cross-section on {date}")

    accounts: list[dict[str, Any]] = []
    holdings: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []

    def record_snapshot(date: str) -> None:
        prices = by_date[date]["close"].to_numpy(dtype=np.float64)
        holdings_val = float(prices @ positions)
        total_val = cash + holdings_val
        accounts.append(
            {
                "date": date,
                "cash": float(cash),
                "holdings_value": holdings_val,
                "account_value": total_val,
                "cumulative_cost": float(cumulative_cost),
                "gross_exposure": holdings_val / total_val if total_val > 0.0 else 0.0,
            }
        )
        for idx, tic in enumerate(tickers):
            holdings.append(
                {
                    "date": date,
                    "tic": tic,
                    "simulated_units": float(positions[idx]),
                    "adjusted_mark_price": float(prices[idx]),
                    "position_value": float(positions[idx] * prices[idx]),
                }
            )

    # Pre-trade capital row at dates[0]
    record_snapshot(dates[0])

    deferred_first_day_tickers: list[str] = []
    for day_idx in range(len(dates) - 1):
        trade_date = dates[day_idx]
        val_date = dates[day_idx + 1]
        if day_idx > 0 and cfg.suspension_policy == "keep_cash":
            # No delayed entries under keep_cash
            record_snapshot(val_date)
            continue

        day_df = by_date[trade_date]
        prices = day_df["close"].to_numpy(dtype=np.float64)
        tradable_flags = (
            np.ones(n_assets, dtype=bool)
            if cfg.tradability_policy == "legacy_finrl"
            else day_df["tradable"].to_numpy(dtype=bool)
        )

        for idx, tic in enumerate(tickers):
            if allocated[idx]:
                continue
            if not tradable_flags[idx]:
                if day_idx == 0:
                    deferred_first_day_tickers.append(tic)
                if day_idx == 0:
                    trades.append(
                        {
                            "sequence": len(trades),
                            "date": trade_date,
                            "valuation_date": val_date,
                            "tic": tic,
                            "side": "buy",
                            "target_budget": per_stock_budget,
                            "filled_shares": 0.0,
                            "execution_price": float(prices[idx]),
                            "cash_before": float(cash),
                            "cash_after": float(cash),
                            "fee": 0.0,
                            "blocked_by_mask": True,
                        }
                    )
                continue

            unit_cost_with_fee = float(prices[idx]) * (1.0 + cfg.buy_cost_pct)
            if cfg.unit_mode == "integer_virtual_units":
                qty = float(math.floor((per_stock_budget + 1e-12) / unit_cost_with_fee))
            else:
                qty = per_stock_budget / unit_cost_with_fee

            allocated[idx] = True
            if qty <= 0.0:
                continue

            cash_before = cash
            gross_notional = qty * float(prices[idx])
            fee = gross_notional * cfg.buy_cost_pct
            total_outlay = gross_notional + fee
            if total_outlay > cash + 1e-9:
                raise RuntimeError(f"Insufficient cash during equal-weight buy for {tic} on {trade_date}")
            cash = max(0.0, cash - total_outlay)
            positions[idx] += qty
            cumulative_cost += fee
            trades.append(
                {
                    "sequence": len(trades),
                    "date": trade_date,
                    "valuation_date": val_date,
                    "tic": tic,
                    "side": "buy",
                    "target_budget": per_stock_budget,
                    "filled_shares": float(qty),
                    "execution_price": float(prices[idx]),
                    "cash_before": float(cash_before),
                    "cash_after": float(cash),
                    "fee": float(fee),
                    "blocked_by_mask": False,
                }
            )

        record_snapshot(val_date)
        # Cash & fee accounting check every session
        spent_gross = sum(t["filled_shares"] * t["execution_price"] for t in trades if not t["blocked_by_mask"])
        if not np.isclose(cash + spent_gross + cumulative_cost, cfg.initial_amount, rtol=1e-11, atol=1e-7):
            raise RuntimeError("Equal-weight cash/fee conservation check failed")

    account_frame = pd.DataFrame(accounts)
    account_frame["nav"] = account_frame["account_value"] / account_frame["account_value"].iloc[0]
    account_frame["daily_return"] = account_frame["account_value"].pct_change(fill_method=None)
    account_frame["drawdown"] = account_frame["nav"] / account_frame["nav"].cummax() - 1.0

    trade_cols = [
        "sequence",
        "date",
        "valuation_date",
        "tic",
        "side",
        "target_budget",
        "filled_shares",
        "execution_price",
        "cash_before",
        "cash_after",
        "fee",
        "blocked_by_mask",
    ]
    trades_frame = pd.DataFrame(trades, columns=trade_cols)
    filled_trades = trades_frame.loc[~trades_frame["blocked_by_mask"]] if len(trades_frame) else trades_frame

    perf = curve_metrics(account_frame["account_value"], cfg.annualization, cfg.risk_free_rate)
    perf.update(
        {
            "cumulative_cost": float(cumulative_cost),
            "trade_count": int(len(filled_trades)),
            "mean_gross_exposure": float(account_frame["gross_exposure"].mean()),
            "gross_turnover": (
                float((filled_trades["filled_shares"].abs() * filled_trades["execution_price"]).sum() / cfg.initial_amount)
                if len(filled_trades)
                else 0.0
            ),
            "Residual_uninvested_cash_final": float(cash),
            "residual_cash_weight_final": float(cash / account_frame["account_value"].iloc[-1]),
            "allocated_stock_count": int(allocated.sum()),
            "never_allocated_tickers": [tic for idx, tic in enumerate(tickers) if not allocated[idx]],
            "first_day_suspended_tickers": deferred_first_day_tickers,
        }
    )
    # Normalize key casing cleanly
    perf["residual_uninvested_cash_final"] = perf.pop("Residual_uninvested_cash_final")

    return {
        "ordered_tickers": tickers,
        "dates": dates,
        "account_value": account_frame,
        "holdings": pd.DataFrame(holdings),
        "trades": trades_frame,
        "metrics": perf,
    }


def run_equal_weight_baseline(
    data_path: str | Path,
    output_root: str | Path,
    *,
    config: EqualWeightConfig | None = None,
    split: str = "validation",
    benchmark_path: str | Path | None = None,
    expected_tickers: list[str] | tuple[str, ...] | None = None,
    output_dir_name: str | None = None,
) -> dict[str, Any]:
    """Load market CSV, run equal-weight buy-and-hold, and write P2-compatible artifacts."""
    if split not in ("train", "validation", "test"):
        raise ValueError("split must be one of ('train', 'validation', 'test')")
    cfg = config or EqualWeightConfig()
    resolved_data = Path(data_path).resolve()
    frame = load_market(resolved_data, list(cfg.tech_indicator_list), cfg.risk_indicator_col)
    sim = simulate_equal_weight(frame, cfg, expected_tickers=expected_tickers)

    code_sha = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
    dir_name = output_dir_name or (
        f"equal_weight_{split}_{cfg.unit_mode}_{cfg.suspension_policy}_{uuid.uuid4().hex[:6]}"
    )
    output_dir = Path(output_root).resolve() / dir_name
    output_dir.mkdir(parents=True, exist_ok=False)

    input_hashes = {"data": sha256(resolved_data)}
    metrics_payload: dict[str, Any] = {"equal_weight": sim["metrics"]}
    curves = sim["account_value"][["date", "nav"]].rename(columns={"nav": "equal_weight"})

    if benchmark_path is not None:
        from backtest.data import align_reference

        resolved_bench = Path(benchmark_path).resolve()
        input_hashes["market_index"] = sha256(resolved_bench)
        bench = align_reference(resolved_bench, sim["dates"], cfg.initial_amount, "market_index")
        metrics_payload["market_index"] = curve_metrics(
            bench["account_value"], cfg.annualization, cfg.risk_free_rate
        )
        curves["market_index"] = bench["account_value"].to_numpy() / cfg.initial_amount
        regimes = regime_metrics(sim["account_value"], bench, cfg.annualization, cfg.risk_free_rate)
    else:
        regimes = pd.DataFrame(
            columns=["regime", "sessions", "return_intervals", "conditional_compounded_return"]
        )

    sim["account_value"].to_csv(output_dir / "account_value.csv", index=False)
    sim["holdings"].to_csv(output_dir / "holdings.csv", index=False)
    sim["trades"].to_csv(output_dir / "trades.csv", index=False)
    curves.to_csv(output_dir / "normalized_curves.csv", index=False)
    regimes.to_csv(output_dir / "regime_metrics.csv", index=False)
    write_json(output_dir / "metrics.json", metrics_payload)

    cfg_dict = asdict(cfg)
    cfg_dict["tech_indicator_list"] = list(cfg.tech_indicator_list)
    manifest = {
        "status": "completed",
        "strategy": "equal_weight_buy_and_hold",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "data_path": str(resolved_data),
        "output": str(output_dir),
        "account_value_csv": str(output_dir / "account_value.csv"),
        "ordered_tickers": sim["ordered_tickers"],
        "dates": {
            "start": sim["dates"][0],
            "end": sim["dates"][-1],
            "observations": len(sim["dates"]),
        },
        "config": cfg_dict,
        "input_sha256": input_hashes,
        "baseline_code_sha256": {"baselines/equal_weight.py": code_sha},
        "output_sha256": {
            p.name: sha256(p)
            for p in sorted(output_dir.iterdir())
            if p.is_file() and p.name != "run_manifest.json"
        },
        "runtime": {"python": platform.python_version()},
    }
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest
