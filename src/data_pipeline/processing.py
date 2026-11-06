"""Pure BaoStock price processing reused from the validated P2 pipeline."""


import numpy as np
import pandas as pd
from stockstats import StockDataFrame


OHLC = ['open', 'high', 'low', 'close']


INDICATORS = ['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma']


FIELDS = ['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount']

NUMERIC = FIELDS[2:]


def invalid_ohlc(df):
    values = df[OHLC]
    eps = 1e-5
    return (values.isna().any(axis=1) | (values <= 0).any(axis=1)
            | (df["high"] + eps < values.max(axis=1))
            | (df["low"] - eps > values.min(axis=1)))


def unique_dates(df, label):
    """Remove exact duplicates, but never pick a conflicting row arbitrarily."""
    df = df.drop_duplicates().copy()
    if df["date"].duplicated().any():
        raise ValueError(f"{label}: conflicting duplicate dates")
    return df.sort_values("date").reset_index(drop=True)


def factor_series(factors, dates):
    """Backward factor uses only events effective on/before each date."""
    left = pd.DataFrame({"date": dates}).sort_values("date")
    if factors.empty:
        return pd.Series(1.0, index=pd.DatetimeIndex(left["date"]))
    right = factors[["dividOperateDate", "backAdjustFactor"]].copy()
    right["date"] = pd.to_datetime(right["dividOperateDate"], format="%Y-%m-%d")
    right["backAdjustFactor"] = pd.to_numeric(right["backAdjustFactor"], errors="raise")
    right = unique_dates(right[["date", "backAdjustFactor"]], "factor")
    merged = pd.merge_asof(left, right, on="date", direction="backward")
    values = merged.set_index("date")["backAdjustFactor"].fillna(1.0)
    if (~np.isfinite(values) | (values <= 0)).any():
        raise ValueError("Invalid backward factor")
    return values


def technical_features(canonical):
    frames = []
    for ticker, group in canonical.groupby("tic", sort=True):
        group = group.sort_values("date").copy().reset_index(drop=True)
        valid = group[OHLC].notna().all(axis=1)
        for col in INDICATORS:
            group[col] = np.nan
        # Restart only on a genuinely missing/quarantined price. Confirmed
        # suspension rows with a real vendor valuation remain in the history.
        segments = (~valid).cumsum()
        for _, idx in group.loc[valid].groupby(segments[valid]).groups.items():
            stock = StockDataFrame.retype(group.loc[idx, OHLC + ["volume"]].copy().reset_index(drop=True))
            for col in INDICATORS:
                values = np.asarray(stock[col], dtype=float)
                group.loc[idx, col] = values
        # Indicators may carry their own prior valid value only during a
        # confirmed suspension. Never backfill, cross a ticker, or fill a
        # retired row. The flag makes this causal imputation auditable.
        group["indicator_carried_during_suspension"] = False
        for col in INDICATORS:
            missing = group[col].isna() & group["is_suspended"]
            prior = group[col].ffill()
            filled = missing & prior.notna()
            group.loc[filled, col] = prior[filled]
            group.loc[filled, "indicator_carried_during_suspension"] = True
        group["day"] = group["date"].dt.dayofweek
        frames.append(group)
    return pd.concat(frames, ignore_index=True).sort_values(["date", "tic"]).reset_index(drop=True)


def add_turbulence(features, lookback):
    prices = features.pivot(index="date", columns="tic", values="close").sort_index()
    returns = prices.pct_change(fill_method=None).to_numpy()
    rows = []
    for i, date in enumerate(prices.index):
        value = np.nan
        count = 0
        if i >= lookback + 1:
            history = returns[i - lookback:i]
            current = returns[i]
            active = np.isfinite(current) & np.isfinite(history).all(axis=0)
            count = int(active.sum())
            if count >= 2:
                hist = history[:, active]
                residual = current[active] - hist.mean(axis=0)
                cov = np.cov(hist, rowvar=False)
                value = max(0.0, float(residual @ np.linalg.pinv(cov, rcond=1e-10) @ residual))
        rows.append({"date": date, "turbulence": value, "risk_asset_count": count})
    return features.merge(pd.DataFrame(rows), on="date", how="left", validate="many_to_one")


def index_regimes(benchmark, cfg):
    frame = benchmark.copy().sort_values("date")
    frame["date"] = pd.to_datetime(frame["date"])
    frame["close"] = pd.to_numeric(frame["close"], errors="raise")
    frame["return_60d"] = frame["close"].pct_change(cfg["lookback_sessions"], fill_method=None)
    frame["regime"] = "sideways"
    frame.loc[frame["return_60d"] > cfg["up_threshold"], "regime"] = "up"
    frame.loc[frame["return_60d"] < cfg["down_threshold"], "regime"] = "down"
    frame.loc[frame["return_60d"].isna(), "regime"] = "warmup"
    frame["benchmark_volatility_20d"] = frame["close"].pct_change(fill_method=None).rolling(20, min_periods=20).std() * np.sqrt(252)
    return frame[["date", "code", "close", "return_60d", "regime", "benchmark_volatility_20d"]].rename(columns={"close": "benchmark_close"})


def coverage(frame, minimum_run):
    groups = frame["regime"].ne(frame["regime"].shift()).cumsum()
    lengths = frame.groupby(["regime", groups]).size()
    result = {}
    for regime in ["up", "sideways", "down"]:
        longest = int(lengths.loc[regime].max()) if regime in lengths.index.get_level_values(0) else 0
        days = int(frame["regime"].eq(regime).sum())
        result[regime] = {"sessions": days, "fraction": days / len(frame) if len(frame) else 0,
                          "longest_contiguous_sessions": longest, "meets_minimum_run": longest >= minimum_run}
    result["all_three_meet_minimum_run"] = all(result[r]["meets_minimum_run"] for r in ["up", "sideways", "down"])
    return result


def clean_index(frame, symbol):
    missing = set(FIELDS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    data = frame[FIELDS].copy()
    data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d", errors="raise")
    data["code"] = data["code"].astype(str).str.strip()
    if set(data["code"]) != {symbol}:
        raise ValueError("Unexpected index code")
    for col in NUMERIC:
        data[col] = pd.to_numeric(data[col], errors="raise")
    before = len(data)
    data = data.drop_duplicates()
    if data["date"].duplicated().any():
        raise ValueError("Conflicting duplicate dates; manual source review required")
    data = data.sort_values("date").reset_index(drop=True)
    values = data[NUMERIC].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Missing/nonfinite index values; no automatic filling")
    if not data[["open", "high", "low", "close"]].gt(0).all().all():
        raise ValueError("Nonpositive index prices")
    if not data[["volume", "amount"]].ge(0).all().all():
        raise ValueError("Negative volume or amount")
    if not data["high"].ge(data[["open", "low", "close"]].max(axis=1)).all():
        raise ValueError("Index high below open/low/close")
    if not data["low"].le(data[["open", "high", "close"]].min(axis=1)).all():
        raise ValueError("Index low above open/high/close")
    return data, {"input_rows": before, "output_rows": len(data),
                  "exact_duplicate_rows_removed": before-len(data),
                  "conflicting_duplicates": 0, "missing_cells": 0,
                  "ohlc_range_violations": 0, "filled_price_rows": 0,
                  "zero_volume_rows": int(data["volume"].eq(0).sum())}


def calculate_metrics(data, regime, annualization=252):
    result = data.sort_values("date").copy().reset_index(drop=True)
    close = result["close"]
    result["daily_return"] = close.pct_change(fill_method=None)
    for window in [5, 20, 60]:
        result[f"return_{window}d"] = close.pct_change(window, fill_method=None)
    for window in [20, 60]:
        result[f"close_{window}_sma"] = close.rolling(window, min_periods=window).mean()
        result[f"volatility_{window}d"] = result["daily_return"].rolling(window, min_periods=window).std(ddof=1) * np.sqrt(annualization)
    lookback = regime["lookback_sessions"]
    trailing = close.pct_change(lookback, fill_method=None)
    result["regime_return"] = trailing
    result["regime"] = np.select(
        [trailing.isna(), trailing > regime["up_threshold"], trailing < regime["down_threshold"]],
        ["warmup", "up", "down"], default="sideways")
    return result


def with_nav(data, prefix):
    result = data.copy()
    if result.empty:
        raise ValueError("Cannot rebase an empty period")
    result[prefix + "nav"] = result["close"] / result["close"].iloc[0]
    result[prefix + "drawdown"] = result["close"] / result["close"].cummax() - 1
    return result


def with_split_baseline(data):
    result = with_nav(data, "split_")
    # The first close establishes the initial capital; it is not a return
    # earned from a position held before this split began.
    result["split_daily_return"] = result["close"].pct_change(fill_method=None)
    return result
