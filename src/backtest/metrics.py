from __future__ import annotations

import numpy as np
import pandas as pd


def curve_metrics(values, annualization=252, risk_free_rate=0.0):
    """First value is initial pre-trade capital; MDD is reported <= 0."""
    series = pd.Series(values, dtype=float).reset_index(drop=True)
    if len(series) < 2 or not np.isfinite(series).all() or (series <= 0).any():
        raise ValueError("An equity curve needs at least two positive finite values")
    if annualization <= 0 or risk_free_rate <= -1:
        raise ValueError("Invalid annualization or risk-free rate")
    returns = series.pct_change(fill_method=None).iloc[1:]
    std = float(returns.std(ddof=1)) if len(returns) > 1 else None
    daily_rf = (1 + risk_free_rate) ** (1 / annualization) - 1
    sharpe = None if std is None or std <= 1e-12 else float((returns.mean()-daily_rf)/std*np.sqrt(annualization))
    return {
        "observations": len(series), "return_intervals": len(returns),
        "initial_value": float(series.iloc[0]), "final_value": float(series.iloc[-1]),
        "total_return": float(series.iloc[-1]/series.iloc[0]-1),
        "annual_return": float(np.expm1(np.log(series.iloc[-1]/series.iloc[0])*annualization/len(returns))),
        "annual_volatility": None if std is None else float(std*np.sqrt(annualization)),
        "max_drawdown": float((series/series.cummax()-1).min()), "sharpe": sharpe,
        "risk_free_rate": risk_free_rate, "annualization": annualization,
    }


def regime_metrics(curve, benchmark, annualization=252, risk_free_rate=0.0):
    """Attribute existing daily returns; never pct_change a filtered regime."""
    if "regime" not in benchmark:
        return pd.DataFrame(columns=["regime", "sessions", "return_intervals", "conditional_compounded_return"])
    joined = curve[["date", "account_value"]].merge(benchmark[["date", "regime"]], on="date", validate="one_to_one")
    joined["daily_return"] = joined["account_value"].pct_change(fill_method=None)
    rows = []
    for label, group in joined.groupby("regime", sort=True):
        returns = group["daily_return"].dropna()
        std = float(returns.std(ddof=1)) if len(returns) > 1 else None
        rf = (1+risk_free_rate)**(1/annualization)-1
        rows.append({"regime": str(label), "sessions": len(group), "return_intervals": len(returns),
            "conditional_compounded_return": float((1+returns).prod()-1) if len(returns) else None,
            "mean_daily_return": float(returns.mean()) if len(returns) else None,
            "annual_volatility": None if std is None else std*np.sqrt(annualization),
            "sharpe": None if std is None or std <= 1e-12 else float((returns.mean()-rf)/std*np.sqrt(annualization))})
    return pd.DataFrame(rows)
