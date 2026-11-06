"""Map BaoStock unadjusted daily rows to the shared stock feature schema."""
import numpy as np
import pandas as pd

from data_pipeline.processing import OHLC, factor_series, invalid_ohlc, unique_dates


def canonical_stock(ticker, reference, factors, dates):
    reference = unique_dates(reference, ticker+" reference")
    reference["date"] = pd.to_datetime(reference["date"], format="%Y-%m-%d")
    numeric_columns = OHLC + ["volume", "preclose", "tradestatus", "adjustflag", "isST", "amount", "turn", "pctChg"]
    for column in numeric_columns:
        reference[column] = pd.to_numeric(reference[column], errors="coerce")
    if set(reference["adjustflag"].dropna()) != {3}:
        raise ValueError("BaoStock daily input must have adjustflag=3")
    if not reference["date"].isin(dates).all():
        raise ValueError("Daily input has dates outside the configured calendar")
    ref = reference.set_index("date").reindex(dates)
    present = ref["code"].notna()
    invalid = invalid_ohlc(ref)
    picked = ref[OHLC].copy()
    picked.loc[~present | invalid, OHLC] = np.nan
    result = pd.DataFrame(index=dates)
    result.index.name = "date"
    result["tic"] = ticker
    result["price_source"] = "baostock_daily"
    result["price_basis"] = "adjusted_virtual_units"
    result["volume_source"] = "baostock_unadjusted"
    for column in OHLC:
        result["raw_"+column] = picked[column]
    result["volume"] = ref["volume"].where(present)
    suspended = present & ref["tradestatus"].eq(0)
    result.loc[suspended & result["volume"].isna(), "volume"] = 0.0
    result["tradestatus"] = ref["tradestatus"].where(present)
    result["is_st"] = ref["isST"].where(present)
    result["is_suspended"] = suspended
    result["is_delisted"] = False
    result["is_reference_missing"] = ~present
    result["is_quarantined"] = present & invalid
    result["tradable"] = present & ~invalid & ref["tradestatus"].eq(1) & result["volume"].gt(0)
    result["corporate_action_factor"] = factor_series(factors, dates).reindex(dates)
    result["corporate_action_factor"] /= result["corporate_action_factor"].iloc[0]
    for column in OHLC:
        result[column] = result["raw_"+column] * result["corporate_action_factor"]
    result["factor_changed"] = ~np.isclose(result["corporate_action_factor"].diff().fillna(0), 0)
    result["source"] = "BaoStock"
    result["baostock_code"] = "sh."+ticker[:6]
    result["factor_source"] = "baostock_backAdjustFactor"
    result["raw_price_basis"] = "as_traded_unadjusted"
    if result[["is_reference_missing", "is_quarantined"]].any().any():
        raise ValueError(f"Unresolved BaoStock prices for {ticker}")
    return result.reset_index()
