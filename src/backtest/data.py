from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

DEFAULT_INDICATORS = ["macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma"]


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def bool_column(series, name):
    if series.isna().any():
        raise ValueError(f"Missing {name} flags")
    mapped = series.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    if mapped.isna().any():
        raise ValueError(f"Invalid {name} flags")
    return mapped.astype(bool)


def load_market(path, indicators, risk_column):
    from train_model.data import load_processed_csv
    frame = load_processed_csv(path, indicators, risk_column)
    if frame["date"].nunique() < 2:
        raise ValueError("At least two trading dates are required")
    for name in ["tradable", "is_suspended", "is_delisted", "feature_available"]:
        if name in frame:
            frame[name] = bool_column(frame[name], name)
    if "tradable" in frame:
        for name in ["is_suspended", "is_delisted"]:
            if name in frame and (frame[name] & frame["tradable"]).any():
                raise ValueError(f"Contradictory {name}/tradable flags")
        if "feature_available" in frame and (~frame["feature_available"] & frame["tradable"]).any():
            raise ValueError("Unavailable features marked tradable")
    if {"raw_close", "corporate_action_factor"}.issubset(frame.columns):
        expected = frame["raw_close"] * frame["corporate_action_factor"]
        if not np.allclose(frame["close"], expected, rtol=1e-10, atol=1e-10):
            raise ValueError("close is not the declared factor-adjusted price")
    return frame


def read_metadata(model_path, metadata_path=None):
    if metadata_path:
        payload = Path(metadata_path).read_text(encoding="utf-8")
    else:
        with zipfile.ZipFile(model_path) as archive:
            if "metadata.txt" not in archive.namelist():
                raise ValueError("Model has no metadata.txt; provide --metadata from its training run")
            item = archive.getinfo("metadata.txt")
            if item.file_size > 2_000_000:
                raise ValueError("Model metadata unexpectedly large")
            payload = archive.read(item).decode("utf-8")
    start = payload.find("{")
    if start < 0:
        raise ValueError("Metadata contains no JSON object")
    result = json.loads(payload[start:])
    if result.get("status", "completed") != "completed":
        raise ValueError("Training run is not completed")
    effective = result.get("effective_config", result)
    for name in ["algorithm", "environment", "data_hashes", "data_paths"]:
        if name not in effective:
            raise ValueError(f"Missing metadata field {name}")
    if effective["algorithm"].lower() not in ["ppo", "ddpg", "td3", "sac", "recurrentppo", "tqc"]:
        raise ValueError("Unsupported project algorithm")
    return result, effective


def resolve_tickers(effective, train_data=None):
    contract = effective.get("data_contract", {})
    if contract.get("ordered_tickers"):
        tickers = contract["ordered_tickers"]
        if len(tickers) != len(set(tickers)) or tickers != sorted(tickers):
            raise ValueError("Invalid ordered tickers in metadata")
        return tickers
    path = Path(train_data or effective["data_paths"]["train_csv"])
    if not path.is_file():
        raise ValueError("Legacy metadata lacks ordered tickers; provide the original --train-data")
    if sha256(path) != effective["data_hashes"]["train_csv"]:
        raise ValueError("Original training CSV hash does not match model metadata")
    return sorted(pd.read_csv(path, usecols=["tic"])["tic"].astype(str).unique())


def check_market(path, indicators=None, risk_column="turbulence"):
    raw = pd.read_csv(path, float_precision="round_trip")
    numeric = ["open", "high", "low", "close", "volume", "day"] + list(indicators or DEFAULT_INDICATORS) + [risk_column]
    existing = [name for name in numeric if name in raw]
    converted = raw[existing].apply(pd.to_numeric, errors="coerce")
    bad_rows = ~np.isfinite(converted.to_numpy(dtype=float)).all(axis=1)
    report = {"file": str(Path(path).resolve()), "rows": len(raw),
        "stocks": int(raw["tic"].nunique()) if "tic" in raw else None,
        "nonfinite_numeric_rows": int(bad_rows.sum()),
        "nonfinite_rows_by_ticker": raw.loc[bad_rows].groupby("tic").size().to_dict() if "tic" in raw else {},
        "missing_columns": sorted(set(numeric + ["date", "tic"]) - set(raw)),
        "legacy_environment_caveat": "Masks need the shared environment; retired NaNs require an explicit universe/exit policy."}
    try:
        load_market(path, indicators or DEFAULT_INDICATORS, risk_column)
        report.update({"input_compatible": True, "error": None})
    except (ValueError, KeyError) as error:
        report.update({"input_compatible": False, "error": str(error)})
    return report


def align_reference(path, dates, initial_amount, name="index"):
    data = pd.read_csv(path, float_precision="round_trip")
    data["date"] = pd.to_datetime(data["date"], errors="raise").dt.strftime("%Y-%m-%d")
    if data["date"].duplicated().any():
        raise ValueError(f"{name}: duplicate dates")
    data = data.sort_values("date")
    data = data.loc[data["date"].between(dates[0], dates[-1])].copy()
    if data["date"].tolist() != list(dates):
        raise ValueError(f"{name}: benchmark dates must exactly match; no silent inner join")
    if "account_value" in data:
        values = pd.to_numeric(data["account_value"], errors="raise")
        if not np.isclose(values.iloc[0], initial_amount):
            raise ValueError(f"{name}: include the initial pre-trade capital row")
    elif "nav" in data:
        nav = pd.to_numeric(data["nav"], errors="raise")
        if not np.isclose(nav.iloc[0], 1):
            raise ValueError(f"{name}: NAV must start at 1")
        values = nav * initial_amount
    elif "close" in data:
        prices = pd.to_numeric(data["close"], errors="raise")
        values = prices/prices.iloc[0]*initial_amount
    else:
        raise ValueError(f"{name}: expected account_value, nav or close")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"{name}: nonpositive/nonfinite curve")
    data["account_value"] = values
    return data.reset_index(drop=True)
