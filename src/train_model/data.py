from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

_BASE_COLUMNS = ("date", "tic", "open", "high", "low", "close", "volume", "day")


def load_processed_csv(
    path: str | Path,
    tech_indicator_list: Iterable[str],
    risk_indicator_col: str,
) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    serialized_index_columns = [
        column for column in frame.columns if str(column).startswith("Unnamed:")
    ]
    if serialized_index_columns:
        frame = frame.drop(columns=serialized_index_columns)

    required_columns = set(_BASE_COLUMNS)
    required_columns.update(tech_indicator_list)
    required_columns.add(risk_indicator_col)
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(
            f"{csv_path} is missing required columns: {', '.join(missing_columns)}"
        )
    if frame.empty:
        raise ValueError(f"CSV file is empty: {csv_path}")

    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.strftime("%Y-%m-%d")
    frame["tic"] = frame["tic"].astype(str)
    if frame.duplicated(["date", "tic"]).any():
        raise ValueError(f"{csv_path} contains duplicate date/ticker rows")

    numeric_columns = required_columns.difference(("date", "tic"))
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame[list(numeric_columns)].isna().any().any():
        raise ValueError(f"{csv_path} contains missing numeric values")

    frame = frame.sort_values(["date", "tic"]).reset_index(drop=True)
    ticker_sets = frame.groupby("date", sort=False)["tic"].agg(frozenset)
    expected_tickers = ticker_sets.iloc[0]
    if any(tickers != expected_tickers for tickers in ticker_sets.iloc[1:]):
        raise ValueError(f"{csv_path} does not contain the same tickers on every date")

    day_index = {date: index for index, date in enumerate(frame["date"].drop_duplicates())}
    frame.index = pd.Index(frame["date"].map(day_index), name="")
    return frame


def load_train_val_data(
    train_path: str | Path,
    val_path: str | Path,
    tech_indicator_list: Iterable[str],
    risk_indicator_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_frame = load_processed_csv(train_path, tech_indicator_list, risk_indicator_col)
    val_frame = load_processed_csv(val_path, tech_indicator_list, risk_indicator_col)

    train_dates = set(train_frame["date"].unique())
    val_dates = set(val_frame["date"].unique())
    if train_dates.intersection(val_dates):
        raise ValueError("Training and validation dates overlap")
    if max(train_dates) >= min(val_dates):
        raise ValueError("Validation dates must follow training dates")
    if set(train_frame["tic"].unique()) != set(val_frame["tic"].unique()):
        raise ValueError("Training and validation data must contain the same tickers")
    return train_frame, val_frame
