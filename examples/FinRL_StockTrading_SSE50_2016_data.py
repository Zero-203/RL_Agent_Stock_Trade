"""Download unprocessed Yahoo daily data for the frozen 2016 SSE 50 pool."""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "sse50_2016.json"


def load_config(path=CONFIG_PATH):
    with path.open(encoding="utf-8") as file:
        config = json.load(file)
    tickers = config["tickers"]
    if len(tickers) != 50 or len(set(tickers)) != 50:
        raise ValueError("The frozen pool must contain 50 distinct tickers")
    if config["request_interval_seconds"] < 0 or config["retry_delay_seconds"] <= 0:
        raise ValueError("Invalid request or retry delay")
    if config["max_attempts"] < 1:
        raise ValueError("max_attempts must be at least one")
    return config


def fetch_ticker(ticker, config, end_date):
    for attempt in range(1, config["max_attempts"] + 1):
        try:
            data = yf.Ticker(ticker).history(
                start=config["start_date"],
                end=end_date,
                interval="1d",
                auto_adjust=False,
                actions=True,
                repair=False,
                keepna=True,
                raise_errors=True,
            )
            if data.empty:
                raise ValueError("Yahoo returned no rows")
            data = data.reset_index()
            data.insert(0, "tic", ticker)
            return data
        except Exception as error:
            if attempt == config["max_attempts"]:
                raise RuntimeError(f"{ticker}: download failed after {attempt} attempts") from error
            delay = config["retry_delay_seconds"] * (2 ** (attempt - 1))
            print(f"{ticker}: {error}; retrying in {delay}s", file=sys.stderr, flush=True)
            time.sleep(delay)


def download(tickers, config, end_date):
    output_dir = ROOT / config["output_directory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for index, ticker in enumerate(tickers):
        if index:
            time.sleep(config["request_interval_seconds"])
        print(f"[{index + 1}/{len(tickers)}] {ticker}", flush=True)
        data = fetch_ticker(ticker, config, end_date)
        data.to_csv(output_dir / f"{ticker}.csv", index=False)
        frames.append(data)
        print(f"{ticker}: {len(data)} rows", flush=True)
    if len(tickers) == 50:
        combined = pd.concat(frames, ignore_index=True)
        path = output_dir / "sse50_2016_raw.csv"
        combined.to_csv(path, index=False)
        print(f"Saved {len(combined)} rows to {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--ticker", help="Download a single ticker from the frozen pool")
    selection.add_argument("--all", action="store_true", help="Download the entire frozen pool")
    args = parser.parse_args()
    config = load_config()
    if args.ticker and args.ticker not in config["tickers"]:
        parser.error("--ticker must belong to the frozen pool")
    proxy = os.getenv("PROXY_SERVER")
    if proxy:
        yf.set_config(proxy=proxy)
    end_date = (datetime.now(timezone(timedelta(hours=8))).date() + timedelta(days=1)).isoformat()
    try:
        download(config["tickers"] if args.all else [args.ticker], config, end_date)
    except Exception as error:
        print(f"Download stopped: {error}", file=sys.stderr)
        if error.__cause__:
            print(f"Cause: {error.__cause__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())