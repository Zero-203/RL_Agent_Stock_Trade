import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "sse50_data", Path(__file__).resolve().parents[1] / "examples" / "FinRL_StockTrading_SSE50_2016_data.py"
)
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def test_frozen_pool_matches_document():
    config = script.load_config()
    document = (script.ROOT / "docs" / "sse50_constituents_2016.md").read_text(encoding="utf-8")
    assert len(config["tickers"]) == len(set(config["tickers"])) == 50
    assert all(f"| {ticker} |" in document for ticker in config["tickers"])
    yahoo = script.yahoo_tickers(config)
    assert len(yahoo) == len(set(yahoo)) == 48
    assert set(config["tickers"]) - set(yahoo) == {"600837.SS", "601989.SS"}


def test_fetch_retries_and_keeps_raw_rows(monkeypatch):
    raw = pd.DataFrame(
        {"Open": [float("nan")], "Close": [3.0], "Adj Close": [2.0]},
        index=pd.DatetimeIndex(["2016-01-04"], name="Date"),
    )
    history = Mock(side_effect=[RuntimeError("rate limit"), raw])
    monkeypatch.setattr(script.yf, "Ticker", lambda ticker: Mock(history=history))
    sleep = Mock()
    monkeypatch.setattr(script.time, "sleep", sleep)
    config = {"start_date": "2016-01-01", "max_attempts": 2, "retry_delay_seconds": 30}

    result = script.fetch_ticker("600000.SS", config, "2026-01-01")

    sleep.assert_called_once_with(30)
    assert history.call_count == 2
    assert history.call_args.kwargs["keepna"] is True
    assert history.call_args.kwargs["auto_adjust"] is False
    assert pd.isna(result.loc[0, "Open"])
    assert result.loc[0, "Adj Close"] == 2.0
    assert result.loc[0, "tic"] == "600000.SS"


def test_failure_does_not_create_combined_csv(monkeypatch, tmp_path):
    monkeypatch.setattr(script, "ROOT", tmp_path)
    monkeypatch.setattr(script.time, "sleep", lambda seconds: None)
    fetch = Mock(side_effect=[pd.DataFrame({"tic": ["600000.SS"]}), RuntimeError("failed")])
    monkeypatch.setattr(script, "fetch_ticker", fetch)
    config = script.load_config()
    config["output_directory"] = "datasets"

    with pytest.raises(RuntimeError, match="failed"):
        script.download(script.yahoo_tickers(config), config, "2026-01-01", save_combined=True)

    assert (tmp_path / "datasets" / "600000.SS.csv").exists()
    assert not (tmp_path / "datasets" / "sse50_2016_yahoo48_raw.csv").exists()


def test_complete_yahoo_pool_is_combined_without_baostock_tickers(monkeypatch, tmp_path):
    monkeypatch.setattr(script, "ROOT", tmp_path)
    monkeypatch.setattr(script.time, "sleep", lambda seconds: None)
    fetch = Mock(side_effect=lambda ticker, config, end: pd.DataFrame({"tic": [ticker], "Close": [1.0]}))
    monkeypatch.setattr(script, "fetch_ticker", fetch)
    config = script.load_config()
    config["output_directory"] = "datasets"
    tickers = script.yahoo_tickers(config)

    script.download(tickers, config, "2026-01-01", save_combined=True)

    combined = pd.read_csv(tmp_path / "datasets" / "sse50_2016_yahoo48_raw.csv")
    assert combined["tic"].tolist() == tickers
    assert fetch.call_count == 48
    assert not (tmp_path / "datasets" / "600837.SS.csv").exists()
    assert not (tmp_path / "datasets" / "601989.SS.csv").exists()


def test_partial_pool_cannot_be_saved_as_combined(monkeypatch, tmp_path):
    monkeypatch.setattr(script, "ROOT", tmp_path)
    config = script.load_config()
    with pytest.raises(ValueError, match="complete Yahoo pool"):
        script.download(script.yahoo_tickers(config)[:-1], config, "2026-01-01", save_combined=True)
    assert list(tmp_path.iterdir()) == []


def test_rejects_mismatched_supplement_codes(tmp_path):
    config = script.load_config()
    config["baostock_supplement"]["codes"] = ["sh.600837", "sh.999999"]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="must belong"):
        script.load_config(path)


def test_rejects_duplicate_pool(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"tickers": ["600000.SS"] * 50}), encoding="utf-8")
    with pytest.raises(ValueError, match="50 distinct"):
        script.load_config(path)