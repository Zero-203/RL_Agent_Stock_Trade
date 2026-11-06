"""Download and prepare the confirmed BaoStock-only 48-stock dataset."""
from __future__ import annotations

import os
for variable in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[variable] = "1"

import argparse
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"src"))

import numpy as np
import pandas as pd

from backtest.data import align_reference, check_market, load_market, sha256, write_json
from data_pipeline.baostock_cache import ensure_cache
from data_pipeline.canonical import canonical_stock
from data_pipeline.processing import (INDICATORS, add_turbulence, calculate_metrics,
    clean_index, coverage, index_regimes, technical_features, with_nav, with_split_baseline)

FILES = {"train": ("train_data.csv", "train_index.csv"),
    "validation": ("val_data.csv", "val_index.csv"), "test": ("trade_data.csv", "test_index.csv")}


def save(frame, path):
    result = frame.copy()
    result["date"] = result["date"].dt.strftime("%Y-%m-%d")
    result.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"config/sse48_data_pipeline.json")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--fetch", action="store_true", help="Download missing or changed cache files")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    contract_path = ROOT/config["universe_file"]
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    tickers = contract["ordered_tickers"]
    if len(tickers) != 48 or tickers != sorted(set(tickers)):
        raise ValueError("Expected the sorted confirmed 48-stock universe")
    name = args.dataset or config["dataset_name"]
    if name in ["", ".", ".."] or Path(name).name != name or any(char in name for char in ["/", "\\"]):
        raise ValueError("Use a simple dataset name")
    target = ROOT/"datasets"/name
    if target.exists():
        raise FileExistsError(f"Dataset exists: {target}. Choose a new --dataset name.")
    cache = (args.cache or ROOT/config["cache_directory"]).resolve()
    cache_manifest = ensure_cache(config, tickers, cache, args.fetch)
    input_paths = [args.config.resolve(), contract_path.resolve(), cache/"cache_manifest.json"]
    input_paths += [cache/relative for relative in cache_manifest["files"]]
    input_hashes = {str(path): sha256(path) for path in input_paths}
    calendar = pd.read_csv(cache/"trade_calendar.csv", dtype=str)
    dates = pd.DatetimeIndex(pd.to_datetime(calendar.loc[calendar["is_trading_day"] == "1", "calendar_date"]))
    dates = dates[(dates >= pd.Timestamp(config["stock_start"])) & (dates <= pd.Timestamp(config["end_inclusive"]))].sort_values()
    frames = []
    for ticker in tickers:
        code = "sh."+ticker[:6]
        reference = pd.read_csv(cache/"daily"/(code+".csv"), dtype=str)
        if set(reference["code"]) != {code}:
            raise ValueError(f"Stock identity mismatch: {ticker}")
        factors = pd.read_csv(cache/"factors"/(code+".csv"), dtype=str)
        frames.append(canonical_stock(ticker, reference, factors, dates))
    canonical = pd.concat(frames, ignore_index=True).sort_values(["date", "tic"]).reset_index(drop=True)
    features = add_turbulence(technical_features(canonical), config["feature_history_sessions"])
    benchmark, _ = clean_index(pd.read_csv(cache/"benchmark_sse50.csv", float_precision="round_trip"), config["benchmark_code"])
    regimes = index_regimes(benchmark, config["regime"]).rename(columns={"code": "benchmark_code"})
    features = features.merge(regimes, on="date", how="left", validate="many_to_one")
    numeric = ["open", "high", "low", "close", "volume"] + INDICATORS + ["turbulence", "benchmark_close", "benchmark_volatility_20d"]
    features["feature_available"] = np.isfinite(features[numeric].to_numpy(dtype=float)).all(axis=1)
    features["tradable"] &= features["feature_available"]
    index_features = calculate_metrics(benchmark, config["regime"])
    index_features = index_features.loc[index_features["date"].isin(dates)].copy()
    index_features = with_nav(index_features, "research_")
    index_features["source"] = "BaoStock"
    index_features["price_basis"] = "SSE50 price index; no stock-level readjustment"
    eligible = dates[config["feature_history_sessions"]+1:]
    model_features = features.loc[features["date"].isin(eligible)]
    staging = ROOT/"datasets"/("."+name+"_build_"+uuid.uuid4().hex[:6])
    staging.mkdir(parents=True)
    save(features, staging/"features_with_warmup.csv")
    summaries, date_sets = {}, []
    for split, (stock_file, index_file) in FILES.items():
        start, end = config["splits"][split]
        selected = model_features.loc[(model_features["date"] >= start) & (model_features["date"] < end)]
        if selected.empty or not selected["feature_available"].all():
            raise ValueError(f"Required features unavailable in {split}")
        if not selected.groupby("date")["tic"].nunique().eq(48).all():
            raise ValueError("Nonrectangular stock panel")
        save(selected, staging/stock_file)
        selected_dates = selected["date"].drop_duplicates()
        index_part = index_features.loc[index_features["date"].isin(selected_dates)]
        save(with_split_baseline(index_part), staging/index_file)
        report = check_market(staging/stock_file)
        if not report["input_compatible"]:
            raise ValueError(report["error"])
        loaded = load_market(staging/stock_file, INDICATORS, "turbulence")
        align_reference(staging/index_file, loaded["date"].drop_duplicates().tolist(), 1_000_000)
        stages = coverage(index_part, config["regime"]["minimum_contiguous_sessions"])
        if not stages["all_three_meet_minimum_run"]:
            raise ValueError(f"Stage coverage insufficient in {split}")
        summaries[split] = {"data_file": stock_file, "index_file": index_file,
            "rows": len(selected), "sessions": len(selected_dates), "stocks": 48,
            "start": selected_dates.min().strftime("%Y-%m-%d"), "end": selected_dates.max().strftime("%Y-%m-%d"),
            "stage_coverage": stages}
        date_sets.append(set(selected_dates))
    if any(date_sets[i] & date_sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Split dates overlap")
    generated_hashes = {path.name: sha256(path) for path in staging.glob("*.csv")}
    comparison = {filename: generated_hashes[filename] == expected for filename, expected in contract["csv_sha256"].items()}
    manifest = {"dataset_name": name, "source": "BaoStock", "stock_count": 48,
        "ordered_tickers": tickers, "price_basis": "adjusted_virtual_units",
        "raw_price_basis": "as_traded_unadjusted", "observation_dimension": 481, "action_dimension": 48,
        "splits": summaries, "outputs_sha256": generated_hashes, "input_sha256": input_hashes,
        "published_reference_csv_hash_matches": comparison,
        "test_strategy_predictions": 0, "main_training_started": False}
    write_json(staging/"dataset_manifest.json", manifest)
    for path, expected in input_hashes.items():
        if sha256(path) != expected:
            raise RuntimeError(f"Input changed during processing: {path}")
    datasets_root = (ROOT/"datasets").resolve()
    if not staging.resolve().is_relative_to(datasets_root) or not target.resolve().is_relative_to(datasets_root):
        raise ValueError("Publication path escaped datasets directory")
    os.rename(staging, target)
    print(json.dumps({"dataset": str(target), "splits": summaries,
        "published_reference_csv_hash_matches": comparison}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
