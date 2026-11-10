from __future__ import annotations

import copy
import json
import platform
from importlib import metadata
from pathlib import Path
from datetime import datetime, timezone
import uuid

import numpy as np
import pandas as pd

from backtest.data import align_reference, load_market, read_metadata, resolve_tickers, sha256, write_json
from backtest.metrics import curve_metrics, regime_metrics
from trading_environment import build_environment, environment_hashes


def _split_bounds(effective, contract, train_data=None, validation_data=None):
    if contract.get("train_end") and contract.get("validation_end"):
        return contract["train_end"], contract["validation_end"]
    dates = {}
    for key in ["train_csv", "val_csv"]:
        relocated = validation_data if key == "val_csv" else train_data
        path = Path(relocated or effective["data_paths"][key])
        if not path.is_file() or sha256(path) != effective["data_hashes"][key]:
            raise ValueError(f"Legacy model: original {key} needed to verify temporal boundaries")
        series = pd.to_datetime(pd.read_csv(path, usecols=["date"])["date"]).dt.strftime("%Y-%m-%d")
        dates[key] = max(series)
    return dates["train_csv"], dates["val_csv"]


def prepare(model_path, data_path, split, train_data=None, validation_data=None, metadata_path=None):
    if split not in ["validation", "test"]:
        raise ValueError("Only validation and test splits are supported")
    outer, effective = read_metadata(model_path, metadata_path)
    cfg = copy.deepcopy(effective["environment"])
    contract = effective.get("data_contract", {})
    if contract.get("execution_timing", "finrl_same_close") != "finrl_same_close":
        raise ValueError("Execution timing disagrees with this shared environment")
    if contract.get("price_basis", cfg.get("price_basis", "adjusted_virtual_units")) != "adjusted_virtual_units":
        raise ValueError("Unsupported price basis")
    indicators = list(cfg["tech_indicator_list"])
    if contract.get("ordered_indicators", indicators) != indicators:
        raise ValueError("Indicator order disagrees with training metadata")
    frame = load_market(data_path, indicators, cfg["risk_indicator_col"])
    tickers = resolve_tickers(effective, train_data)
    if sorted(frame["tic"].unique()) != tickers:
        raise ValueError("Backtest ticker identities/order do not match the trained model")
    if contract.get("environment_code_sha256") and contract["environment_code_sha256"] != environment_hashes():
        raise ValueError("Shared or FinRL environment source differs from training; replay with the recorded version")
    train_end, val_end = _split_bounds(effective, contract, train_data, validation_data)
    data_hash = sha256(data_path)
    if split == "validation":
        if data_hash != effective["data_hashes"]["val_csv"] or min(frame["date"]) <= train_end:
            raise ValueError("Validation must use the exact recorded validation CSV after training")
    elif min(frame["date"]) <= val_end or data_hash in effective["data_hashes"].values():
        raise ValueError("Test dates must follow validation; training/validation CSVs cannot be relabelled test")
    env = build_environment(frame, cfg)
    expected_observation = 1 + (2+len(indicators))*len(tickers)
    if contract.get("observation_dimension", expected_observation) != expected_observation:
        raise ValueError("Observation dimension disagrees with metadata")
    if contract.get("action_dimension", len(tickers)) != len(tickers):
        raise ValueError("Action dimension disagrees with metadata")
    return frame, env, tickers, outer, effective


def apply_threshold(raw_action, threshold_mode, threshold_config, critic=None):
    """Map raw model actions to post-threshold actions; fail explicitly if unimplemented."""
    arr = np.asarray(raw_action, dtype=np.float32)
    if arr.ndim != 1 or not np.isfinite(arr).all() or (np.abs(arr) > 1 + 1e-6).any():
        raise ValueError("Invalid action vector passed to apply_threshold")
    if threshold_mode in (None, "none"):
        return arr.astype(float), {"threshold_mode": "none"}
    raise NotImplementedError(
        f"Threshold mode '{threshold_mode}' is not implemented in this build; "
        "simple threshold is provided by P3, critic threshold by P4."
    )


def replay(model, env, tickers, seed=0, threshold_mode=None, threshold_config=None):
    env.audit_orders = True
    obs, _ = env.reset(seed=seed)
    dates = env.df["date"].drop_duplicates().tolist()
    n = len(tickers)
    recurrent = type(model).__name__.lower() == "recurrentppo"
    states = None
    episode_starts = True
    accounts, holdings, actions, trades = [], [], [], []

    def snapshot(date):
        cash = float(env.state[0])
        prices = np.asarray(env.state[1:1+n], dtype=float)
        positions = np.asarray(env.state[1+n:1+2*n], dtype=float)
        value = float(prices @ positions)
        accounts.append({"date": date, "cash": cash, "holdings_value": value,
            "account_value": cash+value, "cumulative_cost": float(env.cost),
            "gross_exposure": value/(cash+value)})
        for i, ticker in enumerate(tickers):
            holdings.append({"date": date, "tic": ticker, "simulated_units": positions[i],
                "adjusted_mark_price": prices[i], "position_value": positions[i]*prices[i]})

    snapshot(dates[0])
    for day in range(len(dates)-1):
        if recurrent:
            raw_action, states = model.predict(
                np.asarray(obs, dtype=np.float32), state=states,
                episode_start=np.asarray([episode_starts]), deterministic=True,
            )
        else:
            raw_action, _ = model.predict(np.asarray(obs, dtype=np.float32), deterministic=True)
        raw_action = np.asarray(raw_action, dtype=float).reshape(-1)
        if raw_action.shape != (n,) or not np.isfinite(raw_action).all() or (np.abs(raw_action) > 1+1e-6).any():
            raise ValueError("Model returned an invalid action vector")
        action, threshold_info = apply_threshold(raw_action, threshold_mode, threshold_config)
        action = np.clip(action, -1, 1)
        requested = (action*env.hmax).astype(int)
        before_positions = np.asarray(env.state[1+n:1+2*n], dtype=float).copy()
        before_cash, before_cost = float(env.state[0]), float(env.cost)
        prices = np.asarray(env.state[1:1+n], dtype=float).copy()
        env.order_trace = []
        obs, reward, terminated, truncated, _ = env.step(action.copy())
        if recurrent:
            episode_starts = bool(terminated or truncated)
        positions = np.asarray(env.state[1+n:1+2*n], dtype=float)
        fills = positions-before_positions
        fees = float(env.cost)-before_cost
        if not np.isclose(float(env.state[0]), before_cash-float(fills @ prices)-fees, rtol=1e-10, atol=1e-7):
            raise RuntimeError("Cash reconciliation failed")
        if not np.isclose(sum(row["fee"] for row in env.order_trace), fees, rtol=1e-10, atol=1e-7):
            raise RuntimeError("Trade fee reconciliation failed")
        if (positions < -1e-8).any() or float(env.state[0]) < -1e-6:
            raise RuntimeError("Long-only account became negative")
        frame_day = env.df.loc[day]
        flags = frame_day["tradable"] if "tradable" in frame_day else None
        for i, ticker in enumerate(tickers):
            allowed = True if env.tradability_policy == "legacy_finrl" else bool(flags.iloc[i] if isinstance(flags, pd.Series) else flags)
            if not allowed and fills[i] != 0:
                raise RuntimeError("A nontradable asset was traded")
            if not allowed and requested[i] != 0:
                reason = "not_tradable"
            elif fills[i] < 0 and requested[i] >= 0:
                reason = "risk_liquidation"
            elif fills[i] != requested[i]:
                reason = "cash_position_or_environment_limit"
            elif requested[i] == 0:
                reason = "hold_or_integer_rounding"
            else:
                reason = "filled"
            actions.append({"date": dates[day], "valuation_date": dates[day+1], "tic": ticker,
                "model_action": raw_action[i], "thresholded_action": float(action[i]),
                "threshold_mode": threshold_info["threshold_mode"], "requested_shares": int(requested[i]),
                "executed_shares": fills[i], "adjusted_execution_price": prices[i], "reason": reason})
        for trade in env.order_trace:
            if trade["filled_shares"] != 0:
                trade = dict(trade)
                trade["date"] = dates[day]
                trade["tic"] = tickers[trade.pop("ticker_index")]
                trade["valuation_date"] = dates[day+1]
                trades.append(trade)
        snapshot(dates[day+1])
        if not np.isclose(accounts[-1]["account_value"]-accounts[-2]["account_value"], reward/env.reward_scaling, rtol=1e-7, atol=1e-6):
            raise RuntimeError("Reward/account reconciliation failed")
        if (terminated or truncated) and day != len(dates)-2:
            raise RuntimeError("Environment terminated before all requested dates")
    account_frame = pd.DataFrame(accounts)
    account_frame["nav"] = account_frame["account_value"]/account_frame["account_value"].iloc[0]
    account_frame["daily_return"] = account_frame["account_value"].pct_change(fill_method=None)
    account_frame["drawdown"] = account_frame["nav"]/account_frame["nav"].cummax()-1
    trade_columns = ["sequence", "side", "environment_requested_shares", "filled_shares", "execution_price", "cash_before", "cash_after", "position_before", "position_after", "fee", "blocked_by_mask", "date", "tic", "valuation_date"]
    return account_frame, pd.DataFrame(actions), pd.DataFrame(holdings), pd.DataFrame(trades, columns=trade_columns)


def _plots(curves, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    indexed = curves.set_index("date")
    ax = indexed.plot(figsize=(11, 4), title="Normalized portfolio value")
    ax.set_ylabel("NAV")
    ax.figure.tight_layout()
    ax.figure.savefig(output/"equity_curve.png", dpi=140)
    plt.close(ax.figure)
    ax = (indexed/indexed.cummax()-1).plot(figsize=(11, 4), title="Drawdown")
    ax.figure.tight_layout()
    ax.figure.savefig(output/"drawdown.png", dpi=140)
    plt.close(ax.figure)


def run_backtest(model_path, data_path, output_root, *, split="test", benchmark_path=None,
                 baselines=None, train_data=None, validation_data=None, metadata_path=None,
                 annualization=252, risk_free_rate=0.0, device="cpu"):
    from stable_baselines3 import PPO, DDPG, TD3, SAC
    from sb3_contrib import RecurrentPPO, TQC
    import torch
    torch.set_num_threads(1)
    model_path, data_path = Path(model_path).resolve(), Path(data_path).resolve()
    frame, env, tickers, outer, effective = prepare(model_path, data_path, split, train_data, validation_data, metadata_path)
    algorithm = effective["algorithm"].lower()
    classes = {"ppo": PPO, "ddpg": DDPG, "td3": TD3, "sac": SAC, "recurrentppo": RecurrentPPO, "tqc": TQC}
    # Off-policy SB3 load otherwise recreates the training replay buffer,
    # which can consume gigabytes even though evaluation never learns.
    inference_kwargs = {} if algorithm in ("ppo", "recurrentppo") else {"buffer_size": 1}
    model = classes[algorithm].load(str(model_path), device=device, **inference_kwargs)
    if model.observation_space.shape != env.observation_space.shape or model.action_space.shape != env.action_space.shape:
        raise ValueError("Saved model spaces disagree with the evaluation environment")
    dates = frame["date"].drop_duplicates().tolist()
    references = {}
    if benchmark_path:
        references["market_index"] = align_reference(benchmark_path, dates, env.initial_amount, "market_index")
    for name, path in (baselines or {}).items():
        if name in references or name in ["date", "strategy"]:
            raise ValueError(f"Reserved/duplicate baseline name: {name}")
        references[name] = align_reference(path, dates, env.initial_amount, name)
    output = Path(output_root).resolve() / f"{algorithm}_s{effective.get('seed',0)}_{split}_{uuid.uuid4().hex[:10]}"
    output.mkdir(parents=True, exist_ok=False)
    hashes = {"model": sha256(model_path), "data": sha256(data_path)}
    if benchmark_path:
        hashes["market_index"] = sha256(benchmark_path)
    hashes.update({"baseline_"+name: sha256(path) for name, path in (baselines or {}).items()})
    source_paths = {"model": model_path, "data": data_path}
    if benchmark_path:
        source_paths["market_index"] = Path(benchmark_path)
    source_paths.update({"baseline_"+name: Path(path) for name, path in (baselines or {}).items()})
    manifest = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": algorithm, "seed": int(effective.get("seed",0)), "split": split,
        "model_training_run_type": effective.get("run_type", "unknown"),
        "ordered_tickers": tickers, "environment": effective["environment"],
        "environment_code_sha256": environment_hashes(), "input_sha256": hashes,
        "framework_code_sha256": {path.name: sha256(path) for path in Path(__file__).parent.glob("*.py")},
        "model_path": str(model_path), "data_path": str(data_path), "output": str(output),
        "deterministic": True, "execution_timing": "FinRL same-close action at t, mark-to-market at t+1; simplified execution, not an executable next-open claim",
        "inference_replay_buffer_size": None if algorithm in ("ppo", "recurrentppo") else 1,
        "threshold_mode": effective.get("threshold_mode", "none"),
        "price_basis": "adjusted_virtual_units", "corporate_actions_paid_separately": False,
        "annualization": annualization, "risk_free_rate": risk_free_rate,
        "legacy_metadata_inferred": not bool(effective.get("data_contract")),
        "runtime": {"python": platform.python_version(), "packages": {name: metadata.version(name) for name in ["numpy", "pandas", "stable-baselines3", "sb3-contrib", "torch"]}},
        "limitations": ["Adjusted virtual units, no separate dividend/split/merger settlement.",
            "T+1, price limits, real lots, slippage and queue priority are not modeled.",
            "Same-close execution can be optimistic; do not describe as a causally executable trading system.",
            "Market index is a price-index reference without dividend reinvestment or transaction costs.",
            "Regime summary compounds attributed daily returns across noncontiguous days; it is not a separate continuous strategy."]}
    write_json(output/"run_manifest.json", manifest)
    try:
        accounts, actions, holdings, trades = replay(
            model, env, tickers, int(effective.get("seed",0)),
            threshold_mode=effective.get("threshold_mode"),
            threshold_config=effective.get("threshold_config"),
        )
        metrics = {"strategy": curve_metrics(accounts["account_value"], annualization, risk_free_rate)}
        metrics["strategy"].update({"cumulative_cost": float(accounts["cumulative_cost"].iloc[-1]),
            "trade_count": len(trades), "mean_gross_exposure": float(accounts["gross_exposure"].mean()),
            "gross_turnover": float((trades["filled_shares"].abs()*trades["execution_price"]).sum()/env.initial_amount) if len(trades) else 0.0})
        curves = accounts[["date", "nav"]].rename(columns={"nav": "strategy"})
        for name, ref in references.items():
            metrics[name] = curve_metrics(ref["account_value"], annualization, risk_free_rate)
            curves[name] = ref["account_value"].to_numpy()/env.initial_amount
        for filename, table in [("account_value.csv", accounts), ("actions.csv", actions), ("holdings.csv", holdings), ("trades.csv", trades), ("normalized_curves.csv", curves)]:
            table.to_csv(output/filename, index=False)
        stages = regime_metrics(accounts, references.get("market_index", pd.DataFrame()), annualization, risk_free_rate)
        stages.to_csv(output/"regime_metrics.csv", index=False)
        write_json(output/"metrics.json", metrics)
        write_json(output/"model_metadata.json", outer)
        _plots(curves, output)
        if any(sha256(path) != hashes[name] for name, path in source_paths.items()):
            raise RuntimeError("An input changed during evaluation")
        manifest.update({"status": "completed", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "dates": {"start": dates[0], "end": dates[-1], "observations": len(dates)},
            "checks": {"cash_fee_reward_reconciliation": True, "masks_enforced_when_configured": True,
                "all_dates_replayed": len(accounts) == len(dates), "model_and_data_unchanged": True},
            "output_sha256": {path.name: sha256(path) for path in output.iterdir() if path.is_file() and path.name != "run_manifest.json"}})
        write_json(output/"run_manifest.json", manifest)
        return manifest
    except Exception as error:
        manifest.update({"status": "failed", "error": {"type": type(error).__name__, "message": str(error)}})
        write_json(output/"run_manifest.json", manifest)
        raise
    finally:
        env.close()


def summarize_runs(manifests, output_path):
    rows = []
    for path in manifests:
        path = Path(path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            raise ValueError(f"Cannot summarize incomplete run: {path}")
        metrics = json.loads((path.parent/"metrics.json").read_text(encoding="utf-8"))["strategy"]
        rows.append({"algorithm": manifest["algorithm"], "seed": manifest["seed"], "split": manifest["split"],
            "data_sha256": manifest["input_sha256"]["data"],
            "environment": json.dumps(manifest["environment"], sort_keys=True),
            "run_type": manifest["model_training_run_type"],
            "annualization": manifest["annualization"], "risk_free_rate": manifest["risk_free_rate"], **metrics})
    frame = pd.DataFrame(rows)
    for column in ["split", "data_sha256", "environment", "run_type", "annualization", "risk_free_rate"]:
        if frame[column].nunique() != 1:
            raise ValueError(f"Incomparable runs: {column} differs")
    if frame.duplicated(["algorithm", "seed"]).any():
        raise ValueError("Duplicate algorithm/seed; choose one run per seed before aggregation")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    stats = frame.groupby("algorithm")[["total_return", "annual_return", "annual_volatility", "max_drawdown", "sharpe"]].agg(["count", "mean", "std"])
    stats.to_csv(output_path.with_name(output_path.stem+"_by_algorithm.csv"))
    return frame
