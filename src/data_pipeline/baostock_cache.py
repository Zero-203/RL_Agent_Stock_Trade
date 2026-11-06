"""Download restartable, hash-verified BaoStock inputs using relative cache keys."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import time

import pandas as pd

from backtest.data import sha256

STOCK_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST"
INDEX_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,pctChg"


def request_specs(config, tickers):
    end = config["end_inclusive"]
    tasks = [
        ("benchmark_sse50.csv", {"api": "history", "code": config["benchmark_code"],
            "fields": INDEX_FIELDS, "start": config["benchmark_start"], "end": end, "adjustflag": "3"}),
        ("trade_calendar.csv", {"api": "calendar", "start": config["benchmark_start"], "end": end}),
    ]
    for ticker in tickers:
        code = "sh."+ticker[:6]
        tasks.append((f"daily/{code}.csv", {"api": "history", "code": code,
            "fields": STOCK_FIELDS, "start": config["stock_start"], "end": end, "adjustflag": "3"}))
        tasks.append((f"factors/{code}.csv", {"api": "factors", "code": code,
            "start": config["factor_start"], "end": end}))
    return tasks


def _query(client, spec):
    if spec["api"] == "history":
        result = client.query_history_k_data_plus(spec["code"], spec["fields"],
            start_date=spec["start"], end_date=spec["end"], frequency="d", adjustflag="3")
    elif spec["api"] == "factors":
        result = client.query_adjust_factor(spec["code"], start_date=spec["start"], end_date=spec["end"])
    else:
        result = client.query_trade_dates(start_date=spec["start"], end_date=spec["end"])
    if result.error_code != "0":
        raise RuntimeError(f"{result.error_code}: {result.error_msg}")
    rows = []
    while result.next():
        rows.append(result.get_row_data())
    if result.error_code != "0":
        raise RuntimeError(f"Pagination failed: {result.error_msg}")
    if not rows and spec["api"] != "factors":
        raise RuntimeError("BaoStock returned no rows")
    return pd.DataFrame(rows, columns=result.fields)


def _save_manifest(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def ensure_cache(config, tickers, cache, fetch=False):
    cache = Path(cache).resolve()
    manifest_path = cache/"cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"schema_version": 1, "files": {}}
    tasks = request_specs(config, tickers)
    pending = []
    for relative, spec in tasks:
        entry = manifest["files"].get(relative)
        path = cache/relative
        if not entry or entry["spec"] != spec or not path.is_file() or sha256(path) != entry["sha256"]:
            pending.append((relative, spec))
    if not pending:
        print(f"CACHE_READY {len(tasks)} verified files", flush=True)
        return manifest
    if not fetch:
        raise ValueError(f"Cache needs {len(pending)} files. Run the same command with --fetch.")
    import baostock as bs
    socket.setdefaulttimeout(config["socket_timeout_seconds"])
    cache.mkdir(parents=True, exist_ok=True)
    lock = cache/".fetch.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    try:
        login = bs.login()
        if login.error_code != "0":
            raise RuntimeError(f"BaoStock login failed: {login.error_msg}")
        for index, (relative, spec) in enumerate(pending, start=1):
            print(f"FETCH {index}/{len(pending)} {relative}", flush=True)
            for attempt in range(2):
                try:
                    frame = _query(bs, spec)
                    break
                except (RuntimeError, OSError):
                    if attempt == 1:
                        raise
                    bs.logout()
                    time.sleep(3)
                    login = bs.login()
                    if login.error_code != "0":
                        raise RuntimeError("BaoStock reconnect failed")
            path = cache/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".csv.tmp")
            frame.to_csv(temporary, index=False)
            os.replace(temporary, path)
            manifest["files"][relative] = {"spec": spec, "rows": len(frame), "sha256": sha256(path),
                "fetched_at_utc": datetime.now(timezone.utc).isoformat()}
            _save_manifest(manifest_path, manifest)
            time.sleep(config["request_interval_seconds"])
    finally:
        bs.logout()
        lock.unlink()
    return manifest
