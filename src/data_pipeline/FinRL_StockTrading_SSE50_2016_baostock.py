"""Save separate, unadjusted BaoStock history for two delisted constituents."""

import csv
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "sse50_2016.json"
FIELDS = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST"


def save_history(client, code, start_date, end_date, adjustflag, output_dir):
    result = client.query_history_k_data_plus(
        code, FIELDS, start_date=start_date, end_date=end_date,
        frequency="d", adjustflag=adjustflag,
    )
    if result.error_code != "0":
        raise RuntimeError(f"{code}: {result.error_code} {result.error_msg}")
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_path = None
    count = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=output_dir, delete=False
        ) as file:
            temp_path = Path(file.name)
            writer = csv.writer(file)
            writer.writerow(result.fields)
            while result.next():
                writer.writerow(result.get_row_data())
                count += 1
            if result.error_code != "0":
                raise RuntimeError(f"{code}: {result.error_code} {result.error_msg}")
            if not count:
                raise RuntimeError(f"{code}: BaoStock returned no rows")
        os.replace(temp_path, output_dir / f"{code}.csv")
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    return count


def main():
    try:
        import baostock as bs
    except ImportError:
        print("Install the optional dependency: python -m pip install baostock==0.9.4", file=sys.stderr)
        return 1

    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    supplement = config["baostock_supplement"]
    end_date = datetime.now(timezone(timedelta(hours=8))).date().isoformat()
    login = bs.login()
    if login.error_code != "0":
        print(f"BaoStock login failed: {login.error_code} {login.error_msg}", file=sys.stderr)
        return 1
    try:
        for index, code in enumerate(supplement["codes"]):
            if index:
                time.sleep(config["request_interval_seconds"])
            count = save_history(
                bs, code, config["start_date"], end_date,
                supplement["adjustflag"], ROOT / supplement["output_directory"],
            )
            print(f"{code}: {count} rows", flush=True)
    except Exception as error:
        print(f"BaoStock download stopped: {error}", file=sys.stderr)
        return 1
    finally:
        bs.logout()
    return 0


if __name__ == "__main__":
    sys.exit(main())
