"""Download BaoStock daily data for the official SSE 50 index."""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from FinRL_StockTrading_SSE50_2016_baostock import save_history


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "sse50_2016.json"
FIELDS = "date,code,open,high,low,close,volume,amount"


def main():
    try:
        import baostock as bs
    except ImportError:
        print("Install the optional dependency: python -m pip install baostock==0.9.4", file=sys.stderr)
        return 1

    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    index = config["baostock_index"]
    end_date = datetime.now(timezone(timedelta(hours=8))).date().isoformat()
    login = bs.login()
    if login.error_code != "0":
        print(f"BaoStock login failed: {login.error_code} {login.error_msg}", file=sys.stderr)
        return 1
    try:
        output_dir = ROOT / index["output_directory"]
        count = save_history(
            bs, index["code"], config["start_date"], end_date, "3", output_dir, fields=FIELDS,
        )
        print(f"{index['code']}: {count} rows saved to {output_dir / (index['code'] + '.csv')}", flush=True)
    except Exception as error:
        print(f"BaoStock index download stopped: {error}", file=sys.stderr)
        return 1
    finally:
        bs.logout()
    return 0


if __name__ == "__main__":
    sys.exit(main())
