"""Cross-check computed prices against BaoStock's direct HFQ endpoint."""
import argparse
import json
from pathlib import Path
import socket
import time

import baostock as bs
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    args = parser.parse_args()
    data = pd.read_csv(args.dataset_dir/"features_with_warmup.csv", float_precision="round_trip")
    out = args.dataset_dir/"provider_checks"
    out.mkdir(exist_ok=True)
    socket.setdefaulttimeout(45)
    result = bs.login()
    if result.error_code != "0":
        raise RuntimeError(result.error_msg)
    checks = []
    try:
        for ticker in ["600000.SS", "600030.SS", "600519.SS"]:
            code = "sh."+ticker[:6]
            original = data.loc[data["tic"] == ticker].copy()
            request = bs.query_history_k_data_plus(code, "date,code,open,high,low,close",
                start_date=original["date"].min(), end_date=original["date"].max(),
                frequency="d", adjustflag="1")
            if request.error_code != "0":
                raise RuntimeError(request.error_msg)
            rows = []
            while request.next():
                rows.append(request.get_row_data())
            if request.error_code != "0" or not rows:
                raise RuntimeError("Direct adjusted-price query empty/incomplete")
            adjusted = pd.DataFrame(rows, columns=request.fields)
            adjusted.to_csv(out/(code+"_direct_hfq.csv"), index=False)
            factors = pd.read_csv(args.reference_cache/"factors"/(code+".csv"))
            factors = factors.loc[factors["dividOperateDate"] <= original["date"].min()].sort_values("dividOperateDate")
            base = float(factors["backAdjustFactor"].iloc[-1]) if len(factors) else 1.0
            joined = original.merge(adjusted, on="date", suffixes=("_computed", "_direct"), validate="one_to_one")
            assert len(joined) == len(original)
            differences = {}
            for column in ["open", "high", "low", "close"]:
                expected = pd.to_numeric(joined[column+"_direct"])/base
                actual = joined[column+"_computed"]
                np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0.001)
                differences[column] = float((actual-expected).abs().max())
            checks.append({"tic": ticker, "rows": len(joined), "passed": True,
                "direct_api_adjustflag": "1", "normalization_base_factor": base,
                "maximum_absolute_difference": differences})
            print("HFQ_CHECK_PASS", ticker, len(joined), differences, flush=True)
            time.sleep(2)
    finally:
        bs.logout()
    report = {"passed": True, "checks": checks, "relative_tolerance": 1e-6,
        "absolute_tolerance": 0.001, "normalization": "direct HFQ divided by source factor effective at 2016-01-04",
        "comparison_scope": "same provider, separate direct-adjusted interface; precision tolerance accounts for vendor rounding"}
    (out/"adjustment_crosscheck.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
