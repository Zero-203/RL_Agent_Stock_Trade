import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"src"))

import pandas as pd
from data_pipeline.baostock_cache import ensure_cache


class BaoStockCacheTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT/".test_tmp"
        parent.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="cache_", dir=parent)
        self.cache = Path(self.temporary.name)
        self.config = {"end_inclusive": "2026-11-03", "benchmark_code": "sh.000016",
            "benchmark_start": "2015-01-01", "stock_start": "2016-01-01",
            "factor_start": "1990-01-01", "socket_timeout_seconds": 45,
            "request_interval_seconds": 0}

    def tearDown(self):
        self.temporary.cleanup()

    def test_valid_cache_reuses_relative_keys_without_network_login(self):
        frame = pd.DataFrame({"date": ["2016-01-04"], "value": [1]})
        with patch("baostock.login", return_value=SimpleNamespace(error_code="0")), patch("baostock.logout"), patch("data_pipeline.baostock_cache._query", return_value=frame):
            ensure_cache(self.config, [], self.cache, fetch=True)
        stored = json.loads((self.cache/"cache_manifest.json").read_text())
        self.assertEqual(set(stored["files"]), {"benchmark_sse50.csv", "trade_calendar.csv"})
        self.assertTrue(all(not Path(key).is_absolute() for key in stored["files"]))
        with patch("baostock.login") as login:
            ensure_cache(self.config, [], self.cache, fetch=True)
            login.assert_not_called()

    def test_failed_download_keeps_completed_file_and_retries_only_missing(self):
        frame = pd.DataFrame({"date": ["2016-01-04"], "value": [1]})
        with patch("baostock.login", return_value=SimpleNamespace(error_code="0")), patch("baostock.logout"), patch("time.sleep"), patch("data_pipeline.baostock_cache._query", side_effect=[frame, RuntimeError("connection reset"), RuntimeError("connection reset")]):
            with self.assertRaises(RuntimeError):
                ensure_cache(self.config, [], self.cache, fetch=True)
        stored = json.loads((self.cache/"cache_manifest.json").read_text())
        self.assertEqual(set(stored["files"]), {"benchmark_sse50.csv"})
        self.assertFalse((self.cache/".fetch.lock").exists())
        with patch("baostock.login", return_value=SimpleNamespace(error_code="0")), patch("baostock.logout"), patch("data_pipeline.baostock_cache._query", return_value=frame) as query:
            ensure_cache(self.config, [], self.cache, fetch=True)
            self.assertEqual(query.call_count, 1)
            self.assertEqual(query.call_args.args[1]["api"], "calendar")


if __name__ == "__main__":
    unittest.main()
