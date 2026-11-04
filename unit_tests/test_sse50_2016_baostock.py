import csv
import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest


spec = importlib.util.spec_from_file_location(
    "sse50_baostock", Path(__file__).resolve().parents[1] / "examples" / "FinRL_StockTrading_SSE50_2016_baostock.py"
)
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def test_saves_unmodified_rows_and_empty_fields(tmp_path):
    result = Mock(error_code="0", error_msg="success", fields=["date", "code", "volume", "tradestatus"])
    result.next.side_effect = [True, True, False]
    result.get_row_data.side_effect = [
        ["2016-01-04", "sh.600837", "87982647", "1"],
        ["2025-03-04", "sh.600837", "", "0"],
    ]
    client = Mock()
    client.query_history_k_data_plus.return_value = result

    count = script.save_history(client, "sh.600837", "2016-01-01", "2026-11-04", "3", tmp_path)

    assert count == 2
    assert client.query_history_k_data_plus.call_args.kwargs["adjustflag"] == "3"
    with (tmp_path / "sh.600837.csv").open(encoding="utf-8", newline="") as file:
        assert list(csv.reader(file)) == [result.fields, ["2016-01-04", "sh.600837", "87982647", "1"], ["2025-03-04", "sh.600837", "", "0"]]


def test_empty_result_does_not_replace_previous_file(tmp_path):
    path = tmp_path / "sh.601989.csv"
    path.write_text("previous data", encoding="utf-8")
    result = Mock(error_code="0", error_msg="success", fields=["date"])
    result.next.return_value = False
    client = Mock()
    client.query_history_k_data_plus.return_value = result

    with pytest.raises(RuntimeError, match="no rows"):
        script.save_history(client, "sh.601989", "2016-01-01", "2026-11-04", "3", tmp_path)

    assert path.read_text(encoding="utf-8") == "previous data"
    assert list(tmp_path.iterdir()) == [path]


def test_pagination_error_does_not_publish_partial_file(tmp_path):
    result = Mock(error_code="0", error_msg="success", fields=["date"])
    result.get_row_data.return_value = ["2016-01-04"]

    def next_page():
        if result.next.call_count == 1:
            return True
        result.error_code = "1"
        result.error_msg = "page failed"
        return False

    result.next.side_effect = next_page
    client = Mock()
    client.query_history_k_data_plus.return_value = result

    with pytest.raises(RuntimeError, match="page failed"):
        script.save_history(client, "sh.600837", "2016-01-01", "2026-11-04", "3", tmp_path)

    assert list(tmp_path.iterdir()) == []