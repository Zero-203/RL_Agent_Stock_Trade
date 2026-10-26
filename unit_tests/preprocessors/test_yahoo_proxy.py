from finrl.meta.yahoo_utils import get_yahoo_proxy


def test_uses_proxy_server_environment_variable(monkeypatch):
    monkeypatch.setenv("PROXY_SERVER", "http://127.0.0.1:7890")

    assert get_yahoo_proxy() == "http://127.0.0.1:7890"


def test_explicit_proxy_overrides_environment_variable(monkeypatch):
    monkeypatch.setenv("PROXY_SERVER", "http://127.0.0.1:7890")

    assert get_yahoo_proxy("http://127.0.0.1:8080") == "http://127.0.0.1:8080"


def test_returns_none_when_proxy_is_not_configured(monkeypatch):
    monkeypatch.delenv("PROXY_SERVER", raising=False)

    assert get_yahoo_proxy() is None