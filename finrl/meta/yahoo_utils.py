import os


def get_yahoo_proxy(proxy=None):
    return proxy or os.getenv("PROXY_SERVER") or None