import sys
from unittest.mock import MagicMock


def _import_proxy(monkeypatch):
    for dep in ("llmlingua", "torch", "transformers"):
        if dep not in sys.modules:
            monkeypatch.setitem(sys.modules, dep, MagicMock())
    monkeypatch.delitem(sys.modules, "proxy", raising=False)
    import proxy
    return proxy


def test_cache_key_is_deterministic_and_model_scoped(monkeypatch):
    proxy = _import_proxy(monkeypatch)
    k1 = proxy._cache_key("hello world", "kompress", 0.5)
    k2 = proxy._cache_key("hello world", "kompress", 0.5)
    k3 = proxy._cache_key("hello world", "llmlingua2", 0.5)
    k4 = proxy._cache_key("hello world", "kompress", 0.6)
    assert k1 == k2                 # deterministic
    assert k1 != k3                 # model is part of key
    assert k1 != k4                 # rate is part of key
    assert k1.endswith("|kompress|0.5")
    assert len(k1.split("|")[0]) == 64   # sha256 hex
