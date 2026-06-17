import sqlite3
import sys
import pytest
from unittest.mock import MagicMock
from starlette.testclient import TestClient

# ---------------------------------------------------------------------------
# Compression result shape expected by llmlingua_proxy.compress_text
# ---------------------------------------------------------------------------

MOCK_COMPRESS_RESULT = {
    "compressed_prompt": "compressed prompt",
    "origin_tokens": 100,
    "compressed_tokens": 60,
    "ratio": 0.60,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_llmlingua() -> MagicMock:
    """Return a MagicMock that mimics PromptCompressor."""
    mock = MagicMock()
    mock.compress_prompt.return_value = MOCK_COMPRESS_RESULT
    return mock


# ---------------------------------------------------------------------------
# client fixture
#
# Patches _load_backend so the heavy model never loads.  Uses a temp SQLite
# DB so all I/O is hermetic.
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient with model loading patched and a temp DB."""
    # Env vars consumed by the proxy
    monkeypatch.setenv("COMPRESSOR_MODEL", "llmlingua2")
    monkeypatch.setenv("COMPRESS_RATE", "0.5")
    monkeypatch.setenv("COST_PER_MTOK", "3.0")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    # Stub heavy transitive dependencies so importing the proxy is safe.
    for dep in ("llmlingua", "torch", "transformers"):
        if dep not in sys.modules:
            monkeypatch.setitem(sys.modules, dep, MagicMock())

    # Remove a cached module so reload picks up the patched deps.
    monkeypatch.delitem(sys.modules, "llmlingua_proxy", raising=False)

    import llmlingua_proxy as proxy  # noqa: PLC0415

    # Patch _load_backend to return a mock backend dict
    mock_compressor = make_mock_llmlingua()
    mock_backend = {"type": "llmlingua2", "compressor": mock_compressor, "rate": 0.5}
    monkeypatch.setattr(proxy, "_load_backend", lambda: mock_backend)

    # Redirect DB_PATH to temp path
    monkeypatch.setattr(proxy, "DB_PATH", str(tmp_path / "test_metrics.db"))

    # Suppress JSON migration so the test DB stays empty
    monkeypatch.setattr(proxy, "migrate_from_json", lambda conn, json_path="stats.json": None)

    with TestClient(proxy.app) as c:
        yield c
