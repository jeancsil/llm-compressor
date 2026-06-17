import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient

# ---------------------------------------------------------------------------
# Compression result shape expected by llmlingua_proxy.compress_text
# ---------------------------------------------------------------------------

COMPRESS_RESULT = {
    "compressed_prompt": "compressed prompt",
    "origin_tokens": 100,
    "compressed_tokens": 60,
    "ratio": 0.60,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_compressor() -> MagicMock:
    """Return a MagicMock that mimics PromptCompressor."""
    mock = MagicMock()
    mock.compress_prompt.return_value = COMPRESS_RESULT
    return mock


# ---------------------------------------------------------------------------
# tmp_stats_file fixture
#
# Provides a temporary path for STATS_FILE so tests never touch the real
# stats.json on disk.  The fixture patches the module-level string *before*
# any stats I/O happens (the proxy calls load_stats() at import time, so
# patching must happen before the module is first imported — see `client`
# fixture below).
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_stats_file(tmp_path: Path) -> Path:
    return tmp_path / "test_stats.json"


# ---------------------------------------------------------------------------
# client fixture
#
# Patches out PromptCompressor *before* importing llmlingua_proxy so the
# heavy model never loads.  Also redirects STATS_FILE to a temp path so
# load_stats() / save_stats() are hermetic.
#
# Uses monkeypatch + importlib.reload so that each test session gets a clean
# module state.  The reload approach means:
#   - pytest --collect-only does NOT import the real llmlingua module.
#   - Multiple test runs in the same process stay isolated.
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Point STATS_FILE to a temp path before the module initialises its stats.
    stats_path = str(tmp_path / "test_stats.json")

    # Patch the llmlingua package so PromptCompressor is a mock class whose
    # instances return our canned COMPRESS_RESULT.
    mock_compressor_instance = _make_mock_compressor()
    mock_llmlingua = MagicMock()
    mock_llmlingua.PromptCompressor.return_value = mock_compressor_instance
    monkeypatch.setitem(sys.modules, "llmlingua", mock_llmlingua)

    # Also stub heavy transitive dependencies that llmlingua pulls in.
    for dep in ("torch", "transformers"):
        if dep not in sys.modules:
            monkeypatch.setitem(sys.modules, dep, MagicMock())

    # Remove a cached llmlingua_proxy so reload picks up the patched deps.
    monkeypatch.delitem(sys.modules, "llmlingua_proxy", raising=False)

    # Set env vars consumed by the proxy.
    monkeypatch.setenv("COMPRESS_RATE", "0.5")
    monkeypatch.setenv("COST_PER_MTOK", "3.0")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    # Import (or reload) the proxy module now that patches are in place.
    import llmlingua_proxy as proxy  # noqa: PLC0415

    # Redirect STATS_FILE after import so save_stats / load_stats use tmp.
    monkeypatch.setattr(proxy, "STATS_FILE", stats_path)

    # Replace the live compressor instance with the mock.
    monkeypatch.setattr(proxy, "compressor", mock_compressor_instance)

    return TestClient(proxy.app)
