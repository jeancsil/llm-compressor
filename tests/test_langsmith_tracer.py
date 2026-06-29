import asyncio
import importlib
from unittest.mock import MagicMock, patch, call
import pytest


def make_tracer():
    """Fresh tracer instance per test — avoids singleton pollution."""
    import langsmith_tracer as mod
    tracer = mod.LangSmithTracer()
    return tracer


# ---------------------------------------------------------------------------
# enabled / disabled
# ---------------------------------------------------------------------------

def test_disabled_when_no_key(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    t = make_tracer()
    t.init()
    assert not t.enabled


def test_disabled_when_key_empty(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "   ")
    t = make_tracer()
    t.init()
    assert not t.enabled


def test_enabled_when_key_set(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls__test123")
    mock_client = MagicMock()
    with patch("langsmith.Client", return_value=mock_client):
        t = make_tracer()
        t.init()
    assert t.enabled


def test_disabled_when_langsmith_not_installed(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls__test123")
    with patch.dict("sys.modules", {"langsmith": None}):
        t = make_tracer()
        t.init()
    assert not t.enabled


# ---------------------------------------------------------------------------
# log_request — disabled path
# ---------------------------------------------------------------------------

def test_log_request_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    t = make_tracer()
    t.init()
    # Should not raise, should not call anything
    asyncio.run(t.log_request([], [], None, None, "", {}))


# ---------------------------------------------------------------------------
# log_request — enabled path
# ---------------------------------------------------------------------------

def test_log_request_calls_create_and_update(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls__test123")
    mock_client = MagicMock()

    with patch("langsmith.Client", return_value=mock_client):
        t = make_tracer()
        t.init()

    original_msgs = [{"role": "user", "content": "hello world long enough"}]
    compressed_msgs = [{"role": "user", "content": "hello long"}]
    metadata = {"compression_ratio": 0.55, "model": "llmlingua2"}

    async def run():
        await t.log_request(
            original_msgs, compressed_msgs,
            "system original", "system compressed",
            "response text", metadata,
        )
        # Allow the fire-and-forget task to execute
        await asyncio.sleep(0)

    asyncio.run(run())

    assert mock_client.create_run.called
    create_kwargs = mock_client.create_run.call_args.kwargs
    assert create_kwargs["run_type"] == "llm"
    assert create_kwargs["name"] == "proxy-request"
    assert create_kwargs["inputs"]["original_messages"] == original_msgs
    assert create_kwargs["inputs"]["compressed_messages"] == compressed_msgs

    assert mock_client.update_run.called
    update_kwargs = mock_client.update_run.call_args.kwargs
    assert update_kwargs["outputs"]["response"] == "response text"


def test_log_request_swallows_langsmith_errors(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls__test123")
    mock_client = MagicMock()
    mock_client.create_run.side_effect = RuntimeError("LangSmith is down")

    with patch("langsmith.Client", return_value=mock_client):
        t = make_tracer()
        t.init()

    async def run():
        await t.log_request([], [], None, None, "", {})
        await asyncio.sleep(0)

    # Must not raise
    asyncio.run(run())
