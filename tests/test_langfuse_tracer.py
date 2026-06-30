import asyncio
from unittest.mock import MagicMock, patch
import pytest


def make_tracer():
    import langfuse_tracer as mod
    return mod.LangfuseTracer()


# --- enabled / disabled ---

def test_disabled_when_no_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    t = make_tracer()
    t.init()
    assert not t.enabled


def test_disabled_when_keys_empty(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "  ")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "  ")
    t = make_tracer()
    t.init()
    assert not t.enabled


def test_enabled_when_keys_set(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    mock_lf = MagicMock()
    with patch("langfuse.Langfuse", return_value=mock_lf):
        t = make_tracer()
        t.init()
    assert t.enabled


def test_disabled_when_langfuse_not_installed(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    with patch.dict("sys.modules", {"langfuse": None}):
        t = make_tracer()
        t.init()
    assert not t.enabled


# --- log_request disabled ---

def test_log_request_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    t = make_tracer()
    t.init()
    asyncio.run(t.log_request([], [], None, None, "", {}))  # must not raise


# --- log_request enabled ---

def test_log_request_creates_generation(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    mock_lf = MagicMock()
    mock_obs = MagicMock()
    mock_lf.start_observation.return_value = mock_obs

    with patch("langfuse.Langfuse", return_value=mock_lf):
        t = make_tracer()
        t.init()

    async def run():
        await t.log_request(
            [{"role": "user", "content": "hello"}],
            [{"role": "user", "content": "hi"}],
            "system orig", "system compressed",
            "response text",
            {"compression_ratio": 0.55, "anthropic_model": "claude-haiku-4-5"},
        )
        await asyncio.sleep(0)

    asyncio.run(run())
    assert mock_lf.start_observation.called
    kwargs = mock_lf.start_observation.call_args.kwargs
    assert kwargs["as_type"] == "generation"
    assert kwargs["output"] == {"text": "response text"}
    assert kwargs["model"] == "claude-haiku-4-5"
    assert mock_obs.end.called


def test_log_request_passes_tags(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    mock_lf = MagicMock()
    mock_lf.start_observation.return_value = MagicMock()

    with patch("langfuse.Langfuse", return_value=mock_lf):
        t = make_tracer()
        t.init()

    async def run():
        await t.log_request([], [], None, None, "", {}, tags=["streaming", "llmlingua2"])
        await asyncio.sleep(0)

    asyncio.run(run())
    kwargs = mock_lf.start_observation.call_args.kwargs
    assert "streaming" in kwargs["metadata"]["tags"]
    assert "llmlingua2" in kwargs["metadata"]["tags"]


def test_log_request_swallows_errors(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    mock_lf = MagicMock()
    mock_lf.start_observation.side_effect = RuntimeError("langfuse is down")

    with patch("langfuse.Langfuse", return_value=mock_lf):
        t = make_tracer()
        t.init()

    async def run():
        await t.log_request([], [], None, None, "", {})
        await asyncio.sleep(0)

    asyncio.run(run())  # must not raise


# --- add_to_dataset ---

def test_add_to_dataset_calls_create_item(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    mock_lf = MagicMock()

    with patch("langfuse.Langfuse", return_value=mock_lf):
        t = make_tracer()
        t.init()

    async def run():
        await t.add_to_dataset({"input": "hi"}, {"output": "hello"}, "test-ds")
        await asyncio.sleep(0)

    asyncio.run(run())
    assert mock_lf.create_dataset_item.called
    kwargs = mock_lf.create_dataset_item.call_args.kwargs
    assert kwargs["dataset_name"] == "test-ds"
    assert kwargs["input"] == {"input": "hi"}
    assert kwargs["expected_output"] == {"output": "hello"}


def test_add_to_dataset_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    t = make_tracer()
    t.init()
    asyncio.run(t.add_to_dataset({}, {}))  # must not raise


def test_add_to_dataset_swallows_errors(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    mock_lf = MagicMock()
    mock_lf.create_dataset_item.side_effect = RuntimeError("network error")

    with patch("langfuse.Langfuse", return_value=mock_lf):
        t = make_tracer()
        t.init()

    async def run():
        await t.add_to_dataset({}, {})
        await asyncio.sleep(0)

    asyncio.run(run())  # must not raise


# --- attach_feedback ---

def test_attach_feedback_calls_create_score(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    mock_lf = MagicMock()

    with patch("langfuse.Langfuse", return_value=mock_lf):
        t = make_tracer()
        t.init()

    import uuid as _uuid
    trace_id = str(_uuid.uuid4())

    async def run():
        await t.attach_feedback(trace_id, 0.8, "looks good")
        await asyncio.sleep(0)

    asyncio.run(run())
    assert mock_lf.create_score.called
    kwargs = mock_lf.create_score.call_args.kwargs
    assert kwargs["trace_id"] == trace_id
    assert kwargs["value"] == 0.8
    assert kwargs["name"] == "quality"


def test_attach_feedback_clamps_score(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    mock_lf = MagicMock()

    with patch("langfuse.Langfuse", return_value=mock_lf):
        t = make_tracer()
        t.init()

    async def run():
        await t.attach_feedback("tid", 1.5)
        await t.attach_feedback("tid", -0.3)
        await asyncio.sleep(0)

    asyncio.run(run())
    scores = [c.kwargs["value"] for c in mock_lf.create_score.call_args_list]
    assert scores[0] == 1.0
    assert scores[1] == 0.0


def test_attach_feedback_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    t = make_tracer()
    t.init()
    asyncio.run(t.attach_feedback("tid", 1.0))  # must not raise


# --- status ---

def test_status_disabled(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    t = make_tracer()
    t.init()
    s = t.status()
    assert s["enabled"] is False
    assert s["host"] is not None  # always has a default


def test_status_enabled(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    mock_lf = MagicMock()
    with patch("langfuse.Langfuse", return_value=mock_lf):
        t = make_tracer()
        t.init()
    s = t.status()
    assert s["enabled"] is True
    assert s["public_key_set"] is True
