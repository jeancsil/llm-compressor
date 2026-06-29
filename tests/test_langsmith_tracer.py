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


# ---------------------------------------------------------------------------
# tags forwarded to create_run
# ---------------------------------------------------------------------------

def test_log_request_passes_tags(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls__test123")
    mock_client = MagicMock()
    with patch("langsmith.Client", return_value=mock_client):
        t = make_tracer()
        t.init()

    async def run():
        await t.log_request([], [], None, None, "", {}, tags=["llmlingua2", "streaming"])
        await asyncio.sleep(0)

    asyncio.run(run())
    create_kwargs = mock_client.create_run.call_args.kwargs
    assert create_kwargs["tags"] == ["llmlingua2", "streaming"]


def test_log_request_empty_tags_when_none(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls__test123")
    mock_client = MagicMock()
    with patch("langsmith.Client", return_value=mock_client):
        t = make_tracer()
        t.init()

    async def run():
        await t.log_request([], [], None, None, "", {})
        await asyncio.sleep(0)

    asyncio.run(run())
    create_kwargs = mock_client.create_run.call_args.kwargs
    assert create_kwargs["tags"] == []


# ---------------------------------------------------------------------------
# add_to_dataset
# ---------------------------------------------------------------------------

def test_add_to_dataset_uses_existing_dataset(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls__test123")
    mock_client = MagicMock()
    existing_ds = MagicMock()
    existing_ds.id = "ds-uuid-123"
    mock_client.list_datasets.return_value = iter([existing_ds])

    with patch("langsmith.Client", return_value=mock_client):
        t = make_tracer()
        t.init()

    async def run():
        await t.add_to_dataset({"input": "hi"}, {"output": "hello"}, "test-ds")
        await asyncio.sleep(0)

    asyncio.run(run())
    mock_client.create_dataset.assert_not_called()
    mock_client.create_example.assert_called_once()
    call_kwargs = mock_client.create_example.call_args.kwargs
    assert call_kwargs["dataset_id"] == "ds-uuid-123"


def test_add_to_dataset_creates_dataset_when_absent(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls__test123")
    mock_client = MagicMock()
    mock_client.list_datasets.return_value = iter([])
    new_ds = MagicMock()
    new_ds.id = "ds-new-456"
    mock_client.create_dataset.return_value = new_ds

    with patch("langsmith.Client", return_value=mock_client):
        t = make_tracer()
        t.init()

    async def run():
        await t.add_to_dataset({"input": "hi"}, {"output": "hello"}, "test-ds")
        await asyncio.sleep(0)

    asyncio.run(run())
    mock_client.create_dataset.assert_called_once()
    call_kwargs = mock_client.create_example.call_args.kwargs
    assert call_kwargs["dataset_id"] == "ds-new-456"


def test_add_to_dataset_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    t = make_tracer()
    t.init()
    asyncio.run(t.add_to_dataset({}, {}))  # must not raise


def test_add_to_dataset_swallows_errors(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls__test123")
    mock_client = MagicMock()
    mock_client.list_datasets.side_effect = RuntimeError("network error")

    with patch("langsmith.Client", return_value=mock_client):
        t = make_tracer()
        t.init()

    async def run():
        await t.add_to_dataset({}, {})
        await asyncio.sleep(0)

    asyncio.run(run())  # must not raise


# ---------------------------------------------------------------------------
# attach_feedback
# ---------------------------------------------------------------------------

def test_attach_feedback_calls_create_feedback(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls__test123")
    mock_client = MagicMock()
    with patch("langsmith.Client", return_value=mock_client):
        t = make_tracer()
        t.init()

    import uuid as _uuid
    run_id = _uuid.uuid4()

    async def run():
        await t.attach_feedback(run_id, 0.8, "looks good")
        await asyncio.sleep(0)

    asyncio.run(run())
    mock_client.create_feedback.assert_called_once()
    call_kwargs = mock_client.create_feedback.call_args.kwargs
    assert call_kwargs["run_id"] == run_id
    assert call_kwargs["score"] == 0.8
    assert call_kwargs["key"] == "quality"


def test_attach_feedback_clamps_score(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls__test123")
    mock_client = MagicMock()
    with patch("langsmith.Client", return_value=mock_client):
        t = make_tracer()
        t.init()

    import uuid as _uuid

    async def run():
        await t.attach_feedback(_uuid.uuid4(), 1.5)   # above max
        await t.attach_feedback(_uuid.uuid4(), -0.3)  # below min
        await asyncio.sleep(0)

    asyncio.run(run())
    scores = [c.kwargs["score"] for c in mock_client.create_feedback.call_args_list]
    assert scores[0] == 1.0
    assert scores[1] == 0.0


def test_attach_feedback_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    t = make_tracer()
    t.init()
    import uuid as _uuid
    asyncio.run(t.attach_feedback(_uuid.uuid4(), 1.0))  # must not raise


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def test_status_disabled(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    t = make_tracer()
    t.init()
    s = t.status()
    assert s["enabled"] is False
    assert s["project"] is None
    assert s["api_key_set"] is False


def test_status_enabled(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls__test123")
    mock_client = MagicMock()
    with patch("langsmith.Client", return_value=mock_client):
        t = make_tracer()
        t.init("my-project")
    s = t.status()
    assert s["enabled"] is True
    assert s["project"] == "my-project"
    assert s["api_key_set"] is True
