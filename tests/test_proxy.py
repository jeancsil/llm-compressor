import sys
import json
from collections import deque

from starlette.testclient import TestClient


def test_health(client: TestClient):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_init_db_creates_table(tmp_path):
    from proxy import init_db
    conn = init_db(str(tmp_path / "metrics.db"))
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='compressions'")
    assert cur.fetchone() is not None
    cols = [row[1] for row in conn.execute("PRAGMA table_info(compressions)")]
    for col in ("id", "ts", "session_id", "model", "original_tokens", "compressed_tokens", "latency_ms"):
        assert col in cols
    conn.close()


def test_migrate_imports_rows(tmp_path):
    from proxy import init_db, migrate_from_json
    stats_json = tmp_path / "stats.json"
    stats_json.write_text(json.dumps({
        "total_requests": 5,
        "total_original_tokens": 500,
        "total_compressed_tokens": 300,
        "sessions": {},
        "recent_compressions": [
            {"ts": "10:00:00", "session_id": "abc", "original": 100, "compressed": 60, "saved": 40},
            {"ts": "10:01:00", "session_id": "def", "original": 200, "compressed": 120, "saved": 80},
        ],
    }))
    conn = init_db(str(tmp_path / "metrics.db"))
    migrate_from_json(conn, json_path=str(stats_json))
    count = conn.execute("SELECT COUNT(*) FROM compressions").fetchone()[0]
    assert count == 2
    bak = tmp_path / "stats.json.bak"
    assert bak.exists()
    conn.close()


def test_migrate_is_idempotent(tmp_path):
    from proxy import init_db, migrate_from_json
    stats_json = tmp_path / "stats.json"
    stats_json.write_text(json.dumps({
        "recent_compressions": [
            {"ts": "10:00:00", "session_id": "abc", "original": 100, "compressed": 60, "saved": 40},
        ],
    }))
    conn = init_db(str(tmp_path / "metrics.db"))
    migrate_from_json(conn, json_path=str(stats_json))
    migrate_from_json(conn, json_path=str(stats_json))
    count = conn.execute("SELECT COUNT(*) FROM compressions").fetchone()[0]
    assert count == 1
    conn.close()


def test_migrate_no_json(tmp_path):
    from proxy import init_db, migrate_from_json
    conn = init_db(str(tmp_path / "metrics.db"))
    migrate_from_json(conn, json_path=str(tmp_path / "nonexistent.json"))
    conn.close()


def test_load_stats_from_db(tmp_path):
    from proxy import init_db, load_stats_from_db, stats
    conn = init_db(str(tmp_path / "metrics.db"))
    conn.executemany(
        "INSERT INTO compressions (ts, session_id, model, original_tokens, compressed_tokens, latency_ms) VALUES (?,?,?,?,?,?)",
        [
            ("2026-06-14T10:00:00", "aaa", "llmlingua2", 200, 140, 10.0),
            ("2026-06-14T10:01:00", "aaa", "llmlingua2", 150, 110, 12.0),
            ("2026-06-14T10:02:00", "bbb", "llmlingua2", 300, 210, 9.0),
        ],
    )
    conn.commit()
    stats["total_original_tokens"] = 0
    stats["total_compressed_tokens"] = 0
    stats["sessions"] = {}
    stats["recent_compressions"] = deque(maxlen=100)
    load_stats_from_db(conn)
    assert stats["total_original_tokens"] == 650
    assert stats["total_compressed_tokens"] == 460
    assert "aaa" in stats["sessions"]
    assert stats["sessions"]["aaa"]["original_tokens"] == 350
    assert len(stats["recent_compressions"]) == 3
    conn.close()


def test_recover_stats_from_backup(tmp_path):
    from collections import deque
    from proxy import init_db, migrate_from_json, recover_stats_from_backup, load_stats_from_db, stats

    stats_json = tmp_path / "stats.json"
    stats_json.write_text(json.dumps({
        "total_requests": 50,
        "total_original_tokens": 5000,
        "total_compressed_tokens": 3000,
        "sessions": {
            "sess-a": {"requests": 30, "original_tokens": 3000, "compressed_tokens": 1800,
                       "first_seen": "2026-06-01T10:00:00", "last_seen": "2026-06-10T10:00:00"},
            "sess-b": {"requests": 20, "original_tokens": 2000, "compressed_tokens": 1200,
                       "first_seen": "2026-06-01T11:00:00", "last_seen": "2026-06-10T11:00:00"},
        },
        "recent_compressions": [
            {"ts": "10:00:00", "session_id": "sess-a", "original": 100, "compressed": 60, "saved": 40},
        ],
    }))
    conn = init_db(str(tmp_path / "metrics.db"))
    migrate_from_json(conn, json_path=str(stats_json))
    bak = tmp_path / "stats.json.bak"

    stats["total_requests"] = 0
    stats["total_original_tokens"] = 0
    stats["total_compressed_tokens"] = 0
    stats["sessions"] = {}
    stats["recent_compressions"] = deque(maxlen=100)

    recover_stats_from_backup(conn, bak_path=str(bak))
    recover_stats_from_backup(conn, bak_path=str(bak))  # idempotent

    load_stats_from_db(conn)
    assert stats["total_original_tokens"] == 5000
    assert stats["total_compressed_tokens"] == 3000
    assert stats["total_requests"] == 50
    conn.close()


def test_record_compression_writes_to_db(tmp_path):
    """Task 7: record_compression writes a row to SQLite including latency_ms."""
    for dep in ("llmlingua", "torch", "transformers"):
        if dep not in sys.modules:
            import unittest.mock as _mock
            sys.modules[dep] = _mock.MagicMock()

    import proxy as proxy
    from proxy import init_db

    conn = init_db(str(tmp_path / "metrics.db"))
    proxy._db_conn = conn
    proxy.backend = {"type": "llmlingua2", "rate": 0.5}

    proxy.record_compression("sess-xyz", 200, 120, 95.5)

    row = conn.execute("SELECT * FROM compressions").fetchone()
    assert row is not None
    assert row["session_id"] == "sess-xyz"
    assert row["original_tokens"] == 200
    assert row["compressed_tokens"] == 120
    assert row["latency_ms"] == 95.5
    assert row["model"] == "llmlingua2"
    assert proxy.stats["recent_compressions"][0]["latency_ms"] == 95.5
    conn.close()


def test_compress_text_records_latency(tmp_path, monkeypatch):
    """Task 8: compress_text measures wall-clock latency and persists it via record_compression."""
    from unittest.mock import MagicMock

    for dep in ("llmlingua", "torch", "transformers"):
        if dep not in sys.modules:
            monkeypatch.setitem(sys.modules, dep, MagicMock())

    monkeypatch.delitem(sys.modules, "proxy", raising=False)

    import proxy as proxy
    from proxy import init_db
    from tests.conftest import make_mock_llmlingua

    conn = init_db(str(tmp_path / "metrics.db"))
    proxy._db_conn = conn
    proxy.backend = {
        "type": "llmlingua2",
        "compressor": make_mock_llmlingua(),
        "rate": 0.5,
    }

    proxy.compress_text("word " * 50, "sess-test")  # 250 chars → triggers compression

    row = proxy._db_conn.execute("SELECT latency_ms FROM compressions").fetchone()
    assert row is not None
    assert row[0] >= 0
    conn.close()


def test_load_backend_llmlingua2(monkeypatch):
    """Task 9: load_backend with COMPRESSOR_MODEL=llmlingua2 returns a valid backend dict."""
    from unittest.mock import MagicMock

    monkeypatch.setenv("COMPRESSOR_MODEL", "llmlingua2")
    monkeypatch.setenv("COMPRESS_RATE", "0.4")

    mock_cls = MagicMock()
    mock_cls.return_value.compress_prompt.return_value = {
        "compressed_prompt": "compressed x",
        "origin_tokens": 10,
        "compressed_tokens": 6,
        "ratio": 0.6,
    }

    monkeypatch.setattr("llmlingua.PromptCompressor", mock_cls)
    from proxy import load_backend
    b = load_backend()
    assert b["type"] == "llmlingua2"
    assert b["rate"] == 0.4


def test_load_backend_kompress_raises_without_package(monkeypatch):
    """load_backend with COMPRESSOR_MODEL=kompress raises RuntimeError when headroom-ai is absent."""
    import pytest
    import unittest.mock as mock

    monkeypatch.setenv("COMPRESSOR_MODEL", "kompress")
    monkeypatch.delitem(sys.modules, "proxy", raising=False)

    with mock.patch.dict(sys.modules, {"headroom": None, "headroom.transforms": None,
                                        "headroom.transforms.kompress_compressor": None}):
        import proxy as proxy
        monkeypatch.setattr(proxy, "_load_backend", proxy.load_backend)
        with pytest.raises((RuntimeError, ImportError)):
            proxy._load_kompress_backend()


def test_timeseries_empty(client):
    """Task 10: /stats/timeseries returns [] when there are no rows in the DB."""
    r = client.get("/stats/timeseries")
    assert r.status_code == 200
    assert r.json() == []


def test_timeseries_structure(tmp_path, monkeypatch):
    """Task 10: /stats/timeseries returns hourly buckets with required keys."""
    from unittest.mock import MagicMock
    from datetime import datetime, timedelta, timezone

    for dep in ("llmlingua", "torch", "transformers"):
        if dep not in sys.modules:
            monkeypatch.setitem(sys.modules, dep, MagicMock())

    monkeypatch.delitem(sys.modules, "proxy", raising=False)

    import proxy as proxy
    from proxy import init_db
    from tests.conftest import make_mock_llmlingua

    conn = init_db(str(tmp_path / "metrics.db"))

    now = datetime.now(timezone.utc)
    conn.executemany(
        "INSERT INTO compressions (ts, session_id, model, original_tokens, compressed_tokens, latency_ms) VALUES (?,?,?,?,?,?)",
        [
            (
                (now - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%S"),
                "sess1", "llmlingua2", 200, 120, 100.0,
            ),
            (
                (now - timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%S"),
                "sess2", "llmlingua2", 300, 180, 200.0,
            ),
        ],
    )
    conn.commit()

    monkeypatch.setattr(proxy, "_db_conn", conn)
    monkeypatch.setattr(proxy, "backend", {"type": "llmlingua2", "rate": 0.5})
    monkeypatch.setattr(proxy, "_load_backend", lambda: {"type": "llmlingua2", "rate": 0.5})
    monkeypatch.setattr(proxy, "DB_PATH", tmp_path / "metrics.db")
    monkeypatch.setattr(proxy, "_migrate_db_location", lambda: None)
    monkeypatch.setattr(proxy, "migrate_from_json", lambda conn, json_path="stats.json": None)
    monkeypatch.setattr(proxy, "recover_stats_from_backup", lambda conn, bak_path="stats.json.bak": None)

    from starlette.testclient import TestClient
    with TestClient(proxy.app) as c:
        r = c.get("/stats/timeseries")

    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    bucket = data[0]
    for key in ("hour", "requests", "avg_savings_pct", "avg_latency_ms", "total_saved"):
        assert key in bucket, f"missing key: {key}"

    conn.close()


def test_stats_includes_compressor_cost(client):
    """Task 11: /stats returns compressor info and cost_per_mtok and avg_latency_ms."""
    r = client.get("/stats")
    assert r.status_code == 200
    d = r.json()
    assert "compressor" in d
    assert d["compressor"]["model"] == "llmlingua2"
    assert d["compressor"]["param_name"] == "rate"
    assert d["compressor"]["param_value"] == 0.5
    assert "cost_per_mtok" in d
    assert d["cost_per_mtok"] == 3.0
    assert "avg_latency_ms" in d


def test_stats_by_model(tmp_path, monkeypatch):
    """Task 4: /stats returns by_model with per-model aggregated compression stats."""
    from unittest.mock import MagicMock

    for dep in ("llmlingua", "torch", "transformers"):
        if dep not in sys.modules:
            monkeypatch.setitem(sys.modules, dep, MagicMock())

    monkeypatch.delitem(sys.modules, "proxy", raising=False)

    import proxy as proxy
    from proxy import init_db

    from pathlib import Path as _Path
    db_path = tmp_path / "metrics.db"
    monkeypatch.setattr(proxy, "DB_PATH", db_path)
    conn = init_db(str(db_path))
    conn.executemany(
        "INSERT INTO compressions (ts, session_id, model, original_tokens, compressed_tokens, latency_ms) VALUES (?,?,?,?,?,?)",
        [
            ("2026-01-01T10:00:00", "s1", "llmlingua2", 200, 100, 10.0),
            ("2026-01-01T10:01:00", "s2", "llmlingua2", 300, 150, 12.0),
            ("2026-01-01T10:02:00", "s3", "llmlingua2-large", 400, 160, 20.0),
        ],
    )
    conn.commit()
    conn.close()

    mock_backend = {"type": "llmlingua2", "rate": 0.5}
    monkeypatch.setattr(proxy, "_load_backend", lambda: mock_backend)
    monkeypatch.setattr(proxy, "migrate_from_json", lambda conn, json_path="stats.json": None)
    monkeypatch.setattr(proxy, "recover_stats_from_backup", lambda conn, bak_path="stats.json.bak": None)
    monkeypatch.setattr(proxy, "_migrate_db_location", lambda: None)

    with TestClient(proxy.app) as c:
        r = c.get("/stats")

    assert r.status_code == 200
    d = r.json()
    assert "by_model" in d
    models = {m["model"]: m for m in d["by_model"]}
    assert "llmlingua2" in models
    assert "llmlingua2-large" in models
    assert models["llmlingua2"]["requests"] == 2
    assert models["llmlingua2-large"]["requests"] == 1
    assert models["llmlingua2"]["avg_savings_pct"] == 50.0


def test_no_utcnow_in_source():
    """Task 1: Verify that datetime.utcnow is not used in proxy.py."""
    from pathlib import Path
    src = Path(__file__).parent.parent / "proxy.py"
    source = src.read_text()
    assert "utcnow" not in source, "Found deprecated datetime.utcnow in proxy.py"


# ---------------------------------------------------------------------------
# Task 2: chunk_text helpers and multi-chunk compress_llmlingua2
# ---------------------------------------------------------------------------

def test_chunk_text_short_returns_single(monkeypatch):
    """Text under 400 tokens is returned as a single-element list."""
    from unittest.mock import MagicMock

    for dep in ("llmlingua", "torch", "transformers"):
        if dep not in sys.modules:
            monkeypatch.setitem(sys.modules, dep, MagicMock())

    monkeypatch.delitem(sys.modules, "proxy", raising=False)

    import proxy as proxy
    from tests.conftest import make_mock_llmlingua

    mock_backend = {"type": "llmlingua2", "compressor": make_mock_llmlingua(), "rate": 0.5}
    monkeypatch.setattr(proxy, "backend", mock_backend)

    short_text = "word " * 50  # 50 tokens — well under 400
    result = proxy.chunk_text(short_text.strip())
    assert len(result) == 1


def test_chunk_text_splits_long_paragraphs(monkeypatch):
    """Two 250-token paragraphs split into two chunks (each exceeds 400 when combined)."""
    from unittest.mock import MagicMock

    for dep in ("llmlingua", "torch", "transformers"):
        if dep not in sys.modules:
            monkeypatch.setitem(sys.modules, dep, MagicMock())

    monkeypatch.delitem(sys.modules, "proxy", raising=False)

    import proxy as proxy
    from tests.conftest import make_mock_llmlingua

    mock_backend = {"type": "llmlingua2", "compressor": make_mock_llmlingua(), "rate": 0.5}
    monkeypatch.setattr(proxy, "backend", mock_backend)

    para = "word " * 250  # ~250 tokens per paragraph
    text = para.strip() + "\n\n" + para.strip()
    chunks = proxy.chunk_text(text)
    assert len(chunks) == 2
    for c in chunks:
        # Each chunk should be at most CHUNK_MAX_TOKENS words
        assert len(c.split()) <= proxy.CHUNK_MAX_TOKENS


def test_compress_llmlingua2_multi_chunk(monkeypatch):
    """compress_backend calls compress_prompt once per chunk and combines output."""
    from unittest.mock import MagicMock

    for dep in ("llmlingua", "torch", "transformers"):
        if dep not in sys.modules:
            monkeypatch.setitem(sys.modules, dep, MagicMock())

    monkeypatch.delitem(sys.modules, "proxy", raising=False)

    import proxy as proxy
    from tests.conftest import make_mock_llmlingua, MOCK_COMPRESS_RESULT

    mock_compressor = make_mock_llmlingua()
    mock_backend = {"type": "llmlingua2", "compressor": mock_compressor, "rate": 0.5}
    monkeypatch.setattr(proxy, "backend", mock_backend)

    para = "word " * 250
    text = para.strip() + "\n\n" + para.strip()  # 2 chunks

    compressed, orig_tokens, comp_tokens = proxy._compress_llmlingua2(mock_backend, text)

    # compress_prompt should have been called once per chunk (2 total)
    assert mock_compressor.compress_prompt.call_count == 2
    # Token counts should be aggregated across both chunks
    assert orig_tokens == MOCK_COMPRESS_RESULT["origin_tokens"] * 2       # 100 * 2
    assert comp_tokens == MOCK_COMPRESS_RESULT["compressed_tokens"] * 2   # 60 * 2


def test_load_backend_llmlingua2_large(monkeypatch):
    """Task 3: load_backend with COMPRESSOR_MODEL=llmlingua2-large returns a valid backend dict."""
    from unittest.mock import MagicMock

    monkeypatch.setenv("COMPRESSOR_MODEL", "llmlingua2-large")
    monkeypatch.setenv("COMPRESS_RATE", "0.45")

    mock_cls = MagicMock()
    mock_cls.return_value.compress_prompt.return_value = {
        "compressed_prompt": "compressed x",
        "origin_tokens": 10,
        "compressed_tokens": 6,
        "ratio": 0.6,
    }

    monkeypatch.setattr("llmlingua.PromptCompressor", mock_cls)
    from proxy import load_backend
    b = load_backend()
    assert b["type"] == "llmlingua2-large"
    assert b["rate"] == 0.45


def test_tracker_all_pagination(client):
    """Task 1: GET /admin/tracker/all?page=1&page_size=10 returns paginated envelope."""
    r = client.get("/admin/tracker/all?page=1&page_size=10")
    assert r.status_code == 200
    data = r.json()
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert "pages" in data
    assert data["page"] == 1


def test_session_compressions_pagination(client):
    """Task 2: GET /session/{slug}/compressions?page=1&page_size=5 returns paginated envelope."""
    # Non-existent slug should 404
    r = client.get("/session/nosuchslug/compressions?page=1&page_size=5")
    assert r.status_code == 404

    # Create a tracker with session_id to test with real data
    from proxy import _db_conn
    tracker_slug = "test-slug-123"
    tracker_name = "Test Tracker"
    session_id = "session-123"

    _db_conn.execute(
        "INSERT INTO trackers (slug, name, status, session_id, created_at) VALUES (?, ?, ?, ?, datetime('now'))",
        (tracker_slug, tracker_name, "active", session_id),
    )

    # Insert 10 test compressions
    for i in range(10):
        _db_conn.execute(
            "INSERT INTO compressions (ts, session_id, model, original_tokens, compressed_tokens, latency_ms) VALUES (?, ?, ?, ?, ?, ?)",
            (f"2026-06-27T10:0{i}:00", session_id, "llmlingua2", 100 + i * 10, 60 + i * 5, 10.0),
        )
    _db_conn.commit()

    # Test pagination with page_size=5
    r = client.get(f"/session/{tracker_slug}/compressions?page=1&page_size=5")
    assert r.status_code == 200
    data = r.json()

    # Check response structure
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert "page_size" in data
    assert "pages" in data

    # Check pagination values
    assert data["total"] == 10
    assert data["page"] == 1
    assert data["page_size"] == 5
    assert data["pages"] == 2
    assert len(data["items"]) == 5

    # Test page 2
    r = client.get(f"/session/{tracker_slug}/compressions?page=2&page_size=5")
    assert r.status_code == 200
    data = r.json()
    assert data["page"] == 2
    assert len(data["items"]) == 5


def test_langfuse_status_disabled(client):
    resp = client.get("/admin/langfuse-status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["enabled"] is False
    assert data["public_key_set"] is False
    assert data["secret_key_set"] is False


def test_langfuse_status_enabled(client, monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test123")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test456")
    from unittest.mock import MagicMock, patch
    mock_client = MagicMock()
    with patch("langfuse.Langfuse", return_value=mock_client):
        import langfuse_tracer
        langfuse_tracer.tracer.init()
    resp = client.get("/admin/langfuse-status")
    assert resp.status_code == 200
    assert resp.json()["enabled"] is True
    # cleanup
    import langfuse_tracer
    langfuse_tracer.tracer._client = None


def test_session_rtk_commands_pagination(client):
    """Task 3: GET /session/{slug}/rtk-commands?page=1&page_size=5 returns paginated envelope."""
    # Non-existent slug should 404
    r = client.get("/session/nosuchslug/rtk-commands?page=1&page_size=5")
    assert r.status_code == 404

    # Create a tracker with session_id to test with real data
    from proxy import _db_conn
    tracker_slug = "test-rtk-slug-123"
    tracker_name = "Test RTK Tracker"
    session_id = "rtk-session-123"

    _db_conn.execute(
        "INSERT INTO trackers (slug, name, status, session_id, created_at) VALUES (?, ?, ?, ?, datetime('now'))",
        (tracker_slug, tracker_name, "active", session_id),
    )

    # Insert 10 test rtk_events
    for i in range(10):
        _db_conn.execute(
            "INSERT INTO rtk_events (ts, session_id, rtk_cmd, input_tokens, output_tokens, saved_tokens, savings_pct, project_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (f"2026-06-27T10:0{i}:00", session_id, f"rtk gain --iteration {i}", 100 + i * 10, 50 + i * 5, 25 + i * 2, 25.0 + i, f"/path/to/project{i}"),
        )
    _db_conn.commit()

    # Test pagination with page_size=5
    r = client.get(f"/session/{tracker_slug}/rtk-commands?page=1&page_size=5")
    assert r.status_code == 200
    data = r.json()

    # Check response structure
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert "page_size" in data
    assert "pages" in data

    # Check pagination values
    assert data["total"] == 10
    assert data["page"] == 1
    assert data["page_size"] == 5
    assert data["pages"] == 2
    assert len(data["items"]) == 5

    # Test page 2
    r = client.get(f"/session/{tracker_slug}/rtk-commands?page=2&page_size=5")
    assert r.status_code == 200
    data = r.json()
    assert data["page"] == 2
    assert len(data["items"]) == 5
