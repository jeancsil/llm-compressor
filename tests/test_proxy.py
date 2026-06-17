import sys
import json
from collections import deque

from starlette.testclient import TestClient


def test_health(client: TestClient):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_init_db_creates_table(tmp_path):
    from llmlingua_proxy import init_db
    conn = init_db(str(tmp_path / "metrics.db"))
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='compressions'")
    assert cur.fetchone() is not None
    cols = [row[1] for row in conn.execute("PRAGMA table_info(compressions)")]
    for col in ("id", "ts", "session_id", "model", "original_tokens", "compressed_tokens", "latency_ms"):
        assert col in cols
    conn.close()


def test_migrate_imports_rows(tmp_path):
    from llmlingua_proxy import init_db, migrate_from_json
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
    from llmlingua_proxy import init_db, migrate_from_json
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
    from llmlingua_proxy import init_db, migrate_from_json
    conn = init_db(str(tmp_path / "metrics.db"))
    migrate_from_json(conn, json_path=str(tmp_path / "nonexistent.json"))
    conn.close()


def test_load_stats_from_db(tmp_path):
    from llmlingua_proxy import init_db, load_stats_from_db, stats
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


def test_record_compression_writes_to_db(tmp_path):
    """Task 7: record_compression writes a row to SQLite including latency_ms."""
    for dep in ("llmlingua", "torch", "transformers"):
        if dep not in sys.modules:
            import unittest.mock as _mock
            sys.modules[dep] = _mock.MagicMock()

    import llmlingua_proxy as proxy
    from llmlingua_proxy import init_db

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

    monkeypatch.delitem(sys.modules, "llmlingua_proxy", raising=False)

    import llmlingua_proxy as proxy
    from llmlingua_proxy import init_db
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
