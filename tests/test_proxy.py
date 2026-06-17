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
