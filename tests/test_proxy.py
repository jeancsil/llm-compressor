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
