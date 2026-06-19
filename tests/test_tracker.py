import re
import pytest
from starlette.testclient import TestClient


def test_init_db_creates_trackers_table(tmp_path):
    from llmlingua_proxy import init_db
    conn = init_db(str(tmp_path / "metrics.db"))
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='trackers'"
    )
    assert cur.fetchone() is not None
    cols = [row[1] for row in conn.execute("PRAGMA table_info(trackers)")]
    for col in ("slug", "name", "status", "session_id", "created_at", "linked_at"):
        assert col in cols
    conn.close()


def test_make_slug_basic(tmp_path):
    from llmlingua_proxy import init_db, make_slug
    conn = init_db(str(tmp_path / "metrics.db"))
    assert make_slug("Kompress Test 1", conn) == "kompress-test-1"
    assert make_slug("hello world!", conn) == "hello-world"
    conn.close()


def test_make_slug_collision(tmp_path):
    from llmlingua_proxy import init_db, make_slug
    conn = init_db(str(tmp_path / "metrics.db"))
    conn.execute(
        "INSERT INTO trackers (slug, name, status, created_at) VALUES ('my-test','My Test','pending','2026-01-01T00:00:00')"
    )
    conn.commit()
    slug = make_slug("my test", conn)
    assert slug == "my-test-2"
    conn.close()
