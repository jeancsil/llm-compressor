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


def test_create_tracker(client: TestClient):
    r = client.post("/admin/tracker", json={"name": "Kompress Run 1"})
    assert r.status_code == 200
    data = r.json()
    assert data["slug"] == "kompress-run-1"
    assert data["status"] == "pending"
    assert data["session_id"] is None


def test_create_tracker_requires_name(client: TestClient):
    r = client.post("/admin/tracker", json={"name": ""})
    assert r.status_code == 400


def test_create_tracker_409_when_pending_exists(client: TestClient):
    client.post("/admin/tracker", json={"name": "First"})
    r = client.post("/admin/tracker", json={"name": "Second"})
    assert r.status_code == 409


def test_get_tracker_null_when_none(client: TestClient):
    r = client.get("/admin/tracker")
    assert r.status_code == 200
    assert r.json() is None


def test_get_tracker_returns_pending(client: TestClient):
    client.post("/admin/tracker", json={"name": "My Track"})
    r = client.get("/admin/tracker")
    assert r.status_code == 200
    assert r.json()["status"] == "pending"
    assert r.json()["name"] == "My Track"


def test_delete_tracker(client: TestClient):
    client.post("/admin/tracker", json={"name": "To Delete"})
    r = client.delete("/admin/tracker/to-delete")
    assert r.status_code == 200
    assert client.get("/admin/tracker").json() is None


def test_delete_tracker_404(client: TestClient):
    r = client.delete("/admin/tracker/nonexistent")
    assert r.status_code == 404


def test_auto_link_pending_tracker(client: TestClient):
    import sys
    proxy = sys.modules["llmlingua_proxy"]

    # Create a pending tracker
    client.post("/admin/tracker", json={"name": "Auto Link Test"})
    assert client.get("/admin/tracker").json()["status"] == "pending"

    # Simulate a session request arriving
    proxy.record_request("session-abc-123")

    tracker = client.get("/admin/tracker").json()
    assert tracker["status"] == "active"
    assert tracker["session_id"] == "session-abc-123"
    assert tracker["linked_at"] is not None


def test_no_tracker_record_request_is_safe(client: TestClient):
    import sys
    proxy = sys.modules["llmlingua_proxy"]
    # No tracker exists — should not raise
    proxy.record_request("session-xyz")
    assert client.get("/admin/tracker").json() is None
