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
    for col in ("slug", "name", "status", "session_id", "created_at", "linked_at", "closed_at"):
        assert col in cols
    conn.close()


def test_make_slug_basic(tmp_path):
    from llmlingua_proxy import init_db, make_slug
    conn = init_db(str(tmp_path / "metrics.db"))
    s1 = make_slug("Kompress Test 1", conn)
    assert s1.startswith("kompress-test-1-")
    assert len(s1.split("-")[-1]) == 4
    s2 = make_slug("hello world!", conn)
    assert s2.startswith("hello-world-")
    conn.close()


def test_make_slug_no_collision(tmp_path):
    from llmlingua_proxy import init_db, make_slug
    conn = init_db(str(tmp_path / "metrics.db"))
    s1 = make_slug("my test", conn)
    s2 = make_slug("my test", conn)
    assert s1 != s2
    conn.close()


def test_make_slug_hex_suffix(tmp_path):
    from llmlingua_proxy import init_db, make_slug
    conn = init_db(str(tmp_path / "metrics.db"))
    slug = make_slug("My Task", conn)
    suffix = slug.split("-")[-1]
    assert len(suffix) == 4
    assert all(c in "0123456789abcdef" for c in suffix)
    conn.close()


def test_create_tracker(client: TestClient):
    r = client.post("/admin/tracker", json={"name": "Kompress Run 1"})
    assert r.status_code == 200
    data = r.json()
    assert data["slug"].startswith("kompress-run-1-")
    assert data["status"] == "pending"
    assert data["session_id"] is None


def test_create_tracker_requires_name(client: TestClient):
    r = client.post("/admin/tracker", json={"name": ""})
    assert r.status_code == 400


def test_create_multiple_trackers(client: TestClient):
    r1 = client.post("/admin/tracker", json={"name": "First"})
    r2 = client.post("/admin/tracker", json={"name": "Second"})
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["slug"] != r2.json()["slug"]


def test_get_tracker_empty_when_none(client: TestClient):
    r = client.get("/admin/tracker")
    assert r.status_code == 200
    assert r.json() == []


def test_get_tracker_returns_pending(client: TestClient):
    client.post("/admin/tracker", json={"name": "My Track"})
    r = client.get("/admin/tracker")
    assert r.status_code == 200
    trackers = r.json()
    assert len(trackers) == 1
    assert trackers[0]["status"] == "pending"
    assert trackers[0]["name"] == "My Track"


def test_soft_close_keeps_row(client: TestClient):
    import sys
    r = client.post("/admin/tracker", json={"name": "soft test"})
    slug = r.json()["slug"]
    resp = client.delete(f"/admin/tracker/{slug}")
    assert resp.status_code == 200
    assert resp.json() == {"closed": slug}
    conn = sys.modules["llmlingua_proxy"]._db_conn
    row = conn.execute(
        "SELECT status, closed_at FROM trackers WHERE slug=?", (slug,)
    ).fetchone()
    assert row is not None
    assert row[0] == "closed"
    assert row[1] is not None


def test_chip_list_excludes_closed(client: TestClient):
    r = client.post("/admin/tracker", json={"name": "chip test"})
    slug = r.json()["slug"]
    client.delete(f"/admin/tracker/{slug}")
    chips = client.get("/admin/tracker").json()
    assert all(t["slug"] != slug for t in chips)
    assert chips == []


def test_delete_tracker_404(client: TestClient):
    r = client.delete("/admin/tracker/nonexistent")
    assert r.status_code == 404


def test_auto_link_pending_tracker(client: TestClient):
    import sys
    proxy = sys.modules["llmlingua_proxy"]

    client.post("/admin/tracker", json={"name": "Auto Link Test"})
    trackers = client.get("/admin/tracker").json()
    assert len(trackers) == 1
    assert trackers[0]["status"] == "pending"

    proxy.record_request("session-abc-123")

    trackers = client.get("/admin/tracker").json()
    assert len(trackers) == 1
    assert trackers[0]["status"] == "active"
    assert trackers[0]["session_id"] == "session-abc-123"
    assert trackers[0]["linked_at"] is not None


def test_no_tracker_record_request_is_safe(client: TestClient):
    import sys
    proxy = sys.modules["llmlingua_proxy"]
    proxy.record_request("session-xyz")
    assert client.get("/admin/tracker").json() == []


def test_all_trackers_includes_closed(client: TestClient):
    r = client.post("/admin/tracker", json={"name": "all test"})
    slug = r.json()["slug"]
    client.delete(f"/admin/tracker/{slug}")
    data = client.get("/admin/tracker/all").json()
    assert any(t["slug"] == slug and t["status"] == "closed" for t in data)


def test_all_trackers_has_token_fields(client: TestClient):
    client.post("/admin/tracker", json={"name": "token test"})
    data = client.get("/admin/tracker/all").json()
    assert len(data) == 1
    assert "tokens_saved" in data[0]
    assert "requests" in data[0]


def test_play_list_returns_html(client: TestClient):
    r = client.get("/play/list")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "Session History" in r.text


def test_stats_has_tracked_key(client: TestClient):
    data = client.get("/stats").json()
    assert "tracked" in data
    assert "sessions" in data["tracked"]
    assert "tokens_saved" in data["tracked"]


def test_stats_session_filter(client: TestClient):
    import sys
    proxy = sys.modules["llmlingua_proxy"]

    proxy.record_compression("session-A", 500, 300, 50.0)
    proxy.record_compression("session-B", 400, 200, 40.0)

    r = client.get("/stats?session_id=session-A")
    assert r.status_code == 200
    data = r.json()
    assert data["alltime"]["requests"] == 1
    assert data["alltime"]["tokens_saved"] == 200


def test_stats_no_filter_returns_all(client: TestClient):
    import sys
    proxy = sys.modules["llmlingua_proxy"]

    proxy.record_compression("session-A", 500, 300, 50.0)
    proxy.record_compression("session-B", 400, 200, 40.0)

    r = client.get("/stats")
    data = r.json()
    assert data["alltime"]["requests"] == 2


def test_timeseries_session_filter(client: TestClient):
    import sys
    proxy = sys.modules["llmlingua_proxy"]

    proxy.record_compression("session-A", 500, 300, 50.0)
    proxy.record_compression("session-B", 400, 200, 40.0)

    r = client.get("/stats/timeseries?session_id=session-A")
    assert r.status_code == 200
    buckets = r.json()
    total_reqs = sum(b["requests"] for b in buckets)
    assert total_reqs == 1


def test_session_dashboard_404_for_unknown_slug(client: TestClient):
    r = client.get("/dashboard/does-not-exist")
    assert r.status_code == 404


def test_session_dashboard_injects_tracker(client: TestClient):
    r = client.post("/admin/tracker", json={"name": "My Test"})
    slug = r.json()["slug"]
    r = client.get(f"/dashboard/{slug}")
    assert r.status_code == 200
    assert "const TRACKER" in r.text
    assert f'"slug": "{slug}"' in r.text
    assert '"status": "pending"' in r.text


def test_dashboard_slug_returns_html(client: TestClient):
    r = client.post("/admin/tracker", json={"name": "HTML Test"})
    slug = r.json()["slug"]
    r = client.get(f"/dashboard/{slug}")
    assert r.headers["content-type"].startswith("text/html")


def test_session_dashboard_accessible_after_close(client: TestClient):
    r = client.post("/admin/tracker", json={"name": "Keep After Close"})
    slug = r.json()["slug"]
    client.delete(f"/admin/tracker/{slug}")
    r = client.get(f"/dashboard/{slug}")
    assert r.status_code == 200
    assert '"status": "closed"' in r.text
