from llm_compressor import proxy
from llm_compressor import sessions as S


def _conn(tmp_path):
    return proxy.init_db(str(tmp_path / "m.db"))


def test_init_db_creates_sessions_table(tmp_path):
    conn = proxy.init_db(str(tmp_path / "m.db"))
    cols = {r[1] for r in conn.execute("PRAGMA table_info(sessions)")}
    assert cols == {
        "session_id",
        "project",
        "display_name",
        "name_source",
        "first_seen",
        "last_seen",
    }
    # name_source default is 'provisional'
    conn.execute("INSERT INTO sessions (session_id, first_seen, last_seen) VALUES ('s1','t','t')")
    row = conn.execute("SELECT name_source FROM sessions WHERE session_id='s1'").fetchone()
    assert row[0] == "provisional"


def test_trackers_table_still_present(tmp_path):
    # Re-homed from tests/test_tracker.py (Task 10): the `trackers` table and its
    # columns still exist (read by /stats' tracked totals) even though nothing
    # writes to it via the deleted pending/link CRUD flow anymore.
    conn = proxy.init_db(str(tmp_path / "m.db"))
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trackers'")
    assert cur.fetchone() is not None
    cols = [row[1] for row in conn.execute("PRAGMA table_info(trackers)")]
    for col in ("slug", "name", "status", "session_id", "created_at", "linked_at", "closed_at"):
        assert col in cols


def test_session_detail_bootstraps_the_session_record(client):
    # The page reads its session through UI.bootstrap('session-data'), which
    # parses an inert <script type="application/json"> block. The old build
    # interpolated json.dumps() into an executable <script> as window.SESSION,
    # which made every user-settable display name an XSS vector.
    from llm_compressor import proxy

    proxy._db_conn.execute(
        "INSERT INTO sessions (session_id, display_name, name_source, first_seen, last_seen)"
        " VALUES ('dash-sess-1', 'My Test', 'auto', 't', 't')"
    )
    proxy._db_conn.commit()
    r = client.get("/sessions/dash-sess-1")
    assert r.status_code == 200
    assert 'type="application/json" id="session-data"' in r.text
    assert "window.SESSION" not in r.text
    assert '"session_id": "dash-sess-1"' in r.text
    assert '"display_name": "My Test"' in r.text


def test_session_detail_escapes_a_hostile_display_name(client):
    from llm_compressor import proxy

    proxy._db_conn.execute(
        "INSERT INTO sessions (session_id, display_name, name_source, first_seen, last_seen)"
        " VALUES ('dash-sess-x', '</script><img src=x onerror=alert(1)>', 'manual', 't', 't')"
    )
    proxy._db_conn.commit()
    r = client.get("/sessions/dash-sess-x")
    assert r.status_code == 200
    assert "</script><img" not in r.text


def test_legacy_dashboard_url_redirects_to_the_session_page(client):
    from llm_compressor import proxy

    proxy._db_conn.execute(
        "INSERT INTO sessions (session_id, display_name, name_source, first_seen, last_seen)"
        " VALUES ('dash-sess-2', 'HTML Test', 'auto', 't', 't')"
    )
    proxy._db_conn.commit()

    r = client.get("/dashboard/dash-sess-2", follow_redirects=False)
    assert r.status_code == 301
    assert r.headers["location"] == "/sessions/dash-sess-2"

    r = client.get("/dashboard/dash-sess-2")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")


def test_unknown_session_is_a_404_not_a_blank_page(client):
    r = client.get("/sessions/no-such-session-id")
    assert r.status_code == 404


def test_provisional_name_is_session_hex():
    assert S.provisional_name("abcdef1234567890") == "session-abcdef12"


def test_ensure_session_creates_provisional_row(tmp_path):
    c = _conn(tmp_path)
    S.ensure_session(c, "abcdef1234567890")
    row = S.get_session(c, "abcdef1234567890")
    assert row["name_source"] == "provisional"
    assert row["display_name"] == "session-abcdef12"
    assert row["first_seen"] and row["last_seen"]


def test_ensure_session_is_idempotent_and_updates_last_seen(tmp_path):
    c = _conn(tmp_path)
    S.ensure_session(c, "s1s1s1s1s1s1")
    first = S.get_session(c, "s1s1s1s1s1s1")["first_seen"]
    S.ensure_session(c, "s1s1s1s1s1s1")
    assert S.get_session(c, "s1s1s1s1s1s1")["first_seen"] == first  # unchanged


def test_apply_auto_name_sets_auto(tmp_path):
    c = _conn(tmp_path)
    S.ensure_session(c, "s1s1s1s1s1s1")
    S.apply_auto_name(c, "s1s1s1s1s1s1", "fix-dashboard-css")
    row = S.get_session(c, "s1s1s1s1s1s1")
    assert row["display_name"] == "fix-dashboard-css"
    assert row["name_source"] == "auto"


def test_apply_auto_name_prefixes_rtk_project(tmp_path):
    c = _conn(tmp_path)
    S.ensure_session(c, "s1s1s1s1s1s1")
    c.execute(
        "INSERT INTO rtk_events (rtk_id, ts, session_id, rtk_cmd, project_path)"
        " VALUES ('r1', 't', 's1s1s1s1s1s1', 'test-cmd', '/Users/j/code/llm-compressor')"
    )
    c.commit()
    S.apply_auto_name(c, "s1s1s1s1s1s1", "fix-dashboard-css")
    row = S.get_session(c, "s1s1s1s1s1s1")
    assert row["display_name"] == "llm-compressor/fix-dashboard-css"
    assert row["project"] == "llm-compressor"
    assert row["name_source"] == "auto"


def test_apply_auto_name_no_rtk_project_is_bare_slug(tmp_path):
    c = _conn(tmp_path)
    S.ensure_session(c, "s2s2s2s2s2s2")
    S.apply_auto_name(c, "s2s2s2s2s2s2", "add-batch-endpoint")
    assert S.get_session(c, "s2s2s2s2s2s2")["display_name"] == "add-batch-endpoint"


def test_apply_auto_name_never_overwrites_manual(tmp_path):
    c = _conn(tmp_path)
    S.ensure_session(c, "s1s1s1s1s1s1")
    assert S.rename_session(c, "s1s1s1s1s1s1", "my-name") is True
    S.apply_auto_name(c, "s1s1s1s1s1s1", "auto-name")
    row = S.get_session(c, "s1s1s1s1s1s1")
    assert row["display_name"] == "my-name"
    assert row["name_source"] == "manual"


def test_claim_for_naming_is_single_winner(tmp_path):
    c = _conn(tmp_path)
    S.ensure_session(c, "s1s1s1s1s1s1")
    # First claim wins; the second (concurrent) request loses.
    assert S.claim_for_naming(c, "s1s1s1s1s1s1") is True
    assert S.claim_for_naming(c, "s1s1s1s1s1s1") is False
    assert S.get_session(c, "s1s1s1s1s1s1")["name_source"] == "naming"


def test_revert_naming_allows_retry(tmp_path):
    c = _conn(tmp_path)
    S.ensure_session(c, "s1s1s1s1s1s1")
    assert S.claim_for_naming(c, "s1s1s1s1s1s1") is True
    S.revert_naming(c, "s1s1s1s1s1s1")
    row = S.get_session(c, "s1s1s1s1s1s1")
    assert row["name_source"] == "provisional"
    # a fresh turn can claim again
    assert S.claim_for_naming(c, "s1s1s1s1s1s1") is True


def test_revert_naming_never_clobbers_auto_or_manual(tmp_path):
    c = _conn(tmp_path)
    S.ensure_session(c, "s1s1s1s1s1s1")
    S.apply_auto_name(c, "s1s1s1s1s1s1", "fix-thing")  # now 'auto'
    S.revert_naming(c, "s1s1s1s1s1s1")
    assert S.get_session(c, "s1s1s1s1s1s1")["name_source"] == "auto"


def test_rename_rejects_empty(tmp_path):
    c = _conn(tmp_path)
    S.ensure_session(c, "s1s1s1s1s1s1")
    assert S.rename_session(c, "s1s1s1s1s1s1", "   ") is False


def test_list_sessions_paginates(tmp_path):
    c = _conn(tmp_path)
    for i in range(5):
        S.ensure_session(c, f"sess{i:012d}")
    out = S.list_sessions(c, page=1, page_size=2)
    assert out["total"] == 5 and out["pages"] == 3 and len(out["items"]) == 2


def test_list_sessions_savings_combine_compressions_and_rtk(tmp_path):
    c = _conn(tmp_path)
    S.ensure_session(c, "sidZ")
    c.execute(
        "INSERT INTO compressions (ts, session_id, model, original_tokens, compressed_tokens)"
        " VALUES ('t','sidZ','llmlingua2',100,60)"
    )  # +40 saved
    c.execute(
        "INSERT INTO rtk_events (rtk_id, ts, session_id, rtk_cmd, saved_tokens)"
        " VALUES ('r1','t','sidZ','test-cmd',25)"
    )  # +25 saved
    c.commit()
    item = next(i for i in S.list_sessions(c, 1, 25)["items"] if i["session_id"] == "sidZ")
    assert item["tokens_saved"] == 65  # 40 (compression) + 25 (rtk), no fan-out
    assert item["requests"] == 1  # counts compressions only, not rtk rows


def test_messages_registers_session_provisionally(client, monkeypatch):
    # /v1/messages proxies to Anthropic; stub httpx so no network is hit.
    from llm_compressor import proxy

    class _Resp:
        status_code = 200

        def json(self):
            return {"content": [{"text": "hi"}]}

    class _Client:
        def __init__(self, *a, **k): ...
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            return _Resp()

    monkeypatch.setattr(proxy.httpx, "AsyncClient", _Client)
    r = client.post(
        "/v1/messages",
        headers={"x-claude-code-session-id": "abcdef1234567890"},
        json={
            "model": "m",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hello world this is a test"}],
        },
    )
    assert r.status_code == 200
    row = proxy._db_conn.execute(
        "SELECT display_name, name_source FROM sessions WHERE session_id='abcdef1234567890'"
    ).fetchone()
    assert row is not None
    # claim_for_naming() flips name_source: 'provisional' -> 'naming' synchronously
    # in the request path, before the fire-and-forget naming task (apply_auto_name)
    # ever runs. That background task races the test's synchronous read on
    # TestClient's separate event-loop thread, so exactly two outcomes are legit:
    #   - task hasn't applied yet: still 'naming' / the provisional slug
    #   - task already applied:    'auto' / the deterministic heuristic slug
    assert row[1] in ("naming", "auto")
    if row[1] == "naming":
        assert row[0] == "session-abcdef12"
    else:
        assert row[0] == "hello-world-test"


def test_session_compressions_keyed_by_session_id(client):
    from llm_compressor import proxy

    proxy._db_conn.execute(
        "INSERT INTO sessions (session_id, display_name, name_source, first_seen, last_seen)"
        " VALUES ('sid123', 'fix-thing', 'auto', 't', 't')"
    )
    proxy._db_conn.execute(
        "INSERT INTO compressions (ts, session_id, model, original_tokens, compressed_tokens, latency_ms)"
        " VALUES ('t','sid123','llmlingua2',100,60,1.0)"
    )
    proxy._db_conn.commit()
    r = client.get("/session/sid123/compressions")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1 and body["items"][0]["original_tokens"] == 100


def test_admin_sessions_lists_named_sessions(client):
    from llm_compressor import proxy

    proxy._db_conn.execute(
        "INSERT INTO sessions (session_id, display_name, name_source, first_seen, last_seen)"
        " VALUES ('sidA', 'add-endpoint', 'auto', 't', 't')"
    )
    proxy._db_conn.commit()
    r = client.get("/admin/sessions")
    assert r.status_code == 200
    names = [i["display_name"] for i in r.json()["items"]]
    assert "add-endpoint" in names


def test_dashboard_404_on_missing_session(client):
    # Re-homes the 404 assertion the deleted tracker test used to cover.
    r = client.get("/dashboard/no-such-session-id")
    assert r.status_code == 404


def test_rename_endpoint_sets_manual(client):
    from llm_compressor import proxy

    proxy._db_conn.execute(
        "INSERT INTO sessions (session_id, display_name, name_source, first_seen, last_seen)"
        " VALUES ('sidR', 'session-sidR', 'provisional', 't', 't')"
    )
    proxy._db_conn.commit()
    r = client.patch("/session/sidR/name", json={"name": "my custom name"})
    assert r.status_code == 200
    row = proxy._db_conn.execute(
        "SELECT display_name, name_source FROM sessions WHERE session_id='sidR'"
    ).fetchone()
    assert row[0] == "my custom name" and row[1] == "manual"


def test_rename_endpoint_rejects_empty(client):
    from llm_compressor import proxy

    proxy._db_conn.execute(
        "INSERT INTO sessions (session_id, first_seen, last_seen) VALUES ('sidE','t','t')"
    )
    proxy._db_conn.commit()
    assert client.patch("/session/sidE/name", json={"name": "  "}).status_code == 400


def test_rename_endpoint_404_missing(client):
    assert client.patch("/session/nope/name", json={"name": "x"}).status_code == 404


def test_listing_sinks_a_malformed_timestamp_below_real_ones(tmp_path):
    """A bare time must not outrank every dated row.

    `sessions.last_seen` is TEXT and the listing sorts it lexicographically, so
    '22:53:59' compares greater than '2026-08-13T19:12:36' at offset 1 ('2' >
    '0'). One such row -- and real databases have them -- pinned itself to the
    top of page one forever, above genuinely recent sessions, while the UI
    showed its Last seen as "—" because it could not be parsed.
    """
    conn = _conn(tmp_path)
    conn.executemany(
        "INSERT INTO sessions (session_id, display_name, first_seen, last_seen)"
        " VALUES (?, ?, ?, ?)",
        [
            ("bare", "bare-time", "22:01:45", "22:53:59"),
            ("older", "older", "2026-08-01T10:00:00", "2026-08-01T10:00:00"),
            ("newest", "newest", "2026-08-13T19:12:36", "2026-08-13T19:12:36"),
        ],
    )
    conn.commit()

    ids = [i["session_id"] for i in S.list_sessions(conn)["items"]]

    assert ids[0] == "newest", f"malformed timestamp captured the top of the list: {ids}"
    assert ids == ["newest", "older", "bare"]


def test_listing_uses_session_id_index(tmp_path):
    """The sessions listing must not full-scan `compressions` per session.

    Every per-session aggregate in `list_sessions` is a correlated subquery
    keyed on session_id, and the ORDER BY is a computed expression, so SQLite
    evaluates all of them for every session before LIMIT applies. Unindexed,
    that is one full table scan per subquery per session: on a real 186k-row
    install /sessions took 39s to load. Guard both the index and its use.
    """
    conn = _conn(tmp_path)
    idx = {r[1] for r in conn.execute("PRAGMA index_list(compressions)")}
    assert "idx_compressions_session" in idx

    plan = "\n".join(
        str(r[-1])
        for r in conn.execute(
            "EXPLAIN QUERY PLAN SELECT COUNT(*) FROM compressions c WHERE c.session_id = 's1'"
        )
    )
    assert "idx_compressions_session" in plan, plan
