import sqlite3
import proxy
import sessions as S


def _conn(tmp_path):
    return proxy.init_db(str(tmp_path / "m.db"))


def test_init_db_creates_sessions_table(tmp_path):
    conn = proxy.init_db(str(tmp_path / "m.db"))
    cols = {r[1] for r in conn.execute("PRAGMA table_info(sessions)")}
    assert cols == {
        "session_id", "project", "display_name",
        "name_source", "first_seen", "last_seen",
    }
    # name_source default is 'provisional'
    conn.execute(
        "INSERT INTO sessions (session_id, first_seen, last_seen) VALUES ('s1','t','t')"
    )
    row = conn.execute("SELECT name_source FROM sessions WHERE session_id='s1'").fetchone()
    assert row[0] == "provisional"


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
    assert item["requests"] == 1       # counts compressions only, not rtk rows
