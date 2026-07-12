import sqlite3
import proxy


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
