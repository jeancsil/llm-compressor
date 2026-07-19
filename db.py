"""SQLite schema, connection lifecycle, and one-time migration helpers.

Owns the process-wide, runtime-reassigned globals `_db_conn` and `DB_PATH`.
Everywhere else in the codebase (and in proxy.py, which re-exports these for
backward-compatible test patching via a forwarding shim) must read/write them
as `db._db_conn` / `db.DB_PATH` — never `from db import _db_conn` — so that
`lifespan`'s reassignment at startup, and `monkeypatch.setattr` in tests, are
visible to every reader. See the `_ProxyModule` shim + `_FORWARD` table in
proxy.py for the mechanism that keeps `monkeypatch.setattr(proxy, "_db_conn",
...)` (the pre-split test idiom) working without every test needing to be
rewritten to target `db.` directly.

NOTE (deliberate deviation from the Task 13 brief): the brief suggested
deleting the bodies of `migrate_from_json` / `recover_stats_from_backup` as
"cruft" once moved here. They are kept intact instead — `tests/test_proxy.py`
and `tests/test_coverage.py` have standalone tests that exercise the real
migration behavior (not just patch it away), and gutting the bodies would
require deleting those tests too. That's a product/test-coverage decision
beyond a mechanical extraction, so it's flagged here and in the task report
rather than done silently.
"""

import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

# Mutable, runtime-reassigned by lifespan() in proxy.py; see module docstring.
_db_conn = None


def _rtk_data_dir() -> Path:
    """Platform-specific directory where both history.db and metrics.db live."""
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "rtk"
    if system == "Windows":
        return Path(os.environ.get("APPDATA", Path.home())) / "rtk"
    return Path.home() / ".local" / "share" / "rtk"


def _default_db_path() -> Path:
    d = _rtk_data_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / "metrics.db"


DB_PATH = Path(os.environ.get("LLM_COMPRESSOR_DB") or _default_db_path())


def init_db(path: str):
    import sqlite3 as _sqlite3

    conn = _sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = _sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS compressions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            ts               TEXT,
            session_id       TEXT,
            model            TEXT,
            original_tokens  INTEGER,
            compressed_tokens INTEGER,
            latency_ms       REAL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON compressions(ts)")
    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS compression_texts (
            compression_id INTEGER PRIMARY KEY REFERENCES compressions(id),
            original_text  TEXT NOT NULL,
            compressed_text TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trackers (
            slug        TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'pending',
            session_id  TEXT,
            created_at  TEXT NOT NULL,
            linked_at   TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rtk_events (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            rtk_id        INTEGER UNIQUE,
            ts            TEXT NOT NULL,
            session_id    TEXT NOT NULL,
            rtk_cmd       TEXT NOT NULL,
            input_tokens  INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            saved_tokens  INTEGER NOT NULL DEFAULT 0,
            savings_pct   REAL    NOT NULL DEFAULT 0.0
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rtk_events_session ON rtk_events(session_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rtk_events_ts      ON rtk_events(ts)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS compression_cache (
            key               TEXT PRIMARY KEY,
            model             TEXT NOT NULL,
            rate              REAL NOT NULL,
            compressed_text   TEXT NOT NULL,
            original_tokens   INTEGER NOT NULL,
            compressed_tokens INTEGER NOT NULL,
            created_at        TEXT NOT NULL,
            hit_count         INTEGER NOT NULL DEFAULT 0,
            last_hit          TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_last_hit ON compression_cache(last_hit)")
    try:
        conn.execute("ALTER TABLE trackers ADD COLUMN closed_at TEXT")
    except Exception:
        pass  # column already exists
    try:
        conn.execute("ALTER TABLE compressions ADD COLUMN role TEXT DEFAULT 'user'")
    except Exception:
        pass  # column already exists
    try:
        conn.execute("ALTER TABLE compressions ADD COLUMN cache_hit INTEGER DEFAULT 0")
    except Exception:
        pass  # column already exists
    try:
        conn.execute("ALTER TABLE rtk_events ADD COLUMN project_path TEXT DEFAULT ''")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rtk_events_project ON rtk_events(project_path)"
        )
    except Exception:
        pass  # column already exists
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id   TEXT PRIMARY KEY,
            project      TEXT,
            display_name TEXT,
            name_source  TEXT DEFAULT 'provisional',
            first_seen   TEXT,
            last_seen    TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_last_seen ON sessions(last_seen)")
    # Mark when caching went live so hit-ratio stats can exclude the pre-feature
    # backlog of misses. Set once, never overwritten (INSERT OR IGNORE).
    conn.execute(
        "INSERT OR IGNORE INTO meta (key, value) VALUES ('cache_since', ?)",
        (datetime.now(timezone.utc).isoformat(timespec="seconds"),),
    )
    conn.commit()
    return conn


def migrate_from_json(conn, json_path: str = "stats.json") -> None:
    path = Path(json_path)
    if not path.exists():
        return
    existing = conn.execute("SELECT COUNT(*) FROM compressions").fetchone()[0]
    if existing:
        return
    try:
        import shutil

        data = json.loads(path.read_text())
        rows = data.get("recent_compressions", [])
        bak = path.with_suffix(".json.bak")
        shutil.copy2(path, bak)
        print(f"[migration] Backed {path} → {bak}")
        conn.executemany(
            "INSERT INTO compressions (ts, session_id, model, original_tokens, compressed_tokens, latency_ms) VALUES (?,?,'llmlingua2',?,?,0.0)",
            [
                (
                    datetime.now().strftime("%Y-%m-%dT%H:%M:%S") if not r.get("ts") else r["ts"],
                    r.get("session_id", ""),
                    r.get("original", 0),
                    r.get("compressed", 0),
                )
                for r in rows
            ],
        )
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM compressions").fetchone()[0]
        print(f"[migration] Imported {count} rows {path} → metrics.db. Backup {bak}.")
    except Exception as e:
        print(f"[migration] Failed, skipping: {e}")


def recover_stats_from_backup(conn, bak_path: str = "stats.json.bak") -> None:
    """Import full session history from stats.json.bak.

    The initial migration only captured recent_compressions (≤100 rows). This inserts
    one residual synthetic row per session for the token delta not yet in the DB,
    then stores a legacy_request_offset so load_stats_from_db produces the correct total.
    """
    path = Path(bak_path)
    if not path.exists():
        return
    if conn.execute("SELECT value FROM meta WHERE key='backup_recovered'").fetchone():
        return
    try:
        data = json.loads(path.read_text())
        sessions = data.get("sessions", {})
        rows_inserted = 0
        for session_id, sess in sessions.items():
            existing = conn.execute(
                "SELECT COALESCE(SUM(original_tokens),0), COALESCE(SUM(compressed_tokens),0) "
                "FROM compressions WHERE session_id=?",
                (session_id,),
            ).fetchone()
            remaining_orig = int(sess.get("original_tokens", 0)) - int(existing[0])
            remaining_comp = int(sess.get("compressed_tokens", 0)) - int(existing[1])
            if remaining_orig > 0:
                ts = (
                    sess.get("last_seen") or sess.get("first_seen") or datetime.now().isoformat()
                )[:19]
                conn.execute(
                    "INSERT INTO compressions (ts, session_id, model, original_tokens, compressed_tokens, latency_ms) "
                    "VALUES (?,?,'llmlingua2',?,?,0.0)",
                    (ts, session_id, remaining_orig, max(0, remaining_comp)),
                )
                rows_inserted += 1

        db_rows = conn.execute("SELECT COUNT(*) FROM compressions").fetchone()[0]
        total_requests = int(data.get("total_requests", 0))
        legacy_offset = max(0, total_requests - db_rows)
        conn.execute(
            "INSERT OR REPLACE INTO meta VALUES ('legacy_request_offset', ?)", (str(legacy_offset),)
        )
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('backup_recovered', '1')")
        conn.commit()
        print(
            f"[recovery] {rows_inserted} synthetic rows from {bak_path}. Request offset: {legacy_offset}."
        )
    except Exception as e:
        print(f"[recovery] Failed: {e}")


def load_stats_from_db(conn) -> None:
    # `stats` (the in-memory aggregate dict) is still owned by proxy.py until
    # Task 13 Step 6 moves it into stats.py. Reach back via a local import
    # (avoids a circular top-level import, since proxy.py imports this module)
    # and mutate the same dict object proxy.stats already points at.
    import proxy as _proxy

    stats = _proxy.stats

    row = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(original_tokens),0), COALESCE(SUM(compressed_tokens),0) FROM compressions"
    ).fetchone()
    try:
        offset_row = conn.execute(
            "SELECT value FROM meta WHERE key='legacy_request_offset'"
        ).fetchone()
        legacy_offset = int(offset_row[0]) if offset_row else 0
    except Exception:
        legacy_offset = 0
    stats["total_requests"] = row[0] + legacy_offset
    stats["total_original_tokens"] = row[1]
    stats["total_compressed_tokens"] = row[2]

    for r in conn.execute(
        "SELECT session_id, COUNT(*), SUM(original_tokens), SUM(compressed_tokens), MIN(ts), MAX(ts) FROM compressions GROUP BY session_id"
    ):
        stats["sessions"][r[0]] = {
            "requests": r[1],
            "original_tokens": r[2],
            "compressed_tokens": r[3],
            "first_seen": r[4],
            "last_seen": r[5],
            "name": None,
        }

    for r in conn.execute(
        "SELECT ts, session_id, original_tokens, compressed_tokens, latency_ms FROM compressions ORDER BY id DESC LIMIT 100"
    ):
        saved = r[2] - r[3]
        stats["recent_compressions"].append(
            {
                "ts": r[0][11:19],
                "session_id": r[1][:8],
                "original": r[2],
                "compressed": r[3],
                "saved": saved,
                "latency_ms": r[4],
            }
        )

    print(
        f"[stats] Loaded from metrics.db: "
        f"{stats['total_original_tokens']} original, {stats['total_compressed_tokens']} compressed, "
        f"{len(stats['sessions'])} sessions"
    )


# ---------------------------------------------------------------------------
# DB location migration (legacy ./metrics.db → RTK data dir)
# ---------------------------------------------------------------------------


def _migrate_db_location() -> None:
    """Copy metrics.db from the old CWD location to the RTK data directory once.

    Skipped when: old doesn't exist, paths are the same, or new already has data
    (size > 64 KiB means it was populated, not just an empty shell created by a
    previous aborted startup).
    """
    old = Path("metrics.db").resolve()
    new = DB_PATH.resolve()
    if old == new or not old.exists():
        return
    if new.exists() and new.stat().st_size > 65536:
        return  # new DB already has real data
    import shutil

    try:
        shutil.copy2(str(old), str(new))
        print(f"[db] migrated {old} → {new}")
        old.rename(old.with_suffix(".db.migrated"))
    except Exception as e:
        print(f"[db] migration failed, using {old}: {e}")
