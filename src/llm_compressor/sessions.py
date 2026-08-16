"""Session state: provisional + auto + manual names. All fns take an explicit conn.

`record_compression` / `record_request` are the two exceptions (Task 13, Step
7): they read/write the process-wide `stats.stats` dict and `db._db_conn`
directly rather than taking a `conn` parameter, mirroring their original shape
in proxy.py so every existing call site (`compression.py`'s deferred
`sessions.record_compression(...)`, `proxy.py`'s `proxy_messages` route, and
`proxy.record_compression`/`proxy.record_request` direct calls in the test
suite) keeps working unchanged. Neither name is ever monkeypatched by the
test suite, so `proxy.py`'s `from sessions import record_request,
record_compression` re-export carries no staleness risk.
"""

import math
import os

from llm_compressor import backends
from llm_compressor import db
from llm_compressor import stats as _stats


def _now() -> str:
    """Naive ISO UTC, matching every other timestamp written to the DB."""
    return db.utc_now()


def provisional_name(session_id: str) -> str:
    return f"session-{session_id[:8]}"


def project_basename(conn, session_id: str) -> str | None:
    """Best-effort: basename of the most recent non-empty rtk_events.project_path.

    Returns None when RTK never logged a path for this session. Never raises —
    the rtk_events table/column may be absent on a fresh DB.
    """
    if conn is None or not session_id:
        return None
    try:
        row = conn.execute(
            "SELECT project_path FROM rtk_events "
            "WHERE session_id=? AND project_path != '' ORDER BY ts DESC LIMIT 1",
            (session_id,),
        ).fetchone()
    except Exception:
        return None
    if not row or not row[0]:
        return None
    return os.path.basename(str(row[0]).rstrip("/")) or None


def ensure_session(conn, session_id: str, project: str | None = None) -> None:
    if conn is None or not session_id or session_id == "unknown":
        return
    ts = _now()
    conn.execute(
        """INSERT INTO sessions (session_id, project, display_name, name_source, first_seen, last_seen)
           VALUES (?, ?, ?, 'provisional', ?, ?)
           ON CONFLICT(session_id) DO UPDATE SET last_seen=excluded.last_seen""",
        (session_id, project, provisional_name(session_id), ts, ts),
    )
    conn.commit()


def apply_auto_name(conn, session_id: str, display_name: str) -> None:
    if conn is None or not display_name:
        return
    project = project_basename(conn, session_id)
    stored = f"{project}/{display_name}" if project else display_name
    conn.execute(
        """UPDATE sessions SET display_name=?, project=COALESCE(?, project),
                             name_source='auto', last_seen=?
           WHERE session_id=? AND name_source != 'manual'""",
        (stored, project, _now(), session_id),
    )
    conn.commit()


def claim_for_naming(conn, session_id: str) -> bool:
    """Atomically move provisional -> naming. True only for the single winner."""
    if conn is None or not session_id:
        return False
    cur = conn.execute(
        "UPDATE sessions SET name_source='naming', last_seen=? "
        "WHERE session_id=? AND name_source='provisional'",
        (_now(), session_id),
    )
    conn.commit()
    return cur.rowcount == 1


def revert_naming(conn, session_id: str) -> None:
    """Move naming -> provisional so a later turn can retry. No-op otherwise."""
    if conn is None or not session_id:
        return
    conn.execute(
        "UPDATE sessions SET name_source='provisional' WHERE session_id=? AND name_source='naming'",
        (session_id,),
    )
    conn.commit()


def rename_session(conn, session_id: str, display_name: str) -> bool:
    if conn is None:
        return False
    name = (display_name or "").strip()
    if not name:
        return False
    cur = conn.execute(
        "UPDATE sessions SET display_name=?, name_source='manual', last_seen=? WHERE session_id=?",
        (name, _now(), session_id),
    )
    conn.commit()
    return cur.rowcount > 0


def get_session(conn, session_id: str) -> dict | None:
    """Look up one session, falling back to its traffic when unnamed.

    Same reason as `list_sessions`: a session id can have compressions without
    ever getting a `sessions` row, and such a session must still open rather
    than 404. Returns None only when the id is unknown to both tables.
    """
    if conn is None:
        return None
    row = conn.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
    if row:
        return dict(row)
    traffic = conn.execute(
        "SELECT COUNT(*), MIN(ts), MAX(ts) FROM compressions WHERE session_id=?",
        (session_id,),
    ).fetchone()
    if not traffic or not traffic[0]:
        return None
    return {
        "session_id": session_id,
        "project": None,
        "display_name": provisional_name(session_id),
        "name_source": "provisional",
        "first_seen": traffic[1],
        "last_seen": traffic[2],
    }


def list_sessions(conn, page: int = 1, page_size: int = 25) -> dict:
    """One page of sessions, ordered by most recent activity.

    Driven by `compressions`, LEFT JOINed to `sessions` for the name -- not the
    other way round. `sessions` is only populated by `ensure_session`, which
    was added long after traffic started flowing, so on a real install it can
    be empty while `compressions` holds thousands of rows under dozens of
    session ids. Listing from `sessions` renders a blank page in exactly the
    case the page exists to serve. A session with no row here still lists,
    under its provisional `session-<hex>` name.

    Activity aggregates come from correlated subqueries rather than extra
    JOINs so `compressions × rtk_events` cannot fan out and double-count.
    `tokens_saved` is proxy savings + RTK shell savings.
    """
    page = max(1, page)
    page_size = max(1, min(200, page_size))
    if conn is None:
        return {"items": [], "total": 0, "page": page, "page_size": page_size, "pages": 0}

    total = conn.execute(
        "SELECT COUNT(*) FROM (SELECT session_id FROM compressions "
        "UNION SELECT session_id FROM sessions)"
    ).fetchone()[0]
    offset = (page - 1) * page_size

    rows = conn.execute(
        """WITH ids AS (
               SELECT session_id FROM compressions
               UNION
               SELECT session_id FROM sessions
           )
           SELECT ids.session_id                          AS session_id,
                  s.project                               AS project,
                  s.display_name                          AS display_name,
                  COALESCE(s.name_source, 'provisional')  AS name_source,
                  COALESCE(s.first_seen,
                           (SELECT MIN(c.ts) FROM compressions c
                             WHERE c.session_id = ids.session_id)) AS first_seen,
                  COALESCE((SELECT MAX(c.ts) FROM compressions c
                             WHERE c.session_id = ids.session_id),
                           s.last_seen)                   AS last_seen,
                  COALESCE((SELECT SUM(c.original_tokens - c.compressed_tokens)
                            FROM compressions c WHERE c.session_id = ids.session_id), 0)
                  + COALESCE((SELECT SUM(r.saved_tokens)
                              FROM rtk_events r WHERE r.session_id = ids.session_id), 0)
                                                          AS tokens_saved,
                  COALESCE((SELECT COUNT(*) FROM compressions c
                            WHERE c.session_id = ids.session_id), 0) AS requests,
                  (SELECT ROUND(AVG(CAST(c.original_tokens AS REAL)
                                    / NULLIF(c.compressed_tokens, 0)), 2)
                     FROM compressions c WHERE c.session_id = ids.session_id) AS avg_ratio
           FROM ids
           LEFT JOIN sessions s ON s.session_id = ids.session_id
           WHERE ids.session_id IS NOT NULL AND ids.session_id != ''
           -- `last_seen` is TEXT, so DESC is a lexicographic sort and any value
           -- that is not a full ISO-8601 timestamp sorts by its first character.
           -- A bare time like '22:53:59' therefore beats every '2026-..' row
           -- ('2' > '0' at offset 1) and pins itself to the top of page one
           -- permanently -- while the UI renders it as "—" because it cannot be
           -- parsed. Well-formed timestamps sort first, malformed ones sink.
           ORDER BY (last_seen LIKE '____-__-__T%') DESC, last_seen DESC
           LIMIT ? OFFSET ?""",
        (page_size, offset),
    ).fetchall()

    items = []
    for r in rows:
        item = dict(r)
        if not item.get("display_name"):
            item["display_name"] = provisional_name(item["session_id"])
        items.append(item)

    pages = math.ceil(total / page_size) if page_size else 0
    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": pages,
    }


# ---------------------------------------------------------------------------
# Stats recorders (Task 13, Step 7 -- moved from proxy.py)
# ---------------------------------------------------------------------------


def record_compression(
    session_id: str,
    original: int,
    compressed: int,
    latency_ms: float = 0.0,
    original_text: str | None = None,
    compressed_text: str | None = None,
    role: str = "user",
    active_backend: dict | None = None,
    cache_hit: int = 0,
    ok: int = 1,
):
    _stats.stats["total_original_tokens"] += original
    _stats.stats["total_compressed_tokens"] += compressed

    active = active_backend if active_backend is not None else backends.backend
    model_name = active.get("type", "llmlingua2") if active else "llmlingua2"
    ts = db.utc_now()

    if db._db_conn:
        cur = db._db_conn.execute(
            "INSERT INTO compressions (ts, session_id, model, original_tokens, "
            "compressed_tokens, latency_ms, role, cache_hit, ok) VALUES (?,?,?,?,?,?,?,?,?)",
            (ts, session_id, model_name, original, compressed, latency_ms, role, cache_hit, ok),
        )
        if original_text is not None and compressed_text is not None:
            db._db_conn.execute(
                "INSERT INTO compression_texts (compression_id, original_text, compressed_text) VALUES (?,?,?)",
                (cur.lastrowid, original_text, compressed_text),
            )
        db._db_conn.commit()

    sess = _stats.stats["sessions"].setdefault(
        session_id,
        {
            "first_seen": ts,
            "requests": 0,
            "original_tokens": 0,
            "compressed_tokens": 0,
        },
    )
    sess["original_tokens"] += original
    sess["compressed_tokens"] += compressed
    sess["last_seen"] = ts

    _stats.stats["recent_compressions"].appendleft(
        {
            # Full ISO, not the bare %H:%M:%S this used to store -- the UI
            # cannot compute "how long ago" from a time with no date, and an
            # earlier build persisted this same string into compressions.ts,
            # which is where the year-2000 rows came from.
            "ts": ts,
            "session_id": session_id[:8],
            "original": original,
            "compressed": compressed,
            "saved": original - compressed,
            "latency_ms": round(latency_ms, 1),
        }
    )


def record_request(session_id: str):
    _stats.stats["total_requests"] += 1
    now = db.utc_now()
    sess = _stats.stats["sessions"].setdefault(
        session_id,
        {
            "first_seen": now,
            "requests": 0,
            "original_tokens": 0,
            "compressed_tokens": 0,
            "name": None,
        },
    )
    sess["requests"] += 1
    sess["last_seen"] = now
