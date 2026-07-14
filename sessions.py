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
from datetime import datetime, timezone

import backends
import db
import stats as _stats


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
        "UPDATE sessions SET name_source='provisional' "
        "WHERE session_id=? AND name_source='naming'",
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
    if conn is None:
        return None
    row = conn.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
    return dict(row) if row else None


def list_sessions(conn, page: int = 1, page_size: int = 25) -> dict:
    page = max(1, page)
    page_size = max(1, min(200, page_size))
    if conn is None:
        return {"items": [], "total": 0, "page": page, "page_size": page_size, "pages": 0}
    total = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    offset = (page - 1) * page_size
    # Correlated subqueries (not multi-JOIN) so compressions × rtk_events do not
    # fan out and double-count. tokens_saved = proxy compression savings + RTK
    # savings, matching the spec ripple table (sessions ⋈ compressions/rtk_events).
    rows = conn.execute(
        """SELECT s.session_id, s.project, s.display_name, s.name_source,
                  s.first_seen, s.last_seen,
                  COALESCE((SELECT SUM(c.original_tokens - c.compressed_tokens)
                            FROM compressions c WHERE c.session_id = s.session_id), 0)
                  + COALESCE((SELECT SUM(r.saved_tokens)
                              FROM rtk_events r WHERE r.session_id = s.session_id), 0)
                    AS tokens_saved,
                  COALESCE((SELECT COUNT(*) FROM compressions c
                            WHERE c.session_id = s.session_id), 0) AS requests
           FROM sessions s
           ORDER BY s.last_seen DESC
           LIMIT ? OFFSET ?""",
        (page_size, offset),
    ).fetchall()
    pages = math.ceil(total / page_size) if page_size else 0
    return {"items": [dict(r) for r in rows], "total": total,
            "page": page, "page_size": page_size, "pages": pages}


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
):
    _stats.stats["total_original_tokens"] += original
    _stats.stats["total_compressed_tokens"] += compressed

    active = active_backend if active_backend is not None else backends.backend
    model_name = active.get("type", "llmlingua2") if active else "llmlingua2"
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if db._db_conn:
        cur = db._db_conn.execute(
            "INSERT INTO compressions (ts, session_id, model, original_tokens, compressed_tokens, latency_ms, role, cache_hit) VALUES (?,?,?,?,?,?,?,?)",
            (ts, session_id, model_name, original, compressed, latency_ms, role, cache_hit),
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
            "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            "session_id": session_id[:8],
            "original": original,
            "compressed": compressed,
            "saved": original - compressed,
            "latency_ms": round(latency_ms, 1),
        }
    )


def record_request(session_id: str):
    _stats.stats["total_requests"] += 1
    sess = _stats.stats["sessions"].setdefault(
        session_id,
        {
            "first_seen": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "requests": 0,
            "original_tokens": 0,
            "compressed_tokens": 0,
            "name": None,
        },
    )
    sess["requests"] += 1
    sess["last_seen"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
