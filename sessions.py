"""Session state: provisional + auto + manual names. All fns take an explicit conn."""
import math
import os
from datetime import datetime, timezone


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
