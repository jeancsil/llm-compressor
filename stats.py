"""In-memory stats aggregate, rtk integration, and the /stats helper functions.

Owns the process-wide in-memory `stats` dict (started_at, running totals,
per-session totals, and the recent-compressions ring buffer) and `_rtk_db_path`
(the only two names in this module that appear in the 17-name monkeypatch
enumeration from Task 12's split audit). Unlike `_cache`/`backend` in
compression.py/backends.py, `stats` is never *reassigned* wholesale by
`lifespan` or by tests -- it is only ever mutated in place (`stats["x"] = ...`,
`stats["sessions"].setdefault(...)`) -- so `proxy.py` can carry it as a plain
`from stats import stats  # re-export` without the staleness risk that applies
to the reassigned globals. `_rtk_db_path`, by contrast, *is* monkeypatched
directly (`monkeypatch.setattr(proxy, "_rtk_db_path", ...)` in
test_coverage.py), so it must be read module-qualified (`stats._rtk_db_path()`)
by every caller -- including `read_rtk_stats`, its only same-module caller --
and forwarded through the `_ProxyModule` shim + `_FORWARD` table in proxy.py
rather than statically imported.

`record_compression` / `record_request` still live in proxy.py as of this
module's introduction (Task 13, Step 6); they move to sessions.py in Step 7,
at which point they start writing to `stats.stats[...]` (qualified, via
`import stats`) instead of the bare `stats[...]` they use today.

The `_compressor_info` / `_stats_scope` / `_aggregate_stats` /
`_recent_compression_rows` / `_rtk_stats` / `_empty_cache_stats` /
`_cache_stats` / `_tracked_stats` / `_merge_rtk_into_sessions` helpers are
plain functions with no mutable-global risk of their own -- they only *read*
`db._db_conn` and `backends.backend`/`backends.dual_mode`/etc., already
module-qualified from the db.py/backends.py extractions. None of them is
monkeypatched by name in the test suite (only `get_stats()`, the route that
calls them, is exercised end to end via the test client), so `proxy.py` does
not need to re-export most of them -- `read_rtk_stats` and `_cache_stats` are
the two exceptions, called directly as `proxy.read_rtk_stats(...)` /
`proxy._cache_stats()` in test_coverage.py/test_cache.py, but never
monkeypatched, so a plain re-export (like db.py's `init_db`) is safe for those
two.
"""

import sqlite3
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import backends
import db

# ---------------------------------------------------------------------------
# In-memory stats aggregate
# ---------------------------------------------------------------------------

stats = {
    "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "total_requests": 0,
    "total_original_tokens": 0,
    "total_compressed_tokens": 0,
    "sessions": {},
    "recent_compressions": deque(maxlen=100),
}


# ---------------------------------------------------------------------------
# rtk integration (optional -- gracefully absent when rtk not installed)
# ---------------------------------------------------------------------------


def _rtk_db_path() -> Path:
    return db._rtk_data_dir() / "history.db"


def read_rtk_stats(since: str | None = None) -> dict | None:
    rtk_db = _rtk_db_path()
    if not rtk_db.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{rtk_db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        where = "WHERE timestamp >= ?" if since else ""
        args = (since,) if since else ()

        row = cur.execute(
            f"SELECT COUNT(*) as n, SUM(input_tokens) as inp, "
            f"SUM(output_tokens) as out, SUM(saved_tokens) as saved, "
            f"AVG(savings_pct) as avg_pct FROM commands {where}",
            args,
        ).fetchone()

        top = cur.execute(
            f"SELECT rtk_cmd, COUNT(*) as cnt, SUM(saved_tokens) as saved, "
            f"AVG(savings_pct) as avg_pct FROM commands {where} "
            f"GROUP BY rtk_cmd ORDER BY saved DESC LIMIT 8",
            args,
        ).fetchall()

        conn.close()
        return {
            "total_commands": row["n"] or 0,
            "total_input_tokens": row["inp"] or 0,
            "total_output_tokens": row["out"] or 0,
            "total_saved_tokens": row["saved"] or 0,
            "avg_savings_pct": round(row["avg_pct"] or 0, 1),
            "top_commands": [
                {
                    "cmd": r["rtk_cmd"],
                    "count": r["cnt"],
                    "saved": r["saved"],
                    "avg_pct": round(r["avg_pct"], 1),
                }
                for r in top
            ],
        }
    except Exception as e:
        print(f"[rtk] could not read tracking db: {e}")
        return None


# ---------------------------------------------------------------------------
# /stats helpers
#
# get_stats() (proxy.py) is an orchestrator; these pull the per-section logic
# out so each query lives in one place. In particular, today/alltime/recent
# all share the same model/session scoping, so it is defined once in
# _stats_scope().
# ---------------------------------------------------------------------------


def _compressor_info() -> dict:
    """Describe the active (or loading) compression backend for the dashboard."""
    if backends.backend_loading:
        return {
            "model": backends.backend_loading,
            "param_name": "",
            "param_value": "",
            "loading": True,
        }
    if backends.backend and backends.backend.get("type") == "dual":
        return {
            "model": "dual",
            "loading": False,
            "param_name": None,
            "param_value": None,
            "model_system": backends.dual_model_system,
            "model_user": backends.dual_model_user,
        }
    if backends.backend and backends.backend.get("type") == "kompress":
        return {
            "model": "kompress",
            "param_name": "threshold",
            "param_value": backends.backend.get("threshold", 0.5),
            "loading": False,
        }
    backend_key = (
        backends.backend.get("backend_key", "llmlingua2") if backends.backend else "llmlingua2"
    )
    return {
        "model": backend_key,
        "param_name": "rate",
        "param_value": backends.backend.get("rate", 0.5) if backends.backend else 0.5,
        "loading": False,
    }


def _stats_scope(active_model: str, session_id: str | None) -> tuple[str, tuple]:
    """WHERE fragment + args selecting compression rows for the active scope.

    A session filter wins; otherwise dual mode spans all sub-model names while a
    single model matches just itself.
    """
    if session_id:
        return "session_id = ?", (session_id,)
    if active_model == "dual":
        return (
            f"model IN ({', '.join('?' * len(backends.DUAL_SUBMODELS))})",
            backends.DUAL_SUBMODELS,
        )
    return "model = ?", (active_model,)


def _aggregate_stats(scope: str, args: tuple, *, today: bool, with_ratio: bool) -> dict:
    """Aggregate request/savings/latency metrics for a scope, optionally today-only."""
    date_clause = "date(ts) = date('now') AND " if today else ""
    ratio_col = (
        ", ROUND(AVG(CAST(original_tokens AS REAL) / NULLIF(compressed_tokens, 0)), 2) AS avg_ratio"
        if with_ratio
        else ""
    )
    row = db._db_conn.execute(
        f"""
        SELECT COUNT(*) AS requests,
               COALESCE(SUM(original_tokens - compressed_tokens), 0) AS tokens_saved,
               ROUND(AVG((original_tokens - compressed_tokens) * 100.0 / original_tokens), 1) AS avg_savings_pct,
               ROUND(AVG(latency_ms), 1) AS avg_latency_ms,
               COUNT(DISTINCT session_id) AS sessions{ratio_col}
        FROM compressions
        WHERE {date_clause}{scope}
        """,
        args,
    ).fetchone()
    out = {
        "requests": (row[0] if row else 0) or 0,
        "tokens_saved": (row[1] if row else 0) or 0,
        "avg_savings_pct": (row[2] if row else 0.0) or 0.0,
        "avg_latency_ms": (row[3] if row else 0.0) or 0.0,
        "sessions": (row[4] if row else 0) or 0,
    }
    if with_ratio:
        out["avg_ratio"] = (row[5] if row else 0.0) or 0.0
    return out


def _recent_compression_rows(active_model: str, session_id: str | None) -> list:
    """Most recent 20 compression rows for the active scope.

    Uses the same scoping as the today/alltime panels, so dual mode spans all
    sub-model rows rather than matching the non-existent model='dual'.
    """
    scope, args = _stats_scope(active_model, session_id)
    rows = db._db_conn.execute(
        f"""
        SELECT ts, session_id, model, original_tokens, compressed_tokens,
               ROUND((original_tokens - compressed_tokens) * 100.0 / original_tokens, 1) AS savings_pct,
               latency_ms, role
        FROM compressions
        WHERE {scope}
        ORDER BY id DESC
        LIMIT 20
        """,
        args,
    ).fetchall()
    return [
        {
            "ts": r[0],
            "session_id": r[1][:8] if r[1] else "",
            "model": r[2],
            "original_tokens": r[3],
            "compressed_tokens": r[4],
            "savings_pct": r[5] or 0.0,
            "latency_ms": round(r[6], 1) if r[6] is not None else 0.0,
            "role": r[7] or "user",
        }
        for r in rows
    ]


def _rtk_stats(session_id: str | None) -> dict | None:
    """Aggregate rtk shell-layer savings, with a top-commands breakdown."""
    where = "WHERE session_id = ?" if session_id else ""
    args = (session_id,) if session_id else ()
    row = db._db_conn.execute(
        f"""SELECT COUNT(*), COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0),
                   COALESCE(SUM(saved_tokens),0), COALESCE(AVG(savings_pct),0)
            FROM rtk_events {where}""",
        args,
    ).fetchone()
    if not (row and row[0] > 0):
        return None
    top = db._db_conn.execute(
        f"""SELECT rtk_cmd, COUNT(*) AS cnt, SUM(saved_tokens) AS saved, AVG(savings_pct) AS avg_pct
            FROM rtk_events {where}
            GROUP BY rtk_cmd ORDER BY saved DESC LIMIT 8""",
        args,
    ).fetchall()
    return {
        "total_commands": row[0],
        "total_input_tokens": row[1],
        "total_output_tokens": row[2],
        "total_saved_tokens": row[3],
        "avg_savings_pct": round(row[4], 1),
        "top_commands": [
            {"cmd": r[0], "count": r[1], "saved": r[2], "avg_pct": round(r[3], 1)} for r in top
        ],
    }


def _empty_cache_stats() -> dict:
    """Zeroed cache-stats payload, used when no DB is available."""
    zero = {"hits": 0, "total": 0, "hit_ratio": 0.0}
    return {
        "since_deploy": dict(zero),
        "last_24h": dict(zero),
        "entries": 0,
        "time_saved_ms": 0,
        "by_role": {},
    }


def _cache_stats(session_id: str | None = None) -> dict:
    """Cache-hit summary from compressions.cache_hit, windowed to avoid dilution.

    `since_deploy` counts only rows recorded after caching went live (the
    `cache_since` meta marker), so the pre-feature backlog of misses cannot
    permanently depress the ratio. `last_24h` is a rolling window that reflects
    current behaviour. When `session_id` is given, the windows and `by_role`
    breakdown are scoped to that session; `entries` (compression_cache size)
    stays global always -- the dedup cache is content-addressed, not
    session-addressed, so it has no session_id column to scope by.
    """
    if db._db_conn is None:
        return _empty_cache_stats()

    sess_clause = " AND session_id = ?" if session_id else ""
    sess_args = (session_id,) if session_id else ()

    def _window(cutoff) -> dict:
        if cutoff is None:
            return {"hits": 0, "total": 0, "hit_ratio": 0.0}
        row = db._db_conn.execute(
            f"SELECT COALESCE(SUM(cache_hit), 0), COUNT(*) FROM compressions "
            f"WHERE ts >= ?{sess_clause}",
            (cutoff, *sess_args),
        ).fetchone()
        hits, total = int(row[0]), int(row[1])
        return {"hits": hits, "total": total, "hit_ratio": round(hits / total, 4) if total else 0.0}

    since_row = db._db_conn.execute("SELECT value FROM meta WHERE key='cache_since'").fetchone()
    since = since_row[0] if since_row else None
    day_ago = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat(timespec="seconds")

    entries = db._db_conn.execute("SELECT COUNT(*) FROM compression_cache").fetchone()[0]

    by_role = {}
    if since:
        for role, hits, total, avg_miss_ms in db._db_conn.execute(
            f"""SELECT role,
                        COALESCE(SUM(cache_hit), 0),
                        COUNT(*),
                        AVG(CASE WHEN cache_hit = 0 THEN latency_ms END)
                 FROM compressions WHERE ts >= ?{sess_clause} GROUP BY role""",
            (since, *sess_args),
        ).fetchall():
            h, t = int(hits), int(total)
            by_role[role] = {
                "hits": h,
                "total": t,
                "hit_ratio": round(h / t, 4) if t else 0.0,
                "avg_miss_latency_ms": round(avg_miss_ms, 1) if avg_miss_ms else 0.0,
                "time_saved_ms": round(h * (avg_miss_ms or 0.0)),
            }

    sd = _window(since)
    total_time_saved_ms = sum(v["time_saved_ms"] for v in by_role.values())

    return {
        "since_deploy": sd,
        "last_24h": _window(day_ago),
        "entries": int(entries),
        "time_saved_ms": total_time_saved_ms,
        "by_role": by_role,
    }



def _tracked_stats() -> dict:
    """Totals across tracked sessions (active/closed trackers joined to compressions)."""
    row = db._db_conn.execute(
        """
        SELECT COUNT(DISTINCT t.slug),
               COALESCE(SUM(c.original_tokens - c.compressed_tokens), 0)
        FROM trackers t
        JOIN compressions c ON c.session_id = t.session_id
        WHERE t.status IN ('active', 'closed') AND t.session_id IS NOT NULL
        """
    ).fetchone()
    return {"sessions": (row[0] if row else 0) or 0, "tokens_saved": (row[1] if row else 0) or 0}


def _merge_rtk_into_sessions(sessions_out: dict) -> None:
    """Decorate in-memory session entries with their rtk command counts/savings."""
    for sid, cmds, saved in db._db_conn.execute(
        """SELECT session_id, COUNT(*), COALESCE(SUM(saved_tokens), 0)
           FROM rtk_events GROUP BY session_id"""
    ).fetchall():
        if sid in sessions_out:
            sessions_out[sid]["rtk_commands"] = cmds
            sessions_out[sid]["rtk_saved"] = saved
