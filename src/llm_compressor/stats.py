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

from llm_compressor import backends, db

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


def _latency_percentiles(where: str, args: tuple) -> dict:
    """p50 / p95 of real compression latency for a scope.

    Cache hits are excluded here and in every other latency figure. A hit
    writes `latency_ms = 0.0` because no compression ran, so including them
    does not measure a faster compressor -- it measures how often the
    compressor was skipped, and drags the mean toward zero as the cache warms.
    Reported latency should answer "how long does compressing cost me", so it
    is computed over misses only.

    SQLite has no percentile function, so each is an OFFSET into the ordered
    row set. Two small indexed scans, run only on the aggregate endpoints.
    """
    real = f"({where}) AND cache_hit = 0 AND latency_ms > 0"
    n = db._db_conn.execute(f"SELECT COUNT(*) FROM compressions WHERE {real}", args).fetchone()[0]
    if not n:
        return {"p50_latency_ms": 0.0, "p95_latency_ms": 0.0, "latency_samples": 0}

    def at(fraction: float) -> float:
        offset = min(n - 1, int(n * fraction))
        row = db._db_conn.execute(
            f"SELECT latency_ms FROM compressions WHERE {real} "
            f"ORDER BY latency_ms LIMIT 1 OFFSET {offset}",
            args,
        ).fetchone()
        return round(row[0], 1) if row else 0.0

    return {"p50_latency_ms": at(0.50), "p95_latency_ms": at(0.95), "latency_samples": n}


def _aggregate_stats(scope: str, args: tuple, *, today: bool, with_ratio: bool) -> dict:
    """Aggregate request/savings/latency metrics for a scope, optionally today-only."""
    date_clause = "date(ts) = date('now') AND " if today else ""
    ratio_col = (
        ", ROUND(AVG(CAST(original_tokens AS REAL) / NULLIF(compressed_tokens, 0)), 2) AS avg_ratio"
        if with_ratio
        else ""
    )
    where = f"{date_clause}{scope}"
    row = db._db_conn.execute(
        f"""
        SELECT COUNT(*) AS requests,
               COALESCE(SUM(original_tokens - compressed_tokens), 0) AS tokens_saved,
               ROUND(AVG((original_tokens - compressed_tokens) * 100.0 / original_tokens), 1) AS avg_savings_pct,
               ROUND(AVG(CASE WHEN cache_hit = 0 THEN latency_ms END), 1) AS avg_latency_ms,
               COUNT(DISTINCT session_id) AS sessions,
               COALESCE(SUM(original_tokens), 0) AS original_tokens,
               COALESCE(SUM(compressed_tokens), 0) AS compressed_tokens,
               COALESCE(SUM(CASE WHEN ok = 0 THEN 1 ELSE 0 END), 0) AS errors{ratio_col}
        FROM compressions
        WHERE {where}
        """,
        args,
    ).fetchone()
    requests = (row[0] if row else 0) or 0
    errors = (row[7] if row else 0) or 0
    out = {
        "requests": requests,
        "tokens_saved": (row[1] if row else 0) or 0,
        "avg_savings_pct": (row[2] if row else 0.0) or 0.0,
        "avg_latency_ms": (row[3] if row else 0.0) or 0.0,
        "sessions": (row[4] if row else 0) or 0,
        "original_tokens": (row[5] if row else 0) or 0,
        "compressed_tokens": (row[6] if row else 0) or 0,
        "errors": errors,
        "error_rate": round(errors / requests, 4) if requests else 0.0,
    }
    out.update(_latency_percentiles(where, args))
    if with_ratio:
        out["avg_ratio"] = (row[8] if row else 0.0) or 0.0
    return out


#: Supported chart windows -> (SQLite modifier, bucket width in hours).
#: Bucket width is chosen so every range lands in the 24-120 bucket band: dense
#: enough to show shape, sparse enough that marks stay above the 2px spacer.
RANGES: dict[str, tuple[str, int]] = {
    "24h": ("-24 hours", 1),
    "48h": ("-48 hours", 1),
    "7d": ("-7 days", 6),
    "30d": ("-30 days", 24),
}
DEFAULT_RANGE = "24h"


def _cutoff(modifier: str) -> str:
    """SQL for a range boundary in the same text format the `ts` column uses.

    Not `datetime('now', modifier)`: that renders `YYYY-MM-DD HH:MM:SS` with a
    space, while `db.utc_now()` writes `isoformat()` with a `T`. These columns
    are TEXT, so the comparison is lexicographic, and at offset 10 'T' (0x54)
    sorts above ' ' (0x20) -- every row sharing the boundary's *date* compared
    greater regardless of its time. A "last 24 hours" chart silently ran from
    the start of yesterday, i.e. up to 48 hours wide, and the KPI tiles diffed
    two overlapping windows.
    """
    return f"strftime('%Y-%m-%dT%H:%M:%S', 'now', '{modifier}')"


def _bucket_expr(hours: int) -> str:
    """SQL expression flooring `ts` to a bucket of `hours`."""
    if hours == 1:
        return "strftime('%Y-%m-%dT%H:00:00', ts)"
    if hours == 24:
        return "strftime('%Y-%m-%dT00:00:00', ts)"
    return (
        "strftime('%Y-%m-%dT', ts) || "
        f"printf('%02d', (CAST(strftime('%H', ts) AS INTEGER) / {hours}) * {hours}) || ':00:00'"
    )


def timeseries(range_key: str, active_model: str | None, session_id: str | None) -> list[dict]:
    """Bucketed traffic for the flow chart.

    Emits `original_tokens` and `compressed_tokens` per bucket, not just the
    saved delta. The stacked area chart draws compressed-still-paid beneath
    saved, summing to original; with only the delta stored, that chart cannot
    be drawn at all -- which is why the previous version showed a bare savings
    line that no one could sanity-check against their bill.

    Model scoping goes through `_stats_scope`, so `model=dual` expands to its
    sub-model names. The old query compared `model = 'dual'` directly against
    rows that only ever store the resolved sub-model, and so returned an empty
    series -- an empty chart on the exact configuration the project recommends.
    """
    modifier, hours = RANGES.get(range_key, RANGES[DEFAULT_RANGE])
    bucket = _bucket_expr(hours)

    clauses = [f"ts >= {_cutoff(modifier)}", "ts IS NOT NULL"]
    args: tuple = ()
    if session_id:
        clauses.append("session_id = ?")
        args += (session_id,)
    elif active_model:
        scope, scope_args = _stats_scope(active_model, None)
        clauses.append(scope)
        args += tuple(scope_args)

    rows = db._db_conn.execute(
        f"""
        SELECT {bucket} AS bucket,
               COUNT(*) AS requests,
               COALESCE(SUM(original_tokens), 0) AS original_tokens,
               COALESCE(SUM(compressed_tokens), 0) AS compressed_tokens,
               COALESCE(SUM(original_tokens - compressed_tokens), 0) AS total_saved,
               ROUND(AVG((original_tokens - compressed_tokens) * 100.0 / NULLIF(original_tokens, 0)), 1)
                   AS avg_savings_pct,
               ROUND(AVG(CASE WHEN cache_hit = 0 THEN latency_ms END), 1) AS avg_latency_ms,
               COALESCE(SUM(cache_hit), 0) AS cache_hits,
               COALESCE(SUM(CASE WHEN ok = 0 THEN 1 ELSE 0 END), 0) AS errors
        FROM compressions
        WHERE {" AND ".join(clauses)}
        GROUP BY bucket
        ORDER BY bucket
        """,
        args,
    ).fetchall()

    # A window with no rows at all stays empty rather than becoming a lattice of
    # zeros: the chart keys its empty state off an empty series, and a flat
    # axis with no marks reads as "measured, nothing happened" instead of the
    # "nothing recorded yet" this actually is.
    if not rows:
        return []

    found = {
        r[0]: {
            "bucket": r[0],
            # Kept as `hour` too: the key the previous chart consumed.
            "hour": r[0],
            "requests": r[1],
            "original_tokens": r[2],
            "compressed_tokens": r[3],
            "total_saved": r[4],
            "avg_savings_pct": r[5] or 0.0,
            "avg_latency_ms": r[6] or 0.0,
            "cache_hits": r[7],
            "errors": r[8],
        }
        for r in rows
    }
    return [found.get(b, _empty_bucket(b)) for b in _bucket_range(hours, modifier)]


def _empty_bucket(bucket: str) -> dict:
    return {
        "bucket": bucket,
        "hour": bucket,
        "requests": 0,
        "original_tokens": 0,
        "compressed_tokens": 0,
        "total_saved": 0,
        "avg_savings_pct": 0.0,
        "avg_latency_ms": 0.0,
        "cache_hits": 0,
        "errors": 0,
    }


def _bucket_range(hours: int, modifier: str) -> list[str]:
    """Every bucket label in the window, including the quiet ones.

    GROUP BY only emits buckets that have rows, but the chart draws one
    equal-width column per element it is handed -- so an idle hour did not
    render as a gap, it vanished and pulled the later columns leftward. On a
    sparse series that reads as a continuous run of traffic, and the x-axis
    ticks appear to jump backwards in time where a day boundary was skipped.
    Zero-filling makes column position mean elapsed time again.
    """
    amount, unit = modifier.lstrip("-").split(" ")
    delta = timedelta(**{unit: int(amount)})
    now = datetime.now(timezone.utc).replace(tzinfo=None, minute=0, second=0, microsecond=0)
    start = now - delta

    step = timedelta(hours=hours)
    # Floor the start onto the bucket lattice so labels line up with the SQL
    # bucket expression rather than drifting by the window's offset.
    if hours >= 24:
        start = start.replace(hour=0)
    else:
        start = start.replace(hour=(start.hour // hours) * hours)

    out, cur = [], start
    while cur <= now:
        out.append(cur.isoformat(timespec="seconds"))
        cur += step
    return out


def window_summary(range_key: str, active_model: str | None, session_id: str | None) -> dict:
    """Totals for a window and the one before it, so KPI tiles can show a delta.

    A savings number with nothing to compare against cannot answer "is this
    getting better", which is most of why someone opens the page twice.
    """
    modifier, _ = RANGES.get(range_key, RANGES[DEFAULT_RANGE])
    amount, unit = modifier.lstrip("-").split(" ")
    prev_start = f"-{int(amount) * 2} {unit}"

    scope, scope_args = (
        _stats_scope(active_model or "", session_id)
        if (active_model or session_id)
        else ("1=1", ())
    )

    def totals(start: str, end: str | None) -> dict:
        clauses = [f"ts >= {_cutoff(start)}", "ts IS NOT NULL", scope]
        if end:
            clauses.append(f"ts < {_cutoff(end)}")
        return _aggregate_stats(
            " AND ".join(clauses), tuple(scope_args), today=False, with_ratio=True
        )

    current = totals(modifier, None)
    previous = totals(prev_start, modifier)
    return {"range": range_key, "current": current, "previous": previous}


def _recent_compression_rows(active_model: str, session_id: str | None, limit: int = 20) -> list:
    """Most recent compression rows for the active scope.

    Uses the same scoping as the today/alltime panels, so dual mode spans all
    sub-model rows rather than matching the non-existent model='dual'.
    """
    scope, args = _stats_scope(active_model, session_id)
    rows = db._db_conn.execute(
        f"""
        SELECT id, ts, session_id, model, original_tokens, compressed_tokens,
               ROUND((original_tokens - compressed_tokens) * 100.0 / NULLIF(original_tokens, 0), 1) AS savings_pct,
               latency_ms, role, cache_hit, ok
        FROM compressions
        WHERE {scope}
        ORDER BY id DESC
        LIMIT ?
        """,
        (*args, limit),
    ).fetchall()
    return [
        {
            "id": r[0],
            "ts": r[1],
            # Full id as well as the short form: the row links to a session
            # page, and 8 hex characters are not enough to address one.
            "session_id": r[2] or "",
            "session_short": (r[2] or "")[:8],
            "model": r[3],
            "original_tokens": r[4],
            "compressed_tokens": r[5],
            "savings_pct": r[6] or 0.0,
            "latency_ms": round(r[7], 1) if r[7] is not None else 0.0,
            "role": r[8] or "user",
            "cache_hit": bool(r[9]),
            "ok": bool(r[10]) if r[10] is not None else True,
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
    # Naive, to match db.utc_now(): `ts` is TEXT and the compare is
    # lexicographic, so a "+00:00" suffix here sorts above an equal bare
    # timestamp and silently drops rows sitting on the boundary.
    day_ago = (
        (datetime.now(timezone.utc) - timedelta(hours=24))
        .replace(tzinfo=None)
        .isoformat(timespec="seconds")
    )

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
