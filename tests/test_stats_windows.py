"""Range-window filtering for the chart and KPI endpoints.

These all guard one bug class: `compressions.ts` is TEXT holding naive ISO-8601
with a `T` separator, so every range comparison is lexicographic and only works
if the boundary is rendered in exactly that format. SQLite's `datetime('now')`
renders a space instead, which sorts *below* `T`, so a same-day row from long
outside the window compared greater and was included anyway.
"""

from datetime import datetime, timedelta, timezone

from llm_compressor import proxy
from llm_compressor import stats as S


def _ts(hours_ago: float) -> str:
    """A stored-format timestamp `hours_ago` hours in the past."""
    return (
        (datetime.now(timezone.utc) - timedelta(hours=hours_ago))
        .replace(tzinfo=None)
        .isoformat(timespec="seconds")
    )


def _row(conn, ts: str, original: int = 100, compressed: int = 60) -> None:
    conn.execute(
        "INSERT INTO compressions (ts, session_id, model, original_tokens,"
        " compressed_tokens, latency_ms, role, cache_hit, ok)"
        " VALUES (?, 's1', 'llmlingua2', ?, ?, 1.0, 'user', 0, 1)",
        (ts, original, compressed),
    )
    conn.commit()


def test_cutoff_renders_the_columns_own_timestamp_format():
    # The whole bug in one assertion: a space separator here silently widens
    # every window to the start of the boundary's calendar day.
    conn = proxy.init_db(":memory:")
    rendered = conn.execute(f"SELECT {S._cutoff('-24 hours')}").fetchone()[0]
    assert "T" in rendered and " " not in rendered
    assert rendered == datetime.fromisoformat(rendered).isoformat(timespec="seconds")


def test_timeseries_excludes_a_row_older_than_the_range(tmp_path, monkeypatch):
    from llm_compressor import db

    conn = proxy.init_db(str(tmp_path / "m.db"))
    monkeypatch.setattr(db, "_db_conn", conn)

    _row(conn, _ts(2))  # inside 24h
    _row(conn, _ts(40))  # outside 24h, but often the SAME CALENDAR DAY as the
    # -24h boundary -- the case the old comparison let through.

    buckets = S.timeseries("24h", None, None)
    assert sum(b["requests"] for b in buckets) == 1, (
        "a 40h-old row leaked into the 24h window: the range boundary is not "
        "being rendered in the ts column's format"
    )


def test_timeseries_range_widths_are_actually_different(tmp_path, monkeypatch):
    from llm_compressor import db

    conn = proxy.init_db(str(tmp_path / "m.db"))
    monkeypatch.setattr(db, "_db_conn", conn)

    _row(conn, _ts(2))
    _row(conn, _ts(40))

    assert sum(b["requests"] for b in S.timeseries("24h", None, None)) == 1
    assert sum(b["requests"] for b in S.timeseries("48h", None, None)) == 2


def test_window_summary_does_not_double_count_into_previous(tmp_path, monkeypatch):
    """`previous` is the window before `current`, and they must not overlap.

    With a mis-rendered boundary both halves matched the same rows, so every
    delta arrow on the overview read ~0% no matter what the traffic did.
    """
    from llm_compressor import db

    conn = proxy.init_db(str(tmp_path / "m.db"))
    monkeypatch.setattr(db, "_db_conn", conn)

    _row(conn, _ts(2))  # current 24h window
    _row(conn, _ts(30))  # previous 24h window
    _row(conn, _ts(30))

    out = S.window_summary("24h", None, None)
    assert out["current"]["requests"] == 1
    assert out["previous"]["requests"] == 2


def test_cache_24h_window_keeps_a_row_on_the_boundary(tmp_path, monkeypatch):
    """The mirror-image bug: an offset-aware cutoff against a naive column.

    '...T12:00:00+00:00' sorts above '...T12:00:00', so a row landing exactly on
    the boundary fell out of the 24h cache ratio.
    """
    from llm_compressor import db

    conn = proxy.init_db(str(tmp_path / "m.db"))
    monkeypatch.setattr(db, "_db_conn", conn)
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('cache_since', ?)", (_ts(72),))
    conn.commit()

    _row(conn, _ts(23.9))
    out = S._cache_stats(None)
    assert out["last_24h"]["total"] == 1


def test_timeseries_zero_fills_quiet_buckets(tmp_path, monkeypatch):
    """An idle hour must occupy a column, not disappear.

    The chart draws one equal-width column per element, so a dropped bucket
    silently rescales the x-axis: quiet periods vanish and the axis can appear
    to run backwards where a whole day was skipped.
    """
    from llm_compressor import db

    conn = proxy.init_db(str(tmp_path / "m.db"))
    monkeypatch.setattr(db, "_db_conn", conn)

    _row(conn, _ts(1))
    _row(conn, _ts(6))

    out = S.timeseries("24h", None, None)

    assert len(out) >= 24, f"expected a contiguous 24h lattice, got {len(out)} buckets"
    assert sum(b["requests"] for b in out) == 2
    assert any(b["requests"] == 0 for b in out), "quiet hours were dropped, not zero-filled"

    buckets = [b["bucket"] for b in out]
    assert buckets == sorted(buckets), "buckets are not ascending"
    assert len(set(buckets)) == len(buckets), "duplicate buckets"

    # Evenly spaced: the property the chart's equal-width columns depend on.
    from datetime import datetime as _dt

    times = [_dt.fromisoformat(b) for b in buckets]
    gaps = {(b - a).total_seconds() for a, b in zip(times, times[1:])}
    assert gaps == {3600.0}, f"uneven bucket spacing: {gaps}"


def test_timeseries_lattice_matches_sql_bucket_width_for_coarse_ranges(tmp_path, monkeypatch):
    from llm_compressor import db

    conn = proxy.init_db(str(tmp_path / "m.db"))
    monkeypatch.setattr(db, "_db_conn", conn)
    _row(conn, _ts(3))

    out = S.timeseries("7d", None, None)
    from datetime import datetime as _dt

    times = [_dt.fromisoformat(b["bucket"]) for b in out]
    gaps = {(b - a).total_seconds() for a, b in zip(times, times[1:])}
    assert gaps == {6 * 3600.0}
    # The row still lands in exactly one bucket rather than falling off-lattice.
    assert sum(b["requests"] for b in out) == 1
