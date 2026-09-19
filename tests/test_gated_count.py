"""Test for gated_count stats aggregate."""

from llm_compressor import db, sessions, stats


def test_stats_reports_gated_count(tmp_path, monkeypatch):
    conn = db.init_db(str(tmp_path / "m.db"))
    monkeypatch.setattr(db, "_db_conn", conn)
    sessions.record_compression("s", 100, 40, original_text="x"*300, compressed_text="y")           # not gated
    sessions.record_compression("s", 100, 40, gated=1, original_text="x"*300, compressed_text="y")   # gated
    summary = stats._aggregate_stats("1=1", (), today=False, with_ratio=False)
    assert summary["gated_count"] == 1
