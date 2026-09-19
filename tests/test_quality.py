"""Unit tests for quality.py's QualityJudge -- mirrors test_langfuse_tracer.py's shape.

No real network access: `_client.post` is always replaced with an AsyncMock. `init()`
itself is safe to call for real (httpx.AsyncClient(...) just builds a client object,
no socket opened), so enabled/disabled tests exercise the real constructor.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_compressor import compression, quality


def make_judge(monkeypatch, api_key="ts-test-key"):
    """Env-gated init(), like langfuse_tracer's make_tracer() -- real construction,
    no network access performed by init() itself."""
    if api_key is None:
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    else:
        monkeypatch.setenv("TYPESAFE_API_KEY", api_key)
    from llm_compressor import quality as mod

    j = mod.QualityJudge()
    j.init()
    return j, mod


def _mock_response(payload, raise_error=None):
    resp = MagicMock()
    resp.raise_for_status = MagicMock(side_effect=raise_error)
    resp.json = MagicMock(return_value=payload)
    return resp


_GOOD_PAYLOAD = {
    "model": "jev-latest",
    "answers": {
        "meaning_preserved": {
            "type": "score",
            "score": 2.0,
            "legend": {"0": "lost", "1": "drift", "2": "preserved"},
            "probabilities": {"0": 0.05, "1": 0.05, "2": 0.9},
            "confidence": 0.9,
        }
    },
    "usage": {"input_tokens": 10, "output_tokens": 2},
}


# --- init / enabled ---


def test_disabled_when_no_api_key(monkeypatch):
    j, mod = make_judge(monkeypatch, api_key=None)
    assert not j.enabled
    s = j.status()
    assert s["enabled"] is False
    assert s["api_key_set"] is False
    assert s["host"] == mod.TYPESAFE_API_BASE
    assert s["gate_enabled"] is False
    assert s["timeout_s"] == mod.GATE_TIMEOUT


def test_disabled_when_api_key_blank(monkeypatch):
    j, _ = make_judge(monkeypatch, api_key="   ")
    assert not j.enabled


def test_enabled_when_api_key_set(monkeypatch):
    j, _ = make_judge(monkeypatch)
    assert j.enabled
    s = j.status()
    assert s["enabled"] is True
    assert s["api_key_set"] is True


# --- judge_now() (awaited, on-demand path for the per-row "Judge" button,
# and -- with compression_id=None -- the gate's inline verdict call) ---


def test_judge_now_accepts_none_compression_id_and_skips_quality_insert(tmp_path, monkeypatch):
    """The gate calls judge_now(None, ...) BEFORE the compression row exists.
    It must still perform the Jev call and parse the verdict, but skip the
    `compression_quality` INSERT (no valid compression_id to key it on)."""
    from llm_compressor import db

    conn = db.init_db(str(tmp_path / "m.db"))
    monkeypatch.setattr(db, "_db_conn", conn)
    conn.execute(
        "INSERT INTO compression_cache (key, model, rate, compressed_text, "
        "original_tokens, compressed_tokens, created_at, hit_count, last_hit) "
        "VALUES ('k1','llmlingua2',0.5,'COMP',100,60,?,0,?)",
        (db.utc_now(), db.utc_now()),
    )
    conn.commit()

    j, _ = make_judge(monkeypatch)
    j._client = MagicMock()
    j._client.post = AsyncMock(return_value=_mock_response(_GOOD_PAYLOAD))

    result = asyncio.run(j.judge_now(None, "original", "compressed", cache_key="k1"))

    assert result == {
        "score": 2.0,
        "label": "Preserved",
        "detail": "preserved",
        "confidence": 0.9,
        "flagged": 0,
        "meta": {
            "probabilities": {"0": 0.05, "1": 0.05, "2": 0.9},
            "usage": {"input_tokens": 10, "output_tokens": 2},
            "model": "jev-latest",
        },
    }
    assert conn.execute("SELECT COUNT(*) FROM compression_quality").fetchone()[0] == 0
    cache_row = conn.execute(
        "SELECT quality_score, quality_label FROM compression_cache WHERE key='k1'"
    ).fetchone()
    assert tuple(cache_row) == (2.0, "Preserved")  # cache verdict columns updated
    conn.close()


def test_judge_now_returns_none_when_disabled(monkeypatch):
    j, _ = make_judge(monkeypatch, api_key=None)

    result = asyncio.run(j.judge_now(1, "orig", "comp"))

    assert result is None


def test_judge_now_returns_verdict_dict(tmp_path, monkeypatch):
    from llm_compressor import db

    conn = db.init_db(str(tmp_path / "m.db"))
    monkeypatch.setattr(db, "_db_conn", conn)
    conn.execute(
        "INSERT INTO compressions (id, ts, session_id, model, original_tokens, "
        "compressed_tokens, latency_ms, role, cache_hit, ok) VALUES "
        "(5, ?, 's', 'llmlingua2', 100, 60, 5.0, 'user', 0, 1)",
        (db.utc_now(),),
    )
    conn.commit()

    j, _ = make_judge(monkeypatch)
    j._client = MagicMock()
    j._client.post = AsyncMock(return_value=_mock_response(_GOOD_PAYLOAD))

    result = asyncio.run(j.judge_now(5, "original", "compressed"))

    assert result == {
        "score": 2.0,
        "label": "Preserved",
        "detail": "preserved",
        "confidence": 0.9,
        "flagged": 0,
        "meta": {
            "probabilities": {"0": 0.05, "1": 0.05, "2": 0.9},
            "usage": {"input_tokens": 10, "output_tokens": 2},
            "model": "jev-latest",
        },
    }
    conn.close()


def test_judge_now_returns_none_on_malformed_response(tmp_path, monkeypatch):
    from llm_compressor import db

    conn = db.init_db(str(tmp_path / "m.db"))
    monkeypatch.setattr(db, "_db_conn", conn)

    j, _ = make_judge(monkeypatch)
    j._client = MagicMock()
    j._client.post = AsyncMock(return_value=_mock_response({"answers": {}}))

    result = asyncio.run(j.judge_now(1, "a", "b"))

    assert result is None
    conn.close()


# --- _judge() request shape ---


def test_judge_posts_verified_request_contract(monkeypatch):
    from llm_compressor import db

    monkeypatch.setattr(db, "_db_conn", None)  # persistence not under test here
    j, mod = make_judge(monkeypatch)
    j._client = MagicMock()
    j._client.post = AsyncMock(return_value=_mock_response(_GOOD_PAYLOAD))

    asyncio.run(j._judge(1, "original", "compressed", None))

    assert j._client.post.call_args.args[0] == "/v1/systemone"
    body = j._client.post.call_args.kwargs["json"]
    assert body["model"] == "jev-latest"
    assert body["state"] == {"original_text": "original", "compressed_text": "compressed"}
    question = body["questions"]["meaning_preserved"]
    assert question["type"] == "score"
    assert len(question["criteria"]) == len(mod._CRITERIA)


# --- _judge() success path: persistence ---


def test_judge_success_writes_quality_row_and_updates_cache(tmp_path, monkeypatch):
    from llm_compressor import db

    conn = db.init_db(str(tmp_path / "m.db"))
    monkeypatch.setattr(db, "_db_conn", conn)
    conn.execute(
        "INSERT INTO compressions (id, ts, session_id, model, original_tokens, "
        "compressed_tokens, latency_ms, role, cache_hit, ok) VALUES "
        "(42, ?, 's', 'llmlingua2', 100, 60, 5.0, 'user', 0, 1)",
        (db.utc_now(),),
    )
    conn.execute(
        "INSERT INTO compression_cache (key, model, rate, compressed_text, "
        "original_tokens, compressed_tokens, created_at, hit_count, last_hit) "
        "VALUES ('k1','llmlingua2',0.5,'COMP',100,60,?,0,?)",
        (db.utc_now(), db.utc_now()),
    )
    conn.commit()

    j, _ = make_judge(monkeypatch)
    j._client = MagicMock()
    j._client.post = AsyncMock(return_value=_mock_response(_GOOD_PAYLOAD))

    asyncio.run(j._judge(42, "original", "compressed", "k1"))

    row = conn.execute(
        "SELECT compression_id, score, label, detail, confidence, flagged, meta "
        "FROM compression_quality WHERE compression_id=42"
    ).fetchone()
    # `label` is the short chip text (SHORT_LABELS); `detail` is TypeSafe's
    # own legend text for the rounded level, verbatim.
    assert tuple(row)[:6] == (42, 2.0, "Preserved", "preserved", 0.9, 0)
    # `meta` is display-only (probabilities/usage/model), stored as JSON text.
    assert json.loads(row["meta"]) == {
        "probabilities": {"0": 0.05, "1": 0.05, "2": 0.9},
        "usage": {"input_tokens": 10, "output_tokens": 2},
        "model": "jev-latest",
    }

    cache_row = conn.execute(
        "SELECT quality_score, quality_label, quality_detail, quality_flagged "
        "FROM compression_cache WHERE key='k1'"
    ).fetchone()
    assert tuple(cache_row) == (2.0, "Preserved", "preserved", 0)
    conn.close()


def test_judge_success_without_cache_key_skips_cache_update(tmp_path, monkeypatch):
    from llm_compressor import db

    conn = db.init_db(str(tmp_path / "m.db"))
    monkeypatch.setattr(db, "_db_conn", conn)
    conn.execute(
        "INSERT INTO compressions (id, ts, session_id, model, original_tokens, "
        "compressed_tokens, latency_ms, role, cache_hit, ok) VALUES "
        "(1, ?, 's', 'llmlingua2', 100, 60, 5.0, 'user', 0, 1)",
        (db.utc_now(),),
    )
    conn.commit()

    j, _ = make_judge(monkeypatch)
    j._client = MagicMock()
    j._client.post = AsyncMock(return_value=_mock_response(_GOOD_PAYLOAD))

    asyncio.run(j._judge(1, "original", "compressed", None))  # no cache_key

    row = conn.execute(
        "SELECT compression_id FROM compression_quality WHERE compression_id=1"
    ).fetchone()
    assert row is not None  # quality row still written
    conn.close()


def test_judge_flags_low_score(tmp_path, monkeypatch):
    from llm_compressor import db

    conn = db.init_db(str(tmp_path / "m.db"))
    monkeypatch.setattr(db, "_db_conn", conn)
    conn.execute(
        "INSERT INTO compressions (id, ts, session_id, model, original_tokens, "
        "compressed_tokens, latency_ms, role, cache_hit, ok) VALUES "
        "(9, ?, 's', 'llmlingua2', 100, 60, 5.0, 'user', 0, 1)",
        (db.utc_now(),),
    )
    conn.commit()

    low_payload = {
        "answers": {
            "meaning_preserved": {
                "score": 0.2,
                "legend": {"0": "lost", "1": "drift", "2": "preserved"},
                "confidence": 0.4,
            }
        }
    }
    j, _ = make_judge(monkeypatch)
    j._client = MagicMock()
    j._client.post = AsyncMock(return_value=_mock_response(low_payload))

    asyncio.run(j._judge(9, "original", "compressed", None))

    row = conn.execute(
        "SELECT score, label, detail, flagged, meta FROM compression_quality WHERE compression_id=9"
    ).fetchone()
    assert tuple(row)[:4] == (0.2, "Lost", "lost", 1)
    # low_payload has no probabilities/usage/model -> meta round-trips as
    # empty dicts/string, never null (see quality.py's `.get(..., {})` guards).
    assert json.loads(row["meta"]) == {"probabilities": {}, "usage": {}, "model": ""}
    conn.close()


def test_migrate_creates_compression_quality_meta_column(tmp_path):
    """db.init_db()'s ALTER TABLE for `meta` (added alongside the label/detail
    split migration) must be present -- without it, every judged row fails to
    persist with 'table compression_quality has no column named meta'."""
    from llm_compressor import db

    conn = db.init_db(str(tmp_path / "m.db"))
    cols = [r[1] for r in conn.execute("PRAGMA table_info(compression_quality)")]
    assert "meta" in cols
    conn.close()


def test_compressions_has_gated_column(tmp_path):
    from llm_compressor import db

    conn = db.init_db(str(tmp_path / "m.db"))
    cols = {r[1] for r in conn.execute("PRAGMA table_info(compressions)")}
    assert "gated" in cols
    conn.close()


# --- _judge() failure paths: each stage swallows and writes nothing ---


def test_judge_http_error_swallowed(tmp_path, monkeypatch):
    from llm_compressor import db

    conn = db.init_db(str(tmp_path / "m.db"))
    monkeypatch.setattr(db, "_db_conn", conn)

    j, _ = make_judge(monkeypatch)
    j._client = MagicMock()
    j._client.post = AsyncMock(
        return_value=_mock_response({}, raise_error=RuntimeError("529 overloaded"))
    )

    asyncio.run(j._judge(1, "a", "b", None))  # must not raise

    assert conn.execute("SELECT COUNT(*) FROM compression_quality").fetchone()[0] == 0
    conn.close()


def test_judge_network_error_swallowed(tmp_path, monkeypatch):
    from llm_compressor import db

    conn = db.init_db(str(tmp_path / "m.db"))
    monkeypatch.setattr(db, "_db_conn", conn)

    j, _ = make_judge(monkeypatch)
    j._client = MagicMock()
    j._client.post = AsyncMock(side_effect=OSError("connection refused"))

    asyncio.run(j._judge(1, "a", "b", None))  # must not raise

    assert conn.execute("SELECT COUNT(*) FROM compression_quality").fetchone()[0] == 0
    conn.close()


def test_judge_malformed_response_swallowed(tmp_path, monkeypatch):
    from llm_compressor import db

    conn = db.init_db(str(tmp_path / "m.db"))
    monkeypatch.setattr(db, "_db_conn", conn)

    j, _ = make_judge(monkeypatch)
    j._client = MagicMock()
    # Missing the "meaning_preserved" key entirely -> KeyError inside parsing stage.
    j._client.post = AsyncMock(return_value=_mock_response({"answers": {}}))

    asyncio.run(j._judge(1, "a", "b", None))  # must not raise

    assert conn.execute("SELECT COUNT(*) FROM compression_quality").fetchone()[0] == 0
    conn.close()


def test_judge_db_unavailable_swallowed(monkeypatch):
    from llm_compressor import db

    monkeypatch.setattr(db, "_db_conn", None)

    j, _ = make_judge(monkeypatch)
    j._client = MagicMock()
    j._client.post = AsyncMock(return_value=_mock_response(_GOOD_PAYLOAD))

    asyncio.run(j._judge(1, "a", "b", "k1"))  # must not raise despite no DB


# ---------------------------------------------------------------------------
# GET /session/{session_id}/compressions -- routes._row_with_parsed_meta
# decodes compression_quality.meta from its stored JSON-text form so the
# frontend's qualityMetaText() gets a plain object, never a JSON string.
# ---------------------------------------------------------------------------


def test_compressions_endpoint_decodes_meta_json(client):
    from llm_compressor import db

    db._db_conn.execute(
        "INSERT INTO compressions (id, ts, session_id, model, original_tokens, "
        "compressed_tokens, latency_ms, role, cache_hit, ok) VALUES "
        "(11, ?, 'sess-meta', 'llmlingua2', 100, 60, 5.0, 'user', 0, 1)",
        (db.utc_now(),),
    )
    db._db_conn.execute(
        "INSERT INTO compression_quality (compression_id, score, label, detail, "
        "confidence, flagged, judged_at, meta) VALUES (11, 2.0, 'Preserved', "
        "'preserved', 0.9, 0, ?, ?)",
        (
            db.utc_now(),
            json.dumps({"probabilities": {"2": 0.9}, "usage": {}, "model": "jev-latest"}),
        ),
    )
    db._db_conn.commit()

    r = client.get("/session/sess-meta/compressions")
    assert r.status_code == 200
    item = r.json()["items"][0]
    assert item["meta"] == {"probabilities": {"2": 0.9}, "usage": {}, "model": "jev-latest"}


def test_compressions_endpoint_includes_confidence_for_judged_compression(client):
    """`compression_quality.confidence` must ride along in the /compressions
    response so the frontend's qualityMetaText() can render a "Confidence"
    bit -- see session_detail.html's qualityMetaText()."""
    from llm_compressor import db

    db._db_conn.execute(
        "INSERT INTO compressions (id, ts, session_id, model, original_tokens, "
        "compressed_tokens, latency_ms, role, cache_hit, ok) VALUES "
        "(13, ?, 'sess-confidence', 'llmlingua2', 100, 60, 5.0, 'user', 0, 1)",
        (db.utc_now(),),
    )
    db._db_conn.execute(
        "INSERT INTO compression_quality (compression_id, score, label, detail, "
        "confidence, flagged, judged_at, meta) VALUES (13, 2.0, 'Preserved', "
        "'preserved', 0.87, 0, ?, ?)",
        (db.utc_now(), json.dumps({})),
    )
    db._db_conn.commit()

    r = client.get("/session/sess-confidence/compressions")
    assert r.status_code == 200
    item = r.json()["items"][0]
    assert item["confidence"] == 0.87


def test_compressions_endpoint_defaults_meta_to_empty_dict_when_unjudged(client):
    from llm_compressor import db

    db._db_conn.execute(
        "INSERT INTO compressions (id, ts, session_id, model, original_tokens, "
        "compressed_tokens, latency_ms, role, cache_hit, ok) VALUES "
        "(12, ?, 'sess-unjudged', 'llmlingua2', 100, 60, 5.0, 'user', 0, 1)",
        (db.utc_now(),),
    )
    db._db_conn.commit()

    r = client.get("/session/sess-unjudged/compressions")
    assert r.status_code == 200
    item = r.json()["items"][0]
    assert item["meta"] == {}  # no compression_quality row at all -> LEFT JOIN NULL


# ---------------------------------------------------------------------------
# POST /admin/quality/judge/{compression_id} -- the per-row "Judge" button's
# route. Uses the `client` fixture (conftest.py): a real TestClient whose
# lifespan has already run, so `db._db_conn` is a live temp-DB connection and
# `quality.judge` is the process-wide singleton the route imports directly
# (`from llm_compressor import quality`) -- not the proxy forwarding shim, so
# these tests patch `quality.judge` attributes directly rather than going
# through `sys.modules["llm_compressor.proxy"]`.
# ---------------------------------------------------------------------------


def test_judge_route_returns_503_when_disabled(client):
    """quality.judge is disabled by default in the `client` fixture (no
    TYPESAFE_API_KEY set), so the route must short-circuit before touching
    the DB at all."""
    r = client.post("/admin/quality/judge/1")
    assert r.status_code == 503
    assert r.json() == {"error": "quality judging not enabled"}


def test_judge_route_returns_404_when_text_missing(client, monkeypatch):
    from llm_compressor import quality

    monkeypatch.setattr(quality.judge, "_client", MagicMock())  # enabled=True

    r = client.post("/admin/quality/judge/999999")
    assert r.status_code == 404
    assert r.json() == {"error": "compression text not found"}


def test_judge_route_returns_verdict_json_on_success(client, monkeypatch):
    from llm_compressor import db, quality

    db._db_conn.execute(
        "INSERT INTO compressions (id, ts, session_id, model, original_tokens, "
        "compressed_tokens, latency_ms, role, cache_hit, ok) VALUES "
        "(77, ?, 's', 'llmlingua2', 100, 60, 5.0, 'user', 0, 1)",
        (db.utc_now(),),
    )
    db._db_conn.execute(
        "INSERT INTO compression_texts (compression_id, original_text, compressed_text) "
        "VALUES (77, 'original', 'compressed')"
    )
    db._db_conn.commit()

    monkeypatch.setattr(quality.judge, "_client", MagicMock())
    verdict = {
        "score": 2.0,
        "label": "Preserved",
        "detail": "preserved",
        "confidence": 0.9,
        "flagged": 0,
        "meta": {"probabilities": {"2": 0.9}, "usage": {}, "model": "jev-latest"},
    }
    monkeypatch.setattr(quality.judge, "judge_now", AsyncMock(return_value=verdict))

    r = client.post("/admin/quality/judge/77")
    assert r.status_code == 200
    assert r.json() == verdict
    quality.judge.judge_now.assert_awaited_once_with(77, "original", "compressed")


# ---------------------------------------------------------------------------
# db._migrate_quality_label_detail_split -- backfill for rows judged before
# the label/detail split existed. Those rows have TypeSafe's full criteria
# sentence sitting in `label` (what's now `detail`) with `detail` NULL,
# because the `detail` column didn't exist yet when they were written. The
# frontend renders `label` as inline chip text, so an unmigrated row blows
# out the Quality column with a full sentence instead of a short chip.
#
# `db.init_db()` itself already ran this migration once when the fixture's
# schema was created (it's a one-shot, gated by a `meta` marker, same as
# `_normalize_timestamps`), so these tests clear the marker and insert
# pre-split-shaped rows before calling the migration function directly --
# mirroring how a fresh call to init_db() would find them on a real upgrade.
# ---------------------------------------------------------------------------


def test_migrate_quality_label_detail_split_backfills_compression_quality(tmp_path):
    from llm_compressor import db

    conn = db.init_db(str(tmp_path / "m.db"))
    conn.execute("DELETE FROM meta WHERE key='quality_label_detail_migrated'")
    conn.execute(
        "INSERT INTO compressions (id, ts, session_id, model, original_tokens, "
        "compressed_tokens, latency_ms, role, cache_hit, ok) VALUES "
        "(1, ?, 's', 'llmlingua2', 100, 60, 5.0, 'user', 0, 1)",
        (db.utc_now(),),
    )
    # Pre-split shape: the full legend sentence lives in `label`, `detail` is
    # NULL (the column didn't exist yet when this row was written).
    conn.execute(
        "INSERT INTO compression_quality (compression_id, score, label, detail, "
        "confidence, flagged, judged_at) VALUES (1, 1.0, "
        "'Meaning is mostly preserved but with noticeable drift — some nuance, "
        "detail, or phrasing changed enough that it could affect interpretation.', "
        "NULL, 0.8, 0, ?)",
        (db.utc_now(),),
    )
    conn.commit()

    db._migrate_quality_label_detail_split(conn)

    row = conn.execute(
        "SELECT label, detail FROM compression_quality WHERE compression_id=1"
    ).fetchone()
    assert row[0] == "Drifted"  # short chip text, derived from round(score)=1
    assert row[1].startswith("Meaning is mostly preserved")  # full sentence, now the tooltip
    conn.close()


def test_migrate_quality_label_detail_split_backfills_compression_cache(tmp_path):
    from llm_compressor import db

    conn = db.init_db(str(tmp_path / "m.db"))
    conn.execute("DELETE FROM meta WHERE key='quality_label_detail_migrated'")
    conn.execute(
        "INSERT INTO compression_cache (key, model, rate, compressed_text, "
        "original_tokens, compressed_tokens, created_at, hit_count, last_hit, "
        "quality_score, quality_label, quality_flagged, quality_detail) VALUES "
        "('k1','llmlingua2',0.5,'COMP',100,60,?,0,?,2.0,"
        "'Meaning is fully preserved — the compressed text conveys the same facts, "
        "instructions, and intent as the original.',0,NULL)",
        (db.utc_now(), db.utc_now()),
    )
    conn.commit()

    db._migrate_quality_label_detail_split(conn)

    row = conn.execute(
        "SELECT quality_label, quality_detail FROM compression_cache WHERE key='k1'"
    ).fetchone()
    assert row[0] == "Preserved"
    assert row[1].startswith("Meaning is fully preserved")
    conn.close()


def test_migrate_quality_label_detail_split_is_one_shot(tmp_path):
    """Second call is a no-op once the `meta` marker is set -- new rows written
    in the new (post-split) shape must not be touched even if `label` happens
    to look unusual."""
    from llm_compressor import db

    conn = db.init_db(str(tmp_path / "m.db"))
    conn.execute(
        "INSERT INTO compressions (id, ts, session_id, model, original_tokens, "
        "compressed_tokens, latency_ms, role, cache_hit, ok) VALUES "
        "(1, ?, 's', 'llmlingua2', 100, 60, 5.0, 'user', 0, 1)",
        (db.utc_now(),),
    )
    # Already correctly split: label is short, detail is populated. Since the
    # marker from init_db()'s own migration pass is still set, a second call
    # must see it and return immediately without rewriting this row.
    conn.execute(
        "INSERT INTO compression_quality (compression_id, score, label, detail, "
        "confidence, flagged, judged_at) VALUES (1, 2.0, 'Preserved', 'preserved', "
        "0.9, 0, ?)",
        (db.utc_now(),),
    )
    conn.commit()

    db._migrate_quality_label_detail_split(conn)

    row = conn.execute(
        "SELECT label, detail FROM compression_quality WHERE compression_id=1"
    ).fetchone()
    assert row[0] == "Preserved" and row[1] == "preserved"
    conn.close()


# --- gate_decision() classifier ---


def test_gate_decision_blocks_lost_and_drifted():
    from llm_compressor import quality
    assert quality.gate_decision(0.0) == "block"   # Lost
    assert quality.gate_decision(0.4) == "block"    # rounds to 0
    assert quality.gate_decision(1.0) == "block"    # Drifted
    assert quality.gate_decision(1.4) == "block"    # rounds to 1


def test_gate_decision_allows_preserved_and_unknown():
    from llm_compressor import quality
    assert quality.gate_decision(2.0) == "allow"    # Preserved
    assert quality.gate_decision(1.6) == "allow"    # rounds to 2
    assert quality.gate_decision(None) == "allow"   # no verdict -> savings-first


def test_status_reports_gate_config():
    from llm_compressor import quality
    st = quality.judge.status()
    assert st["block_levels"] == ["Lost", "Drifted"]
    assert st["fallback"] == "compressed"
    assert "timeout_s" in st and "gate_enabled" in st


# --- record_compression(gated=...) ---


def test_record_compression_gated_zeroes_savings(tmp_path, monkeypatch):
    from llm_compressor import db, sessions
    conn = db.init_db(str(tmp_path / "m.db"))
    monkeypatch.setattr(db, "_db_conn", conn)
    cid = sessions.record_compression(
        "sess", original=100, compressed=40, gated=1,
        original_text="x" * 300, compressed_text="y",
    )
    row = db._db_conn.execute(
        "SELECT original_tokens, compressed_tokens, gated FROM compressions WHERE id=?",
        (cid,),
    ).fetchone()
    assert tuple(row) == (100, 100, 1)  # realized: compressed forced to original
    conn.close()


# --- gate: compress_text() forwards original on Lost/Drifted, compressed
# otherwise; Jev disabled/error always falls back to compressed. ---


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """Live temp-DB connection wired up as db._db_conn, plus a loaded backend
    so `compress_text`'s `backends._pick_backend(role)` returns non-None
    (the gate tests stub `_compress_with` directly, so the backend's contents
    don't matter -- only that one is present)."""
    from llm_compressor import backends, db

    conn = db.init_db(str(tmp_path / "m.db"))
    monkeypatch.setattr(db, "_db_conn", conn)
    monkeypatch.setattr(backends, "backend", {"type": "llmlingua2", "rate": 0.5})
    monkeypatch.setattr(backends, "dual_mode", False)
    yield conn
    conn.close()


def _stub_judge(monkeypatch, verdict):
    """Make judge_now return a fixed verdict (or None) without any HTTP."""

    async def fake_judge_now(cid, original, compressed, cache_key=None):
        return verdict

    monkeypatch.setattr(quality.judge, "judge_now", fake_judge_now)
    monkeypatch.setattr(quality.judge, "_client", object())  # enabled


def _stub_compress(monkeypatch):
    monkeypatch.setattr(
        compression, "_compress_with", lambda active, text: ("COMPRESSED", 100, 40)
    )
    monkeypatch.setattr(compression, "_cache", None)  # force miss path


def test_gate_forwards_original_on_lost(monkeypatch, fresh_db):
    _stub_compress(monkeypatch)
    _stub_judge(monkeypatch, {"score": 0.0, "label": "Lost"})
    text = "word " * 60
    out = asyncio.run(compression.compress_text(text, "sess"))
    assert out == text  # original forwarded, NOT "COMPRESSED"


def test_gate_forwards_original_on_drifted(monkeypatch, fresh_db):
    _stub_compress(monkeypatch)
    _stub_judge(monkeypatch, {"score": 1.0, "label": "Drifted"})
    out = asyncio.run(compression.compress_text("word " * 60, "sess"))
    assert out == "word " * 60


def test_gate_keeps_compressed_on_preserved(monkeypatch, fresh_db):
    _stub_compress(monkeypatch)
    _stub_judge(monkeypatch, {"score": 2.0, "label": "Preserved"})
    out = asyncio.run(compression.compress_text("word " * 60, "sess"))
    assert out == "COMPRESSED"


def test_gate_keeps_compressed_when_jev_disabled(monkeypatch, fresh_db):
    _stub_compress(monkeypatch)
    monkeypatch.setattr(quality.judge, "_client", None)  # disabled
    out = asyncio.run(compression.compress_text("word " * 60, "sess"))
    assert out == "COMPRESSED"


def test_gate_keeps_compressed_on_jev_error(monkeypatch, fresh_db):
    _stub_compress(monkeypatch)

    async def boom(*a, **k):
        raise RuntimeError("jev down")

    monkeypatch.setattr(quality.judge, "judge_now", boom)
    monkeypatch.setattr(quality.judge, "_client", object())
    out = asyncio.run(compression.compress_text("word " * 60, "sess"))
    assert out == "COMPRESSED"  # error -> savings-first fallback


def test_gate_records_gated_row_on_block(monkeypatch, fresh_db):
    """The gate's DB row records realized tokens (compressed == original) and
    gated=1 when the verdict blocks -- sessions.record_compression's contract
    from Step 2, exercised end-to-end through the gate."""
    _stub_compress(monkeypatch)
    _stub_judge(monkeypatch, {"score": 0.0, "label": "Lost"})
    asyncio.run(compression.compress_text("word " * 60, "sess"))
    row = fresh_db.execute(
        "SELECT original_tokens, compressed_tokens, gated FROM compressions ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert tuple(row) == (100, 100, 1)


def test_gate_records_ungated_row_on_allow(monkeypatch, fresh_db):
    _stub_compress(monkeypatch)
    _stub_judge(monkeypatch, {"score": 2.0, "label": "Preserved"})
    asyncio.run(compression.compress_text("word " * 60, "sess"))
    row = fresh_db.execute(
        "SELECT original_tokens, compressed_tokens, gated FROM compressions ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert tuple(row) == (100, 40, 0)


# --- fix-up: compression_quality must be backfilled for auto-judged rows,
# both on a fresh miss and on the instant-gate (warm-hit-with-verdict) branch,
# or the per-row quality chip (LEFT JOIN on compression_id) never sees the
# verdict the gate already obtained. ---


def test_gate_backfills_compression_quality_on_miss(monkeypatch, fresh_db):
    """judge_now(None, ...) can't write compression_quality itself (no
    compression_id exists yet when it's called from the gate) -- _finish must
    back-fill it once sessions.record_compression() hands back a real id."""
    _stub_compress(monkeypatch)
    _stub_judge(
        monkeypatch,
        {
            "score": 2.0,
            "label": "Preserved",
            "detail": "meaning fully preserved",
            "confidence": 0.9,
            "flagged": 0,
            "meta": {"model": "jev-latest"},
        },
    )
    asyncio.run(compression.compress_text("word " * 60, "sess"))

    comp_id = fresh_db.execute("SELECT id FROM compressions ORDER BY id DESC LIMIT 1").fetchone()[
        0
    ]
    row = fresh_db.execute(
        "SELECT score, label, detail, confidence, flagged FROM compression_quality "
        "WHERE compression_id=?",
        (comp_id,),
    ).fetchone()
    assert tuple(row) == (2.0, "Preserved", "meaning fully preserved", 0.9, 0)


def test_gate_backfills_compression_quality_on_warm_hit_with_verdict(monkeypatch, fresh_db):
    """Same fix, exercised via the instant-gate branch (a cache hit whose row
    already carries a stored verdict, so no Jev call happens). The synthetic
    verdict built from the cache row must carry label/detail/flagged -- not
    just score -- so the backfilled compression_quality row has the real
    verdict, not defaults."""
    cache = compression.CompressionCache(fresh_db, max_mem=10, max_rows=100)
    monkeypatch.setattr(compression, "_cache", cache)

    text = "word " * 60
    key = compression._cache_key(text, "llmlingua2", 0.5)
    cache.put(key, "COMPRESSED", 100, 40, "llmlingua2", 0.5)
    cache.update_quality(key, 1.0, "Drifted", "meaning drifted", 1)  # warm verdict: Drifted

    asyncio.run(compression.compress_text(text, "sess"))

    comp_id = fresh_db.execute("SELECT id FROM compressions ORDER BY id DESC LIMIT 1").fetchone()[
        0
    ]
    row = fresh_db.execute(
        "SELECT score, label, detail, flagged FROM compression_quality WHERE compression_id=?",
        (comp_id,),
    ).fetchone()
    assert tuple(row) == (1.0, "Drifted", "meaning drifted", 1)


def test_gate_keeps_compressed_on_timeout(monkeypatch, fresh_db):
    """Gate timeout (GATE_TIMEOUT exceeded) falls back to forwarding compressed.

    Monkeypatches GATE_TIMEOUT to 0.01 seconds and makes judge_now sleep
    longer, so the timeout branch is exercised. Asserts that the returned
    text is the COMPRESSED text (not the original), confirming the fallback
    is savings-first."""
    _stub_compress(monkeypatch)

    # Make the judge call slow enough to exceed the timeout
    async def slow_judge_now(*args, **kwargs):
        await asyncio.sleep(0.1)  # Sleep longer than timeout
        return {"score": 2.0, "label": "Preserved"}

    monkeypatch.setattr(quality.judge, "judge_now", slow_judge_now)
    monkeypatch.setattr(quality.judge, "_client", object())  # enabled
    monkeypatch.setattr(quality, "GATE_TIMEOUT", 0.01)  # Very short timeout

    text = "word " * 60
    out = asyncio.run(compression.compress_text(text, "sess"))
    assert out == "COMPRESSED"  # timeout -> savings-first fallback
