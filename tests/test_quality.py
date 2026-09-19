"""Unit tests for quality.py's QualityJudge -- mirrors test_langfuse_tracer.py's shape.

No real network access: `_client.post` is always replaced with an AsyncMock. `init()`
itself is safe to call for real (httpx.AsyncClient(...) just builds a client object,
no socket opened), so enabled/disabled tests exercise the real constructor.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock


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
    assert s["sampled_pct"] == int(mod.SAMPLE_RATE * 100)


def test_disabled_when_api_key_blank(monkeypatch):
    j, _ = make_judge(monkeypatch, api_key="   ")
    assert not j.enabled


def test_enabled_when_api_key_set(monkeypatch):
    j, _ = make_judge(monkeypatch)
    assert j.enabled
    s = j.status()
    assert s["enabled"] is True
    assert s["api_key_set"] is True


# --- judge() dispatch / no-ops ---


def test_judge_noop_when_disabled(monkeypatch):
    j, _ = make_judge(monkeypatch, api_key=None)
    j.judge(1, "orig", "comp")  # must not raise -- returns before touching asyncio


def test_judge_noop_when_compression_id_none(monkeypatch):
    j, _ = make_judge(monkeypatch)
    j._judge = AsyncMock()

    async def run():
        j.judge(None, "orig", "comp")
        await asyncio.sleep(0)

    asyncio.run(run())
    j._judge.assert_not_awaited()


def test_judge_schedules_task_with_expected_args(monkeypatch):
    j, _ = make_judge(monkeypatch)
    j._judge = AsyncMock()

    async def run():
        j.judge(7, "original text", "compressed text", cache_key="k1")
        await asyncio.sleep(0)

    asyncio.run(run())
    j._judge.assert_awaited_once_with(7, "original text", "compressed text", "k1")


def test_judge_defaults_cache_key_to_none(monkeypatch):
    j, _ = make_judge(monkeypatch)
    j._judge = AsyncMock()

    async def run():
        j.judge(7, "a", "b")
        await asyncio.sleep(0)

    asyncio.run(run())
    j._judge.assert_awaited_once_with(7, "a", "b", None)


# --- judge_now() (awaited, on-demand path for the per-row "Judge" button) ---


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
