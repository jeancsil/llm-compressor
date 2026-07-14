"""All `@app.*` endpoints. Extracted from proxy.py in Task 14.

Every mutable global read/written here (`db._db_conn`, `backends.backend`,
`backends.backend_loading`, `backends.backend_user`, `backends.backend_system`,
`backends.dual_mode`, `backends.dual_model_system`, `backends.dual_model_user`,
`compression._cache`) is owned by its respective module and accessed
module-qualified, never via a bare `from module import name` — see
scratchpad/split-audit.md's Step 3 contract. `stats` (the in-memory aggregate
dict) is the one exception: it's mutated in place, never reassigned, so the
plain `from stats import stats` re-export stays live.

`langfuse_tracer` is imported locally (inside `langfuse_status()` and
`proxy_messages()`) rather than at module level — see app.py's docstring for
why: tests/conftest.py reloads "langfuse_tracer" fresh per test but never
reloads this module, so a module-level binding would go stale after the
first test.
"""

import copy
import json
import math
import os
import threading
import time
from datetime import datetime, timezone

import httpx
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

import backends
import compression
import db
import stats as _stats
from app import app
from sessions import record_request  # re-export target for tests: proxy.record_request
from stats import _cache_stats, stats
from templates import DASHBOARD_HTML, LIST_HTML, PLAY_HTML

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE = "https://api.anthropic.com"
COST_PER_MTOK = float(os.environ.get("COST_PER_MTOK", "3.0"))

# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------

SKIP_HEADERS = {"host", "content-length", "accept-encoding", "connection", "transfer-encoding"}


def build_headers(request: Request) -> dict:
    headers = {k: v for k, v in request.headers.items() if k.lower() not in SKIP_HEADERS}
    headers["content-type"] = "application/json"
    return headers


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
@app.head("/")
async def health():
    return {"status": "ok"}


@app.get("/stats")
async def get_stats(session_id: str | None = None):
    saved = stats["total_original_tokens"] - stats["total_compressed_tokens"]
    ratio = (
        stats["total_original_tokens"] / stats["total_compressed_tokens"]
        if stats["total_compressed_tokens"] > 0
        else 1.0
    )
    sessions_out = {}
    for sid, s in stats["sessions"].items():
        sv = s["original_tokens"] - s["compressed_tokens"]
        sessions_out[sid] = {**s, "saved_tokens": sv}

    recent = list(stats["recent_compressions"])
    if session_id:
        sessions_out = {sid: v for sid, v in sessions_out.items() if sid == session_id}
        recent = [c for c in recent if c.get("session_id", "") == session_id[:8]]
    avg_latency = sum(c["latency_ms"] for c in recent) / len(recent) if recent else 0.0

    compressor_info = _stats._compressor_info()

    by_model: list = []
    today_stats: dict = {
        "requests": 0,
        "tokens_saved": 0,
        "avg_savings_pct": 0.0,
        "avg_latency_ms": 0.0,
        "sessions": 0,
    }
    alltime_stats: dict = {
        "requests": 0,
        "tokens_saved": 0,
        "avg_savings_pct": 0.0,
        "avg_latency_ms": 0.0,
        "sessions": 0,
    }
    recent_rows: list = []
    rtk_stats: dict | None = None
    tracked_stats: dict = {"sessions": 0, "tokens_saved": 0}

    if db._db_conn is not None:
        active_model = compressor_info["model"]
        sess_args = (session_id,) if session_id else ()

        by_model_rows = db._db_conn.execute(
            f"""
            SELECT model,
                   COUNT(*) AS requests,
                   ROUND(AVG((original_tokens - compressed_tokens) * 100.0 / original_tokens), 1) AS avg_savings_pct,
                   ROUND(AVG(CAST(original_tokens AS REAL) / NULLIF(compressed_tokens, 0)), 2) AS avg_ratio,
                   SUM(original_tokens - compressed_tokens) AS total_saved,
                   ROUND(AVG(latency_ms), 1) AS avg_latency_ms
            FROM compressions
            {"WHERE session_id = ?" if session_id else ""}
            GROUP BY model
            ORDER BY requests DESC
            """,
            sess_args,
        ).fetchall()
        by_model = [dict(r) for r in by_model_rows]

        scope, scope_args = _stats._stats_scope(active_model, session_id)
        today_stats = _stats._aggregate_stats(scope, scope_args, today=True, with_ratio=False)
        alltime_stats = _stats._aggregate_stats(scope, scope_args, today=False, with_ratio=True)
        recent_rows = _stats._recent_compression_rows(active_model, session_id)
        rtk_stats = _stats._rtk_stats(session_id)
        tracked_stats = _stats._tracked_stats()
        _stats._merge_rtk_into_sessions(sessions_out)

    cache_stats = _cache_stats()

    return {
        "started_at": stats["started_at"],
        "total_requests": stats["total_requests"],
        "total_original_tokens": stats["total_original_tokens"],
        "total_compressed_tokens": stats["total_compressed_tokens"],
        "total_saved_tokens": saved,
        "overall_ratio": round(ratio, 2),
        "sessions": sessions_out,
        "recent_compressions": recent,
        "rtk": rtk_stats,
        "compressor": compressor_info,
        "cost_per_mtok": COST_PER_MTOK,
        "avg_latency_ms": round(avg_latency, 1),
        "by_model": by_model,
        "today": today_stats,
        "alltime": alltime_stats,
        "recent": recent_rows,
        "tracked": tracked_stats,
        "cache": cache_stats,
        "dual_mode": backends.dual_mode,
        "model_user": backends.backend_user.get("type") if backends.backend_user else None,
        "model_system": backends.backend_system.get("type") if backends.backend_system else None,
    }


@app.get("/stats/timeseries")
async def get_timeseries(model: str | None = None, session_id: str | None = None):
    if db._db_conn is None:
        return JSONResponse([])
    sess_filter = " AND session_id = ?" if session_id else ""
    sess_args = (session_id,) if session_id else ()
    if model:
        rows = db._db_conn.execute(
            f"""
            SELECT strftime('%Y-%m-%dT%H:00:00', ts) AS hour,
                   COUNT(*) AS requests,
                   ROUND(AVG((original_tokens - compressed_tokens) * 100.0 / original_tokens), 1) AS avg_savings_pct,
                   SUM(original_tokens - compressed_tokens) AS total_saved,
                   ROUND(AVG(latency_ms), 1) AS avg_latency_ms
            FROM compressions
            WHERE ts >= datetime('now', '-48 hours') AND model = ?{sess_filter}
            GROUP BY hour
            ORDER BY hour
            """,
            (model, *sess_args),
        ).fetchall()
    else:
        rows = db._db_conn.execute(
            f"""
            SELECT strftime('%Y-%m-%dT%H:00:00', ts) AS hour,
                   COUNT(*) AS requests,
                   ROUND(AVG((original_tokens - compressed_tokens) * 100.0 / original_tokens), 1) AS avg_savings_pct,
                   SUM(original_tokens - compressed_tokens) AS total_saved,
                   ROUND(AVG(latency_ms), 1) AS avg_latency_ms
            FROM compressions
            WHERE ts >= datetime('now', '-48 hours'){sess_filter}
            GROUP BY hour
            ORDER BY hour
            """,
            sess_args,
        ).fetchall()
    return JSONResponse([dict(r) for r in rows])


@app.post("/rtk/log")
async def rtk_log(request: Request):
    if db._db_conn is None:
        return JSONResponse({"error": "db not ready"}, status_code=503)
    body = await request.json()
    session_id = body.get("session_id", "unknown")
    try:
        db._db_conn.execute(
            """INSERT OR IGNORE INTO rtk_events
               (rtk_id, ts, session_id, rtk_cmd, input_tokens, output_tokens, saved_tokens, savings_pct, project_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                body.get("rtk_id"),
                body.get("ts", datetime.now(timezone.utc).isoformat()),
                session_id,
                body.get("rtk_cmd", ""),
                int(body.get("input_tokens", 0)),
                int(body.get("output_tokens", 0)),
                int(body.get("saved_tokens", 0)),
                float(body.get("savings_pct", 0.0)),
                body.get("project_path", ""),
            ),
        )
        db._db_conn.commit()
        return JSONResponse({"ok": True})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/dashboard/{session_id}", response_class=HTMLResponse)
async def session_dashboard(session_id: str):
    import sessions as _sessions  # local import ok; module is light

    if db._db_conn is None:
        return HTMLResponse("<h1>DB not ready</h1>", status_code=503)
    session = _sessions.get_session(db._db_conn, session_id)
    if session is None:
        return HTMLResponse(f"<h1>Session '{session_id}' not found</h1>", status_code=404)
    bootstrap = f"<script>window.SESSION = {json.dumps(session)};</script>"
    html = DASHBOARD_HTML.replace("</head>", bootstrap + "\n</head>", 1)
    return HTMLResponse(html)


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(DASHBOARD_HTML)


@app.get("/play", response_class=HTMLResponse)
async def play():
    return HTMLResponse(PLAY_HTML)


@app.get("/play/list", response_class=HTMLResponse)
async def play_list():
    return HTMLResponse(LIST_HTML)


@app.post("/play/compress")
async def play_compress(request: Request):
    body = await request.json()
    text = body.get("text", "")
    model = body.get("model", "")

    if model and model not in backends.KNOWN_MODELS:
        return JSONResponse({"error": f"Unknown model: {model}"}, status_code=400)

    orig_chars = len(text)
    orig_tokens_est = max(1, orig_chars // 4)

    active_type = backends.backend.get("type") if backends.backend else None

    if model and model != active_type:
        if backends.backend_loading == model:
            return JSONResponse({"loading": True, "model": model}, status_code=202)
        if backends.backend_loading:
            return JSONResponse(
                {"loading": True, "model": backends.backend_loading}, status_code=202
            )
        # Trigger async model switch
        if db._db_conn is not None:
            db._db_conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('current_model', ?)", (model,)
            )
            db._db_conn.commit()
        backends.backend = None
        backends.backend_loading = model
        _target = model

        def _load():
            try:
                if _target == "kompress":
                    new_backend = backends._load_kompress_backend()
                elif _target == "dual":
                    new_backend = backends._load_dual_backend()
                else:
                    new_backend = backends._load_llmlingua2_backend(backend_key=_target)
                backends.backend = new_backend
            except Exception as e:
                print(f"[play] load {_target}: {e}")
            finally:
                backends.backend_loading = None

        threading.Thread(target=_load, daemon=True).start()
        return JSONResponse({"loading": True, "model": model}, status_code=202)

    if backends.backend is None:
        if backends.backend_loading:
            return JSONResponse(
                {"loading": True, "model": backends.backend_loading}, status_code=202
            )
        return JSONResponse(
            {"error": "No model loaded. Select a model to load it."}, status_code=503
        )

    t0 = time.perf_counter()
    try:
        compressed, _orig_tok, _comp_tok = compression.compress_backend(text)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    comp_chars = len(compressed)
    comp_tokens_est = max(1, comp_chars // 4)
    char_pct = round((1 - comp_chars / max(1, orig_chars)) * 100, 1)
    token_pct = round((1 - comp_tokens_est / orig_tokens_est) * 100, 1)

    return JSONResponse(
        {
            "original": text,
            "compressed": compressed,
            "original_chars": orig_chars,
            "compressed_chars": comp_chars,
            "char_pct": char_pct,
            "original_tokens_est": orig_tokens_est,
            "compressed_tokens_est": comp_tokens_est,
            "token_pct": token_pct,
            "model": backends.backend.get("type") if backends.backend else model,
            "latency_ms": latency_ms,
        }
    )


@app.post("/admin/set-model")
async def set_model(request: Request):
    body = await request.json()
    model = body.get("model")
    if model not in backends.KNOWN_MODELS:
        return JSONResponse(
            {"error": f"Unknown model. Known: {sorted(backends.KNOWN_MODELS)}"}, status_code=400
        )
    if db._db_conn is not None:
        db._db_conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('current_model', ?)",
            (model,),
        )
        db._db_conn.commit()

    # When switching away from dual mode, clear dual globals first
    if backends.dual_mode and model != "dual":
        backends.dual_mode = False
        backends.backend_user = None
        backends.backend_system = None

    backends.backend = None
    backends.backend_loading = model

    if model == "dual":
        # Clear any previously loaded single backend globals
        backends.backend_user = None
        backends.backend_system = None

        def load_dual():
            try:
                new_backend = backends._load_dual_backend()
                backends.backend = new_backend
            except Exception as e:
                print(f"[set-model] failed to load dual: {e}")
            finally:
                backends.backend_loading = None

        threading.Thread(target=load_dual, daemon=True).start()
    else:

        def load():
            try:
                # Use model from closure directly — avoids reading db._db_conn cross-thread
                if model == "kompress":
                    new_backend = backends._load_kompress_backend()
                else:
                    new_backend = backends._load_llmlingua2_backend(backend_key=model)
                backends.backend = new_backend
            except Exception as e:
                print(f"[set-model] failed to load {model}: {e}")
            finally:
                backends.backend_loading = None

        threading.Thread(target=load, daemon=True).start()

    return JSONResponse({"status": "loading", "model": model})


@app.post("/admin/set-dual-models")
async def set_dual_models(request: Request):
    """Configure which models handle system vs user turns in dual mode.

    Body: {"system": "<model>", "user": "<model>"}
    Valid values for each: llmlingua2 | llmlingua2-large | kompress
    Omit a key to leave it unchanged.
    If dual mode is currently active, the affected sub-backends reload immediately.
    """
    body = await request.json()
    new_sys = body.get("system")
    new_usr = body.get("user")

    if not new_sys and not new_usr:
        return JSONResponse(
            {"error": "Provide at least one of 'system' or 'user'"}, status_code=400
        )
    if new_sys and new_sys not in backends.DUAL_SUBMODELS:
        return JSONResponse(
            {"error": f"Invalid system model. Valid: {list(backends.DUAL_SUBMODELS)}"},
            status_code=400,
        )
    if new_usr and new_usr not in backends.DUAL_SUBMODELS:
        return JSONResponse(
            {"error": f"Invalid user model. Valid: {list(backends.DUAL_SUBMODELS)}"},
            status_code=400,
        )

    if new_sys:
        backends.dual_model_system = new_sys
    if new_usr:
        backends.dual_model_user = new_usr

    if db._db_conn is not None:
        if new_sys:
            db._db_conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('dual_model_system', ?)",
                (new_sys,),
            )
        if new_usr:
            db._db_conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('dual_model_user', ?)", (new_usr,)
            )
        db._db_conn.commit()

    if backends.dual_mode:
        backends.backend = None
        backends.backend_loading = "dual"
        backends.backend_user = None
        backends.backend_system = None

        def reload_dual():
            try:
                new_backend = backends._load_dual_backend()
                backends.backend = new_backend
            except Exception as e:
                print(f"[set-dual-models] failed: {e}")
            finally:
                backends.backend_loading = None

        threading.Thread(target=reload_dual, daemon=True).start()
        return JSONResponse(
            {
                "status": "loading",
                "system": backends.dual_model_system,
                "user": backends.dual_model_user,
            }
        )

    return JSONResponse(
        {"status": "ok", "system": backends.dual_model_system, "user": backends.dual_model_user}
    )


@app.delete("/admin/compression-texts")
async def clear_compression_texts(request: Request):
    """Delete stored original/compressed texts without touching compression metrics."""
    if db._db_conn is None:
        return JSONResponse({"error": "db not ready"}, status_code=503)
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    session_id = body.get("session_id")
    if session_id:
        cur = db._db_conn.execute(
            "DELETE FROM compression_texts WHERE compression_id IN "
            "(SELECT id FROM compressions WHERE session_id = ?)",
            (session_id,),
        )
    else:
        cur = db._db_conn.execute("DELETE FROM compression_texts")
    db._db_conn.commit()
    return JSONResponse({"deleted": cur.rowcount, "session_id": session_id})


@app.get("/admin/sessions")
async def get_sessions(page: int = 1, page_size: int = 25):
    import sessions as _sessions  # local import ok; module is light

    return _sessions.list_sessions(db._db_conn, page, page_size)


@app.get("/admin/langfuse-status")
async def langfuse_status():
    import langfuse_tracer  # local: see app.py's docstring — avoids stale-tracer binding

    return JSONResponse(content=langfuse_tracer.tracer.status())


@app.patch("/session/{session_id}/name")
async def rename_session_endpoint(session_id: str, request: Request):
    body = await request.json()
    name = (body.get("name") or "").strip()
    if not name:
        return JSONResponse({"error": "name required"}, status_code=400)
    import sessions
    ok = sessions.rename_session(db._db_conn, session_id, name)
    if not ok:
        return JSONResponse({"error": "session not found"}, status_code=404)
    return JSONResponse({"ok": True, "display_name": name, "name_source": "manual"})


@app.get("/session/{session_id}/compressions")
async def get_session_compressions(session_id: str, page: int = 1, page_size: int = 20):
    if db._db_conn is None:
        return JSONResponse({"error": "db not ready"}, status_code=503)

    # Clamp page and page_size BEFORE any early returns
    page = max(1, page)
    page_size = max(1, min(200, page_size))

    offset = (page - 1) * page_size

    total = db._db_conn.execute(
        "SELECT COUNT(*) FROM compressions WHERE session_id=?", (session_id,)
    ).fetchone()[0]
    rows = db._db_conn.execute(
        """
        SELECT c.id, c.ts, c.model, c.original_tokens, c.compressed_tokens,
               ROUND((c.original_tokens - c.compressed_tokens) * 100.0 / c.original_tokens, 1) AS savings_pct,
               c.latency_ms,
               ct.original_text, ct.compressed_text
        FROM compressions c
        LEFT JOIN compression_texts ct ON ct.compression_id = c.id
        WHERE c.session_id = ?
        ORDER BY c.id DESC
        LIMIT ? OFFSET ?
        """,
        (session_id, page_size, offset),
    ).fetchall()
    pages = math.ceil(total / page_size) if page_size > 0 else 0
    return JSONResponse(
        {
            "items": [dict(r) for r in rows],
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": pages,
        }
    )


@app.get("/session/{session_id}/rtk-commands")
async def get_session_rtk_commands(session_id: str, page: int = 1, page_size: int = 25):
    page = max(1, page)
    page_size = max(1, min(200, page_size))
    if db._db_conn is None:
        return JSONResponse({"error": "db not ready"}, status_code=503)
    offset = (page - 1) * page_size

    total = db._db_conn.execute(
        "SELECT COUNT(*) FROM rtk_events WHERE session_id=?", (session_id,)
    ).fetchone()[0]
    rows = db._db_conn.execute(
        """
        SELECT id, ts, rtk_cmd, input_tokens, output_tokens, saved_tokens,
               ROUND(savings_pct, 1) AS savings_pct, project_path
        FROM rtk_events
        WHERE session_id = ?
        ORDER BY id DESC
        LIMIT ? OFFSET ?
        """,
        (session_id, page_size, offset),
    ).fetchall()
    pages = math.ceil(total / page_size) if page_size > 0 else 0
    return JSONResponse(
        {
            "items": [dict(r) for r in rows],
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": pages,
        }
    )


@app.get("/v1/models")
async def list_models(request: Request):
    headers = build_headers(request)
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{ANTHROPIC_BASE}/v1/models",
            headers=headers,
            params=dict(request.query_params),
        )
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.post("/v1/messages")
async def proxy_messages(request: Request):
    import langfuse_tracer  # local: see app.py's docstring — avoids stale-tracer binding

    session_id = request.headers.get("x-claude-code-session-id", "unknown")
    import naming as _naming
    import sessions as _sessions  # local import ok; module is light

    _sessions.ensure_session(db._db_conn, session_id)
    record_request(session_id)

    body = await request.json()
    try:
        # Cheap pre-check: skip all work once the session is named or already being named.
        _row = _sessions.get_session(db._db_conn, session_id) if db._db_conn else None
        if _row and _row["name_source"] == "provisional":
            _user_turns = [
                (
                    m.get("content")
                    if isinstance(m.get("content"), str)
                    else " ".join(
                        b.get("text", "") for b in m.get("content", []) if isinstance(b, dict)
                    )
                )
                for m in body.get("messages", [])
                if m.get("role") == "user"
            ]
            _signal = _naming.clean_signal([t for t in _user_turns if t])
            # Atomically claim the row BEFORE dispatching. Claude Code fires several
            # /v1/messages per human turn; only the single winner of the claim
            # dispatches, so we never start two naming tasks (spec: "exactly one call").
            if _signal and _sessions.claim_for_naming(db._db_conn, session_id):
                _use_llm = os.environ.get("LLM_COMPRESSOR_LLM_NAMING") == "1"
                _auth = {
                    k: v
                    for k, v in request.headers.items()
                    if k.lower() in ("authorization", "anthropic-version", "anthropic-beta")
                }
                _naming.schedule_naming(db._db_conn, session_id, _signal, _auth, _use_llm)
    except Exception as _exc:  # naming must never break the proxy path
        print(f"[naming] trigger skipped: {_exc}")
    _ls_start_ms = time.monotonic() * 1000
    _ls_original_messages = copy.deepcopy(body.get("messages", []))
    _ls_original_system = copy.deepcopy(body.get("system"))
    if body.get("system"):
        body["system"] = compression.compress_system_field(body["system"], session_id)
    body["messages"] = compression.compress_messages(body["messages"], session_id)
    _ls_compression_latency_ms = round(time.monotonic() * 1000 - _ls_start_ms, 1)
    _sess = stats["sessions"].get(session_id, {})
    _ls_original_tokens = _sess.get("original_tokens", 0)
    _ls_compressed_tokens = _sess.get("compressed_tokens", 0)
    _ls_compression_ratio = (
        round(_ls_compressed_tokens / _ls_original_tokens, 3) if _ls_original_tokens else 1.0
    )
    _ls_tokens_saved = _ls_original_tokens - _ls_compressed_tokens
    _ls_compression_model = (backends.backend or {}).get("type", "unknown")
    _ls_cache_hit = False
    headers = build_headers(request)
    is_streaming = body.get("stream", False)

    if is_streaming:

        async def stream_gen():
            _chunks: list[bytes] = []
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream(
                    "POST",
                    f"{ANTHROPIC_BASE}/v1/messages",
                    headers=headers,
                    json=body,
                    params=dict(request.query_params),
                ) as resp:
                    if resp.status_code >= 400:
                        body_bytes = await resp.aread()
                        print(f"[proxy] Anthropic error {resp.status_code}: {body_bytes.decode()}")
                        yield body_bytes
                        return
                    async for chunk in resp.aiter_bytes():
                        _chunks.append(chunk)
                        yield chunk
            if langfuse_tracer.tracer.enabled:
                _end_ms = time.monotonic() * 1000
                _response_text = compression._extract_text_from_sse(b"".join(_chunks))
                await langfuse_tracer.tracer.log_request(
                    original_messages=_ls_original_messages,
                    compressed_messages=body.get("messages", []),
                    original_system=_ls_original_system,
                    compressed_system=body.get("system"),
                    response_text=_response_text,
                    metadata={
                        "session_id": session_id,
                        "anthropic_model": body.get("model", "unknown"),
                        "compression_model": _ls_compression_model,
                        "compression_ratio": _ls_compression_ratio,
                        "tokens_saved": _ls_tokens_saved,
                        "original_tokens": _ls_original_tokens,
                        "compressed_tokens": _ls_compressed_tokens,
                        "compression_latency_ms": _ls_compression_latency_ms,
                        "total_latency_ms": round(_end_ms - _ls_start_ms, 1),
                        "cache_hit": _ls_cache_hit,
                        "streaming": True,
                    },
                    tags=[
                        _ls_compression_model,
                        "streaming",
                        f"ratio:{int(_ls_compression_ratio * 100)}pct",
                    ],
                )
                if stats["total_requests"] % 10 == 0:
                    await langfuse_tracer.tracer.add_to_dataset(
                        run_inputs={
                            "original_messages": _ls_original_messages,
                            "original_system": _ls_original_system,
                        },
                        run_outputs={"response": _response_text},
                    )

        return StreamingResponse(stream_gen(), media_type="text/event-stream")
    else:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{ANTHROPIC_BASE}/v1/messages",
                headers=headers,
                json=body,
                params=dict(request.query_params),
            )
        if resp.status_code >= 400:
            print(f"[proxy] Anthropic error {resp.status_code}: {resp.text}")
        resp_data = resp.json()
        _end_ms = time.monotonic() * 1000
        if langfuse_tracer.tracer.enabled:
            _response_text = ""
            try:
                _response_text = (resp_data.get("content") or [{}])[0].get("text", "")
            except Exception:
                pass
            await langfuse_tracer.tracer.log_request(
                original_messages=_ls_original_messages,
                compressed_messages=body.get("messages", []),
                original_system=_ls_original_system,
                compressed_system=body.get("system"),
                response_text=_response_text,
                metadata={
                    "session_id": session_id,
                    "anthropic_model": body.get("model", "unknown"),
                    "compression_model": _ls_compression_model,
                    "compression_ratio": _ls_compression_ratio,
                    "tokens_saved": _ls_tokens_saved,
                    "original_tokens": _ls_original_tokens,
                    "compressed_tokens": _ls_compressed_tokens,
                    "compression_latency_ms": _ls_compression_latency_ms,
                    "total_latency_ms": round(_end_ms - _ls_start_ms, 1),
                    "cache_hit": _ls_cache_hit,
                    "streaming": False,
                },
                tags=[
                    _ls_compression_model,
                    "sync",
                    f"ratio:{int(_ls_compression_ratio * 100)}pct",
                ],
            )
            if stats["total_requests"] % 10 == 0:
                await langfuse_tracer.tracer.add_to_dataset(
                    run_inputs={
                        "original_messages": _ls_original_messages,
                        "original_system": _ls_original_system,
                    },
                    run_outputs={"response": _response_text},
                )
        return JSONResponse(content=resp_data, status_code=resp.status_code)
