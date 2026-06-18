import os
import json
import platform
import re
import sqlite3
import time
import httpx
import uvicorn
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from llmlingua import PromptCompressor

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE    = "https://api.anthropic.com"
DB_PATH           = "metrics.db"
COST_PER_MTOK     = float(os.environ.get("COST_PER_MTOK", "3.0"))

# Module-level globals populated by lifespan
backend   = None
_db_conn  = None


# ---------------------------------------------------------------------------
# Stub helpers (replaced in later tasks)
# ---------------------------------------------------------------------------

def init_db(path: str):
    import sqlite3 as _sqlite3
    conn = _sqlite3.connect(path)
    conn.row_factory = _sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS compressions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            ts               TEXT,
            session_id       TEXT,
            model            TEXT,
            original_tokens  INTEGER,
            compressed_tokens INTEGER,
            latency_ms       REAL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON compressions(ts)")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
    )
    conn.commit()
    return conn


def migrate_from_json(conn, json_path: str = "stats.json") -> None:
    path = Path(json_path)
    if not path.exists():
        return
    existing = conn.execute("SELECT COUNT(*) FROM compressions").fetchone()[0]
    if existing:
        return
    try:
        import shutil
        data = json.loads(path.read_text())
        rows = data.get("recent_compressions", [])
        bak = path.with_suffix(".json.bak")
        shutil.copy2(path, bak)
        print(f"[migration] Backed {path} → {bak}")
        conn.executemany(
            "INSERT INTO compressions (ts, session_id, model, original_tokens, compressed_tokens, latency_ms) VALUES (?,?,'llmlingua2',?,?,0.0)",
            [
                (
                    datetime.now().strftime("%Y-%m-%dT%H:%M:%S") if not r.get("ts") else r["ts"],
                    r.get("session_id", ""),
                    r.get("original", 0),
                    r.get("compressed", 0),
                )
                for r in rows
            ],
        )
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM compressions").fetchone()[0]
        print(f"[migration] Imported {count} rows {path} → metrics.db. Backup {bak}.")
    except Exception as e:
        print(f"[migration] Failed, skipping: {e}")


def recover_stats_from_backup(conn, bak_path: str = "stats.json.bak") -> None:
    """Import full session history from stats.json.bak.

    The initial migration only captured recent_compressions (≤100 rows). This inserts
    one residual synthetic row per session for the token delta not yet in the DB,
    then stores a legacy_request_offset so load_stats_from_db produces the correct total.
    """
    path = Path(bak_path)
    if not path.exists():
        return
    if conn.execute("SELECT value FROM meta WHERE key='backup_recovered'").fetchone():
        return
    try:
        data = json.loads(path.read_text())
        sessions = data.get("sessions", {})
        rows_inserted = 0
        for session_id, sess in sessions.items():
            existing = conn.execute(
                "SELECT COALESCE(SUM(original_tokens),0), COALESCE(SUM(compressed_tokens),0) "
                "FROM compressions WHERE session_id=?",
                (session_id,),
            ).fetchone()
            remaining_orig = int(sess.get("original_tokens", 0)) - int(existing[0])
            remaining_comp = int(sess.get("compressed_tokens", 0)) - int(existing[1])
            if remaining_orig > 0:
                ts = (sess.get("last_seen") or sess.get("first_seen") or datetime.now().isoformat())[:19]
                conn.execute(
                    "INSERT INTO compressions (ts, session_id, model, original_tokens, compressed_tokens, latency_ms) "
                    "VALUES (?,?,'llmlingua2',?,?,0.0)",
                    (ts, session_id, remaining_orig, max(0, remaining_comp)),
                )
                rows_inserted += 1

        db_rows = conn.execute("SELECT COUNT(*) FROM compressions").fetchone()[0]
        total_requests = int(data.get("total_requests", 0))
        legacy_offset = max(0, total_requests - db_rows)
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('legacy_request_offset', ?)", (str(legacy_offset),))
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('backup_recovered', '1')")
        conn.commit()
        print(f"[recovery] {rows_inserted} synthetic rows from {bak_path}. Request offset: {legacy_offset}.")
    except Exception as e:
        print(f"[recovery] Failed: {e}")


def load_stats_from_db(conn) -> None:
    row = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(original_tokens),0), COALESCE(SUM(compressed_tokens),0) FROM compressions"
    ).fetchone()
    try:
        offset_row = conn.execute("SELECT value FROM meta WHERE key='legacy_request_offset'").fetchone()
        legacy_offset = int(offset_row[0]) if offset_row else 0
    except Exception:
        legacy_offset = 0
    stats["total_requests"] = row[0] + legacy_offset
    stats["total_original_tokens"] = row[1]
    stats["total_compressed_tokens"] = row[2]

    for r in conn.execute(
        "SELECT session_id, COUNT(*), SUM(original_tokens), SUM(compressed_tokens), MIN(ts), MAX(ts) FROM compressions GROUP BY session_id"
    ):
        stats["sessions"][r[0]] = {
            "requests":          r[1],
            "original_tokens":   r[2],
            "compressed_tokens": r[3],
            "first_seen":        r[4],
            "last_seen":         r[5],
            "name":              None,
        }

    for r in conn.execute(
        "SELECT ts, session_id, original_tokens, compressed_tokens, latency_ms FROM compressions ORDER BY id DESC LIMIT 100"
    ):
        saved = r[2] - r[3]
        stats["recent_compressions"].append({
            "ts":         r[0][11:19],
            "session_id": r[1][:8],
            "original":   r[2],
            "compressed": r[3],
            "saved":      saved,
            "latency_ms": r[4],
        })

    print(
        f"[stats] Loaded from metrics.db: "
        f"{stats['total_original_tokens']} original, {stats['total_compressed_tokens']} compressed, "
        f"{len(stats['sessions'])} sessions"
    )


LLMLINGUA2_MODELS = {
    "llmlingua2": "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    "llmlingua2-large": "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
}


def _load_llmlingua2_backend() -> dict:
    """Load the LLMLingua-2 PromptCompressor and return a backend dict."""
    from llmlingua import PromptCompressor
    rate = float(os.environ.get("COMPRESS_RATE", "0.5"))
    backend_key = os.environ.get("COMPRESSOR_MODEL", "llmlingua2")
    model_name = LLMLINGUA2_MODELS.get(backend_key, LLMLINGUA2_MODELS["llmlingua2"])
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading LLMLingua-2 model ({backend_key}: {model_name})...")
    c = PromptCompressor(
        model_name=model_name,
        use_llmlingua2=True,
        device_map=device,
    )
    print(f"Model ready. (device={device})")
    return {"type": "llmlingua2", "backend_key": backend_key, "compressor": c, "rate": rate}


def _load_kompress_backend() -> dict:
    """Load chopratejas/kompress-v2-base via headroom-ai[ml].

    Auto mode tries ONNX CPU first (not in public HF repo, will skip) then
    falls back to PyTorch on MPS/CPU using model.safetensors (~600 MB).
    """
    try:
        from headroom.transforms.kompress_compressor import KompressCompressor, KompressConfig
    except ImportError:
        raise RuntimeError(
            "headroom-ai[ml] is not installed. Run: pip install 'headroom-ai[ml]'"
        )
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    threshold = float(os.environ.get("COMPRESS_THRESHOLD", "0.5"))
    print(f"Loading kompress-v2-base (device={device}, threshold={threshold})...")
    config = KompressConfig(device=device, score_threshold=threshold)
    compressor = KompressCompressor(config=config)
    compressor.preload()
    print(f"kompress-v2-base ready.")
    return {"type": "kompress", "compressor": compressor, "threshold": threshold}


def load_backend() -> dict:
    """Dispatch to the configured backend loader based on COMPRESSOR_MODEL."""
    model_name = os.environ.get("COMPRESSOR_MODEL", "llmlingua2")
    if model_name == "kompress":
        return _load_kompress_backend()
    return _load_llmlingua2_backend()


# Keep the private alias so existing call-sites (lifespan, tests) still work.
_load_backend = load_backend


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global backend, _db_conn
    _db_conn = init_db(DB_PATH)
    migrate_from_json(_db_conn)
    recover_stats_from_backup(_db_conn)
    load_stats_from_db(_db_conn)
    backend = _load_backend()
    yield
    if _db_conn:
        _db_conn.close()


app = FastAPI(lifespan=lifespan)

# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

stats = {
    "started_at": datetime.now().isoformat(),
    "total_requests": 0,
    "total_original_tokens": 0,
    "total_compressed_tokens": 0,
    "sessions": {},
    "recent_compressions": deque(maxlen=100),
}

def record_compression(session_id: str, original: int, compressed: int, latency_ms: float = 0.0):
    stats["total_original_tokens"] += original
    stats["total_compressed_tokens"] += compressed

    model_name = backend.get("type", "llmlingua2") if backend else "llmlingua2"
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if _db_conn:
        _db_conn.execute(
            "INSERT INTO compressions (ts, session_id, model, original_tokens, compressed_tokens, latency_ms) VALUES (?,?,?,?,?,?)",
            (ts, session_id, model_name, original, compressed, latency_ms),
        )
        _db_conn.commit()

    sess = stats["sessions"].setdefault(session_id, {
        "first_seen": ts,
        "requests": 0,
        "original_tokens": 0,
        "compressed_tokens": 0,
    })
    sess["original_tokens"] += original
    sess["compressed_tokens"] += compressed
    sess["last_seen"] = ts

    stats["recent_compressions"].appendleft({
        "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        "session_id": session_id[:8],
        "original": original,
        "compressed": compressed,
        "saved": original - compressed,
        "latency_ms": round(latency_ms, 1),
    })

def record_request(session_id: str, session_name: str | None = None):
    stats["total_requests"] += 1
    sess = stats["sessions"].setdefault(session_id, {
        "first_seen": datetime.now().isoformat(),
        "requests": 0,
        "original_tokens": 0,
        "compressed_tokens": 0,
        "name": None,
    })
    sess["requests"] += 1
    sess["last_seen"] = datetime.now().isoformat()
    if session_name:
        sess["name"] = session_name

# ---------------------------------------------------------------------------
# rtk integration (optional — gracefully absent when rtk not installed)
# ---------------------------------------------------------------------------

def _rtk_db_path() -> Path:
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "rtk" / "history.db"
    if system == "Windows":
        return Path(os.environ.get("APPDATA", "")) / "rtk" / "history.db"
    return Path.home() / ".local" / "share" / "rtk" / "history.db"

def read_rtk_stats() -> dict | None:
    db = _rtk_db_path()
    if not db.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        row = cur.execute(
            "SELECT COUNT(*) as n, SUM(input_tokens) as inp, "
            "SUM(output_tokens) as out, SUM(saved_tokens) as saved, "
            "AVG(savings_pct) as avg_pct FROM commands"
        ).fetchone()

        top = cur.execute(
            "SELECT rtk_cmd, COUNT(*) as cnt, SUM(saved_tokens) as saved, "
            "AVG(savings_pct) as avg_pct FROM commands "
            "GROUP BY rtk_cmd ORDER BY saved DESC LIMIT 8"
        ).fetchall()

        conn.close()
        return {
            "total_commands":      row["n"]       or 0,
            "total_input_tokens":  row["inp"]      or 0,
            "total_output_tokens": row["out"]      or 0,
            "total_saved_tokens":  row["saved"]    or 0,
            "avg_savings_pct":     round(row["avg_pct"] or 0, 1),
            "top_commands": [
                {"cmd": r["rtk_cmd"], "count": r["cnt"],
                 "saved": r["saved"], "avg_pct": round(r["avg_pct"], 1)}
                for r in top
            ],
        }
    except Exception as e:
        print(f"[rtk] could not read tracking db: {e}")
        return None

# ---------------------------------------------------------------------------
# Chunking helpers (prevent BERT 512-token overflow in LLMLingua-2)
# ---------------------------------------------------------------------------

CHUNK_MAX_TOKENS = 400


def _count_tokens(text: str) -> int:
    """Count tokens using the backend tokenizer (falls back to whitespace split)."""
    try:
        return len(backend["compressor"].tokenizer.tokenize(text))
    except Exception:
        return len(text.split())


def _split_into_segments(text: str) -> list[str]:
    """Return a flat list of natural-boundary segments, splitting finer as needed."""
    paras = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    if len(paras) > 1:
        segs: list[str] = []
        for p in paras:
            if _count_tokens(p) > CHUNK_MAX_TOKENS:
                lines = [ln.strip() for ln in p.split("\n") if ln.strip()]
                if len(lines) > 1:
                    segs.extend(lines)
                else:
                    segs.extend(
                        s.strip()
                        for s in re.split(r"(?<=[?.!])\s+", p)
                        if s.strip()
                    )
            else:
                segs.append(p)
        return segs
    # Single paragraph — try sentence splitting
    sents = [s.strip() for s in re.split(r"(?<=[?.!])\s+", text) if s.strip()]
    if len(sents) > 1:
        return sents
    return [text]


def chunk_text(text: str) -> list[str]:
    """Group segments into chunks of at most CHUNK_MAX_TOKENS tokens each."""
    if _count_tokens(text) <= CHUNK_MAX_TOKENS:
        return [text]
    segments = _split_into_segments(text)
    chunks: list[str] = []
    buf: list[str] = []
    buf_count = 0
    for seg in segments:
        seg_count = _count_tokens(seg)
        if buf and buf_count + seg_count > CHUNK_MAX_TOKENS:
            chunks.append("\n\n".join(buf))
            buf = []
            buf_count = 0
        buf.append(seg)
        buf_count += seg_count
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------

def compress_text(text: str, session_id: str) -> str:
    if len(text) <= 200:
        return text
    if backend is None:
        return text
    t0 = time.perf_counter()
    try:
        compressed, orig, comp = compress_backend(text)
    except Exception as e:
        print(f"[compressor] compression failed, forwarding original: {e}")
        return text
    latency_ms = (time.perf_counter() - t0) * 1000
    model_tag = backend.get("type", "compressor") if backend else "compressor"
    print(f"[{model_tag}] {orig} → {comp} tokens [{session_id[:8]}]")
    record_compression(session_id, orig, comp, latency_ms)
    return compressed


def compress_backend(text: str):
    """Dispatch to the configured compressor. Returns (compressed_text, orig_tokens, comp_tokens)."""
    if backend.get("type") == "kompress":
        return _compress_kompress(text)
    return _compress_llmlingua2(text)


def _compress_llmlingua2(text: str):
    chunks = chunk_text(text)
    if len(chunks) == 1:
        result = backend["compressor"].compress_prompt(
            chunks[0],
            rate=backend.get("rate", 0.5),
            force_tokens=["\n", "?", ".", "!"],
        )
        return result["compressed_prompt"], result["origin_tokens"], result["compressed_tokens"]

    parts: list[str] = []
    total_orig = 0
    total_comp = 0
    for chunk in chunks:
        result = backend["compressor"].compress_prompt(
            chunk,
            rate=backend.get("rate", 0.5),
            force_tokens=["\n", "?", ".", "!"],
        )
        parts.append(result["compressed_prompt"])
        total_orig += result["origin_tokens"]
        total_comp += result["compressed_tokens"]
    return "\n\n".join(parts), total_orig, total_comp


def _compress_kompress(text: str):
    result = backend["compressor"].compress(text)
    return result.compressed, result.original_tokens, result.compressed_tokens

def compress_messages(messages: list, session_id: str) -> list:
    out = []
    for msg in messages:
        if msg.get("role") != "user":
            out.append(msg)
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            out.append({**msg, "content": compress_text(content, session_id)})
        elif isinstance(content, list):
            new_blocks = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    new_blocks.append({**block, "text": compress_text(block["text"], session_id)})
                else:
                    new_blocks.append(block)
            out.append({**msg, "content": new_blocks})
        else:
            out.append(msg)
    return out

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
async def get_stats():
    saved = stats["total_original_tokens"] - stats["total_compressed_tokens"]
    ratio = (stats["total_original_tokens"] / stats["total_compressed_tokens"]
             if stats["total_compressed_tokens"] > 0 else 1.0)
    sessions_out = {}
    for sid, s in stats["sessions"].items():
        sv = s["original_tokens"] - s["compressed_tokens"]
        sessions_out[sid] = {**s, "saved_tokens": sv}

    recent = list(stats["recent_compressions"])
    avg_latency = (
        sum(c["latency_ms"] for c in recent) / len(recent) if recent else 0.0
    )

    if backend and backend.get("type") == "kompress":
        compressor_info = {
            "model": "kompress",
            "param_name": "threshold",
            "param_value": backend.get("threshold", 0.5),
        }
    else:
        backend_key = backend.get("backend_key", "llmlingua2") if backend else "llmlingua2"
        compressor_info = {
            "model": backend_key,
            "param_name": "rate",
            "param_value": backend.get("rate", 0.5) if backend else 0.5,
        }

    return {
        "started_at": stats["started_at"],
        "total_requests": stats["total_requests"],
        "total_original_tokens": stats["total_original_tokens"],
        "total_compressed_tokens": stats["total_compressed_tokens"],
        "total_saved_tokens": saved,
        "overall_ratio": round(ratio, 2),
        "sessions": sessions_out,
        "recent_compressions": recent,
        "rtk": read_rtk_stats(),
        "compressor": compressor_info,
        "cost_per_mtok": COST_PER_MTOK,
        "avg_latency_ms": round(avg_latency, 1),
    }


@app.get("/stats/timeseries")
async def get_timeseries():
    if _db_conn is None:
        return JSONResponse([])
    rows = _db_conn.execute(
        """
        SELECT strftime('%Y-%m-%dT%H:00:00', ts) AS hour,
               COUNT(*) AS requests,
               ROUND(AVG((original_tokens - compressed_tokens) * 100.0 / original_tokens), 1) AS avg_savings_pct,
               SUM(original_tokens - compressed_tokens) AS total_saved,
               ROUND(AVG(latency_ms), 1) AS avg_latency_ms
        FROM compressions
        WHERE ts >= datetime('now', '-48 hours')
        GROUP BY hour
        ORDER BY hour
        """
    ).fetchall()
    return JSONResponse([dict(r) for r in rows])


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(DASHBOARD_HTML)


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
    session_id = request.headers.get("x-claude-code-session-id", "unknown")
    # Claude Code may send a session name — log all x-claude-code-* headers once to inspect
    cc_headers = {k: v for k, v in request.headers.items() if "claude" in k.lower() or "session" in k.lower()}
    print(f"[headers] {cc_headers}")
    session_name = (
        request.headers.get("x-claude-code-session-name")
        or request.headers.get("x-session-name")
    )
    record_request(session_id, session_name)

    body = await request.json()
    body["messages"] = compress_messages(body["messages"], session_id)
    headers = build_headers(request)
    is_streaming = body.get("stream", False)

    if is_streaming:
        async def stream_gen():
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
                        yield chunk
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
        return JSONResponse(content=resp.json(), status_code=resp.status_code)


# ---------------------------------------------------------------------------
# Dashboard HTML
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLMLingua Proxy</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'SF Mono', 'Fira Code', monospace; background: #0d1117; color: #c9d1d9; padding: 20px; max-width: 1140px; margin: 0 auto; }

  /* ── Header ── */
  .header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #21262d; }
  .title { color: #58a6ff; font-size: 14px; font-weight: 700; letter-spacing: .04em; }
  .header-right { display: flex; align-items: center; gap: 16px; font-size: 11px; color: #8b949e; }
  .live-dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #3fb950; margin-right: 4px; animation: pulse 2s infinite; vertical-align: middle; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

  /* ── Two-layer hero (shown when rtk present) ── */
  .layers { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }
  .layer { border: 1px solid #30363d; border-radius: 8px; padding: 18px 20px; }
  .layer-shell { background: #111a12; border-color: #2a4a2e; }
  .layer-api   { background: #111520; border-color: #1e2f50; }
  .layer-label { font-size: 9px; text-transform: uppercase; letter-spacing: .1em; margin-bottom: 10px; display: flex; align-items: center; gap: 8px; }
  .layer-shell .layer-label { color: #3fb950; }
  .layer-api   .layer-label { color: #58a6ff; }
  .layer-badge { font-size: 9px; padding: 1px 6px; border-radius: 3px; }
  .layer-shell .layer-badge { background: #0d2b1a; color: #3fb950; }
  .layer-api   .layer-badge { background: #0d1a35; color: #58a6ff; }
  .layer-pct { font-size: 52px; font-weight: 800; line-height: 1; letter-spacing: -.02em; }
  .layer-shell .layer-pct { color: #3fb950; }
  .layer-api   .layer-pct { color: #58a6ff; }
  .layer-sub { font-size: 10px; color: #8b949e; margin-top: 6px; }
  .layer-stats { margin-top: 12px; display: flex; flex-direction: column; gap: 3px; }
  .layer-stat { display: flex; justify-content: space-between; font-size: 11px; }
  .layer-stat-k { color: #8b949e; }
  .layer-stat-v { color: #c9d1d9; font-weight: 600; }

  /* ── Single hero (rtk absent) ── */
  .hero-solo { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }
  .hero-left { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 28px 20px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; }
  .hero-pct { font-size: 68px; font-weight: 800; color: #3fb950; line-height: 1; letter-spacing: -.02em; }
  .hero-label { font-size: 10px; color: #8b949e; margin-top: 8px; text-transform: uppercase; letter-spacing: .1em; }
  .hero-sublabel { font-size: 9px; color: #484f58; margin-top: 3px; }
  .hero-right { background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 18px 20px; }
  .term-prompt { color: #3fb950; margin-bottom: 10px; font-size: 12px; }
  .term-line { display: flex; gap: 8px; padding: 2px 0; font-size: 12px; }
  .term-key { color: #8b949e; flex: 1; white-space: nowrap; }
  .term-val { color: #f0f6fc; font-weight: 600; text-align: right; }
  .term-val.green { color: #3fb950; }
  .term-val.blue { color: #58a6ff; }

  /* ── rtk install hint ── */
  .rtk-hint { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px 16px; margin-bottom: 12px; font-size: 11px; color: #8b949e; display: flex; align-items: center; gap: 10px; }
  .rtk-hint code { background: #21262d; padding: 1px 6px; border-radius: 3px; color: #c9d1d9; }

  /* ── Metric cards ── */
  .cards { display: grid; grid-template-columns: repeat(7, 1fr); gap: 10px; margin-bottom: 12px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px 14px; }
  .card-label { font-size: 9px; color: #8b949e; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 5px; }
  .card-value { font-size: 20px; font-weight: 700; color: #f0f6fc; }
  .card-value.blue  { color: #58a6ff; }
  .card-value.green { color: #3fb950; }

  /* ── Shared section ── */
  .section { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px 16px; margin-bottom: 12px; }
  .section-title { font-size: 9px; color: #8b949e; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 12px; }

  /* ── rtk command table ── */
  .cmd-table { width: 100%; border-collapse: collapse; font-size: 11px; }
  .cmd-table th { text-align: left; color: #8b949e; padding: 4px 8px; border-bottom: 1px solid #21262d; font-weight: normal; font-size: 9px; text-transform: uppercase; letter-spacing: .06em; }
  .cmd-table td { padding: 6px 8px; border-bottom: 1px solid #21262d; vertical-align: middle; }
  .cmd-table tr:last-child td { border-bottom: none; }
  .cmd-table tr:hover td { background: #1c2128; }
  .cmd-name { color: #c9d1d9; font-family: inherit; }
  .cmd-bar-wrap { width: 120px; background: #21262d; border-radius: 2px; height: 8px; display: inline-block; vertical-align: middle; }
  .cmd-bar-fill { height: 100%; border-radius: 2px; background: #3fb950; }

  /* ── Sparkline ── */
  .sparkline { display: flex; align-items: flex-end; gap: 2px; height: 44px; overflow: hidden; }
  .bar { width: 9px; min-height: 4px; border-radius: 2px 2px 0 0; cursor: default; opacity: .85; transition: opacity .1s; flex-shrink: 0; }
  .bar:hover { opacity: 1; }
  .spark-empty { font-size: 11px; color: #484f58; }
  .spark-legend { display: flex; gap: 16px; margin-top: 8px; font-size: 10px; color: #8b949e; }
  .leg-dot { display: inline-block; width: 8px; height: 8px; border-radius: 1px; margin-right: 4px; vertical-align: middle; }

  /* ── Model badge ── */
  .model-badge { font-size: 10px; background: #0d1a35; color: #58a6ff; padding: 2px 8px; border-radius: 4px; }

  /* ── Time-series chart ── */
  .ts-chart { display: flex; align-items: flex-end; gap: 4px; height: 60px; overflow: hidden; }
  .ts-bar { flex: 1; min-height: 4px; border-radius: 2px 2px 0 0; opacity: .8; transition: opacity .1s; cursor: default; }
  .ts-bar:hover { opacity: 1; }
  .ts-labels { display: flex; justify-content: space-between; margin-top: 4px; font-size: 9px; color: #484f58; }

  /* ── Session bars ── */
  .sess-bars { display: flex; flex-direction: column; gap: 8px; }
  .sess-bar-row { display: flex; align-items: center; gap: 10px; font-size: 11px; }
  .sess-bar-label { width: 90px; color: #8b949e; flex-shrink: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .sess-bar-wrap { flex: 1; background: #21262d; border-radius: 3px; height: 12px; overflow: hidden; }
  .sess-bar-fill { height: 100%; border-radius: 3px; transition: width .3s; }
  .sess-bar-pct   { width: 36px; text-align: right; color: #3fb950; flex-shrink: 0; }
  .sess-bar-saved { width: 54px; text-align: right; color: #484f58; flex-shrink: 0; font-size: 10px; }

  /* ── Session table ── */
  table { width: 100%; border-collapse: collapse; font-size: 11px; }
  th { text-align: left; color: #8b949e; padding: 5px 10px; border-bottom: 1px solid #21262d; font-weight: normal; font-size: 9px; text-transform: uppercase; letter-spacing: .06em; }
  td { padding: 7px 10px; border-bottom: 1px solid #21262d; vertical-align: middle; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: #1c2128; }
  .badge     { display: inline-block; padding: 2px 7px; border-radius: 4px; font-size: 10px; background: #1a3a4a; color: #58a6ff; }
  .name-tag  { display: inline-block; padding: 2px 7px; border-radius: 4px; font-size: 10px; background: #2d1f3d; color: #bc8cff; }
  .ratio-badge { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 10px; }
  .muted { color: #8b949e; }
  .green { color: #3fb950; }

  @media (max-width: 700px) {
    .layers, .hero-solo { grid-template-columns: 1fr; }
    .cards { grid-template-columns: repeat(2, 1fr); }
    .layer-pct, .hero-pct { font-size: 48px; }
  }
</style>
</head>
<body>

<div class="header">
  <div style="display:flex;align-items:center;gap:8px">
    <div class="title">⚡ LLMLingua Proxy</div>
    <span class="model-badge" id="model_badge">—</span>
  </div>
  <div class="header-right">
    <span id="uptime_display">—</span>
    <span><span class="live-dot"></span>live</span>
  </div>
</div>

<!-- Two-layer hero (rtk present) -->
<div class="layers" id="hero_layers" style="display:none">
  <div class="layer layer-shell">
    <div class="layer-label">Shell layer <span class="layer-badge">rtk</span></div>
    <div class="layer-pct" id="rtk_pct">—%</div>
    <div class="layer-sub">CLI output compression</div>
    <div class="layer-stats">
      <div class="layer-stat"><span class="layer-stat-k">commands</span><span class="layer-stat-v" id="rtk_cmds">—</span></div>
      <div class="layer-stat"><span class="layer-stat-k">tokens saved</span><span class="layer-stat-v" id="rtk_saved">—</span></div>
      <div class="layer-stat"><span class="layer-stat-k">avg savings</span><span class="layer-stat-v" id="rtk_avg">—</span></div>
    </div>
  </div>
  <div class="layer layer-api">
    <div class="layer-label">API layer <span class="layer-badge">LLMLingua-2</span></div>
    <div class="layer-pct" id="api_pct_layers">—%</div>
    <div class="layer-sub">Prompt compression · LLMLingua-2 tokenizer units</div>
    <div class="layer-stats">
      <div class="layer-stat"><span class="layer-stat-k">requests</span><span class="layer-stat-v" id="api_reqs_layers">—</span></div>
      <div class="layer-stat"><span class="layer-stat-k">tokens saved</span><span class="layer-stat-v" id="api_saved_layers">—</span></div>
      <div class="layer-stat"><span class="layer-stat-k">ratio</span><span class="layer-stat-v" id="api_ratio_layers">—</span></div>
    </div>
  </div>
</div>

<!-- Single hero (rtk absent) -->
<div class="hero-solo" id="hero_solo">
  <div class="hero-left">
    <div class="hero-pct" id="savings_pct">—%</div>
    <div class="hero-label">tokens saved overall</div>
    <div class="hero-sublabel">LLMLingua-2 tokenizer units</div>
  </div>
  <div class="hero-right">
    <div class="term-prompt">$ llmlingua-proxy --status</div>
    <div class="term-line"><span class="term-key">requests processed</span><span class="term-val blue" id="t_requests">—</span></div>
    <div class="term-line"><span class="term-key">original tokens</span><span class="term-val" id="t_original">—</span></div>
    <div class="term-line"><span class="term-key">compressed tokens</span><span class="term-val" id="t_compressed">—</span></div>
    <div class="term-line"><span class="term-key">tokens saved</span><span class="term-val green" id="t_saved">—</span></div>
    <div class="term-line"><span class="term-key">compression ratio</span><span class="term-val green" id="t_ratio">—</span></div>
    <div class="term-line"><span class="term-key">active sessions</span><span class="term-val blue" id="t_sessions">—</span></div>
  </div>
</div>

<!-- rtk install hint (shown when rtk absent) -->
<div class="rtk-hint" id="rtk_hint">
  <span style="color:#484f58">⬡</span>
  <span>Add shell-layer compression: <code>brew install rtk</code> &nbsp;—&nbsp; rtk compresses CLI output; this proxy compresses API messages. Run both for maximum savings.</span>
</div>

<!-- Metric cards (API layer) -->
<div class="cards">
  <div class="card"><div class="card-label">API Requests</div><div class="card-value blue"  id="card_requests">—</div></div>
  <div class="card"><div class="card-label">API Tokens Saved</div><div class="card-value green" id="card_saved">—</div></div>
  <div class="card"><div class="card-label">Sessions</div><div class="card-value blue"  id="card_sessions">—</div></div>
  <div class="card"><div class="card-label">Avg Ratio</div><div class="card-value green" id="card_ratio">—</div></div>
  <div class="card"><div class="card-label">Uptime</div><div class="card-value"          id="card_uptime">—</div></div>
  <div class="card"><div class="card-label">Latency</div><div class="card-value" id="card_latency">—</div></div>
  <div class="card">
    <div class="card-label">Est. $ Saved</div>
    <div class="card-value green" id="card_cost">—</div>
    <div style="font-size:9px;color:#484f58;margin-top:3px" id="card_cost_label">@ $3.00 / MTok</div>
  </div>
</div>

<!-- rtk top commands (shown when rtk present) -->
<div class="section" id="rtk_section" style="display:none">
  <div class="section-title">rtk — top commands by tokens saved</div>
  <table class="cmd-table">
    <thead><tr><th>Command</th><th>Runs</th><th>Saved</th><th>Avg %</th><th style="width:140px"></th></tr></thead>
    <tbody id="rtk_cmd_body"></tbody>
  </table>
</div>

<!-- LLMLingua sparkline -->
<div class="section">
  <div class="section-title">LLMLingua-2 — recent compressions &nbsp;·&nbsp; height = original size &nbsp;·&nbsp; color = savings %</div>
  <div id="sparkline" class="sparkline"><span class="spark-empty">No compressions yet</span></div>
  <div class="spark-legend">
    <span><span class="leg-dot" style="background:#3fb950"></span>≥40% saved</span>
    <span><span class="leg-dot" style="background:#d29922"></span>20–39% saved</span>
    <span><span class="leg-dot" style="background:#484f58"></span>&lt;20% saved</span>
  </div>
</div>

<!-- Session efficiency bars -->
<div class="section">
  <div class="section-title">LLMLingua-2 — sessions by tokens saved</div>
  <div id="sess_bars" class="sess-bars"><span class="spark-empty">No sessions yet</span></div>
</div>

<!-- 48-hour time series -->
<div class="section">
  <div class="section-title">compression rate — last 48h (hourly) &nbsp;·&nbsp; height = requests &nbsp;·&nbsp; color = avg savings %</div>
  <div id="ts_chart" class="ts-chart"><span class="spark-empty">No data</span></div>
  <div id="ts_labels" class="ts-labels"></div>
</div>

<!-- Session detail table -->
<div class="section">
  <div class="section-title">LLMLingua-2 — session detail</div>
  <table>
    <thead><tr><th>Session</th><th>Name</th><th>Requests</th><th>Saved</th><th>Ratio</th><th>Last seen</th></tr></thead>
    <tbody id="sessions_body"></tbody>
  </table>
</div>

<script>
function fmt(n) {
  if (n == null) return '—';
  if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
  if (n >= 1000)    return (n / 1000).toFixed(1) + 'k';
  return n.toLocaleString();
}
function fmtUptime(secs) {
  if (secs < 60)   return secs + 's';
  if (secs < 3600) return Math.floor(secs / 60) + 'm';
  return Math.floor(secs / 3600) + 'h ' + Math.floor((secs % 3600) / 60) + 'm';
}
function barColor(pct) {
  if (pct >= 40) return '#3fb950';
  if (pct >= 20) return '#d29922';
  return '#484f58';
}
function ratioBadgeStyle(ratio) {
  if (ratio >= 2.0) return 'background:#0d2b1a;color:#3fb950';
  if (ratio >= 1.5) return 'background:#2b2200;color:#d29922';
  return 'background:#1c2128;color:#8b949e';
}

async function refresh() {
  try {
    const r = await fetch('/stats');
    const d = await r.json();

    // ── Uptime ──
    const uptimeSecs = Math.floor((Date.now() - new Date(d.started_at)) / 1000);
    const uptimeStr = fmtUptime(uptimeSecs);
    document.getElementById('uptime_display').textContent = 'up ' + uptimeStr;
    document.getElementById('card_uptime').textContent = uptimeStr;

    // ── API layer numbers ──
    const apiPct = d.total_original_tokens > 0
      ? Math.round((1 - d.total_compressed_tokens / d.total_original_tokens) * 100) : 0;

    // ── Hero: two-layer vs solo ──
    const hasRtk = d.rtk && d.rtk.total_commands > 0;
    document.getElementById('hero_layers').style.display = hasRtk ? 'grid' : 'none';
    document.getElementById('hero_solo').style.display   = hasRtk ? 'none' : 'grid';
    document.getElementById('rtk_hint').style.display    = hasRtk ? 'none' : 'flex';
    document.getElementById('rtk_section').style.display = hasRtk ? 'block' : 'none';

    if (hasRtk) {
      const rtk = d.rtk;
      const rtkPct = rtk.total_input_tokens > 0
        ? Math.round((rtk.total_saved_tokens / rtk.total_input_tokens) * 100) : 0;
      document.getElementById('rtk_pct').textContent    = rtkPct + '%';
      document.getElementById('rtk_cmds').textContent   = fmt(rtk.total_commands);
      document.getElementById('rtk_saved').textContent  = fmt(rtk.total_saved_tokens);
      document.getElementById('rtk_avg').textContent    = rtk.avg_savings_pct + '%';
      document.getElementById('api_pct_layers').textContent   = apiPct + '%';
      document.getElementById('api_reqs_layers').textContent  = fmt(d.total_requests);
      document.getElementById('api_saved_layers').textContent = fmt(d.total_saved_tokens);
      document.getElementById('api_ratio_layers').textContent = d.overall_ratio + '×';

      // rtk command table
      const maxSaved = Math.max(...rtk.top_commands.map(c => c.saved), 1);
      document.getElementById('rtk_cmd_body').innerHTML = rtk.top_commands.map(c => {
        const w = Math.round((c.saved / maxSaved) * 100);
        return '<tr>'
          + '<td class="cmd-name">' + c.cmd + '</td>'
          + '<td class="muted">' + c.count + '</td>'
          + '<td class="green">' + fmt(c.saved) + '</td>'
          + '<td class="muted">' + c.avg_pct + '%</td>'
          + '<td><div class="cmd-bar-wrap"><div class="cmd-bar-fill" style="width:' + w + '%"></div></div></td>'
          + '</tr>';
      }).join('');
    } else {
      // Solo hero
      document.getElementById('savings_pct').textContent = apiPct + '%';
      document.getElementById('t_requests').textContent  = fmt(d.total_requests);
      document.getElementById('t_original').textContent  = fmt(d.total_original_tokens);
      document.getElementById('t_compressed').textContent = fmt(d.total_compressed_tokens);
      document.getElementById('t_saved').textContent     = fmt(d.total_saved_tokens) + ' (' + apiPct + '%)';
      document.getElementById('t_ratio').textContent     = d.overall_ratio + '×';
      document.getElementById('t_sessions').textContent  = Object.keys(d.sessions).length;
    }

    // ── Model badge ──
    if (d.compressor) {
      const c = d.compressor;
      document.getElementById('model_badge').textContent =
        (c.model === 'kompress' ? 'kompress' : 'llmlingua2') + ' · ' + c.param_name + '=' + c.param_value;
    }

    // ── Metric cards ──
    document.getElementById('card_requests').textContent = fmt(d.total_requests);
    document.getElementById('card_saved').textContent    = fmt(d.total_saved_tokens);
    document.getElementById('card_sessions').textContent = Object.keys(d.sessions).length;
    document.getElementById('card_ratio').textContent    = d.overall_ratio + '×';

    // ── Latency card ──
    document.getElementById('card_latency').textContent =
      d.avg_latency_ms != null ? Math.round(d.avg_latency_ms) + ' ms' : '—';

    // ── Cost card ──
    if (d.cost_per_mtok != null && d.total_saved_tokens != null) {
      const cost = (d.total_saved_tokens / 1000000) * d.cost_per_mtok;
      document.getElementById('card_cost').textContent = '$' + cost.toFixed(2);
      document.getElementById('card_cost_label').textContent = '@ $' + d.cost_per_mtok.toFixed(2) + ' / MTok';
    }

    // ── Sparkline ──
    const comps = [...d.recent_compressions].reverse();
    const sparkEl = document.getElementById('sparkline');
    if (comps.length === 0) {
      sparkEl.innerHTML = '<span class="spark-empty">No compressions yet</span>';
    } else {
      const maxOrig = Math.max(...comps.map(c => c.original), 1);
      sparkEl.innerHTML = comps.map(c => {
        const pct = c.original > 0 ? Math.round((1 - c.compressed / c.original) * 100) : 0;
        const h = Math.max(10, Math.round((c.original / maxOrig) * 100));
        const tip = c.ts + ' · ' + c.session_id + ' · ' + pct + '% saved (' + fmt(c.original) + ' → ' + fmt(c.compressed) + ')' +
          (c.latency_ms != null ? ' · ' + c.latency_ms + 'ms' : '');
        return '<div class="bar" style="background:' + barColor(pct) + ';height:' + h + '%" title="' + tip + '"></div>';
      }).join('');
    }

    // ── Session bars ──
    const sessions = Object.entries(d.sessions).sort((a, b) =>
      (b[1].saved_tokens || 0) - (a[1].saved_tokens || 0)
    );
    const maxSess = Math.max(...sessions.map(([, s]) => s.saved_tokens || 0), 1);
    const sessBarEl = document.getElementById('sess_bars');
    if (sessions.length === 0) {
      sessBarEl.innerHTML = '<span class="spark-empty">No sessions yet</span>';
    } else {
      sessBarEl.innerHTML = sessions.slice(0, 8).map(([id, s]) => {
        const sPct = s.original_tokens > 0
          ? Math.round((1 - s.compressed_tokens / s.original_tokens) * 100) : 0;
        const w = Math.round(((s.saved_tokens || 0) / maxSess) * 100);
        const label = s.name ? s.name : id.slice(0, 8);
        return '<div class="sess-bar-row">'
          + '<div class="sess-bar-label" title="' + id + '">' + label + '</div>'
          + '<div class="sess-bar-wrap"><div class="sess-bar-fill" style="width:' + w + '%;background:' + barColor(sPct) + '"></div></div>'
          + '<div class="sess-bar-pct">' + sPct + '%</div>'
          + '<div class="sess-bar-saved">' + fmt(s.saved_tokens) + '</div>'
          + '</div>';
      }).join('');
    }

    // ── Session table ──
    document.getElementById('sessions_body').innerHTML = sessions.map(([id, s]) => {
      const sPct = s.original_tokens > 0
        ? Math.round((1 - s.compressed_tokens / s.original_tokens) * 100) : 0;
      const ratio = s.compressed_tokens > 0
        ? (s.original_tokens / s.compressed_tokens).toFixed(2) : null;
      const nameCell = s.name
        ? '<span class="name-tag">' + s.name + '</span>'
        : '<span class="muted">—</span>';
      const ratioCell = ratio
        ? '<span class="ratio-badge" style="' + ratioBadgeStyle(parseFloat(ratio)) + '">' + ratio + '×</span>'
        : '<span class="muted">—</span>';
      return '<tr>'
        + '<td><span class="badge">' + id.slice(0, 8) + '</span></td>'
        + '<td>' + nameCell + '</td>'
        + '<td>' + s.requests + '</td>'
        + '<td class="green">' + fmt(s.saved_tokens) + ' <span class="muted">(' + sPct + '%)</span></td>'
        + '<td>' + ratioCell + '</td>'
        + '<td class="muted">' + (s.last_seen ? new Date(s.last_seen).toLocaleTimeString() : '—') + '</td>'
        + '</tr>';
    }).join('');

  } catch(e) { console.error(e); }
}

async function refreshTimeseries() {
  try {
    const r = await fetch('/stats/timeseries');
    const buckets = await r.json();
    const chartEl = document.getElementById('ts_chart');
    const labelsEl = document.getElementById('ts_labels');
    if (!buckets || buckets.length === 0) {
      chartEl.innerHTML = '<span class="spark-empty">No data</span>';
      labelsEl.innerHTML = '';
      return;
    }
    const maxReq = Math.max(...buckets.map(b => b.requests), 1);
    chartEl.innerHTML = buckets.map(b => {
      const h = Math.max(4, Math.round((b.requests / maxReq) * 100));
      const tip = b.hour.slice(11, 16) + ' · ' + b.requests + ' req · ' +
        b.avg_savings_pct + '% saved · ' + b.avg_latency_ms + 'ms avg';
      return '<div class="ts-bar" style="height:' + h + '%;background:' + barColor(b.avg_savings_pct) + '" title="' + tip + '"></div>';
    }).join('');
    const first = buckets[0].hour.slice(11, 16);
    const last  = buckets[buckets.length - 1].hour.slice(11, 16);
    labelsEl.innerHTML = '<span>' + first + '</span><span>' + last + '</span>';
  } catch(e) { console.error(e); }
}

refresh();
refreshTimeseries();
setInterval(refresh, 2000);
setInterval(refreshTimeseries, 60000);
</script>
</body>
</html>"""


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9099)
