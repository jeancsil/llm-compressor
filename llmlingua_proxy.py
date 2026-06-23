import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import json
import platform
import re
import secrets
import sqlite3
import threading
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
backend          = None
backend_loading  = None   # set to model name while async load is in progress
_db_conn         = None
backend_user     = None   # kompress instance in dual mode
backend_system   = None   # llmlingua2-large instance in dual mode
dual_mode        = False

KNOWN_MODELS = ("llmlingua2", "llmlingua2-large", "kompress", "dual")


# ---------------------------------------------------------------------------
# Stub helpers (replaced in later tasks)
# ---------------------------------------------------------------------------

def init_db(path: str):
    import sqlite3 as _sqlite3
    conn = _sqlite3.connect(path, check_same_thread=False)
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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS compression_texts (
            compression_id INTEGER PRIMARY KEY REFERENCES compressions(id),
            original_text  TEXT NOT NULL,
            compressed_text TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trackers (
            slug        TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'pending',
            session_id  TEXT,
            created_at  TEXT NOT NULL,
            linked_at   TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rtk_events (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            rtk_id        INTEGER UNIQUE,
            ts            TEXT NOT NULL,
            session_id    TEXT NOT NULL,
            rtk_cmd       TEXT NOT NULL,
            input_tokens  INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            saved_tokens  INTEGER NOT NULL DEFAULT 0,
            savings_pct   REAL    NOT NULL DEFAULT 0.0
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rtk_events_session ON rtk_events(session_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rtk_events_ts      ON rtk_events(ts)")
    try:
        conn.execute("ALTER TABLE trackers ADD COLUMN closed_at TEXT")
    except Exception:
        pass  # column already exists
    try:
        conn.execute("ALTER TABLE compressions ADD COLUMN role TEXT DEFAULT 'user'")
    except Exception:
        pass  # column already exists
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


def make_slug(name: str, conn) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "tracker"
    return f"{base}-{secrets.token_hex(2)}"


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


def _load_llmlingua2_backend(backend_key: str | None = None) -> dict:
    """Load the LLMLingua-2 PromptCompressor and return a backend dict."""
    from llmlingua import PromptCompressor
    import transformers as _tf
    import logging as _logging
    rate = float(os.environ.get("COMPRESS_RATE", "0.5"))
    if backend_key is None:
        backend_key = os.environ.get("COMPRESSOR_MODEL", "llmlingua2")
    model_name = LLMLINGUA2_MODELS.get(backend_key, LLMLINGUA2_MODELS["llmlingua2"])
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading LLMLingua-2 model ({backend_key}: {model_name})...")
    _tf.logging.set_verbosity_error()
    _hf_log = _logging.getLogger("huggingface_hub")
    _prev_hf = _hf_log.level
    _hf_log.setLevel(_logging.ERROR)
    try:
        c = PromptCompressor(
            model_name=model_name,
            use_llmlingua2=True,
            device_map=device,
        )
    finally:
        _tf.logging.set_verbosity_warning()
        _hf_log.setLevel(_prev_hf)
    print(f"Model ready. (device={device})")
    return {"type": backend_key, "backend_key": backend_key, "compressor": c, "rate": rate}


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
    import transformers as _tf
    _prev_level = _tf.logging.get_verbosity()
    _tf.logging.set_verbosity_error()
    compressor.preload()
    _tf.logging.set_verbosity(_prev_level)
    print(f"kompress-v2-base ready.")
    return {"type": "kompress", "compressor": compressor, "threshold": threshold}


def _load_dual_backend() -> dict:
    """Load both backends and set dual-mode globals.

    Load order: system backend first (llmlingua2-large), then user backend (kompress).
    Sets dual_mode = True only after both backends are ready.
    """
    global backend_user, backend_system, dual_mode
    print("Loading dual mode: kompress (user) + llmlingua2-large (system)...")
    backend_system = _load_llmlingua2_backend("llmlingua2-large")
    backend_user   = _load_kompress_backend()
    dual_mode      = True
    print("Dual mode ready.")
    return {"type": "dual", "model_user": "kompress", "model_system": "llmlingua2-large"}


def load_backend() -> dict:
    """Dispatch to the configured backend loader.

    Resolution order:
    1. DB meta table key 'current_model' (persists across restarts after set-model)
    2. COMPRESSOR_MODEL environment variable
    3. Default: llmlingua2
    """
    model_name = os.environ.get("COMPRESSOR_MODEL", "llmlingua2")
    try:
        if _db_conn is not None:
            row = _db_conn.execute(
                "SELECT value FROM meta WHERE key='current_model'"
            ).fetchone()
            if row:
                model_name = row[0]
    except Exception:
        pass  # DB not available; fall back to env var
    if model_name == "dual":
        return _load_dual_backend()
    if model_name == "kompress":
        return _load_kompress_backend()
    return _load_llmlingua2_backend(backend_key=model_name)


# Keep the private alias so existing call-sites (lifespan, tests) still work.
_load_backend = load_backend


def _pick_backend(role: str) -> dict | None:
    if dual_mode and backend_user is not None and backend_system is not None:
        return backend_system if role == "system" else backend_user
    return backend


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
    # Release model references before process exit to avoid MPS semaphore leaks
    backend = None
    import gc
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
            torch.mps.empty_cache()
    except Exception:
        pass
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

def record_compression(
    session_id: str,
    original: int,
    compressed: int,
    latency_ms: float = 0.0,
    original_text: str | None = None,
    compressed_text: str | None = None,
    role: str = "user",
    active_backend: dict | None = None,
):
    stats["total_original_tokens"] += original
    stats["total_compressed_tokens"] += compressed

    active = active_backend if active_backend is not None else backend
    model_name = active.get("type", "llmlingua2") if active else "llmlingua2"
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if _db_conn:
        cur = _db_conn.execute(
            "INSERT INTO compressions (ts, session_id, model, original_tokens, compressed_tokens, latency_ms, role) VALUES (?,?,?,?,?,?,?)",
            (ts, session_id, model_name, original, compressed, latency_ms, role),
        )
        if original_text is not None and compressed_text is not None:
            _db_conn.execute(
                "INSERT INTO compression_texts (compression_id, original_text, compressed_text) VALUES (?,?,?)",
                (cur.lastrowid, original_text, compressed_text),
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

def _try_link_pending_tracker(session_id: str) -> None:
    """Link the oldest pending tracker to session_id if one exists and session_id is known."""
    if _db_conn is None or not session_id or session_id == "unknown":
        return
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result = _db_conn.execute(
        """UPDATE trackers SET status='active', session_id=?, linked_at=?
           WHERE slug = (
             SELECT slug FROM trackers WHERE status='pending'
             ORDER BY created_at ASC LIMIT 1
           )""",
        (session_id, ts),
    )
    if result.rowcount > 0:
        _db_conn.commit()


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

    _try_link_pending_tracker(session_id)

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

def read_rtk_stats(since: str | None = None) -> dict | None:
    db = _rtk_db_path()
    if not db.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        where = "WHERE timestamp >= ?" if since else ""
        args  = (since,) if since else ()

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
_CHUNK_MAX_CHARS = 1400  # ~400 BERT tokens for mixed code/prose


def _count_tokens(text: str) -> int:
    """Count tokens using the backend tokenizer (falls back to whitespace split)."""
    active = backend or backend_user or backend_system
    try:
        return len(active["compressor"].tokenizer.tokenize(text))
    except Exception:
        return len(text.split())


def _char_split(text: str) -> list[str]:
    """Split text into _CHUNK_MAX_CHARS-sized pieces (last resort for code/dense text)."""
    if len(text) <= _CHUNK_MAX_CHARS:
        return [text]
    return [text[i:i + _CHUNK_MAX_CHARS] for i in range(0, len(text), _CHUNK_MAX_CHARS)]


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
                    sents = [s.strip() for s in re.split(r"(?<=[?.!])\s+", p) if s.strip()]
                    segs.extend(sents if len(sents) > 1 else _char_split(p))
            else:
                segs.append(p)
        return segs
    # Single paragraph — try sentence splitting
    sents = [s.strip() for s in re.split(r"(?<=[?.!])\s+", text) if s.strip()]
    if len(sents) > 1:
        return sents
    return _char_split(text)


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

def compress_text(text: str, session_id: str, role: str = "user") -> str:
    if len(text) <= 200:
        return text
    active = _pick_backend(role)
    if active is None:
        return text
    t0 = time.perf_counter()
    try:
        compressed, orig, comp = _compress_with(active, text)
    except Exception as e:
        print(f"[compressor] compression failed, forwarding original: {e}")
        return text
    latency_ms = (time.perf_counter() - t0) * 1000
    model_tag = active.get("type", "compressor")
    print(f"[{model_tag}] {orig} → {comp} tokens [{session_id[:8]}] role={role}")
    record_compression(session_id, orig, comp, latency_ms, text, compressed, role=role, active_backend=active)
    return compressed


def _compress_with(active: dict, text: str):
    """Dispatch to the right compressor using the given backend dict.

    Returns (compressed_text, orig_tokens, comp_tokens).
    """
    if active.get("type") == "kompress":
        return _compress_kompress(active, text)
    return _compress_llmlingua2(active, text)


# Keep old name as alias so /play/compress endpoint still works without changes.
def compress_backend(text: str):
    """Legacy wrapper — dispatches via the global backend. Use _compress_with() for new code."""
    if backend is None:
        raise RuntimeError("No backend loaded")
    return _compress_with(backend, text)


def _compress_llmlingua2(active: dict, text: str):
    chunks = chunk_text(text)
    if len(chunks) == 1:
        result = active["compressor"].compress_prompt(
            chunks[0],
            rate=active.get("rate", 0.5),
            force_tokens=["\n", "?", ".", "!"],
        )
        return result["compressed_prompt"], result["origin_tokens"], result["compressed_tokens"]

    parts: list[str] = []
    total_orig = 0
    total_comp = 0
    for chunk in chunks:
        result = active["compressor"].compress_prompt(
            chunk,
            rate=active.get("rate", 0.5),
            force_tokens=["\n", "?", ".", "!"],
        )
        parts.append(result["compressed_prompt"])
        total_orig += result["origin_tokens"]
        total_comp += result["compressed_tokens"]
    return "\n\n".join(parts), total_orig, total_comp


def _compress_kompress(active: dict, text: str):
    result = active["compressor"].compress(text)
    return result.compressed, result.original_tokens, result.compressed_tokens


def compress_system_field(system_val, session_id: str):
    """Compress the top-level Anthropic API `system` field (string or content-block list)."""
    if isinstance(system_val, str):
        return compress_text(system_val, session_id, role="system")
    if isinstance(system_val, list):
        return [
            {**b, "text": compress_text(b.get("text", ""), session_id, role="system")}
            if isinstance(b, dict) and b.get("type") == "text" else b
            for b in system_val
        ]
    return system_val


def compress_messages(messages: list, session_id: str) -> list:
    out = []
    for msg in messages:
        if msg.get("role") != "user":
            out.append(msg)
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            out.append({**msg, "content": compress_text(content, session_id, role="user")})
        elif isinstance(content, list):
            new_blocks = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    new_blocks.append({**block, "text": compress_text(block["text"], session_id, role="user")})
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
async def get_stats(session_id: str | None = None):
    saved = stats["total_original_tokens"] - stats["total_compressed_tokens"]
    ratio = (stats["total_original_tokens"] / stats["total_compressed_tokens"]
             if stats["total_compressed_tokens"] > 0 else 1.0)
    sessions_out = {}
    for sid, s in stats["sessions"].items():
        sv = s["original_tokens"] - s["compressed_tokens"]
        sessions_out[sid] = {**s, "saved_tokens": sv}

    recent = list(stats["recent_compressions"])
    if session_id:
        sessions_out = {sid: v for sid, v in sessions_out.items() if sid == session_id}
        recent = [c for c in recent if c.get("session_id", "") == session_id[:8]]
    avg_latency = (
        sum(c["latency_ms"] for c in recent) / len(recent) if recent else 0.0
    )

    if backend_loading:
        compressor_info = {
            "model": backend_loading,
            "param_name": "",
            "param_value": "",
            "loading": True,
        }
    elif backend and backend.get("type") == "kompress":
        compressor_info = {
            "model": "kompress",
            "param_name": "threshold",
            "param_value": backend.get("threshold", 0.5),
            "loading": False,
        }
    else:
        backend_key = backend.get("backend_key", "llmlingua2") if backend else "llmlingua2"
        compressor_info = {
            "model": backend_key,
            "param_name": "rate",
            "param_value": backend.get("rate", 0.5) if backend else 0.5,
            "loading": False,
        }

    by_model: list = []
    today_stats: dict = {"requests": 0, "tokens_saved": 0, "avg_savings_pct": 0.0, "avg_latency_ms": 0.0, "sessions": 0}
    alltime_stats: dict = {"requests": 0, "tokens_saved": 0, "avg_savings_pct": 0.0, "avg_latency_ms": 0.0, "sessions": 0}
    recent_rows: list = []

    if _db_conn is not None:
        active_model = compressor_info["model"]
        sess_args   = (session_id,) if session_id else ()

        by_model_rows = _db_conn.execute(
            f"""
            SELECT model,
                   COUNT(*) AS requests,
                   ROUND(AVG((original_tokens - compressed_tokens) * 100.0 / original_tokens), 1) AS avg_savings_pct,
                   ROUND(AVG(CAST(original_tokens AS REAL) / NULLIF(compressed_tokens, 0)), 2) AS avg_ratio,
                   SUM(original_tokens - compressed_tokens) AS total_saved,
                   ROUND(AVG(latency_ms), 1) AS avg_latency_ms
            FROM compressions
            {'WHERE session_id = ?' if session_id else ''}
            GROUP BY model
            ORDER BY requests DESC
            """,
            sess_args,
        ).fetchall()
        by_model = [dict(r) for r in by_model_rows]

        if session_id:
            today_row = _db_conn.execute(
                """
                SELECT
                    COUNT(*) AS requests,
                    COALESCE(SUM(original_tokens - compressed_tokens), 0) AS tokens_saved,
                    ROUND(AVG((original_tokens - compressed_tokens) * 100.0 / original_tokens), 1) AS avg_savings_pct,
                    ROUND(AVG(latency_ms), 1) AS avg_latency_ms,
                    COUNT(DISTINCT session_id) AS sessions
                FROM compressions
                WHERE date(ts) = date('now') AND session_id = ?
                """,
                (session_id,),
            ).fetchone()
        else:
            today_row = _db_conn.execute(
                """
                SELECT
                    COUNT(*) AS requests,
                    COALESCE(SUM(original_tokens - compressed_tokens), 0) AS tokens_saved,
                    ROUND(AVG((original_tokens - compressed_tokens) * 100.0 / original_tokens), 1) AS avg_savings_pct,
                    ROUND(AVG(latency_ms), 1) AS avg_latency_ms,
                    COUNT(DISTINCT session_id) AS sessions
                FROM compressions
                WHERE date(ts) = date('now') AND model = ?
                """,
                (active_model,),
            ).fetchone()
        if today_row:
            today_stats = {
                "requests": today_row[0] or 0,
                "tokens_saved": today_row[1] or 0,
                "avg_savings_pct": today_row[2] or 0.0,
                "avg_latency_ms": today_row[3] or 0.0,
                "sessions": today_row[4] or 0,
            }

        if session_id:
            alltime_row = _db_conn.execute(
                """
                SELECT
                    COUNT(*) AS requests,
                    COALESCE(SUM(original_tokens - compressed_tokens), 0) AS tokens_saved,
                    ROUND(AVG((original_tokens - compressed_tokens) * 100.0 / original_tokens), 1) AS avg_savings_pct,
                    ROUND(AVG(latency_ms), 1) AS avg_latency_ms,
                    COUNT(DISTINCT session_id) AS sessions,
                    ROUND(AVG(CAST(original_tokens AS REAL) / NULLIF(compressed_tokens, 0)), 2) AS avg_ratio
                FROM compressions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        else:
            alltime_row = _db_conn.execute(
                """
                SELECT
                    COUNT(*) AS requests,
                    COALESCE(SUM(original_tokens - compressed_tokens), 0) AS tokens_saved,
                    ROUND(AVG((original_tokens - compressed_tokens) * 100.0 / original_tokens), 1) AS avg_savings_pct,
                    ROUND(AVG(latency_ms), 1) AS avg_latency_ms,
                    COUNT(DISTINCT session_id) AS sessions,
                    ROUND(AVG(CAST(original_tokens AS REAL) / NULLIF(compressed_tokens, 0)), 2) AS avg_ratio
                FROM compressions
                WHERE model = ?
                """,
                (active_model,),
            ).fetchone()
        if alltime_row:
            alltime_stats = {
                "requests": alltime_row[0] or 0,
                "tokens_saved": alltime_row[1] or 0,
                "avg_savings_pct": alltime_row[2] or 0.0,
                "avg_latency_ms": alltime_row[3] or 0.0,
                "sessions": alltime_row[4] or 0,
                "avg_ratio": alltime_row[5] or 0.0,
            }

        if session_id:
            recent_db_rows = _db_conn.execute(
                """
                SELECT ts, session_id, model, original_tokens, compressed_tokens,
                       ROUND((original_tokens - compressed_tokens) * 100.0 / original_tokens, 1) AS savings_pct,
                       latency_ms, role
                FROM compressions
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT 20
                """,
                (session_id,),
            ).fetchall()
        else:
            recent_db_rows = _db_conn.execute(
                """
                SELECT ts, session_id, model, original_tokens, compressed_tokens,
                       ROUND((original_tokens - compressed_tokens) * 100.0 / original_tokens, 1) AS savings_pct,
                       latency_ms, role
                FROM compressions
                WHERE model = ?
                ORDER BY id DESC
                LIMIT 20
                """,
                (active_model,),
            ).fetchall()
        recent_rows = [
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
            for r in recent_db_rows
        ]

    rtk_stats = None
    if _db_conn is not None:
        rtk_where = "WHERE session_id = ?" if session_id else ""
        rtk_args  = (session_id,) if session_id else ()
        rtk_row = _db_conn.execute(
            f"""SELECT COUNT(*), COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0),
                       COALESCE(SUM(saved_tokens),0), COALESCE(AVG(savings_pct),0)
                FROM rtk_events {rtk_where}""",
            rtk_args,
        ).fetchone()
        if rtk_row and rtk_row[0] > 0:
            rtk_top = _db_conn.execute(
                f"""SELECT rtk_cmd, COUNT(*) AS cnt, SUM(saved_tokens) AS saved, AVG(savings_pct) AS avg_pct
                    FROM rtk_events {rtk_where}
                    GROUP BY rtk_cmd ORDER BY saved DESC LIMIT 8""",
                rtk_args,
            ).fetchall()
            rtk_stats = {
                "total_commands":     rtk_row[0],
                "total_input_tokens": rtk_row[1],
                "total_output_tokens":rtk_row[2],
                "total_saved_tokens": rtk_row[3],
                "avg_savings_pct":    round(rtk_row[4], 1),
                "top_commands": [
                    {"cmd": r[0], "count": r[1], "saved": r[2], "avg_pct": round(r[3], 1)}
                    for r in rtk_top
                ],
            }

    tracked_stats = {"sessions": 0, "tokens_saved": 0}
    if _db_conn is not None:
        trow = _db_conn.execute(
            """
            SELECT COUNT(DISTINCT t.slug),
                   COALESCE(SUM(c.original_tokens - c.compressed_tokens), 0)
            FROM trackers t
            JOIN compressions c ON c.session_id = t.session_id
            WHERE t.status IN ('active', 'closed') AND t.session_id IS NOT NULL
            """
        ).fetchone()
        if trow:
            tracked_stats = {"sessions": trow[0] or 0, "tokens_saved": trow[1] or 0}

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
        "dual_mode": dual_mode,
        "model_user": backend_user.get("type") if backend_user else None,
        "model_system": backend_system.get("type") if backend_system else None,
    }


@app.get("/stats/timeseries")
async def get_timeseries(model: str | None = None, session_id: str | None = None):
    if _db_conn is None:
        return JSONResponse([])
    sess_filter = " AND session_id = ?" if session_id else ""
    sess_args   = (session_id,) if session_id else ()
    if model:
        rows = _db_conn.execute(
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
        rows = _db_conn.execute(
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
    if _db_conn is None:
        return JSONResponse({"error": "db not ready"}, status_code=503)
    body = await request.json()
    session_id = body.get("session_id", "unknown")
    try:
        _db_conn.execute(
            """INSERT OR IGNORE INTO rtk_events
               (rtk_id, ts, session_id, rtk_cmd, input_tokens, output_tokens, saved_tokens, savings_pct)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                body.get("rtk_id"),
                body.get("ts", datetime.now(timezone.utc).isoformat()),
                session_id,
                body.get("rtk_cmd", ""),
                int(body.get("input_tokens", 0)),
                int(body.get("output_tokens", 0)),
                int(body.get("saved_tokens", 0)),
                float(body.get("savings_pct", 0.0)),
            ),
        )
        _db_conn.commit()
        _try_link_pending_tracker(session_id)
        return JSONResponse({"ok": True})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/dashboard/{slug}", response_class=HTMLResponse)
async def session_dashboard(slug: str):
    if _db_conn is None:
        return HTMLResponse("<h1>DB not ready</h1>", status_code=503)
    row = _db_conn.execute(
        "SELECT slug, name, status, session_id, created_at, linked_at "
        "FROM trackers WHERE slug=?",
        (slug,),
    ).fetchone()
    if row is None:
        return HTMLResponse(f"<h1>Tracker '{slug}' not found</h1>", status_code=404)
    tracker = dict(row)
    bootstrap = f'<script>window.TRACKER = {json.dumps(tracker)};</script>'
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
    global backend, backend_loading
    body = await request.json()
    text = body.get("text", "")
    model = body.get("model", "")

    if model and model not in KNOWN_MODELS:
        return JSONResponse({"error": f"Unknown model: {model}"}, status_code=400)

    orig_chars = len(text)
    orig_tokens_est = max(1, orig_chars // 4)

    active_type = backend.get("type") if backend else None

    if model and model != active_type:
        if backend_loading == model:
            return JSONResponse({"loading": True, "model": model}, status_code=202)
        if backend_loading:
            return JSONResponse({"loading": True, "model": backend_loading}, status_code=202)
        # Trigger async model switch
        if _db_conn is not None:
            _db_conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('current_model', ?)", (model,)
            )
            _db_conn.commit()
        backend = None
        backend_loading = model
        _target = model
        def _load():
            global backend, backend_loading
            try:
                if _target == "kompress":
                    backend = _load_kompress_backend()
                else:
                    backend = _load_llmlingua2_backend(backend_key=_target)
            except Exception as e:
                print(f"[play] load {_target}: {e}")
            finally:
                backend_loading = None
        threading.Thread(target=_load, daemon=True).start()
        return JSONResponse({"loading": True, "model": model}, status_code=202)

    if backend is None:
        if backend_loading:
            return JSONResponse({"loading": True, "model": backend_loading}, status_code=202)
        return JSONResponse({"error": "No model loaded. Select a model to load it."}, status_code=503)

    t0 = time.perf_counter()
    try:
        compressed, _orig_tok, _comp_tok = compress_backend(text)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    comp_chars = len(compressed)
    comp_tokens_est = max(1, comp_chars // 4)
    char_pct = round((1 - comp_chars / max(1, orig_chars)) * 100, 1)
    token_pct = round((1 - comp_tokens_est / orig_tokens_est) * 100, 1)

    return JSONResponse({
        "original": text, "compressed": compressed,
        "original_chars": orig_chars, "compressed_chars": comp_chars,
        "char_pct": char_pct,
        "original_tokens_est": orig_tokens_est, "compressed_tokens_est": comp_tokens_est,
        "token_pct": token_pct,
        "model": backend.get("type") if backend else model,
        "latency_ms": latency_ms,
    })


@app.post("/admin/set-model")
async def set_model(request: Request):
    global backend, backend_loading, backend_user, backend_system, dual_mode
    body = await request.json()
    model = body.get("model")
    if model not in KNOWN_MODELS:
        return JSONResponse({"error": f"Unknown model. Known: {sorted(KNOWN_MODELS)}"}, status_code=400)
    if _db_conn is not None:
        _db_conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('current_model', ?)",
            (model,),
        )
        _db_conn.commit()

    # When switching away from dual mode, clear dual globals first
    if dual_mode and model != "dual":
        dual_mode = False
        backend_user = None
        backend_system = None

    backend = None
    backend_loading = model

    if model == "dual":
        # Clear any previously loaded single backend globals
        backend_user = None
        backend_system = None

        def load_dual():
            global backend, backend_loading
            try:
                new_backend = _load_dual_backend()
                backend = new_backend
            except Exception as e:
                print(f"[set-model] failed to load dual: {e}")
            finally:
                backend_loading = None

        threading.Thread(target=load_dual, daemon=True).start()
    else:
        def load():
            global backend, backend_loading
            try:
                # Use model from closure directly — avoids reading _db_conn cross-thread
                if model == "kompress":
                    new_backend = _load_kompress_backend()
                else:
                    new_backend = _load_llmlingua2_backend(backend_key=model)
                backend = new_backend
            except Exception as e:
                print(f"[set-model] failed to load {model}: {e}")
            finally:
                backend_loading = None

        threading.Thread(target=load, daemon=True).start()

    return JSONResponse({"status": "loading", "model": model})


@app.delete("/admin/compression-texts")
async def clear_compression_texts(request: Request):
    """Delete stored original/compressed texts without touching compression metrics."""
    if _db_conn is None:
        return JSONResponse({"error": "db not ready"}, status_code=503)
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    session_id = body.get("session_id")
    if session_id:
        cur = _db_conn.execute(
            "DELETE FROM compression_texts WHERE compression_id IN "
            "(SELECT id FROM compressions WHERE session_id = ?)",
            (session_id,),
        )
    else:
        cur = _db_conn.execute("DELETE FROM compression_texts")
    _db_conn.commit()
    return JSONResponse({"deleted": cur.rowcount, "session_id": session_id})


@app.post("/admin/tracker")
async def create_tracker(request: Request):
    body = await request.json()
    name = (body.get("name") or "").strip()
    if not name:
        return JSONResponse({"error": "name required"}, status_code=400)
    if _db_conn is None:
        return JSONResponse({"error": "db not ready"}, status_code=503)
    slug = make_slug(name, _db_conn)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _db_conn.execute(
        "INSERT INTO trackers (slug, name, status, created_at) VALUES (?,?,'pending',?)",
        (slug, name, ts),
    )
    _db_conn.commit()
    return JSONResponse({"slug": slug, "name": name, "status": "pending", "session_id": None, "created_at": ts})


@app.get("/admin/tracker")
async def get_tracker():
    if _db_conn is None:
        return JSONResponse([])
    rows = _db_conn.execute(
        "SELECT slug, name, status, session_id, created_at, linked_at "
        "FROM trackers WHERE status IN ('pending','active') ORDER BY created_at DESC"
    ).fetchall()
    return JSONResponse([dict(r) for r in rows])


@app.delete("/admin/tracker/{slug}")
async def delete_tracker(slug: str):
    if _db_conn is None:
        return JSONResponse({"error": "db not ready"}, status_code=503)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result = _db_conn.execute(
        "UPDATE trackers SET status='closed', closed_at=? WHERE slug=?",
        (ts, slug),
    )
    _db_conn.commit()
    if result.rowcount == 0:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse({"closed": slug})


@app.get("/admin/tracker/all")
async def get_all_trackers():
    if _db_conn is None:
        return JSONResponse([])
    rows = _db_conn.execute(
        """
        SELECT t.slug, t.name, t.status, t.session_id,
               t.created_at, t.linked_at, t.closed_at,
               COALESCE(SUM(c.original_tokens - c.compressed_tokens), 0) AS tokens_saved,
               COUNT(c.id) AS requests
        FROM trackers t
        LEFT JOIN compressions c ON c.session_id = t.session_id
        GROUP BY t.slug
        ORDER BY t.created_at DESC
        """
    ).fetchall()
    return JSONResponse([dict(r) for r in rows])


@app.get("/session/{slug}/compressions")
async def get_session_compressions(slug: str, limit: int = 50):
    if _db_conn is None:
        return JSONResponse({"error": "db not ready"}, status_code=503)
    row = _db_conn.execute("SELECT session_id FROM trackers WHERE slug=?", (slug,)).fetchone()
    if row is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    session_id = row[0]
    if not session_id:
        return JSONResponse([])
    rows = _db_conn.execute(
        """
        SELECT c.id, c.ts, c.model, c.original_tokens, c.compressed_tokens,
               ROUND((c.original_tokens - c.compressed_tokens) * 100.0 / c.original_tokens, 1) AS savings_pct,
               c.latency_ms,
               ct.original_text, ct.compressed_text
        FROM compressions c
        LEFT JOIN compression_texts ct ON ct.compression_id = c.id
        WHERE c.session_id = ?
        ORDER BY c.id DESC
        LIMIT ?
        """,
        (session_id, limit),
    ).fetchall()
    return JSONResponse([dict(r) for r in rows])


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
    if body.get("system"):
        body["system"] = compress_system_field(body["system"], session_id)
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

  /* ── Today tiles ── */
  .tiles { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 12px; }
  .tile { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px 18px; cursor: default; transition: transform .15s, box-shadow .15s; }
  .tile:hover { transform: translateY(-2px); box-shadow: 0 4px 16px rgba(0,0,0,.4); }
  .tile-num { font-size: 28px; font-weight: 800; color: #f0f6fc; line-height: 1; }
  .tile-num.blue  { color: #58a6ff; }
  .tile-num.green { color: #3fb950; }
  .tile-label { font-size: 9px; color: #8b949e; text-transform: uppercase; letter-spacing: .08em; margin-top: 6px; }

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

  /* ── Model select (switcher) ── */
  .model-select { font-size: 12px; background: #161b22; color: #58a6ff; padding: 4px 10px; border-radius: 6px; border: 1px solid #30363d; cursor: pointer; font-family: inherit; outline: none; }
  .model-select:hover:not(:disabled) { border-color: #58a6ff; }
  .model-select:disabled { opacity: 0.5; cursor: wait; }
  .model-loading { font-size: 10px; color: #d29922; margin-left: 6px; display: none; }

  /* ── Time-series chart ── */
  .ts-chart { display: flex; align-items: flex-end; gap: 4px; height: 60px; overflow: hidden; }
  .ts-bar { flex: 1; min-height: 4px; border-radius: 2px 2px 0 0; opacity: .8; transition: opacity .1s; cursor: default; }
  .ts-bar:hover { opacity: 1; }
  .ts-labels { display: flex; justify-content: space-between; margin-top: 4px; font-size: 9px; color: #484f58; }

  /* ── Model comparison table ── */
  .model-bar-wrap { flex: 1; background: #21262d; border-radius: 2px; height: 8px; display: inline-block; vertical-align: middle; min-width: 60px; }
  .model-bar-fill { height: 100%; border-radius: 2px; }

  /* ── Session bars ── */
  .sess-bars { display: flex; flex-direction: column; gap: 8px; }
  .sess-bar-row { display: flex; align-items: center; gap: 10px; font-size: 11px; }
  .sess-bar-label { width: 90px; color: #8b949e; flex-shrink: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .sess-bar-wrap { flex: 1; background: #21262d; border-radius: 3px; height: 12px; overflow: hidden; }
  .sess-bar-fill { height: 100%; border-radius: 3px; transition: width .3s; }
  .sess-bar-pct   { width: 36px; text-align: right; color: #3fb950; flex-shrink: 0; }
  .sess-bar-saved { width: 54px; text-align: right; color: #484f58; flex-shrink: 0; font-size: 10px; }

  /* ── Tables ── */
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

  /* ── Footer ── */
  .footer { font-size: 9px; color: #484f58; text-align: center; padding-top: 8px; }

  @media (max-width: 700px) {
    .layers, .hero-solo { grid-template-columns: 1fr; }
    .tiles { grid-template-columns: repeat(2, 1fr); }
    .cards { grid-template-columns: repeat(2, 1fr); }
    .layer-pct, .hero-pct { font-size: 48px; }
  }

  /* ── Tracker banner ── */
  .tracker-banner { display: none; align-items: center; gap: 12px; background: #0d1a35; border: 1px solid #1e2f50; border-radius: 8px; padding: 10px 16px; margin-bottom: 12px; font-size: 12px; }
  .tracker-banner a { color: #58a6ff; text-decoration: none; flex-shrink: 0; }
  .tracker-banner a:hover { text-decoration: underline; }
  .tracker-sep { color: #30363d; }
  .tracker-name { color: #f0f6fc; font-weight: 700; }
  .tracker-status-active { color: #3fb950; }
  .tracker-status-pending { color: #d29922; }
  /* ── Track button / chip ── */
  .track-btn { font-size: 11px; background: #161b22; color: #58a6ff; border: 1px solid #30363d; border-radius: 6px; padding: 4px 10px; cursor: pointer; font-family: inherit; }
  .track-btn:hover { border-color: #58a6ff; }
  .track-form { display: none; align-items: center; gap: 6px; }
  .track-input { font-size: 11px; background: #0d1117; color: #c9d1d9; border: 1px solid #30363d; border-radius: 5px; padding: 3px 8px; font-family: inherit; outline: none; width: 160px; }
  .track-input:focus { border-color: #58a6ff; }
  .track-start { font-size: 11px; background: #0d1a35; color: #58a6ff; border: 1px solid #1e2f50; border-radius: 5px; padding: 3px 8px; cursor: pointer; font-family: inherit; }
  .track-chip { display: none; align-items: center; gap: 6px; font-size: 11px; }
  .track-chip a { color: #58a6ff; text-decoration: none; }
  .track-chip a:hover { text-decoration: underline; }
  .track-cancel { background: none; color: #8b949e; border: none; cursor: pointer; font-size: 14px; padding: 0 2px; line-height: 1; }

  /* ── Prompt history (session dashboards only) ── */
  .prompt-row { background: #161b22; border: 1px solid #30363d; border-radius: 6px; margin-bottom: 8px; overflow: hidden; }
  .prompt-row-header { display: flex; align-items: center; gap: 12px; padding: 10px 14px; cursor: pointer; user-select: none; }
  .prompt-row-header:hover { background: #1c2128; }
  .prompt-toggle { color: #484f58; font-size: 10px; flex-shrink: 0; width: 10px; }
  .prompt-ts { color: #484f58; font-size: 10px; flex-shrink: 0; width: 52px; }
  .prompt-tokens { font-size: 11px; }
  .prompt-savings { font-size: 11px; color: #3fb950; font-weight: 600; min-width: 44px; }
  .prompt-latency { font-size: 10px; color: #484f58; margin-left: auto; }
  .prompt-no-text { font-size: 10px; color: #484f58; font-style: italic; margin-left: auto; }
  .prompt-body { display: none; padding: 0 14px 14px; }
  .prompt-body.open { display: block; }
  .prompt-panels { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .prompt-panel-label { font-size: 9px; text-transform: uppercase; letter-spacing: .1em; color: #8b949e; margin-bottom: 6px; display: flex; justify-content: space-between; }
  .prompt-text { font-size: 11px; line-height: 1.5; background: #0d1117; border: 1px solid #21262d; border-radius: 4px; padding: 10px 12px; max-height: 320px; overflow-y: auto; white-space: pre-wrap; word-break: break-word; color: #c9d1d9; font-family: inherit; margin: 0; }
  .prompt-text.compressed { border-color: #1e3a2a; background: #0f1e14; color: #aed6ae; }
  @media (max-width: 700px) { .prompt-panels { grid-template-columns: 1fr; } }
</style>
</head>
<body>

<!-- Session tracker banner (populated by JS when TRACKER is defined) -->
<div class="tracker-banner" id="tracker_banner">
  <a href="/dashboard">← global</a>
  <span class="tracker-sep">|</span>
  <span class="tracker-name" id="tracker_banner_name"></span>
  <span id="tracker_banner_status"></span>
</div>

<div class="header">
  <div style="display:flex;align-items:center;gap:8px">
    <div class="title">LLMLingua Proxy</div>
    <span class="model-badge" id="model_badge">—</span>
    <select class="model-select" id="model_select" onchange="switchModel(this.value)" disabled>
      <option value="llmlingua2">llmlingua2</option>
      <option value="llmlingua2-large">llmlingua2-large</option>
      <option value="kompress">kompress</option>
      <option value="dual">dual (system→large · user→kompress)</option>
    </select>
    <span class="model-loading" id="model_loading">loading…</span>
    <!-- Track new session controls -->
    <div id="track_chips"></div>
    <button class="track-btn" id="track_btn" onclick="openTrackForm()">Track new session</button>
    <div class="track-form" id="track_form">
      <input class="track-input" id="track_name" type="text" placeholder="session name" onkeydown="if(event.key==='Enter')submitTracker()" />
      <button class="track-start" onclick="submitTracker()">Start</button>
      <button class="track-cancel" onclick="closeTrackForm()">✕</button>
    </div>
  </div>
  <div class="header-right">
    <a href="/play" style="color:#8b949e;font-size:11px;text-decoration:none;letter-spacing:.02em">▶ play</a>
    <a href="/play/list" style="color:#8b949e;font-size:11px;text-decoration:none;letter-spacing:.02em">▶ history</a>
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
    <div class="layer-label">API layer <span class="layer-badge" id="api_layer_badge">LLMLingua-2</span></div>
    <div class="layer-pct" id="api_pct_layers">—%</div>
    <div class="layer-sub" id="api_layer_sub">Prompt compression · LLMLingua-2 tokenizer units</div>
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
    <div class="hero-sublabel" id="api_layer_sub_solo">LLMLingua-2 tokenizer units</div>
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

<!-- rtk hint (shown when no rtk_events recorded yet) -->
<div class="rtk-hint" id="rtk_hint">
  <span style="color:#484f58">&#x2B21;</span>
  <span>Shell-layer compression not yet logging. Add the PostToolUse hook for Bash in Claude Code settings to capture rtk savings per session.</span>
</div>

<!-- Tracked sessions summary (hidden until data arrives) -->
<div id="tracked_summary" style="display:none;font-size:11px;color:#8b949e;margin-bottom:10px;text-align:center"></div>

<!-- Row 1: Today tiles -->
<div class="tiles">
  <div class="tile">
    <div class="tile-num blue" id="tile_requests">—</div>
    <div class="tile-label">Today · Requests</div>
  </div>
  <div class="tile">
    <div class="tile-num green" id="tile_savings">—</div>
    <div class="tile-label">Today · Avg Savings %</div>
  </div>
  <div class="tile">
    <div class="tile-num" id="tile_latency">—</div>
    <div class="tile-label">Today · Avg Latency ms</div>
  </div>
  <div class="tile">
    <div class="tile-num green" id="tile_tokens_saved">—</div>
    <div class="tile-label">Today · Tokens Saved</div>
  </div>
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

<!-- Row 2: Model comparison -->
<div class="section" id="model_section" style="display:none">
  <div class="section-title">Model comparison</div>
  <table>
    <thead><tr><th>Model</th><th style="width:200px">Requests</th><th>Savings %</th><th>Latency ms</th></tr></thead>
    <tbody id="model_body"></tbody>
  </table>
</div>

<!-- Row 3: 48h timeseries -->
<div class="section">
  <div class="section-title">compression rate — last 48h (hourly) &nbsp;·&nbsp; height = requests &nbsp;·&nbsp; color = avg savings %</div>
  <div id="ts_chart" class="ts-chart"><span class="spark-empty">No data</span></div>
  <div id="ts_labels" class="ts-labels"></div>
</div>

<!-- Row 4: Recent activity -->
<div class="section">
  <div class="section-title">Recent activity</div>
  <table>
    <thead><tr><th>Time</th><th>Model</th><th>Role</th><th>Tokens</th><th>Savings %</th><th>Latency ms</th></tr></thead>
    <tbody id="recent_body"></tbody>
  </table>
</div>

<!-- LLMLingua sparkline -->
<div class="section">
  <div class="section-title">LLMLingua-2 — recent compressions &nbsp;·&nbsp; height = original size &nbsp;·&nbsp; color = savings %</div>
  <div id="sparkline" class="sparkline"><span class="spark-empty">No compressions yet</span></div>
  <div class="spark-legend">
    <span><span class="leg-dot" style="background:#3fb950"></span>&#x2265;40% saved</span>
    <span><span class="leg-dot" style="background:#d29922"></span>20&#x2013;39% saved</span>
    <span><span class="leg-dot" style="background:#484f58"></span>&lt;20% saved</span>
  </div>
</div>

<!-- Session efficiency bars -->
<div class="section">
  <div class="section-title">LLMLingua-2 — sessions by tokens saved</div>
  <div id="sess_bars" class="sess-bars"><span class="spark-empty">No sessions yet</span></div>
</div>

<!-- Session detail table -->
<div class="section">
  <div class="section-title">LLMLingua-2 — session detail</div>
  <table>
    <thead><tr><th>Session</th><th>Name</th><th>Requests</th><th>Saved</th><th>Ratio</th><th>Last seen</th></tr></thead>
    <tbody id="sessions_body"></tbody>
  </table>
</div>

<!-- Prompt compression history (session dashboards only) -->
<div class="section" id="prompt_section" style="display:none">
  <div class="section-title" style="display:flex;align-items:center;gap:10px">
    Prompt compression history
    <span id="prompt_count" style="color:#484f58;font-size:9px"></span>
  </div>
  <div id="prompt_list"><span class="spark-empty">No compressions yet</span></div>
</div>

<div class="footer">All data from metrics.db &nbsp;·&nbsp; updated every 2s</div>

<script>
const TRACKER = (typeof window !== 'undefined' && window.TRACKER) ? window.TRACKER : null;
let FILTER_SESSION = TRACKER ? TRACKER.session_id : null;

function updateBanner() {
  if (!TRACKER) return;
  const bannerEl = document.getElementById('tracker_banner');
  bannerEl.style.display = 'flex';
  document.getElementById('tracker_banner_name').textContent = TRACKER.name;
  const statusEl = document.getElementById('tracker_banner_status');
  if (TRACKER.status === 'active') {
    const sid = (TRACKER.session_id || '').slice(0, 8);
    statusEl.className = 'tracker-status-active';
    statusEl.textContent = 'active · ' + sid;
  } else {
    statusEl.className = 'tracker-status-pending';
    statusEl.innerHTML = '<span class="live-dot" style="background:#d29922;margin-right:4px"></span>waiting for next session…';
  }
}

function openTrackForm() {
  document.getElementById('track_btn').style.display = 'none';
  document.getElementById('track_form').style.display = 'flex';
  document.getElementById('track_name').focus();
}

function closeTrackForm() {
  document.getElementById('track_form').style.display = 'none';
  document.getElementById('track_btn').style.display = '';
  document.getElementById('track_name').value = '';
}

async function submitTracker() {
  const name = (document.getElementById('track_name').value || '').trim();
  if (!name) return;
  try {
    const r = await fetch('/admin/tracker', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
    const data = await r.json();
    if (r.ok) {
      window.location.href = '/dashboard/' + data.slug;
    } else {
      alert(data.error || 'Error creating tracker');
      closeTrackForm();
    }
  } catch(e) { console.error(e); }
}

async function cancelTracker(slug) {
  try { await fetch('/admin/tracker/' + encodeURIComponent(slug), { method: 'DELETE' }); } catch(e) {}
  try {
    const trackers = await fetch('/admin/tracker').then(r => r.json());
    renderTrackerChips(trackers);
  } catch(e) {}
}

function escapeHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function renderTrackerChips(trackers) {
  const container = document.getElementById('track_chips');
  if (!trackers || trackers.length === 0) {
    container.innerHTML = '';
    return;
  }
  container.innerHTML = trackers.map(t => {
    const color = t.status === 'active' ? '#3fb950' : '#d29922';
    const slug = encodeURIComponent(t.slug);
    return '<div class="track-chip" style="display:flex" data-slug="' + escapeHtml(t.slug) + '">'
      + '<a class="track-chip-link" href="/dashboard/' + slug + '">' + escapeHtml(t.name) + '</a>'
      + '<span style="font-size:10px;color:' + color + '">· ' + escapeHtml(t.status) + '</span>'
      + '<button class="track-cancel">✕</button>'
      + '</div>';
  }).join('');
  container.onclick = function(e) {
    const btn = e.target.closest('.track-cancel');
    if (btn) cancelTracker(btn.closest('[data-slug]').dataset.slug);
  };
}

function fmt(n) {
  if (n == null) return '—';
  if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
  if (n >= 1000)    return (n / 1000).toFixed(1) + 'k';
  return n.toLocaleString();
}
// alias used by tile population
var fmtN = fmt;

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

let _prevLoading = false;

async function switchModel(model) {
  const sel = document.getElementById('model_select');
  const lbl = document.getElementById('model_loading');
  sel.disabled = true;
  lbl.style.display = 'inline';
  _prevLoading = true;
  try {
    await fetch('/admin/set-model', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model }),
    });
  } catch(e) { console.error(e); }
}

async function refresh() {
  try {
  // Session tracker: poll for link while pending
  if (TRACKER && TRACKER.status === 'pending') {
    try {
      const list = await fetch('/admin/tracker').then(r => r.json());
      const t = Array.isArray(list) ? list.find(x => x.slug === TRACKER.slug) : null;
      if (t && t.status === 'active') {
        TRACKER.status = 'active';
        TRACKER.session_id = t.session_id;
        FILTER_SESSION = t.session_id;
      }
    } catch(e) {}
    updateBanner();
    if (!FILTER_SESSION) return; // still pending — no stats to show yet
  }
  if (TRACKER) updateBanner();

  // Build stats URL (with session filter when on session dashboard)
  const statsUrl = '/stats' + (FILTER_SESSION ? '?session_id=' + encodeURIComponent(FILTER_SESSION) : '');

    const r = await fetch(statsUrl);
    const d = await r.json();

    // ── Uptime ──
    const uptimeSecs = Math.floor((Date.now() - new Date(d.started_at)) / 1000);
    const uptimeStr = fmtUptime(uptimeSecs);
    document.getElementById('uptime_display').textContent = 'up ' + uptimeStr;
    document.getElementById('card_uptime').textContent = uptimeStr;

    // ── API layer numbers (DB-driven, filtered by active model) ──
    const at = d.alltime || {};
    const apiPctDisplay = at.avg_savings_pct != null ? Math.round(at.avg_savings_pct) : 0;

    // ── Hero: two-layer vs solo ──
    const hasRtk = d.rtk && d.rtk.total_commands > 0;
    document.getElementById('hero_layers').style.display = hasRtk ? 'grid' : 'none';
    document.getElementById('hero_solo').style.display   = hasRtk ? 'none' : 'grid';
    document.getElementById('rtk_hint').style.display    = hasRtk ? 'none' : 'flex';
    document.getElementById('rtk_section').style.display = hasRtk ? 'block' : 'none';

    // ── Update API layer model badge (both hero variants) ──
    const activeModel = d.compressor ? d.compressor.model : 'LLMLingua';
    const apiBadgeEl = document.getElementById('api_layer_badge');
    if (apiBadgeEl) apiBadgeEl.textContent = activeModel;
    const apiSubEl = document.getElementById('api_layer_sub');
    if (apiSubEl) apiSubEl.textContent = 'Prompt compression · ' + activeModel + ' tokenizer units';
    const apiSubSoloEl = document.getElementById('api_layer_sub_solo');
    if (apiSubSoloEl) apiSubSoloEl.textContent = activeModel + ' tokenizer units';

    if (hasRtk) {
      const rtk = d.rtk;
      const rtkPct = rtk.total_input_tokens > 0
        ? Math.round((rtk.total_saved_tokens / rtk.total_input_tokens) * 100) : 0;
      document.getElementById('rtk_pct').textContent    = rtkPct + '%';
      document.getElementById('rtk_cmds').textContent   = fmt(rtk.total_commands);
      document.getElementById('rtk_saved').textContent  = fmt(rtk.total_saved_tokens);
      document.getElementById('rtk_avg').textContent    = rtk.avg_savings_pct + '%';
      document.getElementById('api_pct_layers').textContent   = apiPctDisplay + '%';
      document.getElementById('api_reqs_layers').textContent  = fmt(at.requests);
      document.getElementById('api_saved_layers').textContent = fmt(at.tokens_saved);
      document.getElementById('api_ratio_layers').textContent = at.avg_ratio ? at.avg_ratio + '×' : '—';

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
      // Solo hero (DB-driven, filtered by active model)
      document.getElementById('savings_pct').textContent = apiPctDisplay + '%';
      document.getElementById('t_requests').textContent  = fmt(at.requests);
      document.getElementById('t_sessions').textContent  = at.sessions ?? Object.keys(d.sessions).length;
      document.getElementById('t_saved').textContent     = fmt(at.tokens_saved) + ' (' + apiPctDisplay + '%)';
      document.getElementById('t_ratio').textContent     = at.avg_ratio ? at.avg_ratio + '×' : '—';
    }

    // ── Model badge + select sync ──
    if (d.compressor) {
      const c = d.compressor;
      const sel = document.getElementById('model_select');
      const lbl = document.getElementById('model_loading');
      if (c.loading) {
        sel.disabled = true;
        lbl.style.display = 'inline';
        document.getElementById('model_badge').textContent = (c.model === 'dual' ? 'dual' : c.model) + ' · loading…';
        _prevLoading = true;
      } else {
        sel.value = c.model;
        sel.disabled = false;
        lbl.style.display = 'none';
        if (c.dual_mode) {
          document.getElementById('model_badge').textContent = 'dual · system→large / user→kompress';
        } else {
          document.getElementById('model_badge').textContent =
            c.model + (c.param_name ? ' · ' + c.param_name + '=' + c.param_value : '');
        }
        if (_prevLoading) { _prevLoading = false; refreshTimeseries(); }
      }
    }

    // ── Tracked sessions summary ──
    const tracked = d.tracked || {};
    const trackedEl = document.getElementById('tracked_summary');
    if (trackedEl && !TRACKER && tracked.sessions > 0) {
      trackedEl.textContent = fmt(tracked.tokens_saved) + ' tokens saved across ' + tracked.sessions + ' tracked session' + (tracked.sessions !== 1 ? 's' : '') + ' · ';
      const link = document.createElement('a');
      link.href = '/play/list';
      link.textContent = 'view history';
      link.style.cssText = 'color:#58a6ff;font-size:11px';
      trackedEl.appendChild(link);
      trackedEl.style.display = '';
    } else if (trackedEl) {
      trackedEl.style.display = 'none';
    }

    // ── Today tiles (DB-driven) ──
    document.getElementById('tile_requests').textContent = d.today.requests ?? 0;
    document.getElementById('tile_savings').textContent = (d.today.avg_savings_pct ?? 0) + '%';
    document.getElementById('tile_latency').textContent = (d.today.avg_latency_ms ?? 0) + 'ms';
    document.getElementById('tile_tokens_saved').textContent = fmtN(d.today.tokens_saved ?? 0);

    // ── Row 2: Model comparison ──
    if (d.by_model && d.by_model.length > 0) {
      document.getElementById('model_section').style.display = 'block';
      const maxReq = Math.max(...d.by_model.map(m => m.requests), 1);
      document.getElementById('model_body').innerHTML = d.by_model.map(function(m) {
        const isActive = d.compressor && m.model === d.compressor.model;
        const bg = isActive ? 'background:#1c2128;' : '';
        const w = Math.round((m.requests / maxReq) * 100);
        return '<tr style="' + bg + '">'
          + '<td><span class="model-badge">' + m.model + '</span></td>'
          + '<td><div class="model-bar-wrap" style="width:160px"><div class="model-bar-fill" style="width:' + w + '%;background:' + barColor(m.avg_savings_pct || 0) + ';height:100%"></div></div>'
          + ' <span class="muted" style="font-size:10px">' + m.requests + '</span></td>'
          + '<td class="green">' + (m.avg_savings_pct ?? '—') + '%</td>'
          + '<td class="muted">' + (m.avg_latency_ms != null ? m.avg_latency_ms + ' ms' : '—') + '</td>'
          + '</tr>';
      }).join('');
    }

    // ── Metric cards (DB-driven, filtered by active model) ──
    document.getElementById('card_requests').textContent = fmt(at.requests);
    document.getElementById('card_saved').textContent    = fmt(at.tokens_saved);
    document.getElementById('card_sessions').textContent = at.sessions ?? Object.keys(d.sessions).length;
    document.getElementById('card_ratio').textContent    = at.avg_ratio ? at.avg_ratio + '×' : '—';

    // ── Latency card ──
    document.getElementById('card_latency').textContent =
      at.avg_latency_ms != null ? Math.round(at.avg_latency_ms) + ' ms' : '—';

    // ── Cost card ──
    if (d.cost_per_mtok != null && at.tokens_saved != null) {
      const cost = (at.tokens_saved / 1000000) * d.cost_per_mtok;
      document.getElementById('card_cost').textContent = '$' + cost.toFixed(2);
      document.getElementById('card_cost_label').textContent = '@ $' + d.cost_per_mtok.toFixed(2) + ' / MTok';
    }

    // ── Row 4: Recent activity table ──
    if (d.recent && d.recent.length > 0) {
      document.getElementById('recent_body').innerHTML = d.recent.slice(0, 10).map(function(row) {
        var roleBadge = row.role === 'system'
          ? '<span style="font-size:10px;border-radius:3px;padding:1px 5px;background:#0d1a35;color:#58a6ff;border:1px solid #1e2f50">system</span>'
          : '<span style="font-size:10px;border-radius:3px;padding:1px 5px;background:#0f1e14;color:#aed6ae;border:1px solid #1e3a2a">user</span>';
        return '<tr>'
          + '<td class="muted">' + row.ts.slice(11, 16) + '</td>'
          + '<td><span class="model-badge">' + row.model + '</span></td>'
          + '<td>' + roleBadge + '</td>'
          + '<td>' + fmt(row.original_tokens) + ' &#x2192; ' + fmt(row.compressed_tokens) + '</td>'
          + '<td class="green">' + row.savings_pct + '%</td>'
          + '<td class="muted">' + row.latency_ms + ' ms</td>'
          + '</tr>';
      }).join('');
    } else {
      document.getElementById('recent_body').innerHTML =
        '<tr><td colspan="6" class="muted" style="text-align:center;padding:12px">No data yet</td></tr>';
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

    // Update main dashboard tracker chips (only on global dashboard, not session dashboard)
    if (!TRACKER) {
      try {
        const trackers = await fetch('/admin/tracker').then(r => r.json());
        renderTrackerChips(trackers);
      } catch(e) {}
    }

  } catch(e) { console.error(e); }
}

async function refreshTimeseries() {
  try {
    const sel = document.getElementById('model_select');
    const model = sel ? sel.value : '';
    const params = [];
    if (model) params.push('model=' + encodeURIComponent(model));
    if (FILTER_SESSION) params.push('session_id=' + encodeURIComponent(FILTER_SESSION));
    const r = await fetch('/stats/timeseries' + (params.length ? '?' + params.join('&') : ''));
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

function togglePrompt(id) {
  const body = document.getElementById(id + '_body');
  const arrow = document.getElementById(id + '_arrow');
  if (!body) return;
  const isOpen = body.classList.contains('open');
  body.classList.toggle('open', !isOpen);
  if (arrow) arrow.textContent = isOpen ? '▶' : '▼';
}

async function refreshCompressions() {
  if (!TRACKER || !TRACKER.session_id) return;
  const section = document.getElementById('prompt_section');
  if (!section) return;
  section.style.display = 'block';
  try {
    const rows = await fetch('/session/' + encodeURIComponent(TRACKER.slug) + '/compressions').then(r => r.json());
    const listEl = document.getElementById('prompt_list');
    const countEl = document.getElementById('prompt_count');
    if (!Array.isArray(rows) || rows.length === 0) {
      listEl.innerHTML = '<span class="spark-empty">No compressions yet</span>';
      countEl.textContent = '';
      return;
    }
    countEl.textContent = rows.length + (rows.length === 50 ? '+ ' : ' ') + 'requests';
    const openIds = new Set([...listEl.querySelectorAll('.prompt-body.open')].map(el => el.id));
    listEl.innerHTML = rows.map(r => {
      const pct = r.savings_pct != null ? r.savings_pct : 0;
      const hasText = r.original_text != null;
      const ts = r.ts ? r.ts.slice(11, 16) : '';
      const id = 'pr_' + r.id;
      const origLen = (r.original_text || '').length;
      const compLen = (r.compressed_text || '').length;
      return '<div class="prompt-row">'
        + '<div class="prompt-row-header" data-toggle-id="' + id + '">'
        + '<span class="prompt-toggle" id="' + id + '_arrow">▶</span>'
        + '<span class="prompt-ts">' + ts + '</span>'
        + '<span class="prompt-tokens">' + fmt(r.original_tokens) + ' → ' + fmt(r.compressed_tokens) + ' tok</span>'
        + '<span class="prompt-savings">-' + pct + '%</span>'
        + '<span class="model-badge" style="font-size:9px;padding:1px 5px">' + (r.model || '—') + '</span>'
        + (r.latency_ms ? '<span class="prompt-latency">' + Math.round(r.latency_ms) + 'ms</span>' : '')
        + (hasText ? '' : '<span class="prompt-no-text">text not stored</span>')
        + '</div>'
        + (hasText
          ? '<div class="prompt-body" id="' + id + '_body">'
            + '<div class="prompt-panels">'
            + '<div><div class="prompt-panel-label"><span>Original</span><span style="color:#484f58">' + fmt(origLen) + ' chars</span></div>'
            + '<pre class="prompt-text">' + escapeHtml(r.original_text) + '</pre></div>'
            + '<div><div class="prompt-panel-label"><span>Compressed</span><span style="color:#3fb950">' + fmt(compLen) + ' chars</span></div>'
            + '<pre class="prompt-text compressed">' + escapeHtml(r.compressed_text || '') + '</pre></div>'
            + '</div></div>'
          : '')
        + '</div>';
    }).join('');
    openIds.forEach(bodyId => {
      const body = document.getElementById(bodyId);
      const arrow = document.getElementById(bodyId.replace('_body', '_arrow'));
      if (body) { body.classList.add('open'); if (arrow) arrow.textContent = '▼'; }
    });
    listEl.onclick = function(e) {
      const hdr = e.target.closest('[data-toggle-id]');
      if (hdr) togglePrompt(hdr.dataset.toggleId);
    };
  } catch(e) { console.error(e); }
}

updateBanner();
refresh();
refreshTimeseries();
if (TRACKER) refreshCompressions();
setInterval(refresh, 2000);
setInterval(refreshTimeseries, 10000);
if (TRACKER) setInterval(refreshCompressions, 5000);
</script>
</body>
</html>"""


PLAY_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Playground — LLMLingua</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'SF Mono', 'Fira Code', monospace; background: #0d1117; color: #c9d1d9;
         display: flex; flex-direction: column; height: 100vh; padding: 16px; gap: 12px; }

  .header { display: flex; align-items: center; gap: 10px; padding-bottom: 12px;
             border-bottom: 1px solid #21262d; flex-shrink: 0; }
  .back-link { font-size: 11px; color: #58a6ff; text-decoration: none; }
  .back-link:hover { text-decoration: underline; }
  .title { color: #58a6ff; font-size: 14px; font-weight: 700; letter-spacing: .04em; }
  .sep { color: #30363d; }

  .controls { display: flex; align-items: center; gap: 10px; flex: 1; }
  .model-select { font-size: 12px; background: #161b22; color: #58a6ff; padding: 4px 10px;
                   border-radius: 6px; border: 1px solid #30363d; cursor: pointer;
                   font-family: inherit; outline: none; }
  .model-select:hover:not(:disabled) { border-color: #58a6ff; }
  .model-select:disabled { opacity: 0.5; cursor: wait; }
  .model-loading { font-size: 10px; color: #d29922; display: none; }

  .compress-btn { font-size: 12px; background: #0d1a35; color: #58a6ff; border: 1px solid #1e2f50;
                   border-radius: 6px; padding: 5px 14px; cursor: pointer; font-family: inherit; }
  .compress-btn:hover:not(:disabled) { border-color: #58a6ff; }
  .compress-btn:disabled { opacity: 0.5; cursor: wait; }

  .panels { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; flex: 1; min-height: 0; }
  .panel { display: flex; flex-direction: column; gap: 6px; min-height: 0; }
  .panel-label { font-size: 9px; color: #8b949e; text-transform: uppercase; letter-spacing: .1em;
                  display: flex; align-items: center; justify-content: space-between; flex-shrink: 0; }
  textarea { flex: 1; width: 100%; background: #161b22; color: #c9d1d9; border: 1px solid #30363d;
              border-radius: 8px; padding: 14px; font-family: inherit; font-size: 12px;
              resize: none; outline: none; line-height: 1.6; min-height: 0; }
  textarea.compressed     { border-color: #1e3a2a; background: #0f1e14; color: #aed6ae; }
  textarea.compressed.dim { border-color: #2a2a30; background: #13131a; color: #6a7a6a; }
  textarea.unchanged      { color: #8b949e; }
  .copy-btn { font-size: 10px; background: none; border: none; color: #8b949e;
               cursor: pointer; font-family: inherit; padding: 0; }
  .copy-btn:hover { color: #c9d1d9; }

  .stats-bar { flex-shrink: 0; display: flex; align-items: center; gap: 20px; font-size: 11px;
                background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                padding: 10px 16px; }
  .stats-bar.hidden { visibility: hidden; }
  .stat-label { color: #8b949e; font-size: 10px; margin-right: 4px; }
  .stat-nums  { color: #c9d1d9; }
  .stat-pct   { font-weight: 700; margin-left: 6px; }
  .green { color: #3fb950; }
  .dim   { color: #484f58; }
  .stat-latency { color: #484f58; font-size: 10px; margin-left: auto; }

  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
  .dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%;
          background: #d29922; animation: pulse 1s infinite; vertical-align: middle; margin-right: 4px; }

  @media (max-width: 700px) { .panels { grid-template-columns: 1fr; } }
</style>
</head>
<body>

<div class="header">
  <a class="back-link" href="/dashboard">← dashboard</a>
  <span class="sep">|</span>
  <div class="title">Playground</div>

  <div class="controls">
    <select class="model-select" id="model_select">
      <option value="llmlingua2">llmlingua2</option>
      <option value="llmlingua2-large">llmlingua2-large</option>
      <option value="kompress">kompress</option>
      <option value="dual">dual (system→large · user→kompress)</option>
    </select>
    <span class="model-loading" id="model_loading"><span class="dot"></span>loading model…</span>

    <button class="compress-btn" id="compress_btn" onclick="compress()">▶ Compress</button>
  </div>
</div>

<div class="panels">
  <div class="panel">
    <div class="panel-label">
      <span>Input</span>
      <span id="input_meta" style="color:#484f58;font-size:10px"></span>
    </div>
    <textarea id="input_text" placeholder="Paste text to compress…" oninput="onInput()"></textarea>
  </div>

  <div class="panel">
    <div class="panel-label">
      <span>Compressed</span>
      <div style="display:flex;align-items:center;gap:8px">
        <span id="output_meta" style="color:#484f58;font-size:10px"></span>
        <button class="copy-btn" id="copy_btn" onclick="copyOutput()" style="display:none">Copy</button>
      </div>
    </div>
    <textarea id="output_text" readonly placeholder="Output appears here…"></textarea>
  </div>
</div>

<div class="stats-bar hidden" id="stats_bar">
  <div>
    <span class="stat-label">Chars</span>
    <span class="stat-nums" id="s_chars"></span>
    <span class="stat-pct" id="s_chars_pct"></span>
  </div>
  <div>
    <span class="stat-label">~Claude tokens</span>
    <span class="stat-nums" id="s_toks"></span>
    <span class="stat-pct" id="s_toks_pct"></span>
  </div>
  <span class="stat-latency" id="s_lat"></span>
</div>

<script>
let debounceTimer = null;
let pollTimer = null;
let lastResult = null;
let suppressModelChange = false;

function onInput() {
  const n = document.getElementById('input_text').value.length;
  document.getElementById('input_meta').textContent = n ? fmt(n) + ' chars' : '';
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(compress, 600);
}

function applyDisplay(d) {
  const outEl = document.getElementById('output_text');
  outEl.value = d.compressed;
  outEl.className = d.char_pct > 0 ? 'compressed' : 'unchanged';
  const outLen = d.compressed.length;
  document.getElementById('output_meta').textContent = outLen ? fmt(outLen) + ' chars' : '';
  document.getElementById('copy_btn').style.display = outLen ? 'inline' : 'none';
  renderStats(d);
}

async function compress() {
  clearTimeout(pollTimer);
  const text = document.getElementById('input_text').value;

  if (!text.trim()) {
    lastResult = null; reset(); return;
  }

  const model = document.getElementById('model_select').value;
  setBusy(true);

  try {
    const resp = await fetch('/play/compress', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, model }),
    });
    const data = await resp.json();

    if (resp.status === 202 && data.loading) {
      showLoading();
      pollTimer = setTimeout(compress, 1500);
      return;
    }

    hideLoading();

    if (data.error) {
      document.getElementById('output_text').value = '⚠ ' + data.error;
      document.getElementById('output_text').className = 'unchanged';
      setBusy(false); return;
    }

    lastResult = data;
    applyDisplay(data);
  } catch(e) {
    document.getElementById('output_text').value = '⚠ ' + e.message;
  } finally {
    setBusy(false);
  }
}

function renderStats(d) {
  const bar = document.getElementById('stats_bar');
  bar.className = 'stats-bar';

  document.getElementById('s_chars').textContent = fmt(d.original_chars) + ' → ' + fmt(d.compressed_chars);
  const cpEl = document.getElementById('s_chars_pct');
  cpEl.textContent = d.char_pct > 0 ? '-' + d.char_pct + '%' : '0%';
  cpEl.className = 'stat-pct ' + (d.char_pct > 0 ? 'green' : 'dim');

  document.getElementById('s_toks').textContent = fmt(d.original_tokens_est) + ' → ' + fmt(d.compressed_tokens_est);
  const tpEl = document.getElementById('s_toks_pct');
  tpEl.textContent = d.token_pct > 0 ? '-' + d.token_pct + '%' : '0%';
  tpEl.className = 'stat-pct ' + (d.token_pct > 0 ? 'green' : 'dim');

  document.getElementById('s_lat').textContent = d.latency_ms > 0 ? d.latency_ms + 'ms' : '';
}

function reset() {
  document.getElementById('output_text').value = '';
  document.getElementById('output_text').className = '';
  document.getElementById('output_meta').textContent = '';
  document.getElementById('copy_btn').style.display = 'none';
  document.getElementById('stats_bar').className = 'stats-bar hidden';
  document.getElementById('input_meta').textContent = '';
}

function setBusy(on) {
  const btn = document.getElementById('compress_btn');
  btn.disabled = on;
  btn.textContent = on ? '…' : '▶ Compress';
}

function showLoading() {
  document.getElementById('model_loading').style.display = 'inline';
  document.getElementById('model_select').disabled = true;
}
function hideLoading() {
  document.getElementById('model_loading').style.display = 'none';
  suppressModelChange = true;
  document.getElementById('model_select').disabled = false;
  suppressModelChange = false;
}

function fmt(n) {
  return n >= 1000 ? (n / 1000).toFixed(1) + 'k' : String(n);
}

async function copyOutput() {
  const text = document.getElementById('output_text').value;
  if (!text) return;
  await navigator.clipboard.writeText(text);
  const btn = document.getElementById('copy_btn');
  btn.textContent = 'Copied!';
  setTimeout(() => btn.textContent = 'Copy', 1500);
}

async function init() {
  try {
    const data = await fetch('/stats').then(r => r.json());
    const info = data.compressor_info;
    if (!info) return;
    const sel = document.getElementById('model_select');
    const m = info.model || '';
    if (m && [...sel.options].some(o => o.value === m)) {
      suppressModelChange = true;
      sel.value = m;
      suppressModelChange = false;
    }
    if (info.loading) { showLoading(); pollTimer = setTimeout(init, 2000); }
  } catch(e) {}
}

document.getElementById('model_select').addEventListener('change', () => {
  if (!suppressModelChange) compress();
});

init();
</script>
</body>
</html>"""


LIST_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Session History · LLMLingua</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'SF Mono','Fira Code',monospace; background: #0d1117; color: #c9d1d9; padding: 20px; max-width: 900px; margin: 0 auto; }
  .header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #21262d; }
  .title { color: #58a6ff; font-size: 14px; font-weight: 700; letter-spacing: .04em; }
  a { color: #58a6ff; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .summary { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px; font-size: 12px; color: #8b949e; display: flex; gap: 24px; flex-wrap: wrap; }
  .summary b { color: #c9d1d9; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th { text-align: left; color: #8b949e; font-weight: 600; padding: 8px 12px; border-bottom: 1px solid #21262d; letter-spacing: .04em; font-size: 10px; text-transform: uppercase; }
  td { padding: 10px 12px; border-bottom: 1px solid #161b22; vertical-align: middle; }
  tr:hover td { background: #161b22; }
  .badge { display: inline-block; font-size: 10px; padding: 1px 7px; border-radius: 3px; font-weight: 600; letter-spacing: .03em; }
  .badge-pending { background: #2b2200; color: #d29922; }
  .badge-active  { background: #0d2b1a; color: #3fb950; }
  .badge-closed  { background: #1c2128; color: #484f58; }
  .slug-hint { font-size: 10px; color: #484f58; margin-top: 2px; }
  .empty { text-align: center; color: #484f58; padding: 40px; font-size: 12px; }
</style>
</head>
<body>
<div class="header">
  <span class="title">SESSION HISTORY</span>
  <a href="/dashboard" style="font-size:11px;color:#8b949e">← dashboard</a>
</div>
<div class="summary" id="summary">loading…</div>
<table>
  <thead>
    <tr>
      <th>Name</th>
      <th>Status</th>
      <th>Tokens saved</th>
      <th>Requests</th>
      <th>Created</th>
      <th>Linked</th>
      <th>Closed</th>
    </tr>
  </thead>
  <tbody id="rows"></tbody>
</table>
<script>
function fmt(n) {
  if (!n) return '—';
  if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
  if (n >= 1000) return (n / 1000).toFixed(1) + 'k';
  return n.toLocaleString();
}
function fmtDate(s) {
  if (!s) return '—';
  return s.replace('T', ' ').slice(0, 16);
}
function escapeHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
async function load() {
  const data = await fetch('/admin/tracker/all').then(r => r.json());
  const totalSaved = data.reduce((a, t) => a + (t.tokens_saved || 0), 0);
  const linked = data.filter(t => t.session_id);
  document.getElementById('summary').innerHTML =
    '<span><b>' + data.length + '</b> sessions total</span>' +
    '<span><b>' + linked.length + '</b> linked to a Claude session</span>' +
    '<span><b>' + fmt(totalSaved) + '</b> tokens saved across all tracked</span>';
  const tbody = document.getElementById('rows');
  if (!data.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="empty">No tracked sessions yet.</td></tr>';
    return;
  }
  tbody.innerHTML = data.map(t => {
    const slug = encodeURIComponent(t.slug);
    const badge = '<span class="badge badge-' + escapeHtml(t.status) + '">' + escapeHtml(t.status) + '</span>';
    return '<tr>'
      + '<td><a href="/dashboard/' + slug + '">' + escapeHtml(t.name) + '</a>'
      + '<div class="slug-hint">' + escapeHtml(t.slug) + '</div></td>'
      + '<td>' + badge + '</td>'
      + '<td style="color:#3fb950;font-weight:600">' + fmt(t.tokens_saved) + '</td>'
      + '<td>' + (t.requests || '—') + '</td>'
      + '<td style="color:#484f58">' + fmtDate(t.created_at) + '</td>'
      + '<td style="color:#484f58">' + fmtDate(t.linked_at) + '</td>'
      + '<td style="color:#484f58">' + fmtDate(t.closed_at) + '</td>'
      + '</tr>';
  }).join('');
}
load();
</script>
</body>
</html>"""


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9099)
