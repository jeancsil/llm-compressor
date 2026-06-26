import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import hashlib
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
COST_PER_MTOK     = float(os.environ.get("COST_PER_MTOK", "3.0"))


def _rtk_data_dir() -> Path:
    """Platform-specific directory where both history.db and metrics.db live."""
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "rtk"
    if system == "Windows":
        return Path(os.environ.get("APPDATA", Path.home())) / "rtk"
    return Path.home() / ".local" / "share" / "rtk"


def _default_db_path() -> Path:
    d = _rtk_data_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / "metrics.db"


DB_PATH = Path(os.environ.get("LLM_COMPRESSOR_DB") or _default_db_path())

# Module-level globals populated by lifespan
backend          = None
backend_loading  = None   # set to model name while async load is in progress
_db_conn         = None
backend_user     = None   # kompress instance in dual mode
backend_system   = None   # llmlingua2-large instance in dual mode
dual_mode        = False
dual_model_system = "llmlingua2-large"   # persisted in meta table
dual_model_user   = "kompress"           # persisted in meta table

KNOWN_MODELS    = ("llmlingua2", "llmlingua2-large", "kompress", "dual")
DUAL_SUBMODELS  = ("llmlingua2", "llmlingua2-large", "kompress")


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(text: str, model_tag: str, rate: float) -> str:
    """Exact-match cache key. Model and rate are included because the same text
    yields different output under a different backend/rate."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{digest}|{model_tag}|{rate}"


# ---------------------------------------------------------------------------
# DB helpers
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
    try:
        conn.execute("ALTER TABLE rtk_events ADD COLUMN project_path TEXT DEFAULT ''")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rtk_events_project ON rtk_events(project_path)")
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
            "headroom-ai[ml] is not installed. Run: uv add 'headroom-ai[ml]'"
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


def _load_single_backend(model_name: str) -> dict:
    """Load any non-dual backend by name."""
    if model_name == "kompress":
        return _load_kompress_backend()
    return _load_llmlingua2_backend(backend_key=model_name)


def _load_dual_backend() -> dict:
    """Load both sub-backends and set dual-mode globals.

    Uses the module-level dual_model_system / dual_model_user which are
    persisted in the meta table and configurable at runtime via
    /admin/set-dual-models.
    """
    global backend_user, backend_system, dual_mode
    sys_m = dual_model_system
    usr_m = dual_model_user
    print(f"Loading dual mode: {usr_m} (user) + {sys_m} (system)...")
    backend_system = _load_single_backend(sys_m)
    backend_user   = _load_single_backend(usr_m)
    dual_mode      = True
    print("Dual mode ready.")
    return {"type": "dual", "model_user": usr_m, "model_system": sys_m}


def load_backend() -> dict:
    """Dispatch to the configured backend loader.

    Resolution order:
    1. DB meta table keys 'current_model', 'dual_model_system', 'dual_model_user'
    2. COMPRESSOR_MODEL environment variable
    3. Defaults: llmlingua2 / llmlingua2-large / kompress
    """
    global dual_model_system, dual_model_user
    model_name = os.environ.get("COMPRESSOR_MODEL", "llmlingua2")
    try:
        if _db_conn is not None:
            for key, default in (
                ("current_model",    None),
                ("dual_model_system", None),
                ("dual_model_user",   None),
            ):
                row = _db_conn.execute(
                    "SELECT value FROM meta WHERE key=?", (key,)
                ).fetchone()
                if row:
                    if key == "current_model":
                        model_name = row[0]
                    elif key == "dual_model_system":
                        dual_model_system = row[0]
                    elif key == "dual_model_user":
                        dual_model_user = row[0]
    except Exception:
        pass  # DB not available; fall back to env var / module defaults
    if model_name == "dual":
        return _load_dual_backend()
    return _load_single_backend(model_name)


# Keep the private alias so existing call-sites (lifespan, tests) still work.
_load_backend = load_backend


def _pick_backend(role: str) -> dict | None:
    if dual_mode and backend_user is not None and backend_system is not None:
        return backend_system if role == "system" else backend_user
    return backend


# ---------------------------------------------------------------------------
# DB location migration (legacy ./metrics.db → RTK data dir)
# ---------------------------------------------------------------------------

def _migrate_db_location() -> None:
    """Copy metrics.db from the old CWD location to the RTK data directory once.

    Skipped when: old doesn't exist, paths are the same, or new already has data
    (size > 64 KiB means it was populated, not just an empty shell created by a
    previous aborted startup).
    """
    old = Path("metrics.db").resolve()
    new = DB_PATH.resolve()
    if old == new or not old.exists():
        return
    if new.exists() and new.stat().st_size > 65536:
        return  # new DB already has real data
    import shutil
    try:
        shutil.copy2(str(old), str(new))
        print(f"[db] migrated {old} → {new}")
        old.rename(old.with_suffix(".db.migrated"))
    except Exception as e:
        print(f"[db] migration failed, using {old}: {e}")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global backend, _db_conn
    _migrate_db_location()
    _db_conn = init_db(str(DB_PATH))
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
    "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
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
        "first_seen": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "requests": 0,
        "original_tokens": 0,
        "compressed_tokens": 0,
        "name": None,
    })
    sess["requests"] += 1
    sess["last_seen"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if session_name:
        sess["name"] = session_name

    _try_link_pending_tracker(session_id)

# ---------------------------------------------------------------------------
# rtk integration (optional — gracefully absent when rtk not installed)
# ---------------------------------------------------------------------------

def _rtk_db_path() -> Path:
    return _rtk_data_dir() / "history.db"

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
# /stats helpers
#
# get_stats() is an orchestrator; these pull the per-section logic out so each
# query lives in one place. In particular, today/alltime/recent all share the
# same model/session scoping, so it is defined once in _stats_scope().
# ---------------------------------------------------------------------------

def _compressor_info() -> dict:
    """Describe the active (or loading) compression backend for the dashboard."""
    if backend_loading:
        return {"model": backend_loading, "param_name": "", "param_value": "", "loading": True}
    if backend and backend.get("type") == "dual":
        return {
            "model": "dual", "loading": False, "param_name": None, "param_value": None,
            "model_system": dual_model_system, "model_user": dual_model_user,
        }
    if backend and backend.get("type") == "kompress":
        return {
            "model": "kompress", "param_name": "threshold",
            "param_value": backend.get("threshold", 0.5), "loading": False,
        }
    backend_key = backend.get("backend_key", "llmlingua2") if backend else "llmlingua2"
    return {
        "model": backend_key, "param_name": "rate",
        "param_value": backend.get("rate", 0.5) if backend else 0.5, "loading": False,
    }


def _stats_scope(active_model: str, session_id: str | None) -> tuple[str, tuple]:
    """WHERE fragment + args selecting compression rows for the active scope.

    A session filter wins; otherwise dual mode spans all sub-model names while a
    single model matches just itself.
    """
    if session_id:
        return "session_id = ?", (session_id,)
    if active_model == "dual":
        return f"model IN ({', '.join('?' * len(DUAL_SUBMODELS))})", DUAL_SUBMODELS
    return "model = ?", (active_model,)


def _aggregate_stats(scope: str, args: tuple, *, today: bool, with_ratio: bool) -> dict:
    """Aggregate request/savings/latency metrics for a scope, optionally today-only."""
    date_clause = "date(ts) = date('now') AND " if today else ""
    ratio_col = (
        ", ROUND(AVG(CAST(original_tokens AS REAL) / NULLIF(compressed_tokens, 0)), 2) AS avg_ratio"
        if with_ratio else ""
    )
    row = _db_conn.execute(
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
    rows = _db_conn.execute(
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
            "ts": r[0], "session_id": r[1][:8] if r[1] else "", "model": r[2],
            "original_tokens": r[3], "compressed_tokens": r[4],
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
    row = _db_conn.execute(
        f"""SELECT COUNT(*), COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0),
                   COALESCE(SUM(saved_tokens),0), COALESCE(AVG(savings_pct),0)
            FROM rtk_events {where}""",
        args,
    ).fetchone()
    if not (row and row[0] > 0):
        return None
    top = _db_conn.execute(
        f"""SELECT rtk_cmd, COUNT(*) AS cnt, SUM(saved_tokens) AS saved, AVG(savings_pct) AS avg_pct
            FROM rtk_events {where}
            GROUP BY rtk_cmd ORDER BY saved DESC LIMIT 8""",
        args,
    ).fetchall()
    return {
        "total_commands": row[0], "total_input_tokens": row[1], "total_output_tokens": row[2],
        "total_saved_tokens": row[3], "avg_savings_pct": round(row[4], 1),
        "top_commands": [
            {"cmd": r[0], "count": r[1], "saved": r[2], "avg_pct": round(r[3], 1)} for r in top
        ],
    }


def _tracked_stats() -> dict:
    """Totals across tracked sessions (active/closed trackers joined to compressions)."""
    row = _db_conn.execute(
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
    for sid, cmds, saved in _db_conn.execute(
        """SELECT session_id, COUNT(*), COALESCE(SUM(saved_tokens), 0)
           FROM rtk_events GROUP BY session_id"""
    ).fetchall():
        if sid in sessions_out:
            sessions_out[sid]["rtk_commands"] = cmds
            sessions_out[sid]["rtk_saved"] = saved


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

    compressor_info = _compressor_info()

    by_model: list = []
    today_stats: dict = {"requests": 0, "tokens_saved": 0, "avg_savings_pct": 0.0, "avg_latency_ms": 0.0, "sessions": 0}
    alltime_stats: dict = {"requests": 0, "tokens_saved": 0, "avg_savings_pct": 0.0, "avg_latency_ms": 0.0, "sessions": 0}
    recent_rows: list = []
    rtk_stats: dict | None = None
    tracked_stats: dict = {"sessions": 0, "tokens_saved": 0}

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

        scope, scope_args = _stats_scope(active_model, session_id)
        today_stats   = _aggregate_stats(scope, scope_args, today=True, with_ratio=False)
        alltime_stats = _aggregate_stats(scope, scope_args, today=False, with_ratio=True)
        recent_rows   = _recent_compression_rows(active_model, session_id)
        rtk_stats     = _rtk_stats(session_id)
        tracked_stats = _tracked_stats()
        _merge_rtk_into_sessions(sessions_out)

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
                    new_backend = _load_kompress_backend()
                elif _target == "dual":
                    new_backend = _load_dual_backend()
                else:
                    new_backend = _load_llmlingua2_backend(backend_key=_target)
                backend = new_backend
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


@app.post("/admin/set-dual-models")
async def set_dual_models(request: Request):
    """Configure which models handle system vs user turns in dual mode.

    Body: {"system": "<model>", "user": "<model>"}
    Valid values for each: llmlingua2 | llmlingua2-large | kompress
    Omit a key to leave it unchanged.
    If dual mode is currently active, the affected sub-backends reload immediately.
    """
    global dual_model_system, dual_model_user, backend, backend_loading, backend_user, backend_system
    body = await request.json()
    new_sys = body.get("system")
    new_usr = body.get("user")

    if not new_sys and not new_usr:
        return JSONResponse({"error": "Provide at least one of 'system' or 'user'"}, status_code=400)
    if new_sys and new_sys not in DUAL_SUBMODELS:
        return JSONResponse({"error": f"Invalid system model. Valid: {list(DUAL_SUBMODELS)}"}, status_code=400)
    if new_usr and new_usr not in DUAL_SUBMODELS:
        return JSONResponse({"error": f"Invalid user model. Valid: {list(DUAL_SUBMODELS)}"}, status_code=400)

    if new_sys:
        dual_model_system = new_sys
    if new_usr:
        dual_model_user = new_usr

    if _db_conn is not None:
        if new_sys:
            _db_conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('dual_model_system', ?)", (new_sys,)
            )
        if new_usr:
            _db_conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('dual_model_user', ?)", (new_usr,)
            )
        _db_conn.commit()

    if dual_mode:
        backend = None
        backend_loading = "dual"
        backend_user = None
        backend_system = None

        def reload_dual():
            global backend, backend_loading
            try:
                new_backend = _load_dual_backend()
                backend = new_backend
            except Exception as e:
                print(f"[set-dual-models] failed: {e}")
            finally:
                backend_loading = None

        threading.Thread(target=reload_dual, daemon=True).start()
        return JSONResponse({"status": "loading", "system": dual_model_system, "user": dual_model_user})

    return JSONResponse({"status": "ok", "system": dual_model_system, "user": dual_model_user})


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
               COUNT(c.id) AS requests,
               COALESCE(SUM(r.saved_tokens), 0) AS rtk_saved,
               COUNT(r.id) AS rtk_commands
        FROM trackers t
        LEFT JOIN compressions c ON c.session_id = t.session_id
        LEFT JOIN rtk_events   r ON r.session_id = t.session_id
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


@app.get("/session/{slug}/rtk-commands")
async def get_session_rtk_commands(slug: str, limit: int = 100):
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
        SELECT id, ts, rtk_cmd, input_tokens, output_tokens, saved_tokens,
               ROUND(savings_pct, 1) AS savings_pct, project_path
        FROM rtk_events
        WHERE session_id = ?
        ORDER BY id DESC
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
# UI templates (HTML/CSS/JS) — loaded from the templates/ directory
# ---------------------------------------------------------------------------

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def _load_template(name: str) -> str:
    """Read a UI template shipped alongside proxy.py (see templates/)."""
    return (TEMPLATES_DIR / name).read_text(encoding="utf-8")


DASHBOARD_HTML = _load_template("dashboard.html")
PLAY_HTML      = _load_template("play.html")
LIST_HTML      = _load_template("list.html")


if __name__ == "__main__":  # pragma: no cover
    uvicorn.run(app, host="127.0.0.1", port=9099)
