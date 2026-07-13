import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import copy
import hashlib
import json
import math
import platform
import re
import sqlite3
import sys as _sys
import threading
import time
import types as _types
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

import backends
import db

# Plain re-exports: real function objects, never monkeypatched by name in the
# test suite (tests call them directly), so a static import carries no
# staleness risk. `_db_conn`/`DB_PATH`/`migrate_from_json`/
# `recover_stats_from_backup`/`_migrate_db_location` are deliberately NOT
# imported here — see `_ProxyModule` below.
from db import init_db, load_stats_from_db  # re-export
from langfuse_tracer import tracer as _lf_tracer

# ---------------------------------------------------------------------------
# proxy.py forwarding shim (Task 13, Steps 1 & 4)
# ---------------------------------------------------------------------------
# db.py owns `_db_conn` (mutable, reassigned by lifespan() on every request-cycle
# reset in tests) and `DB_PATH`, plus `migrate_from_json` / `recover_stats_from_backup`
# / `_migrate_db_location`. backends.py owns the backend-selection globals
# (`backend`, `backend_loading`, `backend_user`, `backend_system`, `dual_mode`,
# `dual_model_system`, `dual_model_user`) and the loader functions (`load_backend`,
# `_load_dual_backend`, `_load_kompress_backend`, `_load_llmlingua2_backend`,
# `_load_single_backend`, `_pick_backend`). All of these are monkeypatched away
# via `monkeypatch.setattr(proxy, "<name>", ...)` somewhere in the test suite
# (conftest.py's autouse fixture, or individual tests in test_proxy.py /
# test_coverage.py / test_cache.py). Every real reader of these names elsewhere
# in the codebase (lifespan(), the /play and /admin route handlers, db.py's and
# backends.py's own functions) is module-qualified (`db._db_conn`,
# `backends.backend`, ...) per scratchpad/split-audit.md's Step 3 contract. This
# class makes `proxy.<name>` reads AND `monkeypatch.setattr(proxy, "<name>",
# ...)` writes forward to the single owning attribute on `db` / `backends`, so
# the pre-split test idiom keeps working unmodified instead of silently
# patching a stale, disconnected copy on proxy itself.
#
# `_load_backend` is a special case: pre-split, proxy.py had a plain alias
# `_load_backend = load_backend`. That alias is not recreated as a real name in
# backends.py — it only exists here as a forwarding key pointing at
# `backends.load_backend`, so `monkeypatch.setattr(proxy, "_load_backend", ...)`
# still works even though nothing in backends.py is ever bound to that name.
_FORWARD = {
    "_db_conn": (db, "_db_conn"),
    "DB_PATH": (db, "DB_PATH"),
    "migrate_from_json": (db, "migrate_from_json"),
    "recover_stats_from_backup": (db, "recover_stats_from_backup"),
    "_migrate_db_location": (db, "_migrate_db_location"),
    "backend": (backends, "backend"),
    "backend_loading": (backends, "backend_loading"),
    "backend_user": (backends, "backend_user"),
    "backend_system": (backends, "backend_system"),
    "dual_mode": (backends, "dual_mode"),
    "dual_model_system": (backends, "dual_model_system"),
    "dual_model_user": (backends, "dual_model_user"),
    "_load_backend": (backends, "load_backend"),
    "load_backend": (backends, "load_backend"),
    "_load_dual_backend": (backends, "_load_dual_backend"),
    "_load_kompress_backend": (backends, "_load_kompress_backend"),
    "_load_llmlingua2_backend": (backends, "_load_llmlingua2_backend"),
    "_load_single_backend": (backends, "_load_single_backend"),
    "_pick_backend": (backends, "_pick_backend"),
}


class _ProxyModule(_types.ModuleType):
    def __getattr__(self, name):
        target = _FORWARD.get(name)
        if target is not None:
            module, attr = target
            return getattr(module, attr)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        target = _FORWARD.get(name)
        if target is not None:
            module, attr = target
            setattr(module, attr, value)
            return
        super().__setattr__(name, value)


_sys.modules[__name__].__class__ = _ProxyModule

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE = "https://api.anthropic.com"
COST_PER_MTOK = float(os.environ.get("COST_PER_MTOK", "3.0"))


# Module-level globals populated by lifespan.
# NOTE: backend/backend_loading/backend_user/backend_system/dual_mode/
# dual_model_system/dual_model_user/KNOWN_MODELS/DUAL_SUBMODELS moved to
# backends.py (Task 13, Step 4) — see the _FORWARD shim above for how
# `proxy.<name>` reads/writes still reach them.
_cache = None  # CompressionCache, set in lifespan


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_key(text: str, model_tag: str, rate: float) -> str:
    """Exact-match cache key. Model and rate are included because the same text
    yields different output under a different backend/rate."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{digest}|{model_tag}|{rate}"


CACHE_MEM_SIZE = int(os.environ.get("LLM_COMPRESSOR_CACHE_SIZE", "2000"))
CACHE_MAX_ROWS = int(os.environ.get("LLM_COMPRESSOR_CACHE_MAX_ROWS", "50000"))


class CompressionCache:
    """Hybrid exact-match cache: bounded in-memory LRU over a SQLite backing
    table. Lookup order is memory -> SQLite -> miss. No seed-on-boot; disk hits
    are promoted into memory lazily so RAM stays bounded regardless of table size."""

    def __init__(self, conn, max_mem: int = 2000, max_rows: int = 50000):
        self._conn = conn
        self._max_mem = max_mem
        self._max_rows = max_rows
        self._mem: "OrderedDict[str, tuple[str, int, int]]" = OrderedDict()

    def get(self, key: str):
        if key in self._mem:
            self._mem.move_to_end(key)
            self._touch(key)
            return self._mem[key]
        if self._conn is not None:
            row = self._conn.execute(
                "SELECT compressed_text, original_tokens, compressed_tokens "
                "FROM compression_cache WHERE key = ?",
                (key,),
            ).fetchone()
            if row is not None:
                entry = (row[0], row[1], row[2])
                self._mem_put(key, entry)
                self._touch(key)
                return entry
        return None

    def put(self, key, compressed_text, original_tokens, compressed_tokens, model, rate):
        self._mem_put(key, (compressed_text, original_tokens, compressed_tokens))
        if self._conn is not None:
            # microsecond precision so last_hit ordering (LRU disk eviction) is
            # unambiguous even when many entries land in the same second.
            ts = datetime.now(timezone.utc).isoformat()
            self._conn.execute(
                "INSERT OR REPLACE INTO compression_cache "
                "(key, model, rate, compressed_text, original_tokens, compressed_tokens, "
                " created_at, hit_count, last_hit) VALUES (?,?,?,?,?,?,?,0,?)",
                (key, model, rate, compressed_text, original_tokens, compressed_tokens, ts, ts),
            )
            self._evict_disk()
            self._conn.commit()

    def _mem_put(self, key, entry):
        self._mem[key] = entry
        self._mem.move_to_end(key)
        while len(self._mem) > self._max_mem:
            self._mem.popitem(last=False)

    def _touch(self, key):
        if self._conn is None:
            return
        ts = datetime.now(timezone.utc).isoformat()  # microsecond precision (see put)
        self._conn.execute(
            "UPDATE compression_cache SET hit_count = hit_count + 1, last_hit = ? WHERE key = ?",
            (ts, key),
        )
        self._conn.commit()

    def _evict_disk(self):
        if self._max_rows <= 0:
            return
        count = self._conn.execute("SELECT COUNT(*) FROM compression_cache").fetchone()[0]
        if count > self._max_rows:
            self._conn.execute(
                "DELETE FROM compression_cache WHERE key IN "
                "(SELECT key FROM compression_cache ORDER BY last_hit ASC LIMIT ?)",
                (count - self._max_rows,),
            )


# NOTE: LLMLINGUA2_MODELS, _load_llmlingua2_backend, _load_kompress_backend,
# _load_single_backend, _load_dual_backend, load_backend, and _pick_backend
# moved to backends.py (Task 13, Step 4) — see the _FORWARD shim above for how
# `proxy.<name>` reads/writes/calls still reach them.


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cache
    db._migrate_db_location()
    db._db_conn = db.init_db(str(db.DB_PATH))
    _cache = CompressionCache(db._db_conn, max_mem=CACHE_MEM_SIZE, max_rows=CACHE_MAX_ROWS)
    db.migrate_from_json(db._db_conn)
    db.recover_stats_from_backup(db._db_conn)
    db.load_stats_from_db(db._db_conn)
    backends.backend = backends.load_backend()
    _lf_tracer.init()
    # print handled inside tracer.init()
    yield
    # Flush langfuse traces before shutdown
    if _lf_tracer.enabled and _lf_tracer._client:
        try:
            _lf_tracer._client.flush()
        except Exception:
            pass
    # Release model references before process exit to avoid MPS semaphore leaks
    backends.backend = None
    import gc

    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.synchronize()
            torch.mps.empty_cache()
    except Exception:
        pass
    if db._db_conn:
        db._db_conn.close()


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
    cache_hit: int = 0,
):
    stats["total_original_tokens"] += original
    stats["total_compressed_tokens"] += compressed

    active = active_backend if active_backend is not None else backends.backend
    model_name = active.get("type", "llmlingua2") if active else "llmlingua2"
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if db._db_conn:
        cur = db._db_conn.execute(
            "INSERT INTO compressions (ts, session_id, model, original_tokens, compressed_tokens, latency_ms, role, cache_hit) VALUES (?,?,?,?,?,?,?,?)",
            (ts, session_id, model_name, original, compressed, latency_ms, role, cache_hit),
        )
        if original_text is not None and compressed_text is not None:
            db._db_conn.execute(
                "INSERT INTO compression_texts (compression_id, original_text, compressed_text) VALUES (?,?,?)",
                (cur.lastrowid, original_text, compressed_text),
            )
        db._db_conn.commit()

    sess = stats["sessions"].setdefault(
        session_id,
        {
            "first_seen": ts,
            "requests": 0,
            "original_tokens": 0,
            "compressed_tokens": 0,
        },
    )
    sess["original_tokens"] += original
    sess["compressed_tokens"] += compressed
    sess["last_seen"] = ts

    stats["recent_compressions"].appendleft(
        {
            "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            "session_id": session_id[:8],
            "original": original,
            "compressed": compressed,
            "saved": original - compressed,
            "latency_ms": round(latency_ms, 1),
        }
    )


def record_request(session_id: str):
    stats["total_requests"] += 1
    sess = stats["sessions"].setdefault(
        session_id,
        {
            "first_seen": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "requests": 0,
            "original_tokens": 0,
            "compressed_tokens": 0,
            "name": None,
        },
    )
    sess["requests"] += 1
    sess["last_seen"] = datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# rtk integration (optional — gracefully absent when rtk not installed)
# ---------------------------------------------------------------------------


def _rtk_db_path() -> Path:
    return db._rtk_data_dir() / "history.db"


def read_rtk_stats(since: str | None = None) -> dict | None:
    db = _rtk_db_path()
    if not db.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
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
# Chunking helpers (prevent BERT 512-token overflow in LLMLingua-2)
# ---------------------------------------------------------------------------

CHUNK_MAX_TOKENS = 400
_CHUNK_MAX_CHARS = 1400  # ~400 BERT tokens for mixed code/prose


def _count_tokens(text: str) -> int:
    """Count tokens using the backend tokenizer (falls back to whitespace split)."""
    active = backends.backend or backends.backend_user or backends.backend_system
    try:
        return len(active["compressor"].tokenizer.tokenize(text))
    except Exception:
        return len(text.split())


def _char_split(text: str) -> list[str]:
    """Split text into _CHUNK_MAX_CHARS-sized pieces (last resort for code/dense text)."""
    if len(text) <= _CHUNK_MAX_CHARS:
        return [text]
    return [text[i : i + _CHUNK_MAX_CHARS] for i in range(0, len(text), _CHUNK_MAX_CHARS)]


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
    active = backends._pick_backend(role)
    if active is None:
        return text
    model_tag = active.get("type", "compressor")
    rate = active.get("rate", 0.5)
    key = _cache_key(text, model_tag, rate)

    if _cache is not None:
        hit = _cache.get(key)
        if hit is not None:
            compressed, orig, comp = hit
            print(f"[cache] hit {orig} → {comp} tokens [{session_id[:8]}] role={role}")
            record_compression(
                session_id,
                orig,
                comp,
                latency_ms=0.0,
                original_text=text,
                compressed_text=compressed,
                role=role,
                active_backend=active,
                cache_hit=1,
            )
            return compressed

    t0 = time.perf_counter()
    try:
        compressed, orig, comp = _compress_with(active, text)
    except Exception as e:
        print(f"[compressor] compression failed, forwarding original: {e}")
        return text
    latency_ms = (time.perf_counter() - t0) * 1000
    print(f"[{model_tag}] {orig} → {comp} tokens [{session_id[:8]}] role={role}")
    record_compression(
        session_id,
        orig,
        comp,
        latency_ms,
        text,
        compressed,
        role=role,
        active_backend=active,
        cache_hit=0,
    )
    if _cache is not None:
        _cache.put(key, compressed, orig, comp, model_tag, rate)
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
    if backends.backend is None:
        raise RuntimeError("No backend loaded")
    return _compress_with(backends.backend, text)


def _extract_text_from_sse(raw: bytes) -> str:
    """Pull text_delta content from Anthropic SSE bytes. Best-effort; returns '' on failure."""
    parts = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        try:
            obj = json.loads(payload)
            if obj.get("type") == "content_block_delta":
                delta = obj.get("delta", {})
                if delta.get("type") == "text_delta":
                    parts.append(delta.get("text", ""))
        except Exception:
            pass
    return "".join(parts)


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
            if isinstance(b, dict) and b.get("type") == "text"
            else b
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
                    new_blocks.append(
                        {**block, "text": compress_text(block["text"], session_id, role="user")}
                    )
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


def _cache_stats() -> dict:
    """Cache-hit summary from compressions.cache_hit, windowed to avoid dilution.

    `since_deploy` counts only rows recorded after caching went live (the
    `cache_since` meta marker), so the pre-feature backlog of misses cannot
    permanently depress the ratio. `last_24h` is a rolling window that reflects
    current behaviour.
    """
    if db._db_conn is None:
        return _empty_cache_stats()

    def _window(cutoff) -> dict:
        if cutoff is None:
            return {"hits": 0, "total": 0, "hit_ratio": 0.0}
        row = db._db_conn.execute(
            "SELECT COALESCE(SUM(cache_hit), 0), COUNT(*) FROM compressions WHERE ts >= ?",
            (cutoff,),
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
            """SELECT role,
                      COALESCE(SUM(cache_hit), 0),
                      COUNT(*),
                      AVG(CASE WHEN cache_hit = 0 THEN latency_ms END)
               FROM compressions WHERE ts >= ? GROUP BY role""",
            (since,),
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

    compressor_info = _compressor_info()

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

        scope, scope_args = _stats_scope(active_model, session_id)
        today_stats = _aggregate_stats(scope, scope_args, today=True, with_ratio=False)
        alltime_stats = _aggregate_stats(scope, scope_args, today=False, with_ratio=True)
        recent_rows = _recent_compression_rows(active_model, session_id)
        rtk_stats = _rtk_stats(session_id)
        tracked_stats = _tracked_stats()
        _merge_rtk_into_sessions(sessions_out)

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
        compressed, _orig_tok, _comp_tok = compress_backend(text)
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
    return JSONResponse(content=_lf_tracer.status())


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
    import math

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
        body["system"] = compress_system_field(body["system"], session_id)
    body["messages"] = compress_messages(body["messages"], session_id)
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
            if _lf_tracer.enabled:
                _end_ms = time.monotonic() * 1000
                _response_text = _extract_text_from_sse(b"".join(_chunks))
                await _lf_tracer.log_request(
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
                    await _lf_tracer.add_to_dataset(
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
        if _lf_tracer.enabled:
            _response_text = ""
            try:
                _response_text = (resp_data.get("content") or [{}])[0].get("text", "")
            except Exception:
                pass
            await _lf_tracer.log_request(
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
                await _lf_tracer.add_to_dataset(
                    run_inputs={
                        "original_messages": _ls_original_messages,
                        "original_system": _ls_original_system,
                    },
                    run_outputs={"response": _response_text},
                )
        return JSONResponse(content=resp_data, status_code=resp.status_code)


# ---------------------------------------------------------------------------
# UI templates (HTML/CSS/JS) — loaded from the templates/ directory
# ---------------------------------------------------------------------------

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def _load_template(name: str) -> str:
    """Read a UI template shipped alongside proxy.py (see templates/)."""
    return (TEMPLATES_DIR / name).read_text(encoding="utf-8")


DASHBOARD_HTML = _load_template("dashboard.html")
PLAY_HTML = _load_template("play.html")
LIST_HTML = _load_template("list.html")


if __name__ == "__main__":  # pragma: no cover
    uvicorn.run(app, host="127.0.0.1", port=9099)
