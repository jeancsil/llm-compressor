"""Compression cache, chunking, and the compress_text/system/messages pipeline.

Owns the process-wide, runtime-reassigned global `_cache` (a `CompressionCache`
instance set by `lifespan()` in proxy.py), plus the pure helpers/constants
around it (`_cache_key`, `CACHE_MEM_SIZE`, `CACHE_MAX_ROWS`, the chunking
helpers, and `compress_text` / `compress_system_field` / `compress_messages`).
Everywhere else in the codebase (including proxy.py, which re-exports these for
backward-compatible test patching via a forwarding shim) must read/write
`_cache` as `compression._cache` -- never `from compression import _cache` --
so that `lifespan`'s reassignment at startup and `monkeypatch.setattr` in
tests are visible to every reader. See the `_ProxyModule` shim + `_FORWARD`
table in proxy.py for the mechanism that keeps `monkeypatch.setattr(proxy,
"_cache", ...)` / `monkeypatch.setattr(proxy, "_compress_with", ...)` (the
pre-split test idiom) working without every test needing to be rewritten to
target `compression.` directly.

`compress_text` calls back into `proxy.record_compression` via a deferred,
call-time `import proxy` (avoids a circular top-level import, since proxy.py
imports this module) -- `record_compression` itself moves to sessions.py in
Task 13 Step 7, at which point this becomes `import sessions`.
"""

import hashlib
import json
import os
import re
import time
from collections import OrderedDict
from datetime import datetime, timezone

import backends

# Module-level global populated by lifespan() in proxy.py.
_cache = None  # CompressionCache, set in lifespan


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

    # record_compression still lives in proxy.py until Task 13 Step 7 moves it
    # to sessions.py; deferred import avoids a circular top-level import.
    import proxy as _proxy

    if _cache is not None:
        hit = _cache.get(key)
        if hit is not None:
            compressed, orig, comp = hit
            print(f"[cache] hit {orig} → {comp} tokens [{session_id[:8]}] role={role}")
            _proxy.record_compression(
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
    _proxy.record_compression(
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
