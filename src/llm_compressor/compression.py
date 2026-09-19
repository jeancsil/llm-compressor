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

`compress_text` calls back into `sessions.record_compression` via a deferred,
call-time `import sessions` (avoids a circular top-level import, since
sessions.py imports this module's sibling `stats`/`backends`/`db`, and
proxy.py imports both compression.py and sessions.py at top level).
"""

import asyncio
import hashlib
import json
import os
import re
import time
from collections import OrderedDict
from datetime import datetime, timezone

from llm_compressor import backends

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
                "SELECT compressed_text, original_tokens, compressed_tokens, "
                "quality_score, quality_label, quality_flagged, quality_detail "
                "FROM compression_cache WHERE key = ?",
                (key,),
            ).fetchone()
            if row is not None:
                entry = (row[0], row[1], row[2], row[3], row[4], row[5], row[6])
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

    def update_quality(self, key: str, score, label, detail, flagged) -> None:
        """Patch a resident entry's quality fields in place after judging finishes.
        Only touches `_mem`; the disk row is already updated by quality.py's
        caller. Without this, an entry that's already in memory stays a stale
        (unjudged) tuple for the rest of the process's life -- get() never
        re-reads disk for a key it already has -- so every future hit on a hot
        key would keep calling Jev inline via compress_text()'s gate instead of
        taking the instant-gate branch. A no-op if the key isn't resident
        (already evicted, or never cached in-process); the disk row is still
        correct for the next cold load."""
        entry = self._mem.get(key)
        if entry is None:
            return
        compressed_text, original_tokens, compressed_tokens = entry[0], entry[1], entry[2]
        self._mem[key] = (
            compressed_text,
            original_tokens,
            compressed_tokens,
            score,
            label,
            flagged,
            detail,
        )

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


async def compress_text(text: str, session_id: str, role: str = "user") -> str:
    """Compress `text`, then gate compressed-vs-original on a Jev verdict.

    Every unique compression (one per cache key) is judged by Jev exactly
    once, inline, bounded by `quality.GATE_TIMEOUT`. A verdict of Lost or
    Drifted forwards the ORIGINAL text instead of the compressed text
    (`quality.gate_decision`); Jev disabled/error/timeout/no-verdict all fall
    back to forwarding the compressed text (savings-first -- the gate must
    never add unbounded latency or block the proxy path). Gated rows are
    recorded with realized tokens (`compressed == original`) via
    `sessions.record_compression(..., gated=1)`.
    """
    if len(text) <= 200:
        return text
    active = backends._pick_backend(role)
    if active is None:
        return text
    model_tag = active.get("type", "compressor")
    rate = active.get("rate", 0.5)
    key = _cache_key(text, model_tag, rate)

    # Deferred import: sessions.py itself imports compression's sibling
    # modules (db/backends/stats), and proxy.py imports both compression.py
    # and sessions.py at top level, so this could likely be promoted to a
    # top-level `import sessions` without an actual cycle -- kept deferred to
    # match the established local-import convention for sessions.py used
    # elsewhere in this codebase (proxy.py's proxy_messages/session_dashboard
    # routes) and to avoid re-litigating import-order safety here.
    from llm_compressor import quality, sessions

    async def _verdict_for(compressed: str, cache_key: str) -> dict | None:
        """Await Jev once, bounded by GATE_TIMEOUT. None on disabled/error/timeout."""
        if not quality.judge.enabled:
            return None
        try:
            return await asyncio.wait_for(
                quality.judge.judge_now(None, text, compressed, cache_key=cache_key),
                timeout=quality.GATE_TIMEOUT,
            )
        except Exception as exc:
            print(f"[gate] verdict unavailable, forwarding compressed: {exc}")
            return None

    async def _finish(compressed, orig, comp, cache_hit, verdict, latency_ms=0.0) -> str:
        """Apply the gate decision, record the row, and return what to forward."""
        raw = verdict.get("score") if verdict else None
        gated = 1 if quality.gate_decision(raw) == "block" else 0
        forwarded = text if gated else compressed
        compression_id = sessions.record_compression(
            session_id,
            orig,
            comp,
            latency_ms=latency_ms,
            original_text=text,
            compressed_text=compressed,
            role=role,
            active_backend=active,
            cache_hit=cache_hit,
            gated=gated,
        )
        if verdict is not None and compression_id is not None:
            # Backfill compression_quality for THIS row so the per-row quality
            # chip (LEFT JOIN on compression_id -- routes.py) reflects the
            # verdict the gate already obtained, whether fresh from Jev
            # (judge_now was called with compression_id=None, so it couldn't
            # write this row itself) or backfilled from a warm cache hit's
            # already-stored verdict (the instant-gate branch below).
            _record_gate_verdict(compression_id, verdict)
        return forwarded

    if _cache is not None:
        hit = _cache.get(key)
        if hit is not None:
            # `hit` is 3-tuple wide for an entry that was `put()` this run (no
            # quality data yet) and 7-tuple wide once promoted from disk, where
            # the cache table's quality_* columns are carried along -- see
            # CompressionCache.get(). Unpack the quality score defensively
            # rather than assuming every hit has all seven elements.
            compressed, orig, comp = hit[0], hit[1], hit[2]
            q_score = hit[3] if len(hit) > 3 else None
            q_label = hit[4] if len(hit) > 4 else None
            q_flagged = hit[5] if len(hit) > 5 else None
            q_detail = hit[6] if len(hit) > 6 else None
            print(f"[cache] hit {orig} → {comp} tokens [{session_id[:8]}] role={role}")
            if q_score is not None:
                # Warm hit WITH a stored verdict: gate instantly, no Jev call.
                # Carry the cache row's label/detail/flagged along too, not
                # just the score, so the backfilled compression_quality row
                # (see _finish) gets the real verdict instead of defaults.
                synthetic_verdict = {
                    "score": q_score,
                    "label": q_label,
                    "detail": q_detail,
                    "flagged": q_flagged,
                }
                return await _finish(compressed, orig, comp, 1, synthetic_verdict)
            # Warm hit WITHOUT a verdict (pre-gate row / never judged): judge
            # once now, then gate. The verdict lands in compression_cache
            # (see judge_now/_judge) so the *next* hit takes the branch above.
            verdict = await _verdict_for(compressed, key)
            return await _finish(compressed, orig, comp, 1, verdict)

    t0 = time.perf_counter()
    try:
        compressed, orig, comp = _compress_with(active, text)
    except Exception as e:
        print(f"[compressor] compression failed, forwarding original: {e}")
        # Record the failure instead of returning silently. The prompt still
        # goes through uncompressed -- that is the right call for the request --
        # but a run where the compressor throws on every call used to look
        # identical to a quiet one: no rows, no savings, no error anywhere in
        # the UI. `ok=0` with zero savings makes the passthrough countable.
        latency_ms = (time.perf_counter() - t0) * 1000
        tokens = _count_tokens(text)
        sessions.record_compression(
            session_id,
            tokens,
            tokens,
            latency_ms=latency_ms,
            role=role,
            active_backend=active,
            cache_hit=0,
            ok=0,
        )
        return text
    latency_ms = (time.perf_counter() - t0) * 1000
    print(f"[{model_tag}] {orig} → {comp} tokens [{session_id[:8]}] role={role}")
    if _cache is not None:
        # Put the fresh entry into the cache BEFORE judging: judge_now's
        # `UPDATE compression_cache ... WHERE key=?` (in quality.py's _judge)
        # needs the row to already exist, else the verdict silently fails to
        # persist and every future hit on this key re-judges from scratch.
        _cache.put(key, compressed, orig, comp, model_tag, rate)
    verdict = await _verdict_for(compressed, key)
    return await _finish(compressed, orig, comp, 0, verdict, latency_ms=latency_ms)


def _record_gate_verdict(compression_id: int, verdict: dict) -> None:
    """Best-effort: persist `verdict` to `compression_quality` for the row
    `compress_text()`'s gate just recorded, keyed on its real compression_id.

    `judge_now(None, ...)` (the gate's call, in `_verdict_for` above) cannot
    write this row itself -- the compression row doesn't exist yet when it's
    called. This is the other half: once `sessions.record_compression()` has
    returned a real id, back-fill `compression_quality` so the per-row
    quality chip (`routes.py`'s `LEFT JOIN compression_quality ON
    compression_id`) reflects the verdict the gate already obtained, whether
    freshly judged or carried over from a warm cache hit's stored verdict.
    Mirrors quality.py's own INSERT; never raises into the proxy path.
    """
    try:
        from llm_compressor import db

        if db._db_conn is None:
            return
        db._db_conn.execute(
            "INSERT OR REPLACE INTO compression_quality "
            "(compression_id, score, label, detail, confidence, flagged, judged_at, meta) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                compression_id,
                verdict.get("score"),
                verdict.get("label"),
                verdict.get("detail"),
                verdict.get("confidence"),
                verdict.get("flagged") or 0,
                db.utc_now(),
                json.dumps(verdict.get("meta") or {}),
            ),
        )
        db._db_conn.commit()
    except Exception as exc:
        print(f"[quality] failed to persist gate verdict: {exc}")


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


async def compress_system_field(system_val, session_id: str):
    """Compress the top-level Anthropic API `system` field (string or content-block list)."""
    if isinstance(system_val, str):
        return await compress_text(system_val, session_id, role="system")
    if isinstance(system_val, list):
        text_idx = [
            i for i, b in enumerate(system_val) if isinstance(b, dict) and b.get("type") == "text"
        ]
        compressed = await asyncio.gather(
            *[compress_text(system_val[i]["text"], session_id, role="system") for i in text_idx]
        )
        new_blocks = list(system_val)
        for i, c in zip(text_idx, compressed):
            new_blocks[i] = {**system_val[i], "text": c}
        return new_blocks
    return system_val


async def compress_messages(messages: list, session_id: str) -> list:
    """Compress every `role=user` message's text content; judges independent
    text blocks concurrently (`asyncio.gather`) since each block's Jev call is
    independent of the others."""
    out = []
    for msg in messages:
        if msg.get("role") != "user":
            out.append(msg)
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            out.append({**msg, "content": await compress_text(content, session_id, role="user")})
        elif isinstance(content, list):
            text_idx = [
                i
                for i, b in enumerate(content)
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            compressed = await asyncio.gather(
                *[compress_text(content[i]["text"], session_id, role="user") for i in text_idx]
            )
            new_blocks = list(content)
            for i, c in zip(text_idx, compressed):
                new_blocks[i] = {**content[i], "text": c}
            out.append({**msg, "content": new_blocks})
        else:
            out.append(msg)
    return out
