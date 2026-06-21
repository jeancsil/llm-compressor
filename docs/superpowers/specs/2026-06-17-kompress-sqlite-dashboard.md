# Spec: kompress-v2 model switching + SQLite metrics + dashboard improvements

**Date:** 2026-06-17  
**Repo:** `/Users/jeancsil/code/llm-compressor`  
**Single file to modify:** `llmlingua_proxy.py` (plus `pyproject.toml` for deps)

---

## Goal

Three independent improvements shipped together:

1. **Pluggable compressor** — add `chopratejas/kompress-v2-base` as an alternative to LLMLingua-2, selectable at startup via env var.
2. **Persistent SQLite metrics** — replace the JSON file with a SQLite DB so time-series data survives restarts.
3. **Richer dashboard** — add time-series chart, active model badge, per-compression latency, and estimated cost savings.

---

## 1. Pluggable compressor

### What changes

A new env var `COMPRESSOR_MODEL` (default: `llmlingua2`, alternative: `kompress`) controls which model is loaded at startup. The rest of the proxy is unaware of which model is active.

### LLMLingua-2 (existing, unchanged behaviour)

- Loaded via `llmlingua.PromptCompressor` exactly as today.
- Respects `COMPRESS_RATE` env var (float, default `0.5`).
- `force_tokens` stays as-is (`['\n', '?', '.', '!']`).

### kompress-v2-base (new)

- Package: `kompress` (PyPI, `pip install kompress`). Also add `transformers` if not already present.
- Model: `chopratejas/kompress-v2-base` — downloaded via HuggingFace on first run, cached normally.
- Respects `COMPRESS_THRESHOLD` env var (float, default `0.5`). Lower = keep more tokens. See table in model card for trade-offs.
- Device: same `mps` logic as LLMLingua-2 (fall back to `cpu` if MPS unavailable).
- Compression mechanics: tokenize input → run model → filter tokens where `final_scores >= threshold` → decode surviving tokens.

### Shared interface

Both models are wrapped behind a single internal function:

```
compress_text(text: str, session_id: str) -> (compressed_str, original_tokens, compressed_tokens, latency_ms)
```

`compress_text` decides which backend to call based on what was loaded at startup. The 200-char skip-short guard and error-fallback (return original on exception) remain unchanged.

### Startup log

```
[compressor] model=kompress-v2-base  threshold=0.50  device=mps
```
or
```
[compressor] model=llmlingua-2  rate=0.50  device=mps
```

### New dependencies (add to `pyproject.toml`)

- `kompress` (PyPI)
- `transformers` (if not already pulled in by llmlingua)

---

## 2. SQLite persistence

### Problem with current approach

`stats.json` stores aggregate counters only. There is no time-series — you cannot chart compression rate over the last 24 hours, and the in-memory `recent_compressions` deque (100 entries) resets on restart.

### New: `metrics.db`

A SQLite file created in the same directory as the script. One table:

```sql
CREATE TABLE IF NOT EXISTS compressions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT    NOT NULL,          -- ISO-8601 UTC
    session_id    TEXT    NOT NULL,
    model         TEXT    NOT NULL,          -- "llmlingua2" or "kompress"
    original_tokens   INTEGER NOT NULL,
    compressed_tokens INTEGER NOT NULL,
    latency_ms    REAL    NOT NULL
);
```

Every call to `compress_text` that succeeds writes one row. Failures (where original is returned) are not written.

### Migration from stats.json

Run once on startup, before anything else, if `stats.json` exists and `metrics.db` does not yet contain any rows:

1. **Backup** — copy `stats.json` to `stats.json.bak` in the same directory. Log the path.
2. **Import** — `stats.json` contains `recent_compressions` (up to 100 entries) with `ts`, `session_id`, `original`, `compressed`. Write each entry as a row in `compressions`, using `model=llmlingua2` (since all historic data predates kompress support) and `latency_ms=0.0` (not recorded historically).
3. **Verify** — after import, query `SELECT COUNT(*) FROM compressions` and assert it equals `len(recent_compressions)` from the JSON. Log the result:
   ```
   [migration] Imported 87 rows from stats.json → metrics.db. Backup at stats.json.bak.
   ```
4. If `stats.json` exists and `metrics.db` already has rows, skip migration entirely (idempotent).
5. If import fails for any reason, log the error and continue — the proxy must still start.

### Aggregation

- In-memory `stats` dict stays for fast reads (total counts, recent compressions deque).
- On startup, re-derive aggregate totals from `metrics.db` so they survive restarts. Replace the `stats.json` load with a SQLite query:
  ```sql
  SELECT COUNT(*), SUM(original_tokens), SUM(compressed_tokens) FROM compressions
  ```
- Session aggregates are also rebuilt from SQLite on startup.
- `stats.json` is no longer written or read after migration. Remove the `save_stats()` / `load_stats()` functions.

### New API endpoint: `/stats/timeseries`

Returns hourly buckets for the last 48 hours:

```json
[
  { "hour": "2026-06-17T10:00:00", "requests": 12, "avg_savings_pct": 34, "total_saved": 4210, "avg_latency_ms": 145 },
  ...
]
```

Query:
```sql
SELECT strftime('%Y-%m-%dT%H:00:00', ts) AS hour,
       COUNT(*) AS requests,
       ROUND(AVG((original_tokens - compressed_tokens) * 100.0 / original_tokens), 1) AS avg_savings_pct,
       SUM(original_tokens - compressed_tokens) AS total_saved,
       ROUND(AVG(latency_ms), 1) AS avg_latency_ms
FROM compressions
WHERE ts >= datetime('now', '-48 hours')
GROUP BY hour
ORDER BY hour ASC
```

---

## 3. Dashboard improvements

The dashboard is a single embedded HTML string in `llmlingua_proxy.py`. All changes are to that string. The dashboard auto-refreshes every 2 seconds via `fetch('/stats')` and `fetch('/stats/timeseries')` (new call added).

### 3a. Active model badge

In the header, next to "LLMLingua Proxy", show the active compressor:

- If kompress: `kompress-v2  threshold=0.50` in blue
- If llmlingua2: `LLMLingua-2  rate=0.50` in blue

This comes from a new field in `GET /stats`:
```json
{ "compressor": { "model": "kompress", "param_name": "threshold", "param_value": 0.5 } }
```

### 3b. Time-series chart

A new section below the sparkline. Shows the last 48 hours of compression rate as a bar chart, one bar per hour. Bars are colored by savings % using the same `barColor()` scale already in the dashboard. Empty hours get a faint placeholder bar. X-axis labels show time (e.g. "14:00", "15:00"). Data comes from `/stats/timeseries`.

### 3c. Latency

Add `latency_ms` to each entry in `recent_compressions`. Update the sparkline tooltip to include it:

```
14:32:01 · abc12345 · 34% saved (1.2k → 810) · 148ms
```

Also add a new metric card: **Avg Latency** showing the average `latency_ms` across the last 100 compressions (computed from the in-memory deque).

### 3d. Cost estimation

Add a metric card: **Est. $ Saved**

Formula: `saved_tokens / 1_000_000 * cost_per_mtok`

`cost_per_mtok` comes from `COST_PER_MTOK` env var (float, default `3.0` — matches claude-3-5-sonnet input pricing). Shown as `$X.XX`. A small grey label beneath the card reads "@ $3.00/MTok" so the user knows the assumption.

Also add `cost_per_mtok` to `GET /stats` response so the dashboard can render it dynamically.

---

## What does NOT change

- Proxy behaviour (`/v1/messages`, `/v1/models`) — identical to today.
- Session tracking via `x-claude-code-session-id` header — unchanged.
- rtk integration — unchanged.
- Two-layer dashboard hero (rtk present vs absent) — unchanged.
- MPS device selection — unchanged.
- The 200-char skip-short guard — unchanged.
- Error fallback in `compress_text` (return original on exception) — unchanged.

---

## File changes summary

| File | Change |
|------|--------|
| `llmlingua_proxy.py` | All logic changes: model loader, SQLite, new endpoints, dashboard HTML |
| `pyproject.toml` | Add `kompress`, add `transformers` if missing |

No new files needed. No new CLI commands. No Docker.

---

## Verification checklist

- [ ] `COMPRESSOR_MODEL=kompress uv run python llmlingua_proxy.py` starts without error, prints kompress startup log
- [ ] `COMPRESSOR_MODEL=llmlingua2 uv run python llmlingua_proxy.py` starts and behaves identically to today
- [ ] `GET /stats` includes `compressor` field and `cost_per_mtok`
- [ ] `GET /stats/timeseries` returns hourly buckets, empty array on fresh start
- [ ] `metrics.db` is created on first run; totals survive a restart
- [ ] On first run with existing `stats.json`: `stats.json.bak` is created, all `recent_compressions` entries are imported into `metrics.db`, row count matches, migration log line printed
- [ ] On second run: migration is skipped (idempotent), no duplicate rows
- [ ] If `stats.json` is absent: startup proceeds normally with empty DB
- [ ] Dashboard shows model badge in header
- [ ] Dashboard shows time-series section (empty bars on fresh start, fills in after compressions)
- [ ] Dashboard sparkline tooltips include latency
- [ ] Dashboard shows Avg Latency and Est. $ Saved cards
- [ ] `COST_PER_MTOK=15.0` changes the card value accordingly
