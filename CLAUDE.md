# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## What this is

FastAPI proxy that intercepts Anthropic API calls and compresses prompts via LLMLingua-2 (or kompress) before forwarding. Installable package at `src/llm_compressor/` (entry point `llm_compressor.cli:main`; `proxy.py` re-exports the FastAPI app and doubles as the `python -m llm_compressor.proxy` server entrypoint).

## Common tasks

```bash
make install    # Install dependencies via uv
make dashboard  # Open the live dashboard in the browser
make stats      # Print compression stats as JSON
make rtk-stats  # Print rtk shell-layer savings
```

### wrap CLI (primary way to run the proxy)

`cli.py` provides `llm-compressor wrap <cmd>` — starts the proxy as a background process, injects `ANTHROPIC_BASE_URL`, runs the command, then kills the proxy on exit. This is the default workflow; `make start`/`make stop` (foreground daemon) is only for running the proxy persistently across multiple terminals.

```bash
uv run llm-compressor wrap claude
uv run llm-compressor wrap aider
```

Entry point registered in `pyproject.toml`; install with `uv pip install -e .` then the binary lives at `.venv/bin/llm-compressor`.

### Persistent daemon (alternative)

```bash
make start      # Start proxy in foreground (Ctrl-C to stop)
make stop       # Stop the proxy
make restart    # Stop then start fresh
make check      # Verify proxy is reachable
```

### Langfuse observability (optional)

```bash
make install-langfuse   # Install langfuse SDK into the venv
make langfuse-status    # Check whether proxy has Langfuse enabled
make langfuse-test      # Send a test request and verify a trace appears
```

## Key endpoints

- `POST /v1/messages` — proxy target; compresses system field + user messages, forwards to Anthropic
- `GET /stats` — JSON compression stats
- `GET /dashboard` — live HTML dashboard (auto-refreshes)
- `POST /admin/set-model` — switch active compression model
- `GET /admin/langfuse-status` — Langfuse enabled/key status
- `GET /v1/models` — passthrough to Anthropic

## Architecture

- Backend loads at startup (MPS device when available, else CPU) — slow cold start expected
- `compress_system_field()` / `compress_messages()` / `compress_text()` (compression.py) are `async`; `compress_messages()` judges independent text blocks concurrently via `asyncio.gather`
- `compress_system_field()` compresses the top-level `system` field (CLAUDE.md, RTK.md, injected context)
- `compress_messages()` compresses `role=user` messages; skips text ≤200 chars; skips `assistant` turns
- Every unique compression is quality-gated: `compress_text()` awaits a Jev verdict (`quality.judge_now`) and forwards the ORIGINAL text instead of the compressed text on a Lost/Drifted verdict (`quality.gate_decision`); gated rows are recorded with realized tokens (`compressed_tokens == original_tokens`, `compressions.gated=1`)
- Stats persisted to `metrics.db` (SQLite); `compressions` table includes `role` column (`system` | `user`) and a `gated` column (`0`/`1`)
- Session tracked via `x-claude-code-session-id` header
- `langfuse_tracer.py` — optional fire-and-forget tracing module; no-ops when keys are absent

## Compression models

| Model | Notes |
|---|---|
| `llmlingua2` | Default; ~47% savings |
| `llmlingua2-large` | More aggressive; ~52% savings; 3× slower |
| `kompress` | Precision-oriented; ~27% savings; lower distortion |
| `dual` | Routes system→llmlingua2-large, user→kompress; loads both (~1.5 GB RAM) |

Switch via the dashboard dropdown (preferred) or directly:

```bash
curl -s -X POST http://127.0.0.1:9099/admin/set-model \
  -H 'Content-Type: application/json' -d '{"model": "dual"}'
```

## Environment flags

- `LLM_COMPRESSOR_LLM_NAMING=1` — opt-in, default off. When set, auto-naming sessions calls Haiku (`claude-haiku-4-5-20251001`) using the caller's forwarded OAuth bearer token (never a standalone `ANTHROPIC_API_KEY`), reusing the user's subscription out-of-band. Off by default; falls back to a deterministic heuristic slug.
- `QUALITY_GATE_TIMEOUT` — seconds the proxy waits inline for a Jev verdict
  before forwarding the compressed text anyway (default 5.0). Every unique
  compression is judged by Jev once; a verdict of Lost or Drifted forwards
  the ORIGINAL text instead of the compressed text. Jev disabled/error/timeout
  ⇒ compressed forwarded (savings-first). `QUALITY_SAMPLE_RATE` /
  `QUALITY_CACHE_SAMPLE_RATE` no longer govern judging.

## Config

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# Point Claude Code at proxy:
export ANTHROPIC_BASE_URL=http://127.0.0.1:9099
```

### Langfuse tracing (optional)

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_HOST=https://cloud.langfuse.com  # default; omit for cloud
```

The proxy enables tracing automatically when both keys are present at startup. Traces are sent fire-and-forget — errors are logged but never affect the proxy.

## Dependencies

Managed via `uv`. Python 3.12 required.

## Not committed to git

- `docs/superpowers/` — local plans/specs from superpowers skill runs
- `.superpowers/` — SDD progress ledger and task artifacts
- `metrics.db` — runtime database
