# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## What this is

FastAPI proxy that intercepts Anthropic API calls and compresses prompts via LLMLingua-2 (or kompress) before forwarding. Single-file app: `proxy.py`.

## Common tasks

```bash
make install    # Install dependencies via uv
make start      # Start proxy in foreground (Ctrl-C to stop)
make stop       # Stop the proxy
make restart    # Stop then start fresh
make check      # Verify proxy is reachable
make dashboard  # Open the live dashboard in the browser
make stats      # Print compression stats as JSON
make rtk-stats  # Print rtk shell-layer savings
```

## Key endpoints

- `POST /v1/messages` — proxy target; compresses system field + user messages, forwards to Anthropic
- `GET /stats` — JSON compression stats
- `GET /dashboard` — live HTML dashboard (auto-refreshes)
- `POST /admin/set-model` — switch active compression model
- `GET /v1/models` — passthrough to Anthropic

## Architecture

- Backend loads at startup (MPS device when available, else CPU) — slow cold start expected
- `compress_system_field()` compresses the top-level `system` field (CLAUDE.md, RTK.md, injected context)
- `compress_messages()` compresses `role=user` messages; skips text ≤200 chars; skips `assistant` turns
- Stats persisted to `metrics.db` (SQLite); `compressions` table includes `role` column (`system` | `user`)
- Session tracked via `x-claude-code-session-id` header

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

## Config

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# Point Claude Code at proxy:
export ANTHROPIC_BASE_URL=http://127.0.0.1:9099
```

## Dependencies

Managed via `uv`. Python 3.12 required.

## Not committed to git

- `docs/superpowers/` — local plans/specs from superpowers skill runs
- `.superpowers/` — SDD progress ledger and task artifacts
- `metrics.db` — runtime database
