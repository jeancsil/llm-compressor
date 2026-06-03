> This file provides guidance to Claude Code when working with code in this repository.

## What this is

FastAPI proxy that intercepts Anthropic API calls, compresses prompts via LLMLingua-2 before forwarding. Single-file app: `llmlingua_proxy.py`.

## Run

```bash
uv run python llmlingua_proxy.py
# Starts on http://127.0.0.1:9099
```

## Key endpoints

- `POST /v1/messages` — proxy target; compresses user messages then forwards to Anthropic
- `GET /stats` — JSON compression stats
- `GET /dashboard` — live HTML dashboard (auto-refreshes)
- `GET /v1/models` — passthrough to Anthropic

## Architecture

- `PromptCompressor` loads at startup (MPS device, LLMLingua-2 multilingual model) — slow cold start expected
- `compress_messages()` only compresses `role=user` messages; skips text ≤200 chars
- In-memory stats only — resets on restart
- Session tracked via `x-claude-code-session-id` header

## Config

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# Point Claude Code at proxy:
export ANTHROPIC_BASE_URL=http://127.0.0.1:9099
```

## Dependencies

Managed via `uv`. Python 3.12 required.
