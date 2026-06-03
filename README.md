# llmlingua-proxy

FastAPI proxy that sits between Claude Code and the Anthropic API. Every outbound prompt is compressed by [LLMLingua-2](https://github.com/microsoft/LLMLingua) before being forwarded, reducing token usage without changing how you work.

**Does not use [rtk](https://github.com/rtk-ai/rtk).** The two tools operate at different layers and complement each other:

| Tool | Layer | What it compresses |
|---|---|---|
| **llmlingua-proxy** (this) | API | Conversation messages before they're billed |
| **rtk** | Shell | CLI command output before it enters the context window |

Running both gives you savings at both layers. The dashboard automatically detects rtk and switches to a two-layer view when it is present.

## Requirements

- Python 3.12
- [uv](https://github.com/astral-sh/uv) — `brew install uv`
- An Anthropic API key

## Install

```bash
git clone <this repo>
cd llm-lingua
uv sync
```

`uv sync` reads `pyproject.toml` and installs all dependencies including LLMLingua-2 into an isolated virtualenv. No `pip install` needed.

## Start the proxy

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uv run python llmlingua_proxy.py
```

First run downloads the LLMLingua-2 BERT model (~500 MB) and loads it onto the MPS device. Expect a 20–40 second cold start. Once you see:

```
Model ready.
INFO:     Uvicorn running on http://127.0.0.1:9099
```

the proxy is ready.

## Configure Claude Code

In a separate terminal (or add to `~/.zshrc`):

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:9099
claude
```

That's all. Claude Code now routes through the proxy transparently.

To stop compressing:

```bash
unset ANTHROPIC_BASE_URL
```

## Dashboard

While the proxy is running, open `http://127.0.0.1:9099/dashboard` in a browser. It auto-refreshes every 2 seconds and shows:

- Overall compression ratio (LLMLingua-2 tokenizer units)
- Per-session efficiency bars and ratio badges
- Sparkline of recent requests colored by savings percentage
- Full session table with request counts and last-seen times

> **Note on token counts:** the saved/compressed figures use LLMLingua-2's BERT-based tokenizer, not Claude's billing tokenizer. They are a good proxy for relative savings but do not map 1:1 to your Anthropic invoice.

### rtk integration

If [rtk](https://github.com/rtk-ai/rtk) is installed, the dashboard automatically reads its tracking database and switches to a two-layer view:

- **Shell layer** — rtk's total commands, tokens saved, and a top-commands breakdown
- **API layer** — LLMLingua-2's per-session stats (existing view)

No configuration required. The proxy reads rtk's SQLite database at the standard platform path in read-only mode:

| OS | Path |
|---|---|
| macOS | `~/Library/Application Support/rtk/history.db` |
| Linux | `~/.local/share/rtk/history.db` |
| Windows | `%APPDATA%\rtk\history.db` |

Install rtk:

```bash
brew install rtk   # macOS
```

## How it works

```
Claude Code → POST /v1/messages → proxy → Anthropic API
                     ↓
             LLMLingua-2 compresses
             each user message
             (skips messages ≤ 200 chars)
```

1. The proxy receives the full request body from Claude Code.
2. `compress_messages()` runs LLMLingua-2 on every `role=user` text block longer than 200 characters. System and assistant turns are forwarded as-is.
3. The compressed body is forwarded to `api.anthropic.com`. Streaming responses are piped through unchanged.
4. Stats are persisted to `stats.json` between restarts.

If LLMLingua-2 fails on a particular input (e.g. very short or malformed text), the original message is forwarded and the error is logged — requests never fail due to compression.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/messages` | Main proxy target; compresses then forwards |
| `GET` | `/v1/models` | Passthrough to Anthropic |
| `GET` | `/stats` | JSON compression statistics |
| `GET` | `/dashboard` | Live HTML dashboard |
| `GET` | `/` | Health check |

## Configuration

All configuration is via environment variables:

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | **Required.** Your Anthropic API key |
| `ANTHROPIC_BASE_URL` | — | Set to `http://127.0.0.1:9099` in the Claude Code terminal |

The compression rate (default `0.5`) and minimum text length (default `200` chars) are constants in `llmlingua_proxy.py` at the top of `compress_text()`.
