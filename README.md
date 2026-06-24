# LLM-Compressor

<img src="assets/logo.svg" width="64" alt="LLM-Compressor logo">

> A local proxy that compresses every Claude Code prompt before it's billed — **no workflow changes, one env var.**

![savings summary](assets/savings-hero.svg)

---

If you use Claude Code daily, every request resends the full conversation history plus your entire `CLAUDE.md`. Those tokens add up fast. LLM-Compressor sits transparently between Claude Code and the Anthropic API and shrinks each payload with a local compression model before forwarding it. Claude never notices. Your invoice does.

<a href="https://github.com/jeancsil/llm-compressor">⭐ Star on GitHub</a> &nbsp;·&nbsp; <a href="#install"><strong>Install in 3 steps ↓</strong></a>

---

## By the numbers

The chart below is generated from real usage data — `metrics.db` logged by this proxy across ~5 weeks of daily Claude Code sessions.

![daily savings timeline](assets/savings-timeline.svg)

**For API key users** this is direct invoice reduction at Sonnet 4.6 input rates ($3/MTok). **For Pro subscribers** (flat €18–$20/month) it translates to roughly 35% more Claude Code turns per 5-hour usage window before hitting limits — and directly reduces cost if you buy extra usage credits.

> Token counts use the compression model's tokenizer, not Claude's billing tokenizer — a good proxy for relative savings, not a 1:1 invoice mapping. System-field tokens (CLAUDE.md etc.) are prompt-cached by Anthropic after turn 1 and billed at $0.30/MTok, not $3.00/MTok — so their monetary value is lower than raw token counts suggest. User-message tokens dominate (97% of savings) and are not cached.

---

## How it works

LLM-Compressor stacks with [rtk](https://github.com/rtk-ai/rtk) to save tokens at two independent layers:

![two-layer architecture](assets/two-layer.svg)

**Does not conflict with rtk.** The two tools operate at different layers:

| Tool | Layer | What it compresses |
|---|---|---|
| **rtk** | Shell | CLI command output before it enters the context window |
| **LLM-Compressor** (this) | API | Conversation messages before they're billed |

Running both compounds the savings. The dashboard automatically detects rtk and switches to a two-layer view when it is present.

---

## Install

**Requirements:** Python 3.12 · [uv](https://github.com/astral-sh/uv) (`brew install uv`) · Anthropic API key

```bash
git clone https://github.com/jeancsil/llm-compressor
cd llm-compressor
uv sync
```

`uv sync` installs all dependencies into an isolated virtualenv. No `pip install` needed.

## Start the proxy

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uv run python proxy.py
```

The first run downloads the compression model and loads it. Cold start is 20–90 seconds depending on the model. Once you see:

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

---

## Dashboard

While the proxy is running, open `http://127.0.0.1:9099/dashboard` in a browser. It auto-refreshes every 2 seconds and shows:

- Overall compression ratio
- Per-session efficiency bars and ratio badges
- Sparkline of recent requests colored by savings percentage
- Full session table with request counts and last-seen times

### rtk integration

If [rtk](https://github.com/rtk-ai/rtk) is installed, the dashboard automatically reads its tracking database and switches to a two-layer view:

- **Shell layer** — rtk's total commands, tokens saved, and a top-commands breakdown
- **API layer** — per-session compression stats (existing view)

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

---

## Compression models

| Model ID | Underlying model | Download | Notes |
|---|---|---|---|
| `llmlingua2` | [microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank](https://huggingface.co/microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank) | **677 MB** | Default; ~47% savings |
| `llmlingua2-large` | [microsoft/llmlingua-2-xlm-roberta-large-meetingbank](https://huggingface.co/microsoft/llmlingua-2-xlm-roberta-large-meetingbank) | **2.1 GB** | More aggressive; ~52% savings; 3× slower |
| `kompress` | [chopratejas/kompress-v2-base](https://huggingface.co/chopratejas/kompress-v2-base) | **301 MB** | Precision-oriented; ~27% savings; lower distortion |
| `dual` | llmlingua2-large + kompress (both loaded) | **~2.4 GB** | Routes system→llmlingua2-large, user→kompress |

Models are downloaded from HuggingFace on first use and cached in `~/.cache/huggingface/hub`.

Switch via dashboard dropdown or API:

```bash
curl -s -X POST http://127.0.0.1:9099/admin/set-model \
  -H 'Content-Type: application/json' -d '{"model": "kompress"}'
```

---

## Compression modes

Every Anthropic API call is **stateless**: the client resends the full conversation on each request. The `system` field (CLAUDE.md, RTK.md, injected context) and all previous `user` turns are retransmitted every time — compressing saves tokens on every call, not just the first.

### What gets compressed

| Part | Compressed? | Reason |
|---|---|---|
| `system` field | **Yes** | Heaviest payload; pure boilerplate sent on every call |
| `user` messages | **Yes** | User intent; compression applied with care |
| `assistant` messages | **No** | Model reads its own prior reasoning; compressing them causes self-confusion |

### Single-model mode (default)

One compression model handles everything. Select from the dashboard or via API:

```bash
curl -s -X POST http://127.0.0.1:9099/admin/set-model \
  -H 'Content-Type: application/json' \
  -d '{"model": "llmlingua2-large"}'
```

Available models: `llmlingua2`, `llmlingua2-large`, `kompress`, `dual`

### Dual mode

Select **"dual (system→large · user→kompress)"** from the dashboard or via API:

```bash
curl -s -X POST http://127.0.0.1:9099/admin/set-model \
  -H 'Content-Type: application/json' \
  -d '{"model": "dual"}'
```

Dual mode loads both models simultaneously (~2.4 GB RAM). Cold start takes 60–90 seconds.

### Auditing compressions

Compression is logged to `metrics.db` with the `role` column (`system` or `user`). Query directly:

```bash
sqlite3 metrics.db \
  "SELECT role, model, COUNT(*), ROUND(AVG((1.0 - compressed_tokens*1.0/original_tokens)*100),1) AS avg_savings_pct FROM compressions GROUP BY role, model"
```

The dashboard recent-activity table shows each row's role with a color-coded badge (blue = system, green = user).

---

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

The compression rate (default `0.5`) and minimum text length (default `200` chars) are constants in `proxy.py` at the top of `compress_text()`.
