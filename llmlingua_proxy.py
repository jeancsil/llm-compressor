import os
import json
import httpx
import uvicorn
from collections import deque
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from llmlingua import PromptCompressor

app = FastAPI()

print("Loading LLMLingua-2 model...")
compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    use_llmlingua2=True,
    device_map="mps",
)
print("Model ready.")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE    = "https://api.anthropic.com"
STATS_FILE        = "stats.json"

# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

stats = {
    "started_at": datetime.now().isoformat(),
    "total_requests": 0,
    "total_original_tokens": 0,
    "total_compressed_tokens": 0,
    "sessions": {},
    "recent_compressions": deque(maxlen=100),
}

def load_stats():
    if not os.path.exists(STATS_FILE):
        return
    try:
        with open(STATS_FILE) as f:
            data = json.load(f)
        stats["total_requests"]          = data.get("total_requests", 0)
        stats["total_original_tokens"]   = data.get("total_original_tokens", 0)
        stats["total_compressed_tokens"] = data.get("total_compressed_tokens", 0)
        stats["sessions"]                = data.get("sessions", {})
        stats["recent_compressions"]     = deque(data.get("recent_compressions", []), maxlen=100)
        print(f"[stats] Loaded from {STATS_FILE}")
    except Exception as e:
        print(f"[stats] Could not load {STATS_FILE}: {e}")

def save_stats():
    try:
        with open(STATS_FILE, "w") as f:
            json.dump({
                "total_requests":          stats["total_requests"],
                "total_original_tokens":   stats["total_original_tokens"],
                "total_compressed_tokens": stats["total_compressed_tokens"],
                "sessions":                stats["sessions"],
                "recent_compressions":     list(stats["recent_compressions"]),
            }, f)
    except Exception as e:
        print(f"[stats] Could not save: {e}")

load_stats()

def record_compression(session_id: str, original: int, compressed: int):
    stats["total_original_tokens"] += original
    stats["total_compressed_tokens"] += compressed

    sess = stats["sessions"].setdefault(session_id, {
        "first_seen": datetime.now().isoformat(),
        "requests": 0,
        "original_tokens": 0,
        "compressed_tokens": 0,
    })
    sess["original_tokens"] += original
    sess["compressed_tokens"] += compressed
    sess["last_seen"] = datetime.now().isoformat()

    stats["recent_compressions"].appendleft({
        "ts": datetime.now().strftime("%H:%M:%S"),
        "session_id": session_id[:8],
        "original": original,
        "compressed": compressed,
        "saved": original - compressed,
    })
    save_stats()

def record_request(session_id: str, session_name: str | None = None):
    stats["total_requests"] += 1
    sess = stats["sessions"].setdefault(session_id, {
        "first_seen": datetime.now().isoformat(),
        "requests": 0,
        "original_tokens": 0,
        "compressed_tokens": 0,
        "name": None,
    })
    sess["requests"] += 1
    sess["last_seen"] = datetime.now().isoformat()
    if session_name:
        sess["name"] = session_name
    save_stats()

# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------

def compress_text(text: str, session_id: str) -> str:
    if len(text) <= 200:
        return text
    try:
        result = compressor.compress_prompt(
            text,
            rate=0.5,
            force_tokens=['\n', '?', '.', '!'],
        )
    except Exception as e:
        print(f"[LLMLingua-2] compression failed, forwarding original: {e}")
        return text
    orig = result['origin_tokens']
    comp = result['compressed_tokens']
    print(f"[LLMLingua-2] {orig} → {comp} tokens ({result['ratio']}) [{session_id[:8]}]")
    record_compression(session_id, orig, comp)
    return result['compressed_prompt']

def compress_messages(messages: list, session_id: str) -> list:
    out = []
    for msg in messages:
        if msg.get("role") != "user":
            out.append(msg)
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            out.append({**msg, "content": compress_text(content, session_id)})
        elif isinstance(content, list):
            new_blocks = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    new_blocks.append({**block, "text": compress_text(block["text"], session_id)})
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
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
@app.head("/")
async def health():
    return {"status": "ok"}


@app.get("/stats")
async def get_stats():
    saved = stats["total_original_tokens"] - stats["total_compressed_tokens"]
    ratio = (stats["total_original_tokens"] / stats["total_compressed_tokens"]
             if stats["total_compressed_tokens"] > 0 else 1.0)
    sessions_out = {}
    for sid, s in stats["sessions"].items():
        sv = s["original_tokens"] - s["compressed_tokens"]
        sessions_out[sid] = {**s, "saved_tokens": sv}
    return {
        "started_at": stats["started_at"],
        "total_requests": stats["total_requests"],
        "total_original_tokens": stats["total_original_tokens"],
        "total_compressed_tokens": stats["total_compressed_tokens"],
        "total_saved_tokens": saved,
        "overall_ratio": round(ratio, 2),
        "sessions": sessions_out,
        "recent_compressions": list(stats["recent_compressions"]),
    }


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(DASHBOARD_HTML)


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
# Dashboard HTML
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLMLingua Proxy</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'SF Mono', 'Fira Code', monospace; background: #0d1117; color: #c9d1d9; padding: 20px; max-width: 1100px; margin: 0 auto; }

  .header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #21262d; }
  .title { color: #58a6ff; font-size: 14px; font-weight: 700; letter-spacing: .04em; }
  .header-right { display: flex; align-items: center; gap: 16px; font-size: 11px; color: #8b949e; }
  .live-dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #3fb950; margin-right: 4px; animation: pulse 2s infinite; vertical-align: middle; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

  .hero { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }
  .hero-left { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 28px 20px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; }
  .hero-pct { font-size: 68px; font-weight: 800; color: #3fb950; line-height: 1; letter-spacing: -.02em; }
  .hero-label { font-size: 10px; color: #8b949e; margin-top: 8px; text-transform: uppercase; letter-spacing: .1em; }
  .hero-sublabel { font-size: 9px; color: #484f58; margin-top: 3px; }

  .hero-right { background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 18px 20px; }
  .term-prompt { color: #3fb950; margin-bottom: 10px; font-size: 12px; }
  .term-line { display: flex; gap: 8px; padding: 2px 0; font-size: 12px; }
  .term-key { color: #8b949e; flex: 1; white-space: nowrap; }
  .term-val { color: #f0f6fc; font-weight: 600; text-align: right; }
  .term-val.green { color: #3fb950; }
  .term-val.blue { color: #58a6ff; }

  .cards { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-bottom: 12px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px 14px; }
  .card-label { font-size: 9px; color: #8b949e; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 5px; }
  .card-value { font-size: 20px; font-weight: 700; color: #f0f6fc; }
  .card-value.blue { color: #58a6ff; }
  .card-value.green { color: #3fb950; }

  .section { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px 16px; margin-bottom: 12px; }
  .section-title { font-size: 9px; color: #8b949e; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 12px; }

  .sparkline { display: flex; align-items: flex-end; gap: 2px; height: 44px; overflow: hidden; }
  .bar { width: 9px; min-height: 4px; border-radius: 2px 2px 0 0; cursor: default; opacity: .85; transition: opacity .1s; flex-shrink: 0; }
  .bar:hover { opacity: 1; }
  .spark-empty { font-size: 11px; color: #484f58; }
  .spark-legend { display: flex; gap: 16px; margin-top: 8px; font-size: 10px; color: #8b949e; }
  .leg-dot { display: inline-block; width: 8px; height: 8px; border-radius: 1px; margin-right: 4px; vertical-align: middle; }

  .sess-bars { display: flex; flex-direction: column; gap: 8px; }
  .sess-bar-row { display: flex; align-items: center; gap: 10px; font-size: 11px; }
  .sess-bar-label { width: 90px; color: #8b949e; flex-shrink: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .sess-bar-wrap { flex: 1; background: #21262d; border-radius: 3px; height: 12px; overflow: hidden; }
  .sess-bar-fill { height: 100%; border-radius: 3px; transition: width .3s; }
  .sess-bar-pct { width: 36px; text-align: right; color: #3fb950; flex-shrink: 0; }
  .sess-bar-saved { width: 54px; text-align: right; color: #484f58; flex-shrink: 0; font-size: 10px; }

  table { width: 100%; border-collapse: collapse; font-size: 11px; }
  th { text-align: left; color: #8b949e; padding: 5px 10px; border-bottom: 1px solid #21262d; font-weight: normal; font-size: 9px; text-transform: uppercase; letter-spacing: .06em; }
  td { padding: 7px 10px; border-bottom: 1px solid #21262d; vertical-align: middle; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: #1c2128; }
  .badge { display: inline-block; padding: 2px 7px; border-radius: 4px; font-size: 10px; background: #1a3a4a; color: #58a6ff; }
  .name-tag { display: inline-block; padding: 2px 7px; border-radius: 4px; font-size: 10px; background: #2d1f3d; color: #bc8cff; }
  .ratio-badge { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 10px; }
  .muted { color: #8b949e; }
  .green { color: #3fb950; }

  @media (max-width: 700px) {
    .hero { grid-template-columns: 1fr; }
    .cards { grid-template-columns: repeat(2, 1fr); }
    .hero-pct { font-size: 52px; }
  }
</style>
</head>
<body>

<div class="header">
  <div class="title">⚡ LLMLingua Proxy</div>
  <div class="header-right">
    <span id="uptime_display">—</span>
    <span><span class="live-dot"></span>live</span>
  </div>
</div>

<div class="hero">
  <div class="hero-left">
    <div class="hero-pct" id="savings_pct">—%</div>
    <div class="hero-label">tokens saved overall</div>
    <div class="hero-sublabel">LLMLingua-2 tokenizer units</div>
  </div>
  <div class="hero-right">
    <div class="term-prompt">$ llmlingua-proxy --status</div>
    <div class="term-line"><span class="term-key">requests processed</span><span class="term-val blue" id="t_requests">—</span></div>
    <div class="term-line"><span class="term-key">original tokens</span><span class="term-val" id="t_original">—</span></div>
    <div class="term-line"><span class="term-key">compressed tokens</span><span class="term-val" id="t_compressed">—</span></div>
    <div class="term-line"><span class="term-key">tokens saved</span><span class="term-val green" id="t_saved">—</span></div>
    <div class="term-line"><span class="term-key">compression ratio</span><span class="term-val green" id="t_ratio">—</span></div>
    <div class="term-line"><span class="term-key">active sessions</span><span class="term-val blue" id="t_sessions">—</span></div>
  </div>
</div>

<div class="cards">
  <div class="card"><div class="card-label">Requests</div><div class="card-value blue" id="card_requests">—</div></div>
  <div class="card"><div class="card-label">Tokens Saved</div><div class="card-value green" id="card_saved">—</div></div>
  <div class="card"><div class="card-label">Sessions</div><div class="card-value blue" id="card_sessions">—</div></div>
  <div class="card"><div class="card-label">Avg Ratio</div><div class="card-value green" id="card_ratio">—</div></div>
  <div class="card"><div class="card-label">Uptime</div><div class="card-value" id="card_uptime">—</div></div>
</div>

<div class="section">
  <div class="section-title">Recent compressions — height = original size &nbsp;·&nbsp; color = savings %</div>
  <div id="sparkline" class="sparkline"><span class="spark-empty">No compressions yet</span></div>
  <div class="spark-legend">
    <span><span class="leg-dot" style="background:#3fb950"></span>≥40% saved</span>
    <span><span class="leg-dot" style="background:#d29922"></span>20–39% saved</span>
    <span><span class="leg-dot" style="background:#484f58"></span>&lt;20% saved</span>
  </div>
</div>

<div class="section">
  <div class="section-title">Sessions — compression efficiency (sorted by tokens saved)</div>
  <div id="sess_bars" class="sess-bars"><span class="spark-empty">No sessions yet</span></div>
</div>

<div class="section">
  <div class="section-title">Sessions — detail</div>
  <table>
    <thead><tr><th>Session</th><th>Name</th><th>Requests</th><th>Saved</th><th>Ratio</th><th>Last seen</th></tr></thead>
    <tbody id="sessions_body"></tbody>
  </table>
</div>

<script>
function fmt(n) {
  if (n == null) return '—';
  if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
  if (n >= 1000) return (n / 1000).toFixed(1) + 'k';
  return n.toLocaleString();
}

function fmtUptime(secs) {
  if (secs < 60) return secs + 's';
  if (secs < 3600) return Math.floor(secs / 60) + 'm';
  return Math.floor(secs / 3600) + 'h ' + Math.floor((secs % 3600) / 60) + 'm';
}

function barColor(pct) {
  if (pct >= 40) return '#3fb950';
  if (pct >= 20) return '#d29922';
  return '#484f58';
}

function ratioBadgeStyle(ratio) {
  if (ratio >= 2.0) return 'background:#0d2b1a;color:#3fb950';
  if (ratio >= 1.5) return 'background:#2b2200;color:#d29922';
  return 'background:#1c2128;color:#8b949e';
}

async function refresh() {
  try {
    const r = await fetch('/stats');
    const d = await r.json();

    const savingsPct = d.total_original_tokens > 0
      ? Math.round((1 - d.total_compressed_tokens / d.total_original_tokens) * 100)
      : 0;

    document.getElementById('savings_pct').textContent = savingsPct + '%';
    document.getElementById('t_requests').textContent = fmt(d.total_requests);
    document.getElementById('t_original').textContent = fmt(d.total_original_tokens);
    document.getElementById('t_compressed').textContent = fmt(d.total_compressed_tokens);
    document.getElementById('t_saved').textContent = fmt(d.total_saved_tokens) + ' (' + savingsPct + '%)';
    document.getElementById('t_ratio').textContent = d.overall_ratio + '×';
    document.getElementById('t_sessions').textContent = Object.keys(d.sessions).length;

    document.getElementById('card_requests').textContent = fmt(d.total_requests);
    document.getElementById('card_saved').textContent = fmt(d.total_saved_tokens);
    document.getElementById('card_sessions').textContent = Object.keys(d.sessions).length;
    document.getElementById('card_ratio').textContent = d.overall_ratio + '×';

    const uptimeSecs = Math.floor((Date.now() - new Date(d.started_at)) / 1000);
    const uptimeStr = fmtUptime(uptimeSecs);
    document.getElementById('card_uptime').textContent = uptimeStr;
    document.getElementById('uptime_display').textContent = 'up ' + uptimeStr;

    // Sparkline
    const comps = [...d.recent_compressions].reverse();
    const sparkEl = document.getElementById('sparkline');
    if (comps.length === 0) {
      sparkEl.innerHTML = '<span class="spark-empty">No compressions yet</span>';
    } else {
      const maxOrig = Math.max(...comps.map(c => c.original), 1);
      sparkEl.innerHTML = comps.map(c => {
        const pct = c.original > 0 ? Math.round((1 - c.compressed / c.original) * 100) : 0;
        const heightPct = Math.max(10, Math.round((c.original / maxOrig) * 100));
        const tip = c.ts + ' · ' + c.session_id + ' · ' + pct + '% saved (' + fmt(c.original) + ' → ' + fmt(c.compressed) + ')';
        return '<div class="bar" style="background:' + barColor(pct) + ';height:' + heightPct + '%" title="' + tip + '"></div>';
      }).join('');
    }

    // Session efficiency bars
    const sessions = Object.entries(d.sessions).sort((a, b) =>
      (b[1].saved_tokens || 0) - (a[1].saved_tokens || 0)
    );
    const maxSaved = Math.max(...sessions.map(([, s]) => s.saved_tokens || 0), 1);
    const sessBarEl = document.getElementById('sess_bars');
    if (sessions.length === 0) {
      sessBarEl.innerHTML = '<span class="spark-empty">No sessions yet</span>';
    } else {
      sessBarEl.innerHTML = sessions.slice(0, 8).map(([id, s]) => {
        const sPct = s.original_tokens > 0
          ? Math.round((1 - s.compressed_tokens / s.original_tokens) * 100) : 0;
        const barW = Math.round(((s.saved_tokens || 0) / maxSaved) * 100);
        const label = s.name ? s.name : id.slice(0, 8);
        return '<div class="sess-bar-row">'
          + '<div class="sess-bar-label" title="' + id + '">' + label + '</div>'
          + '<div class="sess-bar-wrap"><div class="sess-bar-fill" style="width:' + barW + '%;background:' + barColor(sPct) + '"></div></div>'
          + '<div class="sess-bar-pct">' + sPct + '%</div>'
          + '<div class="sess-bar-saved">' + fmt(s.saved_tokens) + '</div>'
          + '</div>';
      }).join('');
    }

    // Session table
    document.getElementById('sessions_body').innerHTML = sessions.map(([id, s]) => {
      const sPct = s.original_tokens > 0
        ? Math.round((1 - s.compressed_tokens / s.original_tokens) * 100) : 0;
      const ratio = s.compressed_tokens > 0
        ? (s.original_tokens / s.compressed_tokens).toFixed(2) : null;
      const nameCell = s.name
        ? '<span class="name-tag">' + s.name + '</span>'
        : '<span class="muted">—</span>';
      const ratioCell = ratio
        ? '<span class="ratio-badge" style="' + ratioBadgeStyle(parseFloat(ratio)) + '">' + ratio + '×</span>'
        : '<span class="muted">—</span>';
      return '<tr>'
        + '<td><span class="badge">' + id.slice(0, 8) + '</span></td>'
        + '<td>' + nameCell + '</td>'
        + '<td>' + s.requests + '</td>'
        + '<td class="green">' + fmt(s.saved_tokens) + ' <span class="muted">(' + sPct + '%)</span></td>'
        + '<td>' + ratioCell + '</td>'
        + '<td class="muted">' + (s.last_seen ? new Date(s.last_seen).toLocaleTimeString() : '—') + '</td>'
        + '</tr>';
    }).join('');

  } catch(e) { console.error(e); }
}

refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>"""


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9099)
