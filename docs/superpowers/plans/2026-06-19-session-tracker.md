# Session Tracker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Track new session" button to the dashboard that creates a named tracker, auto-links the next incoming session, and shows a full filtered dashboard at `/dashboard/{slug}`.

**Architecture:** A new `trackers` SQLite table stores pending/active trackers (one at a time). Three new admin endpoints manage tracker lifecycle. The existing `/stats` and `/stats/timeseries` endpoints gain an optional `session_id` query param that narrows all DB queries to one session. The session dashboard reuses `DASHBOARD_HTML` with a server-injected `const TRACKER = {...}` bootstrap that makes the JS filter all fetches and show a status banner.

**Tech Stack:** Python 3.12, FastAPI, SQLite (via stdlib sqlite3), vanilla JS in `DASHBOARD_HTML`

## Global Constraints

- All changes go in `llmlingua_proxy.py` (single-file app) and `tests/test_tracker.py`
- Python 3.12+ — use `str | None` union syntax, not `Optional`
- FastAPI query params declared as function arguments, not `Query()`
- No new dependencies
- `DASHBOARD_HTML` is the single HTML/JS string at the bottom of the file — all UI changes go there
- Tests use the existing `client` fixture from `tests/conftest.py` unchanged

---

### Task 1: `trackers` table in `init_db` + `make_slug` helper

**Files:**
- Modify: `llmlingua_proxy.py` — `init_db()` function + new `make_slug()` function
- Test: `tests/test_tracker.py` (create file)

**Interfaces:**
- Produces: `make_slug(name: str, conn: sqlite3.Connection) -> str` — called by Task 2

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tracker.py`:

```python
import re
import pytest
from starlette.testclient import TestClient


def test_init_db_creates_trackers_table(tmp_path):
    from llmlingua_proxy import init_db
    conn = init_db(str(tmp_path / "metrics.db"))
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='trackers'"
    )
    assert cur.fetchone() is not None
    cols = [row[1] for row in conn.execute("PRAGMA table_info(trackers)")]
    for col in ("slug", "name", "status", "session_id", "created_at", "linked_at"):
        assert col in cols
    conn.close()


def test_make_slug_basic(tmp_path):
    from llmlingua_proxy import init_db, make_slug
    conn = init_db(str(tmp_path / "metrics.db"))
    assert make_slug("Kompress Test 1", conn) == "kompress-test-1"
    assert make_slug("hello world!", conn) == "hello-world"
    conn.close()


def test_make_slug_collision(tmp_path):
    from llmlingua_proxy import init_db, make_slug
    conn = init_db(str(tmp_path / "metrics.db"))
    conn.execute(
        "INSERT INTO trackers (slug, name, status, created_at) VALUES ('my-test','My Test','pending','2026-01-01T00:00:00')"
    )
    conn.commit()
    slug = make_slug("my test", conn)
    assert slug == "my-test-2"
    conn.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/jeancsil/code/llm-lingua && uv run pytest tests/test_tracker.py -v
```

Expected: 3 failures — `trackers` table missing, `make_slug` not defined.

- [ ] **Step 3: Add `trackers` table to `init_db`**

Inside `init_db()`, after the existing `conn.execute("CREATE TABLE IF NOT EXISTS meta ...")` call, add:

```python
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trackers (
            slug        TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'pending',
            session_id  TEXT,
            created_at  TEXT NOT NULL,
            linked_at   TEXT
        )
        """
    )
```

- [ ] **Step 4: Add `make_slug` helper**

Add this function after `recover_stats_from_backup` and before `load_stats_from_db`:

```python
def make_slug(name: str, conn) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "tracker"
    slug, i = base, 2
    while conn.execute("SELECT 1 FROM trackers WHERE slug=?", (slug,)).fetchone():
        slug, i = f"{base}-{i}", i + 1
    return slug
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_tracker.py -v
```

Expected: all 3 pass.

- [ ] **Step 6: Commit**

```bash
git add llmlingua_proxy.py tests/test_tracker.py
git commit -m "feat: add trackers table and make_slug helper"
```

---

### Task 2: Tracker CRUD endpoints (POST, GET, DELETE)

**Files:**
- Modify: `llmlingua_proxy.py` — add 3 routes after `DELETE /admin/tracker/{slug}` (after the existing `/admin/set-model` route)
- Test: `tests/test_tracker.py`

**Interfaces:**
- Consumes: `make_slug(name, conn)` from Task 1, `_db_conn` global
- Produces:
  - `POST /admin/tracker` → `{"slug": str, "name": str, "status": "pending", "session_id": None, "created_at": str}`
  - `GET /admin/tracker` → tracker dict or `null`
  - `DELETE /admin/tracker/{slug}` → `{"deleted": str}` or 404

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tracker.py`:

```python
def test_create_tracker(client: TestClient):
    r = client.post("/admin/tracker", json={"name": "Kompress Run 1"})
    assert r.status_code == 200
    data = r.json()
    assert data["slug"] == "kompress-run-1"
    assert data["status"] == "pending"
    assert data["session_id"] is None


def test_create_tracker_requires_name(client: TestClient):
    r = client.post("/admin/tracker", json={"name": ""})
    assert r.status_code == 400


def test_create_tracker_409_when_pending_exists(client: TestClient):
    client.post("/admin/tracker", json={"name": "First"})
    r = client.post("/admin/tracker", json={"name": "Second"})
    assert r.status_code == 409


def test_get_tracker_null_when_none(client: TestClient):
    r = client.get("/admin/tracker")
    assert r.status_code == 200
    assert r.json() is None


def test_get_tracker_returns_pending(client: TestClient):
    client.post("/admin/tracker", json={"name": "My Track"})
    r = client.get("/admin/tracker")
    assert r.status_code == 200
    assert r.json()["status"] == "pending"
    assert r.json()["name"] == "My Track"


def test_delete_tracker(client: TestClient):
    client.post("/admin/tracker", json={"name": "To Delete"})
    r = client.delete("/admin/tracker/to-delete")
    assert r.status_code == 200
    assert client.get("/admin/tracker").json() is None


def test_delete_tracker_404(client: TestClient):
    r = client.delete("/admin/tracker/nonexistent")
    assert r.status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_tracker.py -v -k "tracker"
```

Expected: failures on all 7 new tests — routes don't exist.

- [ ] **Step 3: Add the three endpoints**

Add after the existing `set_model` route in `llmlingua_proxy.py` (after the `@app.post("/admin/set-model")` block):

```python
@app.post("/admin/tracker")
async def create_tracker(request: Request):
    body = await request.json()
    name = (body.get("name") or "").strip()
    if not name:
        return JSONResponse({"error": "name required"}, status_code=400)
    if _db_conn is None:
        return JSONResponse({"error": "db not ready"}, status_code=503)
    pending = _db_conn.execute(
        "SELECT slug FROM trackers WHERE status='pending'"
    ).fetchone()
    if pending:
        return JSONResponse(
            {"error": "a pending tracker already exists", "slug": pending[0]},
            status_code=409,
        )
    slug = make_slug(name, _db_conn)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _db_conn.execute(
        "INSERT INTO trackers (slug, name, status, created_at) VALUES (?,?,'pending',?)",
        (slug, name, ts),
    )
    _db_conn.commit()
    return JSONResponse({"slug": slug, "name": name, "status": "pending", "session_id": None, "created_at": ts})


@app.get("/admin/tracker")
async def get_tracker():
    if _db_conn is None:
        return JSONResponse(None)
    row = _db_conn.execute(
        "SELECT slug, name, status, session_id, created_at, linked_at "
        "FROM trackers ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    return JSONResponse(dict(row) if row else None)


@app.delete("/admin/tracker/{slug}")
async def delete_tracker(slug: str):
    if _db_conn is None:
        return JSONResponse({"error": "db not ready"}, status_code=503)
    result = _db_conn.execute("DELETE FROM trackers WHERE slug=?", (slug,))
    _db_conn.commit()
    if result.rowcount == 0:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse({"deleted": slug})
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_tracker.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add llmlingua_proxy.py tests/test_tracker.py
git commit -m "feat: add tracker CRUD endpoints (POST/GET/DELETE /admin/tracker)"
```

---

### Task 3: Auto-link pending tracker in `record_request`

**Files:**
- Modify: `llmlingua_proxy.py` — `record_request()` function
- Test: `tests/test_tracker.py`

**Interfaces:**
- Consumes: `_db_conn` global, `trackers` table from Task 1
- Produces: when a request arrives and a pending tracker exists, sets `status='active'`, `session_id`, `linked_at`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tracker.py`:

```python
def test_auto_link_pending_tracker(client: TestClient):
    import sys
    proxy = sys.modules["llmlingua_proxy"]

    # Create a pending tracker
    client.post("/admin/tracker", json={"name": "Auto Link Test"})
    assert client.get("/admin/tracker").json()["status"] == "pending"

    # Simulate a session request arriving
    proxy.record_request("session-abc-123")

    tracker = client.get("/admin/tracker").json()
    assert tracker["status"] == "active"
    assert tracker["session_id"] == "session-abc-123"
    assert tracker["linked_at"] is not None


def test_no_tracker_record_request_is_safe(client: TestClient):
    import sys
    proxy = sys.modules["llmlingua_proxy"]
    # No tracker exists — should not raise
    proxy.record_request("session-xyz")
    assert client.get("/admin/tracker").json() is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_tracker.py::test_auto_link_pending_tracker tests/test_tracker.py::test_no_tracker_record_request_is_safe -v
```

Expected: `test_auto_link_pending_tracker` fails (tracker stays pending).

- [ ] **Step 3: Add auto-link logic to `record_request`**

At the **end** of `record_request()`, after `if session_name: sess["name"] = session_name`, add:

```python
    if _db_conn is not None:
        pending = _db_conn.execute(
            "SELECT slug FROM trackers WHERE status='pending'"
        ).fetchone()
        if pending:
            ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
            _db_conn.execute(
                "UPDATE trackers SET status='active', session_id=?, linked_at=? WHERE slug=?",
                (session_id, ts, pending[0]),
            )
            _db_conn.commit()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_tracker.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add llmlingua_proxy.py tests/test_tracker.py
git commit -m "feat: auto-link pending tracker on first incoming session request"
```

---

### Task 4: `session_id` filter on `/stats` and `/stats/timeseries`

**Files:**
- Modify: `llmlingua_proxy.py` — `get_stats()` and `get_timeseries()` functions
- Test: `tests/test_tracker.py`

**Interfaces:**
- Produces:
  - `GET /stats?session_id=X` — all DB-driven fields (`today`, `alltime`, `recent`, `by_model`) filtered to session X
  - `GET /stats/timeseries?session_id=X` — time-series filtered to session X

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tracker.py`:

```python
def test_stats_session_filter(client: TestClient):
    import sys
    proxy = sys.modules["llmlingua_proxy"]

    # Record two compressions in different sessions
    proxy.record_compression("session-A", 500, 300, 50.0)
    proxy.record_compression("session-B", 400, 200, 40.0)

    r = client.get("/stats?session_id=session-A")
    assert r.status_code == 200
    data = r.json()
    # alltime for session-A only: 1 request, 200 tokens saved
    assert data["alltime"]["requests"] == 1
    assert data["alltime"]["tokens_saved"] == 200


def test_stats_no_filter_returns_all(client: TestClient):
    import sys
    proxy = sys.modules["llmlingua_proxy"]

    proxy.record_compression("session-A", 500, 300, 50.0)
    proxy.record_compression("session-B", 400, 200, 40.0)

    r = client.get("/stats")
    data = r.json()
    assert data["alltime"]["requests"] == 2


def test_timeseries_session_filter(client: TestClient):
    import sys
    proxy = sys.modules["llmlingua_proxy"]

    proxy.record_compression("session-A", 500, 300, 50.0)
    proxy.record_compression("session-B", 400, 200, 40.0)

    r = client.get("/stats/timeseries?session_id=session-A")
    assert r.status_code == 200
    buckets = r.json()
    total_reqs = sum(b["requests"] for b in buckets)
    assert total_reqs == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_tracker.py -k "filter" -v
```

Expected: all 3 fail — `session_id` param is currently ignored.

- [ ] **Step 3: Add `session_id` param and filter logic to `get_stats`**

Change the function signature from:
```python
async def get_stats():
```
to:
```python
async def get_stats(session_id: str | None = None):
```

Then, directly after `active_model = compressor_info["model"]` (before the `by_model_rows` query), add:

```python
        sess_filter = " AND session_id = ?" if session_id else ""
        sess_args   = (session_id,) if session_id else ()
```

Then replace each of the four DB queries as follows:

**`by_model_rows` query** — change from:
```python
        by_model_rows = _db_conn.execute(
            """
            SELECT model, ...
            FROM compressions
            GROUP BY model
            ORDER BY requests DESC
            """
        ).fetchall()
```
to:
```python
        by_model_rows = _db_conn.execute(
            f"""
            SELECT model,
                   COUNT(*) AS requests,
                   ROUND(AVG((original_tokens - compressed_tokens) * 100.0 / original_tokens), 1) AS avg_savings_pct,
                   ROUND(AVG(CAST(original_tokens AS REAL) / NULLIF(compressed_tokens, 0)), 2) AS avg_ratio,
                   SUM(original_tokens - compressed_tokens) AS total_saved,
                   ROUND(AVG(latency_ms), 1) AS avg_latency_ms
            FROM compressions
            {'WHERE session_id = ?' if session_id else ''}
            GROUP BY model
            ORDER BY requests DESC
            """,
            sess_args,
        ).fetchall()
```

**`today_row` query** — change `WHERE date(ts) = date('now') AND model = ?` to:
```python
        today_row = _db_conn.execute(
            f"""
            SELECT
                COUNT(*) AS requests,
                COALESCE(SUM(original_tokens - compressed_tokens), 0) AS tokens_saved,
                ROUND(AVG((original_tokens - compressed_tokens) * 100.0 / original_tokens), 1) AS avg_savings_pct,
                ROUND(AVG(latency_ms), 1) AS avg_latency_ms,
                COUNT(DISTINCT session_id) AS sessions
            FROM compressions
            WHERE date(ts) = date('now') AND model = ?{sess_filter}
            """,
            (active_model, *sess_args),
        ).fetchone()
```

**`alltime_row` query** — change `WHERE model = ?` to:
```python
        alltime_row = _db_conn.execute(
            f"""
            SELECT
                COUNT(*) AS requests,
                COALESCE(SUM(original_tokens - compressed_tokens), 0) AS tokens_saved,
                ROUND(AVG((original_tokens - compressed_tokens) * 100.0 / original_tokens), 1) AS avg_savings_pct,
                ROUND(AVG(latency_ms), 1) AS avg_latency_ms,
                COUNT(DISTINCT session_id) AS sessions,
                ROUND(AVG(CAST(original_tokens AS REAL) / NULLIF(compressed_tokens, 0)), 2) AS avg_ratio
            FROM compressions
            WHERE model = ?{sess_filter}
            """,
            (active_model, *sess_args),
        ).fetchone()
```

**`recent_db_rows` query** — change `WHERE model = ?` to:
```python
        recent_db_rows = _db_conn.execute(
            f"""
            SELECT ts, session_id, model, original_tokens, compressed_tokens,
                   ROUND((original_tokens - compressed_tokens) * 100.0 / original_tokens, 1) AS savings_pct,
                   latency_ms
            FROM compressions
            WHERE model = ?{sess_filter}
            ORDER BY id DESC
            LIMIT 20
            """,
            (active_model, *sess_args),
        ).fetchall()
```

- [ ] **Step 4: Add `session_id` param to `get_timeseries`**

Change the function signature from:
```python
async def get_timeseries(model: str | None = None):
```
to:
```python
async def get_timeseries(model: str | None = None, session_id: str | None = None):
```

Then add at the top of the function body (before the `if model:` branch):
```python
    sess_filter = " AND session_id = ?" if session_id else ""
    sess_args   = (session_id,) if session_id else ()
```

Replace the `if model:` block with:
```python
    if model:
        rows = _db_conn.execute(
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
        rows = _db_conn.execute(
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
```

- [ ] **Step 5: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass, including the 3 new filter tests.

- [ ] **Step 6: Commit**

```bash
git add llmlingua_proxy.py tests/test_tracker.py
git commit -m "feat: add session_id filter to /stats and /stats/timeseries"
```

---

### Task 5: `GET /dashboard/{slug}` route + session dashboard bootstrap

**Files:**
- Modify: `llmlingua_proxy.py` — add route, update `DASHBOARD_HTML`
- Test: `tests/test_tracker.py`

**Interfaces:**
- Consumes: `trackers` table, `DASHBOARD_HTML` string
- Produces:
  - `GET /dashboard/{slug}` → same HTML with `const TRACKER = {...}` injected before `</head>`
  - `GET /dashboard/nonexistent` → 404 plain HTML

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tracker.py`:

```python
def test_session_dashboard_404_for_unknown_slug(client: TestClient):
    r = client.get("/dashboard/does-not-exist")
    assert r.status_code == 404


def test_session_dashboard_injects_tracker(client: TestClient):
    client.post("/admin/tracker", json={"name": "My Test"})
    r = client.get("/dashboard/my-test")
    assert r.status_code == 200
    assert "const TRACKER" in r.text
    assert '"slug": "my-test"' in r.text
    assert '"status": "pending"' in r.text


def test_dashboard_slug_returns_html(client: TestClient):
    client.post("/admin/tracker", json={"name": "HTML Test"})
    r = client.get("/dashboard/html-test")
    assert r.headers["content-type"].startswith("text/html")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_tracker.py -k "dashboard" -v
```

Expected: all 3 fail — route doesn't exist.

- [ ] **Step 3: Add the route**

Add the following route in `llmlingua_proxy.py` **immediately before** the `@app.get("/dashboard", response_class=HTMLResponse)` route:

```python
@app.get("/dashboard/{slug}", response_class=HTMLResponse)
async def session_dashboard(slug: str):
    if _db_conn is None:
        return HTMLResponse("<h1>DB not ready</h1>", status_code=503)
    row = _db_conn.execute(
        "SELECT slug, name, status, session_id, created_at, linked_at "
        "FROM trackers WHERE slug=?",
        (slug,),
    ).fetchone()
    if row is None:
        return HTMLResponse(f"<h1>Tracker '{slug}' not found</h1>", status_code=404)
    tracker = dict(row)
    bootstrap = f'<script>const TRACKER = {json.dumps(tracker)};</script>'
    html = DASHBOARD_HTML.replace("</head>", bootstrap + "\n</head>", 1)
    return HTMLResponse(html)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add llmlingua_proxy.py tests/test_tracker.py
git commit -m "feat: add GET /dashboard/{slug} route with TRACKER bootstrap injection"
```

---

### Task 6: Session dashboard JS — banner, pending polling, filtered fetches

**Files:**
- Modify: `llmlingua_proxy.py` — `DASHBOARD_HTML` string (CSS, HTML, JS sections)

No new tests for this task — the JS is tested visually by opening `/dashboard/{slug}` in a browser.

- [ ] **Step 1: Add CSS for the tracker banner**

Inside `DASHBOARD_HTML`, in the `<style>` block, add before the closing `</style>` tag:

```css
  /* ── Tracker banner ── */
  .tracker-banner { display: none; align-items: center; gap: 12px; background: #0d1a35; border: 1px solid #1e2f50; border-radius: 8px; padding: 10px 16px; margin-bottom: 12px; font-size: 12px; }
  .tracker-banner a { color: #58a6ff; text-decoration: none; flex-shrink: 0; }
  .tracker-banner a:hover { text-decoration: underline; }
  .tracker-sep { color: #30363d; }
  .tracker-name { color: #f0f6fc; font-weight: 700; }
  .tracker-status-active { color: #3fb950; }
  .tracker-status-pending { color: #d29922; }
```

- [ ] **Step 2: Add banner HTML element**

In `DASHBOARD_HTML`, immediately after the `<body>` tag and before the `<div class="header">` element, add:

```html
<!-- Session tracker banner (populated by JS when TRACKER is defined) -->
<div class="tracker-banner" id="tracker_banner">
  <a href="/dashboard">← global</a>
  <span class="tracker-sep">|</span>
  <span class="tracker-name" id="tracker_banner_name"></span>
  <span id="tracker_banner_status"></span>
</div>
```

- [ ] **Step 3: Add TRACKER detection and `updateBanner` at the top of the `<script>` block**

In `DASHBOARD_HTML`, immediately after `<script>` (the opening of the script block), add:

```javascript
const TRACKER = (typeof window !== 'undefined' && window.TRACKER) ? window.TRACKER : null;
let FILTER_SESSION = TRACKER ? TRACKER.session_id : null;

function updateBanner() {
  if (!TRACKER) return;
  const bannerEl = document.getElementById('tracker_banner');
  bannerEl.style.display = 'flex';
  document.getElementById('tracker_banner_name').textContent = TRACKER.name;
  const statusEl = document.getElementById('tracker_banner_status');
  if (TRACKER.status === 'active') {
    const sid = (TRACKER.session_id || '').slice(0, 8);
    statusEl.className = 'tracker-status-active';
    statusEl.textContent = 'active · ' + sid;
  } else {
    statusEl.className = 'tracker-status-pending';
    statusEl.innerHTML = '<span class="live-dot" style="background:#d29922;margin-right:4px"></span>waiting for next session…';
  }
}
```

- [ ] **Step 4: Update `refresh()` to filter fetches and poll when pending**

In `DASHBOARD_HTML`, at the **top** of the `async function refresh()` body (the very first lines inside the function), add:

```javascript
  // Session tracker: poll for link while pending
  if (TRACKER && TRACKER.status === 'pending') {
    try {
      const t = await fetch('/admin/tracker').then(r => r.json());
      if (t && t.slug === TRACKER.slug && t.status === 'active') {
        TRACKER.status = 'active';
        TRACKER.session_id = t.session_id;
        FILTER_SESSION = t.session_id;
      }
    } catch(e) {}
    updateBanner();
    if (!FILTER_SESSION) return; // still pending — no stats to show yet
  }
  if (TRACKER) updateBanner();

  // Build stats URL (with session filter when on session dashboard)
  const statsUrl = '/stats' + (FILTER_SESSION ? '?session_id=' + encodeURIComponent(FILTER_SESSION) : '');
```

Then change the existing `const r = await fetch('/stats');` line to:

```javascript
    const r = await fetch(statsUrl);
```

- [ ] **Step 5: Update `refreshTimeseries()` to include session filter**

In `refreshTimeseries()`, replace the existing line:
```javascript
    const r = await fetch('/stats/timeseries' + (model ? '?model=' + encodeURIComponent(model) : ''));
```
with:
```javascript
    const params = [];
    if (model) params.push('model=' + encodeURIComponent(model));
    if (FILTER_SESSION) params.push('session_id=' + encodeURIComponent(FILTER_SESSION));
    const r = await fetch('/stats/timeseries' + (params.length ? '?' + params.join('&') : ''));
```

- [ ] **Step 6: Call `updateBanner()` on page load**

At the bottom of `DASHBOARD_HTML`, before the `setInterval` calls, add:

```javascript
updateBanner();
```

- [ ] **Step 7: Smoke-test in browser**

With the proxy running (`uv run python llmlingua_proxy.py`):

1. Open `http://127.0.0.1:9099/dashboard`
2. No banner should be visible
3. Call `POST /admin/tracker` with `{"name": "smoke test"}` via curl or the UI:
   ```bash
   curl -s -X POST http://127.0.0.1:9099/admin/tracker -H 'Content-Type: application/json' -d '{"name":"smoke test"}' | python3 -m json.tool
   ```
4. Open `http://127.0.0.1:9099/dashboard/smoke-test`
5. Banner should appear showing "← global | smoke test | waiting for next session…" with a pulsing yellow dot
6. The main stats tiles should show "—" (empty state while pending)

- [ ] **Step 8: Commit**

```bash
git add llmlingua_proxy.py
git commit -m "feat: session dashboard banner and pending-state polling"
```

---

### Task 7: Main dashboard — Track button, inline form, status chip

**Files:**
- Modify: `llmlingua_proxy.py` — `DASHBOARD_HTML` string (header HTML + JS)

No automated tests — verified visually.

- [ ] **Step 1: Add CSS for the track button and chip**

In the `<style>` block, before `</style>`, add:

```css
  /* ── Track button / chip ── */
  .track-btn { font-size: 11px; background: #161b22; color: #58a6ff; border: 1px solid #30363d; border-radius: 6px; padding: 4px 10px; cursor: pointer; font-family: inherit; }
  .track-btn:hover { border-color: #58a6ff; }
  .track-form { display: none; align-items: center; gap: 6px; }
  .track-input { font-size: 11px; background: #0d1117; color: #c9d1d9; border: 1px solid #30363d; border-radius: 5px; padding: 3px 8px; font-family: inherit; outline: none; width: 160px; }
  .track-input:focus { border-color: #58a6ff; }
  .track-start { font-size: 11px; background: #0d1a35; color: #58a6ff; border: 1px solid #1e2f50; border-radius: 5px; padding: 3px 8px; cursor: pointer; font-family: inherit; }
  .track-chip { display: none; align-items: center; gap: 6px; font-size: 11px; }
  .track-chip a { color: #58a6ff; text-decoration: none; }
  .track-chip a:hover { text-decoration: underline; }
  .track-cancel { background: none; color: #8b949e; border: none; cursor: pointer; font-size: 14px; padding: 0 2px; line-height: 1; }
```

- [ ] **Step 2: Add track controls to the header**

In `DASHBOARD_HTML`, locate the header `<div style="display:flex;align-items:center;gap:8px">` that contains `model_badge` and `model_select`. After the closing `<span class="model-loading" ...>` span, add:

```html
    <!-- Track new session controls -->
    <div class="track-chip" id="track_chip">
      <a id="track_chip_link" href="#"></a>
      <span id="track_chip_status" style="font-size:10px"></span>
      <button class="track-cancel" id="track_cancel_btn" onclick="">✕</button>
    </div>
    <button class="track-btn" id="track_btn" onclick="openTrackForm()">Track new session</button>
    <div class="track-form" id="track_form">
      <input class="track-input" id="track_name" type="text" placeholder="session name" />
      <button class="track-start" onclick="submitTracker()">Start</button>
      <button class="track-cancel" onclick="closeTrackForm()">✕</button>
    </div>
```

- [ ] **Step 3: Add JS functions for track controls**

In the `<script>` block, after the `updateBanner()` function (which was added in Task 6), add:

```javascript
function openTrackForm() {
  document.getElementById('track_btn').style.display = 'none';
  document.getElementById('track_form').style.display = 'flex';
  document.getElementById('track_name').focus();
}

function closeTrackForm() {
  document.getElementById('track_form').style.display = 'none';
  document.getElementById('track_btn').style.display = '';
  document.getElementById('track_name').value = '';
}

async function submitTracker() {
  const name = (document.getElementById('track_name').value || '').trim();
  if (!name) return;
  try {
    const r = await fetch('/admin/tracker', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
    const data = await r.json();
    if (r.ok) {
      window.location.href = '/dashboard/' + data.slug;
    } else {
      alert(data.error || 'Error creating tracker');
      closeTrackForm();
    }
  } catch(e) { console.error(e); }
}

async function cancelTracker(slug) {
  try { await fetch('/admin/tracker/' + encodeURIComponent(slug), { method: 'DELETE' }); } catch(e) {}
  document.getElementById('track_chip').style.display = 'none';
  document.getElementById('track_btn').style.display = '';
}

function updateTrackerChip(tracker) {
  const chip = document.getElementById('track_chip');
  const btn  = document.getElementById('track_btn');
  if (!tracker) {
    chip.style.display = 'none';
    btn.style.display = '';
    return;
  }
  btn.style.display = 'none';
  chip.style.display = 'flex';
  const color = tracker.status === 'active' ? '#3fb950' : '#d29922';
  document.getElementById('track_chip_link').href = '/dashboard/' + tracker.slug;
  document.getElementById('track_chip_link').textContent = tracker.name;
  const statusEl = document.getElementById('track_chip_status');
  statusEl.style.color = color;
  statusEl.textContent = '· ' + tracker.status;
  const cancelBtn = document.getElementById('track_cancel_btn');
  cancelBtn.onclick = () => cancelTracker(tracker.slug);
}
```

- [ ] **Step 4: Add tracker chip polling to `refresh()`**

In `refresh()`, at the very end of the `try` block (just before `} catch(e) { console.error(e); }`), add:

```javascript
    // Update main dashboard tracker chip (only on global dashboard, not session dashboard)
    if (!TRACKER) {
      try {
        const tr = await fetch('/admin/tracker').then(r => r.json());
        updateTrackerChip(tr);
      } catch(e) {}
    }
```

- [ ] **Step 5: Smoke-test in browser**

With the proxy running:

1. Open `http://127.0.0.1:9099/dashboard`
2. "Track new session" button should appear in the header
3. Click it → inline form appears (name input + Start + ✕)
4. Type "llmlingua test" and click Start
5. Should redirect to `http://127.0.0.1:9099/dashboard/llmlingua-test`
6. Banner shows "← global | llmlingua test | ⬤ waiting for next session…"
7. Go back to `http://127.0.0.1:9099/dashboard`
8. Header now shows chip: "llmlingua test · pending ✕"
9. Click ✕ → chip disappears, "Track new session" button returns

- [ ] **Step 6: End-to-end test: tracker links on first request**

1. Create a new tracker via the button
2. Send one real API request through the proxy (or simulate with curl):
   ```bash
   curl -s -X POST http://127.0.0.1:9099/v1/messages \
     -H 'Content-Type: application/json' \
     -H 'x-api-key: test' \
     -H 'anthropic-version: 2023-06-01' \
     -H 'x-claude-code-session-id: my-test-session-001' \
     -d '{"model":"claude-haiku-4-5-20251001","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}' 2>&1 | head -5
   ```
3. Go back to `http://127.0.0.1:9099/dashboard/your-slug`
4. Banner should flip to "active · my-test" and the tiles should populate with data for that session

- [ ] **Step 7: Commit**

```bash
git add llmlingua_proxy.py
git commit -m "feat: main dashboard track button, form, and chip with live polling"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Covered by |
|---|---|
| `trackers` table with slug/name/status/session_id/created_at/linked_at | Task 1 |
| `make_slug` URL-safe with collision handling | Task 1 |
| `POST /admin/tracker` with 409 on duplicate pending | Task 2 |
| `GET /admin/tracker` returns current or null | Task 2 |
| `DELETE /admin/tracker/{slug}` with 404 | Task 2 |
| One tracker at a time (pending enforcement) | Task 2 — 409 check |
| Auto-link on first request regardless of new/existing session | Task 3 |
| `GET /stats?session_id=` filters today/alltime/recent/by_model | Task 4 |
| `GET /stats/timeseries?session_id=` | Task 4 |
| `GET /dashboard/{slug}` injects TRACKER bootstrap | Task 5 |
| 404 for unknown slug | Task 5 |
| Banner with ← global link, name, status | Task 6 |
| Pending state: pulsing yellow dot + "waiting..." | Task 6 |
| Active state: session_id shown | Task 6 |
| Poll `/admin/tracker` every 2s while pending, flip without reload | Task 6 |
| Filter all fetches by session_id once linked | Task 6 |
| "Track new session" button in main header | Task 7 |
| Inline form (no modal) with name input | Task 7 |
| Redirect to `/dashboard/{slug}` on submit | Task 7 |
| Status chip with cancel button | Task 7 |
| Main dashboard polls tracker state every 2s | Task 7 |

**No placeholders found.** All steps contain actual code.

**Type consistency:**
- `make_slug(name: str, conn) -> str` — used in Task 2's `create_tracker`
- `FILTER_SESSION` — set in Task 6 from `TRACKER.session_id`, used in Task 6 fetch URLs
- `updateTrackerChip(tracker)` — defined in Task 7, called in Task 7's `refresh()` addition
- `sess_filter` / `sess_args` — defined locally in each endpoint, not shared
