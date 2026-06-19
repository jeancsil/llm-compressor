# Session Tracker — Design Spec

**Date:** 2026-06-19  
**Status:** Approved

## Problem

The proxy dashboard shows aggregate stats across all sessions. There's no way to isolate a single session's data to compare, e.g., kompress vs llmlingua2 effectiveness on a specific Claude Code session.

## Goal

Allow the user to click "Track new session" in the dashboard, give it a name, and get a session-scoped dashboard at `/dashboard/{slug}` that shows the same full UI filtered to that one session. The next session_id that hits the proxy auto-links to the tracker.

## Data Model

New table in `metrics.db`:

```sql
CREATE TABLE trackers (
    slug        TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    session_id  TEXT,
    created_at  TEXT NOT NULL,
    linked_at   TEXT
)
```

- `slug` is derived from the name (URL-safe, e.g. "Kompress Test 1" → `kompress-test-1`, with `-2` suffix if taken)
- `status` is `'pending'` (waiting for first session) or `'active'` (linked to a session_id)
- Only one tracker may be in `pending` state at a time (enforced by backend)
- `session_id` and `linked_at` are null until auto-linked

## Backend

### New endpoints

```
POST /admin/tracker
  body: { "name": "kompress test 1" }
  Returns: { slug, name, status, session_id, created_at }
  Error 409 if a pending tracker already exists

GET /admin/tracker
  Returns the current tracker (pending or active), or null if none

DELETE /admin/tracker/{slug}
  Removes the tracker; returns 404 if not found
```

### Filter params on existing endpoints

```
GET /stats?session_id=abc123
GET /stats/timeseries?model=kompress&session_id=abc123
```

When `session_id` is present, all DB queries in both endpoints add `AND session_id = ?`. Behavior is unchanged when the param is absent.

### Auto-link logic

In `record_request()`: if a `pending` tracker exists in the DB, set its `session_id` to the incoming session_id, set `linked_at` to now, flip `status` to `'active'`, and commit. This runs on the very first request after tracker creation.

### Slug derivation

```python
def make_slug(name: str, conn) -> str:
    base = re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')
    slug = base
    i = 2
    while conn.execute("SELECT 1 FROM trackers WHERE slug=?", (slug,)).fetchone():
        slug = f"{base}-{i}"
        i += 1
    return slug
```

## Frontend

### Main dashboard (`/dashboard`)

- Header gets a "Track new session" button (right of the model select)
- Clicking reveals an inline form: text input for name + "Start tracking" button
- On submit: `POST /admin/tracker` → redirect to `/dashboard/{slug}` in same tab
- If a tracker already exists (pending or active), the button is replaced by a status chip: `"tracking: {name} · {status}"` with an ✕ cancel button that calls `DELETE /admin/tracker/{slug}`
- Main dashboard polls `GET /admin/tracker` every 2s to keep the header chip in sync

### Session dashboard (`/dashboard/{slug}`)

Served by a new route `GET /dashboard/{slug}`. The server injects tracker state into the page as a JS bootstrap:

```html
<script>
const TRACKER = { slug: "kompress-test-1", name: "Kompress Test 1",
                  status: "pending", session_id: null };
</script>
```

**Banner (top of page, below header):**
- Shows: tracker name, status badge, session_id (first 8 chars once linked), "← global dashboard" link
- While pending: "Waiting for next session…" with a pulsing indicator
- While active: session_id shown, no pulsing

**Data fetching:**
- All `fetch('/stats')` → `fetch('/stats?session_id=' + TRACKER.session_id)` once `session_id` is known
- All `fetch('/stats/timeseries...')` → same with `&session_id=...`
- While pending, stats calls are skipped; the dashboard shows an empty state

**Polling while pending:**
- Polls `GET /admin/tracker` every 2s
- When status flips to `active`, sets `TRACKER.session_id`, stops the pending poll, and starts the normal 2s stats refresh cycle — no page reload

**No behavior changes** to the rest of the dashboard JS; all existing tiles, cards, timeseries, sparklines, and session tables render the same way, just with filtered data.

## Constraints

- One tracker at a time: creating a second while one is pending returns HTTP 409
- Active trackers can be deleted and replaced; only one pending tracker is allowed
- The session dashboard is read-only — no model switching from `/dashboard/{slug}`
- No authentication; same trust model as the rest of the proxy (local use only)

## Out of Scope

- Multiple simultaneous trackers
- Tracker history / archiving past trackers
- Pinning a tracker to a specific model
- Annotations or notes on a tracked session
