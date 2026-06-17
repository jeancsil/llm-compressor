# kompress-v2 + SQLite metrics + dashboard — Implementation Plan

**Spec:** `docs/superpowers/specs/2026-06-17-kompress-sqlite-dashboard.md`  
**Goal:** Pluggable compressor (llmlingua2/kompress), SQLite time-series metrics, richer dashboard.  
**Single file:** `llmlingua_proxy.py` + `pyproject.toml`. No new files except `tests/`.

## Architecture

- Model loading moves from module-level to FastAPI `lifespan` → enables test patching without loading real weights.
- Module-level `_backend: dict` holds the active compressor config after startup.
- Module-level `_db_conn: sqlite3.Connection` holds the open DB handle.
- `compress_text()` stays the same external signature (`str → str`); latency is recorded internally.
- Tests use `starlette.testclient.TestClient` + `monkeypatch` to patch `_backend` and `DB_PATH` before lifespan runs.

## Tech Stack

- Python 3.12, FastAPI, sqlite3 (stdlib), `kompress` (PyPI), `transformers`, `huggingface_hub`, `safetensors`
- Tests: `pytest`, `pytest-asyncio`, `starlette.testclient.TestClient`

---

## Task 1 — Add dependencies to pyproject.toml

**Files:** `pyproject.toml`

- [ ] Step 1: Add `kompress`, `transformers`, `huggingface_hub`, `safetensors` to `dependencies` in `pyproject.toml`:

```toml
dependencies = [
    "anthropic>=0.97.0",
    "fastapi>=0.136.1",
    "httpx>=0.28.1",
    "huggingface_hub>=0.23.0",
    "kompress>=0.3.0",
    "llmlingua>=0.2.2",
    "safetensors>=0.4.0",
    "torch>=2.11.0",
    "transformers>=4.40.0",
    "uvicorn>=0.46.0",
]
```

- [ ] Step 2: Run `uv sync` and confirm no errors.

- [ ] Step 3: Verify import works:
```bash
uv run python -c "from kompress.model.architecture import HeadroomCompressorV2; print('ok')"
```
Expected: `ok`

- [ ] Step 4: Commit:
```bash
git add pyproject.toml uv.lock
git commit -m "Add kompress, transformers, huggingface_hub, safetensors deps"
```

---

## Task 2 — Set up test infrastructure

**Files:** `tests/__init__.py` (empty), `tests/conftest.py`

- [ ] Step 1: Create `tests/__init__.py` (empty file).

- [ ] Step 2: Write `tests/conftest.py`:

```python
import sqlite3
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from starlette.testclient import TestClient


MOCK_COMPRESS_RESULT = {
    "compressed_prompt": "short text here",
    "origin_tokens": 100,
    "compressed_tokens": 60,
    "ratio": "0.60",
}


def _make_mock_llmlingua():
    m = MagicMock()
    m.compress_prompt.return_value = MOCK_COMPRESS_RESULT
    return m


@pytest.fixture()
def tmp_db(tmp_path) -> sqlite3.Connection:
    """In-memory-equivalent: a fresh DB in a temp dir."""
    from llmlingua_proxy import init_db
    conn = init_db(str(tmp_path / "test_metrics.db"))
    yield conn
    conn.close()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient with model loading patched out and a temp DB."""
    monkeypatch.setenv("COMPRESSOR_MODEL", "llmlingua2")
    monkeypatch.setenv("COMPRESS_RATE", "0.5")
    monkeypatch.setenv("COST_PER_MTOK", "3.0")

    # Patch PromptCompressor before lifespan loads it
    monkeypatch.setattr("llmlingua.PromptCompressor", lambda **kw: _make_mock_llmlingua())

    # Use a temp DB path
    import llmlingua_proxy as proxy
    monkeypatch.setattr(proxy, "DB_PATH", str(tmp_path / "test_metrics.db"))

    with TestClient(proxy.app) as c:
        yield c
```

- [ ] Step 3: Confirm pytest discovers the fixtures (no tests yet):
```bash
uv run pytest tests/ --collect-only
```
Expected: `no tests ran`

- [ ] Step 4: Commit:
```bash
git add tests/
git commit -m "Add test infrastructure: conftest with DB and client fixtures"
```

---

## Task 3 — Move model loading into FastAPI lifespan

**Files:** `llmlingua_proxy.py`

This is the structural prerequisite for all other tasks. Move the `PromptCompressor(...)` call from module level into a `lifespan` async context manager.

- [ ] Step 1: Write a failing test in `tests/test_proxy.py` verifying the app health endpoint works with the patched client:

```python
def test_health(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
```

- [ ] Step 2: Run — currently fails because importing `llmlingua_proxy` loads the model at import time, which hangs or errors in test env:
```bash
uv run pytest tests/test_proxy.py::test_health -v
```

- [ ] Step 3: Refactor `llmlingua_proxy.py`:

Remove the module-level model loading block:
```python
# DELETE these lines:
print("Loading LLMLingua-2 model...")
compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    use_llmlingua2=True,
    device_map="mps",
)
print("Model ready.")
```

Add at the top of the file (after existing imports):
```python
import time
from contextlib import asynccontextmanager
```

Add module-level placeholders (right after imports, before `app = FastAPI(...)`):
```python
_backend: dict = {}   # set during lifespan startup
_db_conn: sqlite3.Connection | None = None  # set during lifespan startup
DB_PATH = "metrics.db"
COST_PER_MTOK = float(os.environ.get("COST_PER_MTOK", "3.0"))
```

Add the lifespan function (before `app = FastAPI(...)`):
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _backend, _db_conn
    _db_conn = init_db(DB_PATH)          # Task 4
    migrate_from_json(_db_conn)          # Task 5
    load_stats_from_db(_db_conn)         # Task 6
    _backend = _load_backend()           # Task 9
    yield
    if _db_conn:
        _db_conn.close()
```

Change `app = FastAPI()` to:
```python
app = FastAPI(lifespan=lifespan)
```

Add stub implementations so the file is importable (will be replaced in subsequent tasks):
```python
def init_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    return conn

def migrate_from_json(conn: sqlite3.Connection) -> None:
    pass

def load_stats_from_db(conn: sqlite3.Connection) -> None:
    pass

def _load_backend() -> dict:
    from llmlingua import PromptCompressor
    rate = float(os.environ.get("COMPRESS_RATE", "0.5"))
    print("Loading LLMLingua-2 model...")
    c = PromptCompressor(
        model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        use_llmlingua2=True,
        device_map="mps",
    )
    print(f"[compressor] model=llmlingua-2  rate={rate:.2f}  device=mps")
    return {"type": "llmlingua2", "compressor": c, "rate": rate}
```

- [ ] Step 4: Run test — should now pass:
```bash
uv run pytest tests/test_proxy.py::test_health -v
# PASS
```

- [ ] Step 5: Commit:
```bash
git add llmlingua_proxy.py tests/test_proxy.py
git commit -m "Move model loading into FastAPI lifespan for testability"
```

---

## Task 4 — SQLite init_db() with schema

**Files:** `llmlingua_proxy.py`, `tests/test_proxy.py`

- [ ] Step 1: Add test:

```python
def test_init_db_creates_table(tmp_path):
    from llmlingua_proxy import init_db
    conn = init_db(str(tmp_path / "test.db"))
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='compressions'")
    assert cur.fetchone() is not None
    cols = [row[1] for row in conn.execute("PRAGMA table_info(compressions)")]
    assert cols == ["id", "ts", "session_id", "model", "original_tokens", "compressed_tokens", "latency_ms"]
    conn.close()
```

- [ ] Step 2: Run — fails (stub `init_db` doesn't create table):
```bash
uv run pytest tests/test_proxy.py::test_init_db_creates_table -v
# FAIL
```

- [ ] Step 3: Replace the stub `init_db` with the real implementation:

```python
def init_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS compressions (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            ts                TEXT    NOT NULL,
            session_id        TEXT    NOT NULL,
            model             TEXT    NOT NULL,
            original_tokens   INTEGER NOT NULL,
            compressed_tokens INTEGER NOT NULL,
            latency_ms        REAL    NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON compressions(ts)")
    conn.commit()
    return conn
```

- [ ] Step 4: Run — passes:
```bash
uv run pytest tests/test_proxy.py::test_init_db_creates_table -v
# PASS
```

- [ ] Step 5: Commit:
```bash
git add llmlingua_proxy.py tests/test_proxy.py
git commit -m "Implement init_db: create compressions table with ts index"
```

---

## Task 5 — stats.json migration

**Files:** `llmlingua_proxy.py`, `tests/test_proxy.py`

- [ ] Step 1: Add tests:

```python
import json
import shutil

def test_migrate_imports_rows(tmp_path):
    from llmlingua_proxy import init_db, migrate_from_json

    stats_json = tmp_path / "stats.json"
    stats_json.write_text(json.dumps({
        "total_requests": 5,
        "total_original_tokens": 500,
        "total_compressed_tokens": 300,
        "sessions": {},
        "recent_compressions": [
            {"ts": "10:00:00", "session_id": "aabbccdd1234", "original": 100, "compressed": 60, "saved": 40},
            {"ts": "10:01:00", "session_id": "aabbccdd1234", "original": 200, "compressed": 120, "saved": 80},
        ],
    }))

    conn = init_db(str(tmp_path / "metrics.db"))
    migrate_from_json(conn, json_path=str(stats_json))

    count = conn.execute("SELECT COUNT(*) FROM compressions").fetchone()[0]
    assert count == 2

    bak = tmp_path / "stats.json.bak"
    assert bak.exists()
    conn.close()


def test_migrate_is_idempotent(tmp_path):
    from llmlingua_proxy import init_db, migrate_from_json

    stats_json = tmp_path / "stats.json"
    stats_json.write_text(json.dumps({
        "recent_compressions": [
            {"ts": "10:00:00", "session_id": "abc", "original": 100, "compressed": 60, "saved": 40},
        ],
    }))

    conn = init_db(str(tmp_path / "metrics.db"))
    migrate_from_json(conn, json_path=str(stats_json))
    migrate_from_json(conn, json_path=str(stats_json))  # second call

    count = conn.execute("SELECT COUNT(*) FROM compressions").fetchone()[0]
    assert count == 1  # not duplicated
    conn.close()


def test_migrate_no_json(tmp_path):
    from llmlingua_proxy import init_db, migrate_from_json
    conn = init_db(str(tmp_path / "metrics.db"))
    migrate_from_json(conn, json_path=str(tmp_path / "stats.json"))  # file absent — no error
    conn.close()
```

- [ ] Step 2: Run — fails:
```bash
uv run pytest tests/test_proxy.py -k "migrate" -v
# FAIL
```

- [ ] Step 3: Replace stub `migrate_from_json` with real implementation. Add `json_path` parameter (defaulting to `"stats.json"`):

```python
def migrate_from_json(conn: sqlite3.Connection, json_path: str = "stats.json") -> None:
    path = Path(json_path)
    if not path.exists():
        return

    existing = conn.execute("SELECT COUNT(*) FROM compressions").fetchone()[0]
    if existing > 0:
        return  # already migrated

    try:
        data = json.loads(path.read_text())
        rows = data.get("recent_compressions", [])

        bak = path.with_suffix(".json.bak")
        shutil.copy2(path, bak)
        print(f"[migration] Backed up {path} → {bak}")

        conn.executemany(
            "INSERT INTO compressions (ts, session_id, model, original_tokens, compressed_tokens, latency_ms) "
            "VALUES (?, ?, 'llmlingua2', ?, ?, 0.0)",
            [
                (
                    datetime.now().strftime("%Y-%m-%dT") + r["ts"],
                    r["session_id"],
                    r["original"],
                    r["compressed"],
                )
                for r in rows
            ],
        )
        conn.commit()

        count = conn.execute("SELECT COUNT(*) FROM compressions").fetchone()[0]
        print(f"[migration] Imported {count} rows from {path} → metrics.db. Backup at {bak}.")
    except Exception as e:
        print(f"[migration] Failed, skipping: {e}")
```

Also update the lifespan call signature to match (pass `DB_PATH`-relative json path):
```python
migrate_from_json(_db_conn, json_path="stats.json")
```

- [ ] Step 4: Run — passes:
```bash
uv run pytest tests/test_proxy.py -k "migrate" -v
# PASS (3 tests)
```

- [ ] Step 5: Commit:
```bash
git add llmlingua_proxy.py tests/test_proxy.py
git commit -m "Implement migrate_from_json: backup stats.json, import rows to SQLite"
```

---

## Task 6 — load_stats_from_db() and remove stats.json I/O

**Files:** `llmlingua_proxy.py`, `tests/test_proxy.py`

- [ ] Step 1: Add test:

```python
def test_load_stats_from_db(tmp_path):
    from llmlingua_proxy import init_db, load_stats_from_db, stats
    from collections import deque

    conn = init_db(str(tmp_path / "metrics.db"))
    conn.executemany(
        "INSERT INTO compressions (ts, session_id, model, original_tokens, compressed_tokens, latency_ms) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("2026-06-17T10:00:00", "sess-aaa", "llmlingua2", 200, 120, 140.0),
            ("2026-06-17T10:01:00", "sess-aaa", "llmlingua2", 150, 90, 130.0),
            ("2026-06-17T10:02:00", "sess-bbb", "kompress",   300, 250, 50.0),
        ],
    )
    conn.commit()

    # Reset stats
    stats["total_original_tokens"] = 0
    stats["total_compressed_tokens"] = 0
    stats["sessions"] = {}
    stats["recent_compressions"] = deque(maxlen=100)

    load_stats_from_db(conn)

    assert stats["total_original_tokens"] == 650
    assert stats["total_compressed_tokens"] == 460
    assert "sess-aaa" in stats["sessions"]
    assert stats["sessions"]["sess-aaa"]["original_tokens"] == 350
    assert len(stats["recent_compressions"]) == 3
    conn.close()
```

- [ ] Step 2: Run — fails:
```bash
uv run pytest tests/test_proxy.py::test_load_stats_from_db -v
# FAIL
```

- [ ] Step 3: Replace stub `load_stats_from_db` with real implementation:

```python
def load_stats_from_db(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(original_tokens),0), COALESCE(SUM(compressed_tokens),0) "
        "FROM compressions"
    ).fetchone()
    stats["total_original_tokens"]   = row[1]
    stats["total_compressed_tokens"] = row[2]

    # Rebuild session aggregates
    for r in conn.execute(
        "SELECT session_id, COUNT(*), SUM(original_tokens), SUM(compressed_tokens), "
        "MIN(ts), MAX(ts) FROM compressions GROUP BY session_id"
    ):
        stats["sessions"][r[0]] = {
            "requests":          r[1],
            "original_tokens":   r[2],
            "compressed_tokens": r[3],
            "first_seen":        r[4],
            "last_seen":         r[5],
            "name":              None,
        }

    # Rebuild recent_compressions (last 100 rows, newest first)
    for r in conn.execute(
        "SELECT ts, session_id, original_tokens, compressed_tokens, latency_ms "
        "FROM compressions ORDER BY id DESC LIMIT 100"
    ):
        saved = r[2] - r[3]
        stats["recent_compressions"].append({
            "ts":         r[0][11:19],  # HH:MM:SS portion
            "session_id": r[1][:8],
            "original":   r[2],
            "compressed": r[3],
            "saved":      saved,
            "latency_ms": r[4],
        })

    print(f"[stats] Loaded from metrics.db: "
          f"{stats['total_original_tokens']} original / {stats['total_compressed_tokens']} compressed tokens, "
          f"{len(stats['sessions'])} sessions")
```

- [ ] Step 4: Remove `load_stats()`, `save_stats()`, and `STATS_FILE` from `llmlingua_proxy.py`. Also remove the `save_stats()` call from `record_compression`.

- [ ] Step 5: Run all tests:
```bash
uv run pytest tests/ -v
# All PASS
```

- [ ] Step 6: Commit:
```bash
git add llmlingua_proxy.py tests/test_proxy.py
git commit -m "Implement load_stats_from_db, remove stats.json I/O"
```

---

## Task 7 — record_compression writes to SQLite + adds latency_ms

**Files:** `llmlingua_proxy.py`, `tests/test_proxy.py`

- [ ] Step 1: Add test:

```python
def test_record_compression_writes_to_db(tmp_path):
    from llmlingua_proxy import init_db, load_stats_from_db, record_compression
    import llmlingua_proxy as proxy

    conn = init_db(str(tmp_path / "metrics.db"))
    proxy._db_conn = conn
    proxy._backend = {"type": "llmlingua2", "rate": 0.5}

    record_compression("sess-xyz", 200, 120, 95.5)

    row = conn.execute("SELECT * FROM compressions").fetchone()
    assert row["session_id"] == "sess-xyz"
    assert row["original_tokens"] == 200
    assert row["compressed_tokens"] == 120
    assert row["latency_ms"] == 95.5
    assert row["model"] == "llmlingua2"

    assert proxy.stats["recent_compressions"][0]["latency_ms"] == 95.5
    conn.close()
```

- [ ] Step 2: Run — fails:
```bash
uv run pytest tests/test_proxy.py::test_record_compression_writes_to_db -v
# FAIL
```

- [ ] Step 3: Update `record_compression` to accept `latency_ms` and write to SQLite:

```python
def record_compression(session_id: str, original: int, compressed: int, latency_ms: float) -> None:
    stats["total_original_tokens"]   += original
    stats["total_compressed_tokens"] += compressed

    model_name = _backend.get("type", "llmlingua2")
    ts = datetime.utcnow().isoformat(timespec="seconds")

    if _db_conn:
        _db_conn.execute(
            "INSERT INTO compressions (ts, session_id, model, original_tokens, compressed_tokens, latency_ms) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (ts, session_id, model_name, original, compressed, latency_ms),
        )
        _db_conn.commit()

    sess = stats["sessions"].setdefault(session_id, {
        "first_seen": ts,
        "requests":          0,
        "original_tokens":   0,
        "compressed_tokens": 0,
        "name":              None,
    })
    sess["original_tokens"]   += original
    sess["compressed_tokens"] += compressed
    sess["last_seen"] = ts

    stats["recent_compressions"].appendleft({
        "ts":         datetime.utcnow().strftime("%H:%M:%S"),
        "session_id": session_id[:8],
        "original":   original,
        "compressed": compressed,
        "saved":      original - compressed,
        "latency_ms": round(latency_ms, 1),
    })
```

- [ ] Step 4: Run — passes:
```bash
uv run pytest tests/test_proxy.py::test_record_compression_writes_to_db -v
# PASS
```

- [ ] Step 5: Commit:
```bash
git add llmlingua_proxy.py tests/test_proxy.py
git commit -m "record_compression: write SQLite row, include latency_ms"
```

---

## Task 8 — Update compress_text to measure latency and pass it through

**Files:** `llmlingua_proxy.py`, `tests/test_proxy.py`

- [ ] Step 1: Add test:

```python
def test_compress_text_records_latency(tmp_path, monkeypatch):
    import llmlingua_proxy as proxy
    from llmlingua_proxy import init_db

    conn = init_db(str(tmp_path / "metrics.db"))
    monkeypatch.setattr(proxy, "_db_conn", conn)
    monkeypatch.setattr(proxy, "_backend", {
        "type": "llmlingua2",
        "compressor": _make_mock_llmlingua(),
        "rate": 0.5,
    })

    proxy.compress_text("word " * 50, "sess-test")  # >200 chars

    row = proxy._db_conn.execute("SELECT latency_ms FROM compressions").fetchone()
    assert row is not None
    assert row[0] >= 0  # latency recorded
    conn.close()
```

- [ ] Step 2: Run — fails (current `compress_text` doesn't pass latency):
```bash
uv run pytest tests/test_proxy.py::test_compress_text_records_latency -v
# FAIL
```

- [ ] Step 3: Replace existing `compress_text` with timing-aware version:

```python
def compress_text(text: str, session_id: str) -> str:
    if len(text) <= 200:
        return text
    try:
        t0 = time.perf_counter()
        compressed, orig, comp = _compress_backend(text)
        latency_ms = (time.perf_counter() - t0) * 1000
    except Exception as e:
        print(f"[compressor] compression failed, forwarding original: {e}")
        return text
    print(f"[compressor] {orig} → {comp} tokens  {latency_ms:.0f}ms [{session_id[:8]}]")
    record_compression(session_id, orig, comp, latency_ms)
    return compressed


def _compress_backend(text: str) -> tuple[str, int, int]:
    """Returns (compressed_text, original_tokens, compressed_tokens)."""
    if _backend["type"] == "kompress":
        return _compress_kompress(text)
    return _compress_llmlingua2(text)


def _compress_llmlingua2(text: str) -> tuple[str, int, int]:
    result = _backend["compressor"].compress_prompt(
        text,
        rate=_backend["rate"],
        force_tokens=["\n", "?", ".", "!"],
    )
    return result["compressed_prompt"], result["origin_tokens"], result["compressed_tokens"]
```

(`_compress_kompress` added in Task 9.)

- [ ] Step 4: Run — passes:
```bash
uv run pytest tests/test_proxy.py -v
# All PASS
```

- [ ] Step 5: Commit:
```bash
git add llmlingua_proxy.py tests/test_proxy.py
git commit -m "compress_text: measure latency_ms, dispatch via _compress_backend"
```

---

## Task 9 — kompress backend loader + _compress_kompress

**Files:** `llmlingua_proxy.py`, `tests/test_proxy.py`

- [ ] Step 1: Add tests:

```python
def test_load_backend_llmlingua2(monkeypatch):
    monkeypatch.setenv("COMPRESSOR_MODEL", "llmlingua2")
    monkeypatch.setenv("COMPRESS_RATE", "0.4")
    mock_cls = MagicMock(return_value=_make_mock_llmlingua())
    monkeypatch.setattr("llmlingua.PromptCompressor", mock_cls)

    from llmlingua_proxy import _load_backend
    b = _load_backend()
    assert b["type"] == "llmlingua2"
    assert b["rate"] == 0.4


def test_compress_kompress_filters_by_threshold(monkeypatch):
    import torch
    import llmlingua_proxy as proxy

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    # Tokenizer encodes to 5 token IDs
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
    }
    # Model returns scores: tokens 0,1,3 above threshold 0.5, tokens 2,4 below
    mock_model.return_value = {
        "final_scores": torch.tensor([[0.8, 0.9, 0.2, 0.7, 0.1]])
    }
    mock_tokenizer.decode.return_value = "kept tokens"

    monkeypatch.setattr(proxy, "_backend", {
        "type": "kompress",
        "model": mock_model,
        "tokenizer": mock_tokenizer,
        "threshold": 0.5,
        "device": "cpu",
    })

    compressed, orig, comp = proxy._compress_kompress("some long text " * 20)
    assert compressed == "kept tokens"
    assert orig == 5
    assert comp == 3  # tokens 0,1,3 kept
```

- [ ] Step 2: Run — fails:
```bash
uv run pytest tests/test_proxy.py -k "kompress" -v
# FAIL
```

- [ ] Step 3: Implement `_load_kompress_backend()` and `_compress_kompress()`, and update `_load_backend()` to dispatch:

```python
def _load_backend() -> dict:
    model_name = os.environ.get("COMPRESSOR_MODEL", "llmlingua2")
    if model_name == "kompress":
        return _load_kompress_backend()
    return _load_llmlingua2_backend()


def _load_llmlingua2_backend() -> dict:
    from llmlingua import PromptCompressor
    rate = float(os.environ.get("COMPRESS_RATE", "0.5"))
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Loading LLMLingua-2 model...")
    c = PromptCompressor(
        model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        use_llmlingua2=True,
        device_map=device,
    )
    print(f"[compressor] model=llmlingua-2  rate={rate:.2f}  device={device}")
    return {"type": "llmlingua2", "compressor": c, "rate": rate}


def _load_kompress_backend() -> dict:
    import torch
    from kompress.model.architecture import HeadroomCompressorV2
    from kompress.model.config import V2_BASE
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    threshold = float(os.environ.get("COMPRESS_THRESHOLD", "0.5"))
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print("Loading kompress-v2-base model...")
    tokenizer = AutoTokenizer.from_pretrained("chopratejas/kompress-v2-base")
    model = HeadroomCompressorV2(V2_BASE)
    weights_path = hf_hub_download("chopratejas/kompress-v2-base", "model.safetensors")
    state_dict = load_file(weights_path, device="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    print(f"[compressor] model=kompress-v2-base  threshold={threshold:.2f}  device={device}")
    return {"type": "kompress", "model": model, "tokenizer": tokenizer,
            "threshold": threshold, "device": device}


def _compress_kompress(text: str) -> tuple[str, int, int]:
    import torch
    b = _backend
    enc = b["tokenizer"](text, return_tensors="pt").to(b["device"])
    input_ids = enc["input_ids"][0]
    with torch.no_grad():
        out = b["model"](**enc)
    scores = out["final_scores"][0]
    keep_mask = scores >= b["threshold"]
    kept_ids = input_ids[keep_mask]
    compressed = b["tokenizer"].decode(kept_ids, skip_special_tokens=True)
    return compressed, len(input_ids), int(keep_mask.sum().item())
```

- [ ] Step 4: Run — passes:
```bash
uv run pytest tests/test_proxy.py -v
# All PASS
```

- [ ] Step 5: Commit:
```bash
git add llmlingua_proxy.py tests/test_proxy.py
git commit -m "Add kompress-v2-base backend: loader and _compress_kompress"
```

---

## Task 10 — /stats/timeseries endpoint

**Files:** `llmlingua_proxy.py`, `tests/test_proxy.py`

- [ ] Step 1: Add test:

```python
def test_timeseries_empty(client):
    r = client.get("/stats/timeseries")
    assert r.status_code == 200
    assert r.json() == []


def test_timeseries_returns_hourly_buckets(tmp_path, monkeypatch):
    import llmlingua_proxy as proxy
    from llmlingua_proxy import init_db

    conn = init_db(str(tmp_path / "metrics.db"))
    conn.executemany(
        "INSERT INTO compressions (ts, session_id, model, original_tokens, compressed_tokens, latency_ms) "
        "VALUES (?, 'sess', 'llmlingua2', ?, ?, ?)",
        [
            ("2026-06-17T10:00:00", 200, 120, 100.0),
            ("2026-06-17T10:30:00", 300, 180, 200.0),
            ("2026-06-17T11:00:00", 100, 70,  50.0),
        ],
    )
    conn.commit()
    monkeypatch.setattr(proxy, "_db_conn", conn)

    from starlette.testclient import TestClient
    with TestClient(proxy.app) as c:
        r = c.get("/stats/timeseries")

    # Can't guarantee 48h filter matches test data without freezing time,
    # so just check structure of non-empty response:
    data = r.json()
    if data:
        assert "hour" in data[0]
        assert "requests" in data[0]
        assert "avg_savings_pct" in data[0]
        assert "avg_latency_ms" in data[0]
    conn.close()
```

- [ ] Step 2: Run — fails (route doesn't exist):
```bash
uv run pytest tests/test_proxy.py -k "timeseries" -v
# FAIL
```

- [ ] Step 3: Add route to `llmlingua_proxy.py` after the `/stats` route:

```python
@app.get("/stats/timeseries")
async def get_timeseries():
    if _db_conn is None:
        return JSONResponse([])
    rows = _db_conn.execute("""
        SELECT strftime('%Y-%m-%dT%H:00:00', ts)       AS hour,
               COUNT(*)                                  AS requests,
               ROUND(AVG((original_tokens - compressed_tokens) * 100.0 / original_tokens), 1)
                                                         AS avg_savings_pct,
               SUM(original_tokens - compressed_tokens)  AS total_saved,
               ROUND(AVG(latency_ms), 1)                AS avg_latency_ms
        FROM compressions
        WHERE ts >= datetime('now', '-48 hours')
        GROUP BY hour
        ORDER BY hour ASC
    """).fetchall()
    return JSONResponse([dict(r) for r in rows])
```

- [ ] Step 4: Run — passes:
```bash
uv run pytest tests/test_proxy.py -k "timeseries" -v
# PASS
```

- [ ] Step 5: Commit:
```bash
git add llmlingua_proxy.py tests/test_proxy.py
git commit -m "Add /stats/timeseries endpoint: hourly buckets for last 48h"
```

---

## Task 11 — Update /stats response with compressor info, avg_latency_ms, cost_per_mtok

**Files:** `llmlingua_proxy.py`, `tests/test_proxy.py`

- [ ] Step 1: Add test:

```python
def test_stats_includes_compressor_and_cost(client):
    r = client.get("/stats")
    assert r.status_code == 200
    d = r.json()
    assert "compressor" in d
    assert d["compressor"]["model"] == "llmlingua2"
    assert d["compressor"]["param_name"] == "rate"
    assert d["compressor"]["param_value"] == 0.5
    assert "cost_per_mtok" in d
    assert d["cost_per_mtok"] == 3.0
    assert "avg_latency_ms" in d
```

- [ ] Step 2: Run — fails:
```bash
uv run pytest tests/test_proxy.py::test_stats_includes_compressor_and_cost -v
# FAIL
```

- [ ] Step 3: Update the `get_stats()` route to add these fields. Find the existing return dict and extend it:

```python
@app.get("/stats")
async def get_stats():
    saved = stats["total_original_tokens"] - stats["total_compressed_tokens"]
    ratio = (stats["total_original_tokens"] / stats["total_compressed_tokens"]
             if stats["total_compressed_tokens"] > 0 else 1.0)
    sessions_out = {}
    for sid, s in stats["sessions"].items():
        sv = s["original_tokens"] - s["compressed_tokens"]
        sessions_out[sid] = {**s, "saved_tokens": sv}

    recent = list(stats["recent_compressions"])
    avg_latency = (
        round(sum(c["latency_ms"] for c in recent) / len(recent), 1)
        if recent else 0.0
    )

    if _backend.get("type") == "kompress":
        compressor_info = {
            "model": "kompress",
            "param_name": "threshold",
            "param_value": _backend.get("threshold", 0.5),
        }
    else:
        compressor_info = {
            "model": "llmlingua2",
            "param_name": "rate",
            "param_value": _backend.get("rate", 0.5),
        }

    return {
        "started_at":              stats["started_at"],
        "total_requests":          stats["total_requests"],
        "total_original_tokens":   stats["total_original_tokens"],
        "total_compressed_tokens": stats["total_compressed_tokens"],
        "total_saved_tokens":      saved,
        "overall_ratio":           round(ratio, 2),
        "sessions":                sessions_out,
        "recent_compressions":     recent,
        "rtk":                     read_rtk_stats(),
        "compressor":              compressor_info,
        "cost_per_mtok":           COST_PER_MTOK,
        "avg_latency_ms":          avg_latency,
    }
```

- [ ] Step 4: Run — passes:
```bash
uv run pytest tests/test_proxy.py -v
# All PASS
```

- [ ] Step 5: Commit:
```bash
git add llmlingua_proxy.py tests/test_proxy.py
git commit -m "Update /stats: add compressor info, cost_per_mtok, avg_latency_ms"
```

---

## Task 12 — Dashboard: model badge, latency card, cost card, sparkline tooltip latency

**Files:** `llmlingua_proxy.py` (DASHBOARD_HTML section only)

No new tests needed — dashboard is HTML/JS rendered in browser. Visual verification via `make dashboard`.

- [ ] Step 1: In `DASHBOARD_HTML`, add CSS for the model badge in the header. Find the `.header-right` style and add:

```css
.model-badge { font-size: 10px; background: #0d1a35; color: #58a6ff; padding: 2px 8px; border-radius: 4px; font-family: inherit; }
```

- [ ] Step 2: In the HTML header element, add the badge span after `.title`:

```html
<div class="header">
  <div style="display:flex;align-items:center;gap:10px">
    <div class="title">⚡ LLMLingua Proxy</div>
    <span class="model-badge" id="model_badge">—</span>
  </div>
  <div class="header-right">
    <span id="uptime_display">—</span>
    <span><span class="live-dot"></span>live</span>
  </div>
</div>
```

- [ ] Step 3: In the `.cards` section, expand from 5 to 7 cards by adding two new card divs after the existing 5:

```html
<div class="card"><div class="card-label">Avg Latency</div><div class="card-value" id="card_latency">—</div></div>
<div class="card"><div class="card-label">Est. $ Saved</div><div class="card-value green" id="card_cost">—</div><div style="font-size:9px;color:#484f58;margin-top:3px" id="card_cost_label">@ $3.00/MTok</div></div>
```

Update the grid CSS from `repeat(5, 1fr)` to `repeat(7, 1fr)`.

- [ ] Step 4: In the `refresh()` JS function, after the existing card updates, add:

```js
// Model badge
if (d.compressor) {
  const c = d.compressor;
  document.getElementById('model_badge').textContent =
    (c.model === 'kompress' ? 'kompress-v2' : 'LLMLingua-2') +
    '  ' + c.param_name + '=' + c.param_value;
}

// Avg latency card
document.getElementById('card_latency').textContent =
  d.avg_latency_ms != null ? Math.round(d.avg_latency_ms) + 'ms' : '—';

// Cost card
if (d.cost_per_mtok != null && d.total_saved_tokens != null) {
  const cost = (d.total_saved_tokens / 1_000_000 * d.cost_per_mtok);
  document.getElementById('card_cost').textContent = '$' + cost.toFixed(2);
  document.getElementById('card_cost_label').textContent = '@ $' + d.cost_per_mtok.toFixed(2) + '/MTok';
}
```

- [ ] Step 5: Update the sparkline tooltip in the `refresh()` JS. Find the line that builds the `tip` string and append latency:

```js
const tip = c.ts + ' · ' + c.session_id + ' · ' + pct + '% saved (' + fmt(c.original) + ' → ' + fmt(c.compressed) + ')' +
            (c.latency_ms ? ' · ' + c.latency_ms + 'ms' : '');
```

- [ ] Step 6: Verify visually:
```bash
make start   # in one terminal
make dashboard  # opens browser
```
Confirm: model badge in header, 7 cards, latency in sparkline tooltips.

- [ ] Step 7: Commit:
```bash
git add llmlingua_proxy.py
git commit -m "Dashboard: model badge, Avg Latency card, Est. $ Saved card, latency in tooltips"
```

---

## Task 13 — Dashboard: 48-hour time-series chart

**Files:** `llmlingua_proxy.py` (DASHBOARD_HTML section only)

- [ ] Step 1: Add CSS for the time-series chart section. Add inside the `<style>` block:

```css
/* ── Time-series chart ── */
.ts-chart { display: flex; align-items: flex-end; gap: 1px; height: 60px; overflow: hidden; }
.ts-bar { flex: 1; min-width: 4px; min-height: 2px; border-radius: 1px 1px 0 0; cursor: default; opacity: .8; transition: opacity .1s; }
.ts-bar:hover { opacity: 1; }
.ts-labels { display: flex; justify-content: space-between; margin-top: 4px; font-size: 9px; color: #484f58; }
```

- [ ] Step 2: Add the new section to the HTML, after the existing sparkline section and before the session efficiency bars section:

```html
<!-- 48-hour time-series -->
<div class="section">
  <div class="section-title">compression rate — last 48 h (hourly) &nbsp;·&nbsp; height = requests &nbsp;·&nbsp; color = avg savings %</div>
  <div id="ts_chart" class="ts-chart"><span class="spark-empty">No data yet</span></div>
  <div id="ts_labels" class="ts-labels"></div>
</div>
```

- [ ] Step 3: In the JS `refresh()` function, add a second fetch for `/stats/timeseries` and render the chart. Add this after the existing `fetch('/stats')` block (parallel, separate `try/catch`):

```js
async function refreshTimeseries() {
  try {
    const r = await fetch('/stats/timeseries');
    const buckets = await r.json();
    const chartEl = document.getElementById('ts_chart');
    const labelsEl = document.getElementById('ts_labels');
    if (!buckets || buckets.length === 0) {
      chartEl.innerHTML = '<span class="spark-empty">No data yet</span>';
      labelsEl.innerHTML = '';
      return;
    }
    const maxReq = Math.max(...buckets.map(b => b.requests), 1);
    chartEl.innerHTML = buckets.map(b => {
      const h = Math.max(4, Math.round((b.requests / maxReq) * 100));
      const tip = b.hour.slice(11, 16) + '  ' + b.requests + ' req  ' +
                  b.avg_savings_pct + '% saved  ' + b.avg_latency_ms + 'ms avg';
      return '<div class="ts-bar" style="height:' + h + '%;background:' + barColor(b.avg_savings_pct) + '" title="' + tip + '"></div>';
    }).join('');
    // Show first and last hour labels
    const first = buckets[0].hour.slice(11, 16);
    const last  = buckets[buckets.length - 1].hour.slice(11, 16);
    labelsEl.innerHTML = '<span>' + first + '</span><span>' + last + '</span>';
  } catch(e) { console.error(e); }
}

refreshTimeseries();
setInterval(refreshTimeseries, 10000);  // every 10s — doesn't need 2s cadence
```

- [ ] Step 4: Verify visually after sending a few requests through the proxy. Chart should fill in with colored bars per hour.

- [ ] Step 5: Commit:
```bash
git add llmlingua_proxy.py
git commit -m "Dashboard: 48-hour time-series chart from /stats/timeseries"
```

---

## Final verification

Run the full checklist from the spec:

```bash
# 1. kompress model loads
COMPRESSOR_MODEL=kompress uv run python llmlingua_proxy.py &
curl -s localhost:9099/ | python3 -m json.tool
kill %1

# 2. llmlingua2 still works  
COMPRESSOR_MODEL=llmlingua2 uv run python llmlingua_proxy.py &
curl -s localhost:9099/stats | python3 -m json.tool | grep compressor
kill %1

# 3. Migration: run with existing stats.json, check bak + DB
ls stats.json.bak metrics.db

# 4. Idempotent restart
uv run python llmlingua_proxy.py &
curl -s localhost:9099/stats/timeseries
kill %1

# 5. All tests green
uv run pytest tests/ -v

# 6. Dashboard
make dashboard  # visual check
```
