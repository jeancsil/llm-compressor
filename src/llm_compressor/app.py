"""FastAPI app instance + lifespan (startup/shutdown wiring).

Extracted from proxy.py in Task 14. Owns no globals of its own: every
mutable name touched here (`db._db_conn`, `compression._cache`,
`backends.backend`) is owned by its respective module and written
module-qualified, per scratchpad/split-audit.md's Step 3 contract — the
same contract Task 13's proxy.py forwarding shim already relies on.

`_lf_tracer` is looked up via a local `import langfuse_tracer` inside
`lifespan()` rather than a module-level `from langfuse_tracer import
tracer as _lf_tracer`. This matters: tests/conftest.py's `client` fixture
deletes both "proxy" and "langfuse_tracer" from sys.modules and reimports
them fresh before every test (to get a clean, un-flushed tracer), but this
module (app.py) is imported once and never reloaded. A module-level binding
here would freeze onto the *first* test's tracer instance forever; a local
import resolves `sys.modules["langfuse_tracer"]` fresh on every call, so it
always sees whichever tracer instance the current test's `import proxy`
just (re)created.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from llm_compressor import backends
from llm_compressor import compression
from llm_compressor import db
from llm_compressor.compression import CACHE_MAX_ROWS, CACHE_MEM_SIZE


@asynccontextmanager
async def lifespan(app: FastAPI):
    from llm_compressor import langfuse_tracer  # local: see module docstring — avoids stale-tracer binding

    db._migrate_db_location()
    db._db_conn = db.init_db(str(db.DB_PATH))
    compression._cache = compression.CompressionCache(
        db._db_conn, max_mem=CACHE_MEM_SIZE, max_rows=CACHE_MAX_ROWS
    )
    db.migrate_from_json(db._db_conn)
    db.recover_stats_from_backup(db._db_conn)
    db.load_stats_from_db(db._db_conn)
    backends.backend = backends.load_backend()
    langfuse_tracer.tracer.init()
    # print handled inside tracer.init()
    yield
    # Flush langfuse traces before shutdown
    if langfuse_tracer.tracer.enabled and langfuse_tracer.tracer._client:
        try:
            langfuse_tracer.tracer._client.flush()
        except Exception:
            pass
    # Release model references before process exit to avoid MPS semaphore leaks
    backends.backend = None
    import gc

    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.synchronize()
            torch.mps.empty_cache()
    except Exception:
        pass
    if db._db_conn:
        db._db_conn.close()


app = FastAPI(lifespan=lifespan)
