"""Back-compat shim (Task 14): proxy.py is now a thin re-export layer.

Real code lives in db.py / backends.py / compression.py / stats.py /
sessions.py / templates.py / app.py / routes.py. This module exists so
`import proxy` / `proxy.app` / `from proxy import X` keep working for the
test suite, `cli.py`'s entrypoint, and `python proxy.py` — which is the
actual server entrypoint (see Makefile's `make start` and `cli.py`'s `wrap`
subcommand, both of which launch `python proxy.py` as a subprocess).
"""

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import sys as _sys
import types as _types

import httpx  # noqa: F401  (re-export: tests patch proxy.httpx.AsyncClient)
import uvicorn

import backends
import compression
import db
import routes  # noqa: E402,F401  (registers @app.* endpoints on `app`)
import stats as _stats
from app import app  # noqa: F401
from compression import (  # re-export
    _CHUNK_MAX_CHARS,
    CHUNK_MAX_TOKENS,
    CompressionCache,
    _cache_key,
    _char_split,
    _compress_kompress,
    _compress_llmlingua2,
    _count_tokens,
    _split_into_segments,
    chunk_text,
    compress_messages,
    compress_system_field,
    compress_text,
)

# Plain re-exports: real function objects, never monkeypatched by name in the
# test suite (tests call them directly), so a static import carries no
# staleness risk. `_db_conn`/`DB_PATH`/`migrate_from_json`/
# `recover_stats_from_backup`/`_migrate_db_location` (db.py), `_cache`/
# `_compress_with`/`compress_backend` (compression.py), and `_rtk_db_path`
# (stats.py) are deliberately NOT imported here — see `_ProxyModule` below.
# `compress_backend` is monkeypatched via `proxy.compress_backend` in
# test_coverage.py::test_play_compress_backend_raises (a multi-line
# `monkeypatch.setattr(...)` call easy to miss by grep), and `_rtk_db_path` is
# monkeypatched the same way in test_coverage.py's rtk tests, so both must
# forward through the shim like `_cache`/`_compress_with` rather than be a
# plain static re-export.
from db import init_db, load_stats_from_db  # re-export
from routes import build_headers  # noqa: E402,F401  (proxy.build_headers, plain re-export)
from sessions import record_compression, record_request  # re-export
from stats import _cache_stats, read_rtk_stats, stats  # re-export

# Declares the compression.py/sessions.py/stats.py/routes.py re-exports
# above as intentional public surface so ruff's F401 (unused-import) doesn't
# flag them. As of Task 14 (routes.py now owns get_stats()/proxy_messages()/
# etc.), *none* of these names has a bare call site left inside proxy.py
# itself — every one is exposed purely because tests call it directly as
# `proxy.<name>` (e.g. `proxy.init_db(...)`, `proxy.record_compression(...)`,
# `proxy.build_headers(request)`, `proxy.stats[...]`). None of them is ever
# reassigned/monkeypatched by name in the test suite, so a plain static
# import carries no staleness risk — see `_ProxyModule` below for the 17
# names that *are* monkeypatched and therefore need forwarding instead.
__all__ = [
    "CHUNK_MAX_TOKENS",
    "_CHUNK_MAX_CHARS",
    "CompressionCache",
    "_cache_key",
    "_char_split",
    "_compress_kompress",
    "_compress_llmlingua2",
    "_count_tokens",
    "_split_into_segments",
    "chunk_text",
    "compress_messages",
    "compress_system_field",
    "compress_text",
    "_cache_stats",
    "read_rtk_stats",
    "record_compression",
    "record_request",
    "init_db",
    "load_stats_from_db",
    "stats",
    "build_headers",
]

# ---------------------------------------------------------------------------
# proxy.py forwarding shim (Task 13, Steps 1 & 4)
# ---------------------------------------------------------------------------
# db.py owns `_db_conn` (mutable, reassigned by lifespan() on every request-cycle
# reset in tests) and `DB_PATH`, plus `migrate_from_json` / `recover_stats_from_backup`
# / `_migrate_db_location`. backends.py owns the backend-selection globals
# (`backend`, `backend_loading`, `backend_user`, `backend_system`, `dual_mode`,
# `dual_model_system`, `dual_model_user`) and the loader functions (`load_backend`,
# `_load_dual_backend`, `_load_kompress_backend`, `_load_llmlingua2_backend`,
# `_load_single_backend`, `_pick_backend`). compression.py owns the
# runtime-reassigned cache global `_cache` and the compressor-dispatch function
# `_compress_with`/`compress_backend`. stats.py owns `_rtk_db_path` (its only
# same-module caller, `read_rtk_stats`, reads it as `stats._rtk_db_path()` so
# the reassignment is visible there too). All of these are monkeypatched away
# via `monkeypatch.setattr(proxy, "<name>", ...)` somewhere in the test suite
# (conftest.py's autouse fixture, or individual tests in test_proxy.py /
# test_coverage.py / test_cache.py). Every real reader of these names elsewhere
# in the codebase (lifespan(), the /play and /admin route handlers, db.py's,
# backends.py's, compression.py's, and stats.py's own functions) is
# module-qualified (`db._db_conn`, `backends.backend`, `compression._cache`,
# `stats._rtk_db_path`, ...) per scratchpad/split-audit.md's Step 3 contract.
# This class makes `proxy.<name>` reads AND `monkeypatch.setattr(proxy,
# "<name>", ...)` writes forward to the single owning attribute on `db` /
# `backends` / `compression` / `stats`, so the pre-split test idiom keeps
# working unmodified instead of silently patching a stale, disconnected copy
# on proxy itself.
#
# `_load_backend` is a special case: pre-split, proxy.py had a plain alias
# `_load_backend = load_backend`. That alias is not recreated as a real name in
# backends.py — it only exists here as a forwarding key pointing at
# `backends.load_backend`, so `monkeypatch.setattr(proxy, "_load_backend", ...)`
# still works even though nothing in backends.py is ever bound to that name.
_FORWARD = {
    "_db_conn": (db, "_db_conn"),
    "DB_PATH": (db, "DB_PATH"),
    "migrate_from_json": (db, "migrate_from_json"),
    "recover_stats_from_backup": (db, "recover_stats_from_backup"),
    "_migrate_db_location": (db, "_migrate_db_location"),
    "backend": (backends, "backend"),
    "backend_loading": (backends, "backend_loading"),
    "backend_user": (backends, "backend_user"),
    "backend_system": (backends, "backend_system"),
    "dual_mode": (backends, "dual_mode"),
    "dual_model_system": (backends, "dual_model_system"),
    "dual_model_user": (backends, "dual_model_user"),
    "_load_backend": (backends, "load_backend"),
    "load_backend": (backends, "load_backend"),
    "_load_dual_backend": (backends, "_load_dual_backend"),
    "_load_kompress_backend": (backends, "_load_kompress_backend"),
    "_load_llmlingua2_backend": (backends, "_load_llmlingua2_backend"),
    "_load_single_backend": (backends, "_load_single_backend"),
    "_pick_backend": (backends, "_pick_backend"),
    "_cache": (compression, "_cache"),
    "_compress_with": (compression, "_compress_with"),
    "compress_backend": (compression, "compress_backend"),
    "_rtk_db_path": (_stats, "_rtk_db_path"),
}


class _ProxyModule(_types.ModuleType):
    def __getattr__(self, name):
        target = _FORWARD.get(name)
        if target is not None:
            module, attr = target
            return getattr(module, attr)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        target = _FORWARD.get(name)
        if target is not None:
            module, attr = target
            setattr(module, attr, value)
            return
        super().__setattr__(name, value)


_sys.modules[__name__].__class__ = _ProxyModule


if __name__ == "__main__":  # pragma: no cover
    uvicorn.run(app, host="127.0.0.1", port=9099)
