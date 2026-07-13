"""Compression backend loading, selection, and dual-mode state.

Owns the process-wide, runtime-reassigned globals `backend`, `backend_loading`,
`backend_user`, `backend_system`, `dual_mode`, `dual_model_system`, and
`dual_model_user`, plus the read-only `KNOWN_MODELS`, `DUAL_SUBMODELS`, and
`LLMLINGUA2_MODELS`. Everywhere else in the codebase (including proxy.py, which
re-exports these for backward-compatible test patching via a forwarding shim)
must read/write the mutable globals as `backends.backend`, `backends.dual_mode`,
etc. -- never `from backends import backend` -- so that `lifespan`'s
reassignment at startup, request-handler updates, and `monkeypatch.setattr` in
tests are visible to every reader. See the `_ProxyModule` shim + `_FORWARD`
table in proxy.py for the mechanism that keeps `monkeypatch.setattr(proxy,
"backend", ...)` / `monkeypatch.setattr(proxy, "_load_backend", ...)` (the
pre-split test idiom) working without every test needing to be rewritten to
target `backends.` directly. Note `_load_backend` was a plain alias for
`load_backend` in proxy.py pre-split; it is not recreated here as a real name
-- it only exists as a forwarding key in proxy.py's `_FORWARD` table that
points at `load_backend`.
"""

import os

from llmlingua import PromptCompressor

import db

# Module-level globals populated by lifespan() / the admin endpoints in proxy.py.
backend = None
backend_loading = None  # set to model name while async load is in progress
backend_user = None  # kompress instance in dual mode
backend_system = None  # llmlingua2-large instance in dual mode
dual_mode = False
dual_model_system = "llmlingua2-large"  # persisted in meta table
dual_model_user = "kompress"  # persisted in meta table

KNOWN_MODELS = ("llmlingua2", "llmlingua2-large", "kompress", "dual")
DUAL_SUBMODELS = ("llmlingua2", "llmlingua2-large", "kompress")

LLMLINGUA2_MODELS = {
    "llmlingua2": "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    "llmlingua2-large": "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
}


def _load_llmlingua2_backend(backend_key: str | None = None) -> dict:
    """Load the LLMLingua-2 PromptCompressor and return a backend dict."""
    import logging as _logging

    import transformers as _tf

    rate = float(os.environ.get("COMPRESS_RATE", "0.5"))
    if backend_key is None:
        backend_key = os.environ.get("COMPRESSOR_MODEL", "llmlingua2")
    model_name = LLMLINGUA2_MODELS.get(backend_key, LLMLINGUA2_MODELS["llmlingua2"])
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading LLMLingua-2 model ({backend_key}: {model_name})...")
    _tf.logging.set_verbosity_error()
    _hf_log = _logging.getLogger("huggingface_hub")
    _prev_hf = _hf_log.level
    _hf_log.setLevel(_logging.ERROR)
    try:
        c = PromptCompressor(
            model_name=model_name,
            use_llmlingua2=True,
            device_map=device,
        )
    finally:
        _tf.logging.set_verbosity_warning()
        _hf_log.setLevel(_prev_hf)
    print(f"Model ready. (device={device})")
    return {"type": backend_key, "backend_key": backend_key, "compressor": c, "rate": rate}


def _load_kompress_backend() -> dict:
    """Load chopratejas/kompress-v2-base via headroom-ai[ml].

    Auto mode tries ONNX CPU first (not in public HF repo, will skip) then
    falls back to PyTorch on MPS/CPU using model.safetensors (~600 MB).
    """
    try:
        from headroom.transforms.kompress_compressor import KompressCompressor, KompressConfig
    except ImportError:
        raise RuntimeError("headroom-ai[ml] is not installed. Run: uv add 'headroom-ai[ml]'")
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    threshold = float(os.environ.get("COMPRESS_THRESHOLD", "0.5"))
    print(f"Loading kompress-v2-base (device={device}, threshold={threshold})...")
    config = KompressConfig(device=device, score_threshold=threshold)
    compressor = KompressCompressor(config=config)
    import transformers as _tf

    _prev_level = _tf.logging.get_verbosity()
    _tf.logging.set_verbosity_error()
    compressor.preload()
    _tf.logging.set_verbosity(_prev_level)
    print("kompress-v2-base ready.")
    return {"type": "kompress", "compressor": compressor, "threshold": threshold}


def _load_single_backend(model_name: str) -> dict:
    """Load any non-dual backend by name."""
    if model_name == "kompress":
        return _load_kompress_backend()
    return _load_llmlingua2_backend(backend_key=model_name)


def _load_dual_backend() -> dict:
    """Load both sub-backends and set dual-mode globals.

    Uses the module-level dual_model_system / dual_model_user which are
    persisted in the meta table and configurable at runtime via
    /admin/set-dual-models.
    """
    global backend_user, backend_system, dual_mode
    sys_m = dual_model_system
    usr_m = dual_model_user
    print(f"Loading dual mode: {usr_m} (user) + {sys_m} (system)...")
    backend_system = _load_single_backend(sys_m)
    backend_user = _load_single_backend(usr_m)
    dual_mode = True
    print("Dual mode ready.")
    return {"type": "dual", "model_user": usr_m, "model_system": sys_m}


def load_backend() -> dict:
    """Dispatch to the configured backend loader.

    Resolution order:
    1. DB meta table keys 'current_model', 'dual_model_system', 'dual_model_user'
    2. COMPRESSOR_MODEL environment variable
    3. Defaults: llmlingua2 / llmlingua2-large / kompress
    """
    global dual_model_system, dual_model_user
    model_name = os.environ.get("COMPRESSOR_MODEL", "llmlingua2")
    try:
        if db._db_conn is not None:
            for key, default in (
                ("current_model", None),
                ("dual_model_system", None),
                ("dual_model_user", None),
            ):
                row = db._db_conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
                if row:
                    if key == "current_model":
                        model_name = row[0]
                    elif key == "dual_model_system":
                        dual_model_system = row[0]
                    elif key == "dual_model_user":
                        dual_model_user = row[0]
    except Exception:
        pass  # DB not available; fall back to env var / module defaults
    if model_name == "dual":
        return _load_dual_backend()
    return _load_single_backend(model_name)


def _pick_backend(role: str) -> dict | None:
    if dual_mode and backend_user is not None and backend_system is not None:
        return backend_system if role == "system" else backend_user
    return backend
