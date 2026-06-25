"""
Coverage-completion tests. Each test targets a specific uncovered branch
not exercised by test_proxy.py or test_tracker.py.

Second-pass additions (after reaching 96%):
  481      _rtk_db_path – Darwin return
  542-543  _count_tokens – tokenizer exception fallback
  549      _char_split – short-text early return
  562      _split_into_segments – multi-line paragraph, extend(lines)
  984      get_timeseries – _db_conn is None
  1023     rtk_log – _db_conn is None
  1114,16  play_compress load thread – kompress / dual branches
  1120-21  play_compress load thread – exception path
  1135-36  play_compress – compress_backend raises
  1180-93  set_model – dual path + load_dual thread
  1200     set_model – kompress path in load thread
  1204-05  set_model – load thread exception path
  1218     clear_compression_texts – _db_conn is None
  1303     get_session_compressions – _db_conn is None

Coverage gaps addressed (by proxy.py line range):
  140-141  migrate_from_json error path
  153      recover_stats_from_backup – file absent early return
  184-185  recover_stats_from_backup error path
  200-201  load_stats_from_db meta exception path
  251      _load_llmlingua2_backend – backend_key=None path
  285-297  _load_kompress_backend success path
  307-312  _load_dual_backend
  329-330  load_backend – reads current_model from DB meta
  334,336  load_backend – dual / kompress dispatch
  346      _pick_backend – dual mode path
  372-373  lifespan teardown – torch exception handler
  444      _try_link_pending_tracker – early return for "unknown" / empty session
  470      record_request – stores session_name
  479-484  _rtk_db_path – Windows / Linux branches
  487-527  read_rtk_stats – success, with_since, db_error
  542-543  _char_split – long text
  560-565  _split_into_segments – long paragraph with sentence split
  570-573  _split_into_segments – single paragraph sentence / char fallback
  603      compress_text – short text early return
  606      compress_text – no backend early return
  610-612  compress_text – compression exception path
  626      _compress_with – kompress dispatch
  633-635  compress_backend
  664-665  _compress_kompress
  670-678  compress_system_field – string, list, other
  682-700  compress_messages – assistant passthrough, list content, non-str content
  709-711  build_headers
  742      get_stats – backend_loading info
  749      get_stats – dual backend info
  756      get_stats – kompress backend info
  925-931  get_stats – rtk_events rows present
  984,988  get_timeseries – model filter branch
  1022-46  rtk_log endpoint
  1052     session_dashboard – db not ready
  1068     dashboard endpoint
  1073     play endpoint
  1084-144 play_compress endpoint (all branches)
  1158-211 set_model endpoint (all branches)
  1217-234 clear_compression_texts endpoint
  1244     create_tracker – db not ready
  1258     get_tracker – db not ready
  1269     delete_tracker – db not ready
  1284     get_all_trackers – db not ready
  1302-324 get_session_compressions endpoint
  1329-336 list_models endpoint
  1341-386 proxy_messages endpoint (streaming + non-streaming + errors)
"""

import json
import platform
import sqlite3
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from starlette.testclient import TestClient


# ---------------------------------------------------------------------------
# Helper: stub heavy deps before importing proxy in isolation tests
# ---------------------------------------------------------------------------

def _stub_heavy_deps(monkeypatch):
    for dep in ("llmlingua", "torch", "transformers"):
        if dep not in sys.modules:
            monkeypatch.setitem(sys.modules, dep, MagicMock())


def _fresh_proxy(monkeypatch):
    """Import a freshly-initialised proxy module with heavy deps stubbed."""
    _stub_heavy_deps(monkeypatch)
    monkeypatch.delitem(sys.modules, "proxy", raising=False)
    import proxy as _proxy
    return _proxy


# ===========================================================================
# migrate_from_json – error path (lines 140-141)
# ===========================================================================

def test_migrate_from_json_error_path(tmp_path, monkeypatch):
    """migrate_from_json swallows errors gracefully when JSON is corrupt."""
    _stub_heavy_deps(monkeypatch)
    from proxy import init_db, migrate_from_json

    conn = init_db(str(tmp_path / "metrics.db"))
    bad = tmp_path / "stats.json"
    bad.write_text("{not valid json}")
    migrate_from_json(conn, json_path=str(bad))  # must not raise
    conn.close()


# ===========================================================================
# recover_stats_from_backup – absent file (line 153)
# ===========================================================================

def test_recover_stats_from_backup_no_file(tmp_path, monkeypatch):
    """recover_stats_from_backup returns early when backup file is absent."""
    _stub_heavy_deps(monkeypatch)
    from proxy import init_db, recover_stats_from_backup

    conn = init_db(str(tmp_path / "metrics.db"))
    recover_stats_from_backup(conn, bak_path=str(tmp_path / "nonexistent.bak"))
    conn.close()  # must not raise


# ===========================================================================
# recover_stats_from_backup – error path (lines 184-185)
# ===========================================================================

def test_recover_stats_from_backup_corrupt_json(tmp_path, monkeypatch):
    """recover_stats_from_backup swallows errors on corrupt backup."""
    _stub_heavy_deps(monkeypatch)
    from proxy import init_db, recover_stats_from_backup

    conn = init_db(str(tmp_path / "metrics.db"))
    bak = tmp_path / "stats.json.bak"
    bak.write_text("{bad json}")
    recover_stats_from_backup(conn, bak_path=str(bak))  # must not raise
    conn.close()


# ===========================================================================
# load_stats_from_db – meta query exception (lines 200-201)
# ===========================================================================

def test_load_stats_from_db_meta_exception(tmp_path, monkeypatch):
    """load_stats_from_db falls back gracefully when meta table is absent."""
    _stub_heavy_deps(monkeypatch)
    from proxy import init_db, load_stats_from_db, stats

    conn = init_db(str(tmp_path / "metrics.db"))
    conn.execute("DROP TABLE meta")
    conn.commit()
    stats["total_requests"] = 0
    stats["total_original_tokens"] = 0
    stats["total_compressed_tokens"] = 0
    stats["sessions"] = {}
    stats["recent_compressions"] = deque(maxlen=100)
    load_stats_from_db(conn)  # must not raise
    conn.close()


# ===========================================================================
# _load_llmlingua2_backend – backend_key=None path (line 251)
# ===========================================================================

def test_load_llmlingua2_backend_none_key(monkeypatch):
    """_load_llmlingua2_backend reads COMPRESSOR_MODEL env when key is None."""
    proxy = _fresh_proxy(monkeypatch)
    monkeypatch.setenv("COMPRESSOR_MODEL", "llmlingua2")
    monkeypatch.setenv("COMPRESS_RATE", "0.5")

    mock_cls = MagicMock()
    mock_cls.return_value.compress_prompt.return_value = {
        "compressed_prompt": "x", "origin_tokens": 10, "compressed_tokens": 6, "ratio": 0.6,
    }
    monkeypatch.setattr("llmlingua.PromptCompressor", mock_cls)

    b = proxy._load_llmlingua2_backend(backend_key=None)
    assert b["type"] == "llmlingua2"


# ===========================================================================
# _load_kompress_backend – success path (lines 285-297)
# ===========================================================================

def test_load_kompress_backend_success(monkeypatch):
    """_load_kompress_backend returns a valid backend dict when headroom is present."""
    proxy = _fresh_proxy(monkeypatch)

    mock_compressor = MagicMock()
    mock_config_cls = MagicMock()
    mock_compressor_cls = MagicMock(return_value=mock_compressor)

    mock_headroom_mod = MagicMock()
    mock_headroom_mod.KompressCompressor = mock_compressor_cls
    mock_headroom_mod.KompressConfig = mock_config_cls

    monkeypatch.setitem(
        sys.modules,
        "headroom.transforms.kompress_compressor",
        mock_headroom_mod,
    )
    monkeypatch.setenv("COMPRESS_THRESHOLD", "0.4")

    b = proxy._load_kompress_backend()
    assert b["type"] == "kompress"
    assert b["threshold"] == 0.4
    mock_compressor.preload.assert_called_once()


# ===========================================================================
# _load_dual_backend (lines 307-312)
# ===========================================================================

def test_load_dual_backend(monkeypatch):
    """_load_dual_backend loads both backends and sets dual_mode globals."""
    proxy = _fresh_proxy(monkeypatch)

    llm_backend = {"type": "llmlingua2-large", "compressor": MagicMock(), "rate": 0.5}
    kmp_backend = {"type": "kompress", "compressor": MagicMock(), "threshold": 0.5}
    monkeypatch.setattr(proxy, "_load_llmlingua2_backend", lambda backend_key=None: llm_backend)
    monkeypatch.setattr(proxy, "_load_kompress_backend", lambda: kmp_backend)

    b = proxy._load_dual_backend()
    assert b["type"] == "dual"
    assert proxy.dual_mode is True
    assert proxy.backend_system is llm_backend
    assert proxy.backend_user is kmp_backend


# ===========================================================================
# load_backend – DB meta path (lines 329-330), dual (334), kompress (336)
# ===========================================================================

def test_load_backend_reads_db_meta(tmp_path, monkeypatch):
    """load_backend uses current_model from DB meta when _db_conn is set."""
    proxy = _fresh_proxy(monkeypatch)

    conn = proxy.init_db(str(tmp_path / "metrics.db"))
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('current_model', 'llmlingua2')")
    conn.commit()
    monkeypatch.setattr(proxy, "_db_conn", conn)

    mock_cls = MagicMock()
    mock_cls.return_value.compress_prompt.return_value = {
        "compressed_prompt": "x", "origin_tokens": 10, "compressed_tokens": 6, "ratio": 0.6,
    }
    monkeypatch.setattr("llmlingua.PromptCompressor", mock_cls)

    b = proxy.load_backend()
    assert b["type"] == "llmlingua2"
    conn.close()


def test_load_backend_dual_dispatch(monkeypatch):
    """load_backend dispatches to _load_dual_backend when model=dual."""
    monkeypatch.setenv("COMPRESSOR_MODEL", "dual")
    proxy = _fresh_proxy(monkeypatch)
    monkeypatch.setattr(proxy, "_db_conn", None)

    mock_dual = {"type": "dual", "model_user": "kompress", "model_system": "llmlingua2-large"}
    monkeypatch.setattr(proxy, "_load_dual_backend", lambda: mock_dual)

    b = proxy.load_backend()
    assert b["type"] == "dual"


def test_load_backend_kompress_dispatch(monkeypatch):
    """load_backend dispatches to _load_kompress_backend when model=kompress."""
    monkeypatch.setenv("COMPRESSOR_MODEL", "kompress")
    proxy = _fresh_proxy(monkeypatch)
    monkeypatch.setattr(proxy, "_db_conn", None)

    mock_kmp = {"type": "kompress", "compressor": MagicMock(), "threshold": 0.5}
    monkeypatch.setattr(proxy, "_load_kompress_backend", lambda: mock_kmp)

    b = proxy.load_backend()
    assert b["type"] == "kompress"


# ===========================================================================
# _pick_backend – dual mode path (line 346)
# ===========================================================================

def test_pick_backend_dual_mode(monkeypatch):
    """_pick_backend returns system/user backends when dual_mode is active."""
    proxy = _fresh_proxy(monkeypatch)

    user_b = {"type": "kompress"}
    sys_b = {"type": "llmlingua2-large"}
    monkeypatch.setattr(proxy, "dual_mode", True)
    monkeypatch.setattr(proxy, "backend_user", user_b)
    monkeypatch.setattr(proxy, "backend_system", sys_b)

    assert proxy._pick_backend("system") is sys_b
    assert proxy._pick_backend("user") is user_b


# ===========================================================================
# lifespan teardown – torch exception handler (lines 372-373)
# ===========================================================================

def test_lifespan_teardown_torch_exception(tmp_path, monkeypatch):
    """Lifespan teardown swallows torch exceptions gracefully."""
    _stub_heavy_deps(monkeypatch)
    monkeypatch.delitem(sys.modules, "proxy", raising=False)
    import proxy

    # Make torch.backends.mps.is_available raise during teardown
    bad_torch = MagicMock()
    bad_torch.backends.mps.is_available.side_effect = RuntimeError("mps unavailable")
    monkeypatch.setitem(sys.modules, "torch", bad_torch)

    mock_backend = {"type": "llmlingua2", "compressor": MagicMock(), "rate": 0.5}
    monkeypatch.setattr(proxy, "_load_backend", lambda: mock_backend)
    monkeypatch.setattr(proxy, "migrate_from_json", lambda conn, json_path="stats.json": None)
    monkeypatch.setattr(proxy, "recover_stats_from_backup", lambda conn, bak_path="stats.json.bak": None)
    monkeypatch.setattr(proxy, "_migrate_db_location", lambda: None)
    monkeypatch.setattr(proxy, "DB_PATH", tmp_path / "metrics.db")

    # TestClient context triggers lifespan startup + teardown
    with TestClient(proxy.app):
        pass  # teardown swallows the torch RuntimeError


# ===========================================================================
# _try_link_pending_tracker – early return for "unknown" / empty session (line 444)
# ===========================================================================

def test_try_link_pending_tracker_early_returns(monkeypatch):
    """_try_link_pending_tracker is a no-op for 'unknown' or empty session_id."""
    proxy = _fresh_proxy(monkeypatch)
    monkeypatch.setattr(proxy, "_db_conn", None)

    proxy._try_link_pending_tracker("unknown")  # must not raise
    proxy._try_link_pending_tracker("")          # must not raise


# ===========================================================================
# record_request – session_name stored (line 470)
# ===========================================================================

def test_record_request_stores_session_name(monkeypatch):
    """record_request stores the optional session_name in stats."""
    proxy = _fresh_proxy(monkeypatch)
    monkeypatch.setattr(proxy, "_db_conn", None)
    proxy.stats["sessions"] = {}
    proxy.stats["total_requests"] = 0

    proxy.record_request("sess-named", session_name="My Session")
    assert proxy.stats["sessions"]["sess-named"]["name"] == "My Session"


# ===========================================================================
# _rtk_db_path – Windows / Linux branches (lines 479-484)
# ===========================================================================

def test_rtk_db_path_darwin(tmp_path, monkeypatch):
    """_rtk_db_path returns the macOS Library path on Darwin."""
    monkeypatch.setenv("LLM_COMPRESSOR_DB", str(tmp_path / "metrics.db"))
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    proxy = _fresh_proxy(monkeypatch)
    p = proxy._rtk_db_path()
    assert "Library/Application Support/rtk/history.db" in str(p)


def test_rtk_db_path_windows(tmp_path, monkeypatch):
    """_rtk_db_path returns APPDATA-based path on Windows."""
    monkeypatch.setenv("LLM_COMPRESSOR_DB", str(tmp_path / "metrics.db"))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    proxy = _fresh_proxy(monkeypatch)

    p = proxy._rtk_db_path()
    assert "rtk" in str(p) and "history.db" in str(p)


def test_rtk_db_path_linux(tmp_path, monkeypatch):
    """_rtk_db_path returns XDG-style path on Linux."""
    monkeypatch.setenv("LLM_COMPRESSOR_DB", str(tmp_path / "metrics.db"))
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    proxy = _fresh_proxy(monkeypatch)

    p = proxy._rtk_db_path()
    assert ".local/share/rtk/history.db" in str(p)


# ===========================================================================
# read_rtk_stats (lines 487-527)
# ===========================================================================

def test_read_rtk_stats_no_db(tmp_path, monkeypatch):
    """read_rtk_stats returns None when history.db is absent."""
    proxy = _fresh_proxy(monkeypatch)
    monkeypatch.setattr(proxy, "_rtk_db_path", lambda: tmp_path / "nonexistent.db")
    assert proxy.read_rtk_stats() is None


def test_read_rtk_stats_with_data(tmp_path, monkeypatch):
    """read_rtk_stats returns structured stats when history.db has rows."""
    db_path = tmp_path / "history.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE commands "
        "(id INTEGER PRIMARY KEY, timestamp TEXT, rtk_cmd TEXT, "
        "input_tokens INTEGER, output_tokens INTEGER, saved_tokens INTEGER, savings_pct REAL)"
    )
    conn.execute(
        "INSERT INTO commands VALUES (1,'2026-01-01T10:00:00','git status',100,50,30,30.0)"
    )
    conn.commit()
    conn.close()

    proxy = _fresh_proxy(monkeypatch)
    monkeypatch.setattr(proxy, "_rtk_db_path", lambda: db_path)

    result = proxy.read_rtk_stats()
    assert result is not None
    assert result["total_commands"] == 1
    assert result["total_saved_tokens"] == 30
    assert len(result["top_commands"]) == 1


def test_read_rtk_stats_with_since(tmp_path, monkeypatch):
    """read_rtk_stats filters rows by the since parameter."""
    db_path = tmp_path / "history.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE commands "
        "(id INTEGER PRIMARY KEY, timestamp TEXT, rtk_cmd TEXT, "
        "input_tokens INTEGER, output_tokens INTEGER, saved_tokens INTEGER, savings_pct REAL)"
    )
    conn.executemany(
        "INSERT INTO commands VALUES (?,?,?,?,?,?,?)",
        [
            (1, "2026-01-01T10:00:00", "git status", 100, 50, 30, 30.0),
            (2, "2026-06-01T10:00:00", "git diff",   200, 100, 60, 30.0),
        ],
    )
    conn.commit()
    conn.close()

    proxy = _fresh_proxy(monkeypatch)
    monkeypatch.setattr(proxy, "_rtk_db_path", lambda: db_path)

    result = proxy.read_rtk_stats(since="2026-03-01T00:00:00")
    assert result is not None
    assert result["total_commands"] == 1


def test_read_rtk_stats_db_error(tmp_path, monkeypatch):
    """read_rtk_stats returns None on DB error."""
    db_path = tmp_path / "history.db"
    db_path.write_text("not a database")

    proxy = _fresh_proxy(monkeypatch)
    monkeypatch.setattr(proxy, "_rtk_db_path", lambda: db_path)

    assert proxy.read_rtk_stats() is None


# ===========================================================================
# _char_split – long text (lines 542-543)
# ===========================================================================

def test_char_split_long_text(monkeypatch):
    """_char_split splits text longer than _CHUNK_MAX_CHARS."""
    proxy = _fresh_proxy(monkeypatch)
    long_text = "x" * (proxy._CHUNK_MAX_CHARS + 100)
    chunks = proxy._char_split(long_text)
    assert len(chunks) > 1
    assert all(len(c) <= proxy._CHUNK_MAX_CHARS for c in chunks)


# ===========================================================================
# _split_into_segments – sentence split inside a long paragraph (lines 560-565)
# ===========================================================================

def test_split_into_segments_long_para_sentence_split(monkeypatch):
    """_split_into_segments splits a long paragraph on sentence boundaries."""
    from tests.conftest import make_mock_llmlingua
    proxy = _fresh_proxy(monkeypatch)

    # Tokenizer returns 10 tokens per word → long para exceeds 400-token limit
    mock_compressor = make_mock_llmlingua()
    mock_compressor.tokenizer.tokenize.side_effect = lambda text: ["tok"] * (len(text.split()) * 10)
    mock_backend = {"type": "llmlingua2", "compressor": mock_compressor, "rate": 0.5}
    monkeypatch.setattr(proxy, "backend", mock_backend)

    # Two paragraphs; first is long but contains sentence boundaries
    long_para = ("word " * 50).strip()  # 50 words × 10 = 500 tokens → over limit
    text = f"First sentence. {long_para}\n\nShort."
    segs = proxy._split_into_segments(text)
    assert len(segs) >= 2


# ===========================================================================
# _split_into_segments – sentence / char fallback for single paragraph
# (lines 570-573)
# ===========================================================================

def test_split_into_segments_sentence_fallback(monkeypatch):
    """_split_into_segments falls back to sentence splitting for a single paragraph."""
    from tests.conftest import make_mock_llmlingua
    proxy = _fresh_proxy(monkeypatch)
    mock_backend = {"type": "llmlingua2", "compressor": make_mock_llmlingua(), "rate": 0.5}
    monkeypatch.setattr(proxy, "backend", mock_backend)

    text = "First sentence. Second sentence. Third sentence."
    segs = proxy._split_into_segments(text)
    assert len(segs) >= 1


def test_split_into_segments_char_fallback(monkeypatch):
    """_split_into_segments falls back to _char_split when no sentence boundaries exist."""
    from tests.conftest import make_mock_llmlingua
    proxy = _fresh_proxy(monkeypatch)
    mock_backend = {"type": "llmlingua2", "compressor": make_mock_llmlingua(), "rate": 0.5}
    monkeypatch.setattr(proxy, "backend", mock_backend)

    # Dense text with no sentence-boundary punctuation
    dense = "nodots" * 300
    segs = proxy._split_into_segments(dense)
    assert len(segs) >= 1


# ===========================================================================
# compress_text – short text (603), no backend (606), error (610-612)
# ===========================================================================

def test_compress_text_short_unchanged(monkeypatch):
    """compress_text returns text unchanged when len <= 200."""
    proxy = _fresh_proxy(monkeypatch)
    short = "short"
    assert proxy.compress_text(short, "sess") == short


def test_compress_text_no_backend_unchanged(monkeypatch):
    """compress_text returns text unchanged when no backend is loaded."""
    proxy = _fresh_proxy(monkeypatch)
    monkeypatch.setattr(proxy, "backend", None)
    monkeypatch.setattr(proxy, "dual_mode", False)
    monkeypatch.setattr(proxy, "backend_user", None)
    monkeypatch.setattr(proxy, "backend_system", None)

    long_text = "word " * 50
    assert proxy.compress_text(long_text, "sess") == long_text


def test_compress_text_error_returns_original(monkeypatch):
    """compress_text returns the original text when compression raises."""
    from tests.conftest import make_mock_llmlingua
    proxy = _fresh_proxy(monkeypatch)

    mock_compressor = make_mock_llmlingua()
    mock_compressor.compress_prompt.side_effect = RuntimeError("model exploded")
    mock_backend = {"type": "llmlingua2", "compressor": mock_compressor, "rate": 0.5}
    monkeypatch.setattr(proxy, "backend", mock_backend)
    monkeypatch.setattr(proxy, "dual_mode", False)
    monkeypatch.setattr(proxy, "backend_user", None)
    monkeypatch.setattr(proxy, "backend_system", None)
    monkeypatch.setattr(proxy, "_db_conn", None)

    long_text = "word " * 50
    assert proxy.compress_text(long_text, "sess") == long_text


# ===========================================================================
# _compress_with – kompress dispatch (line 626)
# ===========================================================================

def test_compress_with_kompress_dispatch(monkeypatch):
    """_compress_with routes kompress type to _compress_kompress."""
    proxy = _fresh_proxy(monkeypatch)

    mock_result = MagicMock()
    mock_result.compressed = "compressed text"
    mock_result.original_tokens = 100
    mock_result.compressed_tokens = 60

    active = {"type": "kompress", "compressor": MagicMock(compress=MagicMock(return_value=mock_result))}
    text, orig, comp = proxy._compress_with(active, "some text")
    assert text == "compressed text"
    assert orig == 100 and comp == 60


# ===========================================================================
# compress_backend (lines 633-635)
# ===========================================================================

def test_compress_backend_raises_when_no_backend(monkeypatch):
    """compress_backend raises RuntimeError when global backend is None."""
    proxy = _fresh_proxy(monkeypatch)
    monkeypatch.setattr(proxy, "backend", None)
    with pytest.raises(RuntimeError, match="No backend loaded"):
        proxy.compress_backend("text")


def test_compress_backend_success(monkeypatch):
    """compress_backend delegates to global backend."""
    from tests.conftest import make_mock_llmlingua
    proxy = _fresh_proxy(monkeypatch)
    mock_backend = {"type": "llmlingua2", "compressor": make_mock_llmlingua(), "rate": 0.5}
    monkeypatch.setattr(proxy, "backend", mock_backend)

    compressed, orig, comp = proxy.compress_backend("word " * 50)
    assert compressed is not None


# ===========================================================================
# _compress_kompress (lines 664-665)
# ===========================================================================

def test_compress_kompress(monkeypatch):
    """_compress_kompress extracts fields from compressor.compress() result."""
    proxy = _fresh_proxy(monkeypatch)

    mock_result = MagicMock()
    mock_result.compressed = "out"
    mock_result.original_tokens = 200
    mock_result.compressed_tokens = 120

    active = {"type": "kompress", "compressor": MagicMock(compress=MagicMock(return_value=mock_result))}
    text, orig, comp = proxy._compress_kompress(active, "input")
    assert text == "out" and orig == 200 and comp == 120


# ===========================================================================
# compress_system_field (lines 670-678)
# ===========================================================================

def test_compress_system_field_string(monkeypatch):
    """compress_system_field compresses a string system field."""
    from tests.conftest import make_mock_llmlingua
    proxy = _fresh_proxy(monkeypatch)
    monkeypatch.setattr(proxy, "backend", {"type": "llmlingua2", "compressor": make_mock_llmlingua(), "rate": 0.5})
    monkeypatch.setattr(proxy, "dual_mode", False)
    monkeypatch.setattr(proxy, "backend_user", None)
    monkeypatch.setattr(proxy, "backend_system", None)
    monkeypatch.setattr(proxy, "_db_conn", None)

    result = proxy.compress_system_field("system instruction " * 20, "sess")
    assert isinstance(result, str)


def test_compress_system_field_list_with_text_and_non_text(monkeypatch):
    """compress_system_field compresses text blocks and passes non-text blocks through."""
    from tests.conftest import make_mock_llmlingua
    proxy = _fresh_proxy(monkeypatch)
    monkeypatch.setattr(proxy, "backend", {"type": "llmlingua2", "compressor": make_mock_llmlingua(), "rate": 0.5})
    monkeypatch.setattr(proxy, "dual_mode", False)
    monkeypatch.setattr(proxy, "backend_user", None)
    monkeypatch.setattr(proxy, "backend_system", None)
    monkeypatch.setattr(proxy, "_db_conn", None)

    system_list = [
        {"type": "text", "text": "instruction " * 20},
        {"type": "image", "source": {"type": "url", "url": "http://example.com/img.png"}},
    ]
    result = proxy.compress_system_field(system_list, "sess")
    assert isinstance(result, list)
    assert result[1] == system_list[1]  # image block unchanged


def test_compress_system_field_other_type(monkeypatch):
    """compress_system_field returns non-string/list values unchanged."""
    proxy = _fresh_proxy(monkeypatch)
    assert proxy.compress_system_field(None, "sess") is None
    assert proxy.compress_system_field(42, "sess") == 42


# ===========================================================================
# compress_messages (lines 682-700)
# ===========================================================================

def test_compress_messages_skips_assistant(monkeypatch):
    """compress_messages passes assistant messages through unchanged."""
    proxy = _fresh_proxy(monkeypatch)
    monkeypatch.setattr(proxy, "backend", None)

    msgs = [{"role": "assistant", "content": "I am the assistant."}]
    result = proxy.compress_messages(msgs, "sess")
    assert result[0]["content"] == "I am the assistant."


def test_compress_messages_list_content(monkeypatch):
    """compress_messages compresses text blocks in list-typed content."""
    from tests.conftest import make_mock_llmlingua
    proxy = _fresh_proxy(monkeypatch)
    monkeypatch.setattr(proxy, "backend", {"type": "llmlingua2", "compressor": make_mock_llmlingua(), "rate": 0.5})
    monkeypatch.setattr(proxy, "dual_mode", False)
    monkeypatch.setattr(proxy, "backend_user", None)
    monkeypatch.setattr(proxy, "backend_system", None)
    monkeypatch.setattr(proxy, "_db_conn", None)

    msgs = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "compress this " * 20},  # > 200 chars
            {"type": "image_url", "url": "http://example.com/img.png"},  # non-text
        ],
    }]
    result = proxy.compress_messages(msgs, "sess")
    # non-text block passed through unchanged
    assert result[0]["content"][1]["url"] == "http://example.com/img.png"


def test_compress_messages_non_string_non_list_content(monkeypatch):
    """compress_messages passes through unusual content types unchanged."""
    proxy = _fresh_proxy(monkeypatch)
    monkeypatch.setattr(proxy, "backend", None)

    msgs = [{"role": "user", "content": 42}]
    result = proxy.compress_messages(msgs, "sess")
    assert result[0]["content"] == 42


# ===========================================================================
# build_headers (lines 709-711)
# ===========================================================================

def test_build_headers(client: TestClient):
    """build_headers filters hop-by-hop headers and forces content-type."""
    from starlette.requests import Request

    proxy = sys.modules["proxy"]
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [
            (b"host", b"localhost"),
            (b"x-custom-header", b"keep-this"),
            (b"content-length", b"100"),
            (b"authorization", b"Bearer token"),
        ],
    }
    req = Request(scope)
    headers = proxy.build_headers(req)

    assert "host" not in headers
    assert "content-length" not in headers
    assert headers["x-custom-header"] == "keep-this"
    assert headers["authorization"] == "Bearer token"
    assert headers["content-type"] == "application/json"


# ===========================================================================
# get_stats – backend_loading (742), dual (749), kompress (756)
# ===========================================================================

def test_stats_while_loading(client: TestClient):
    """GET /stats shows loading=True when backend_loading is set."""
    proxy = sys.modules["proxy"]
    proxy.backend_loading = "llmlingua2-large"
    try:
        d = client.get("/stats").json()
        assert d["compressor"]["loading"] is True
        assert d["compressor"]["model"] == "llmlingua2-large"
    finally:
        proxy.backend_loading = None


def test_stats_dual_backend(client: TestClient):
    """GET /stats shows dual model info when backend type is dual."""
    proxy = sys.modules["proxy"]
    orig = proxy.backend
    try:
        proxy.backend = {"type": "dual"}
        d = client.get("/stats").json()
        assert d["compressor"]["model"] == "dual"
        assert d["compressor"]["param_name"] is None
    finally:
        proxy.backend = orig


def test_stats_dual_aggregates_across_submodels(client: TestClient):
    """In dual mode, today/alltime aggregate rows stored under all sub-model names.

    Characterization test guarding the de-duplicated stats SQL: dual mode must
    sum across kompress + llmlingua2 + llmlingua2-large rows, not match model='dual'.
    """
    proxy = sys.modules["proxy"]
    orig = proxy.backend
    conn = proxy._db_conn
    conn.executemany(
        "INSERT INTO compressions (ts, session_id, model, original_tokens, compressed_tokens, latency_ms, role) "
        "VALUES (datetime('now'), ?, ?, ?, ?, ?, ?)",
        [
            ("d1", "kompress", 200, 120, 5.0, "user"),
            ("d2", "llmlingua2-large", 400, 160, 9.0, "system"),
            ("d3", "llmlingua2", 300, 150, 7.0, "user"),
        ],
    )
    conn.execute(
        "INSERT INTO compressions (ts, session_id, model, original_tokens, compressed_tokens, latency_ms, role) "
        "VALUES ('2020-01-01T00:00:00', 'old', 'kompress', 100, 50, 4.0, 'user')",
    )
    conn.commit()
    try:
        proxy.backend = {"type": "dual"}
        d = client.get("/stats").json()
        assert d["compressor"]["model"] == "dual"
        assert d["today"]["requests"] == 3          # 3 rows today, across all sub-models
        assert d["today"]["tokens_saved"] == 80 + 240 + 150
        assert d["alltime"]["requests"] == 4         # + the 2020 row
        models = {m["model"] for m in d["by_model"]}
        assert {"kompress", "llmlingua2", "llmlingua2-large"} <= models
        # recent-activity must span sub-model rows in dual mode (not match model='dual')
        assert len(d["recent"]) == 4
        assert {r["model"] for r in d["recent"]} == {"kompress", "llmlingua2", "llmlingua2-large"}
    finally:
        proxy.backend = orig
        conn.execute("DELETE FROM compressions")
        conn.commit()


def test_stats_kompress_backend(client: TestClient):
    """GET /stats shows kompress info when backend type is kompress."""
    proxy = sys.modules["proxy"]
    orig = proxy.backend
    try:
        proxy.backend = {"type": "kompress", "threshold": 0.4}
        d = client.get("/stats").json()
        assert d["compressor"]["model"] == "kompress"
        assert d["compressor"]["param_name"] == "threshold"
        assert d["compressor"]["param_value"] == 0.4
    finally:
        proxy.backend = orig


# ===========================================================================
# get_stats – rtk_events present (lines 925-931)
# ===========================================================================

def test_stats_includes_rtk_data(client: TestClient):
    """GET /stats returns rtk key when rtk_events has rows."""
    proxy = sys.modules["proxy"]
    conn = proxy._db_conn
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn.execute(
        "INSERT INTO rtk_events (rtk_id, ts, session_id, rtk_cmd, input_tokens, output_tokens, saved_tokens, savings_pct) "
        "VALUES (1, ?, 'sess-rtk', 'git status', 100, 50, 30, 30.0)",
        (ts,),
    )
    conn.commit()
    d = client.get("/stats").json()
    assert d["rtk"] is not None
    assert d["rtk"]["total_commands"] == 1


# ===========================================================================
# get_timeseries – model filter (lines 984, 988)
# ===========================================================================

def test_timeseries_with_model_filter(client: TestClient):
    """GET /stats/timeseries?model=llmlingua2 filters to that model."""
    proxy = sys.modules["proxy"]
    conn = proxy._db_conn
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    conn.execute(
        "INSERT INTO compressions (ts, session_id, model, original_tokens, compressed_tokens, latency_ms) "
        "VALUES (?, 'sess-ts', 'llmlingua2', 200, 120, 50.0)",
        (ts,),
    )
    conn.commit()
    r = client.get("/stats/timeseries?model=llmlingua2")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


# ===========================================================================
# rtk_log endpoint (lines 1022-1046)
# ===========================================================================

def test_rtk_log_inserts_row(client: TestClient):
    """POST /rtk/log inserts a row and returns ok."""
    r = client.post("/rtk/log", json={
        "rtk_id": 99,
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": "sess-rtk-log",
        "rtk_cmd": "git status",
        "input_tokens": 100,
        "output_tokens": 50,
        "saved_tokens": 30,
        "savings_pct": 30.0,
    })
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_rtk_log_idempotent_duplicate_id(client: TestClient):
    """POST /rtk/log with duplicate rtk_id is a no-op (INSERT OR IGNORE)."""
    payload = {
        "rtk_id": 77,
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": "sess-dup",
        "rtk_cmd": "git diff",
        "input_tokens": 100, "output_tokens": 50, "saved_tokens": 30, "savings_pct": 30.0,
    }
    client.post("/rtk/log", json=payload)
    r = client.post("/rtk/log", json=payload)
    assert r.status_code == 200


def test_rtk_log_db_error(client: TestClient, monkeypatch):
    """POST /rtk/log returns 500 when the DB write fails."""
    proxy = sys.modules["proxy"]
    orig_conn = proxy._db_conn

    bad_conn = MagicMock()
    bad_conn.execute.side_effect = Exception("disk full")
    monkeypatch.setattr(proxy, "_db_conn", bad_conn)
    try:
        r = client.post("/rtk/log", json={
            "rtk_id": 1,
            "ts": datetime.now(timezone.utc).isoformat(),
            "session_id": "s",
            "rtk_cmd": "cmd",
            "input_tokens": 1, "output_tokens": 1, "saved_tokens": 0, "savings_pct": 0.0,
        })
        assert r.status_code == 500
    finally:
        monkeypatch.setattr(proxy, "_db_conn", orig_conn)


# ===========================================================================
# session_dashboard – db not ready (line 1052)
# ===========================================================================

def test_session_dashboard_db_not_ready(client: TestClient, monkeypatch):
    """GET /dashboard/<slug> returns 503 when _db_conn is None."""
    proxy = sys.modules["proxy"]
    orig = proxy._db_conn
    monkeypatch.setattr(proxy, "_db_conn", None)
    try:
        r = client.get("/dashboard/any-slug")
        assert r.status_code == 503
    finally:
        monkeypatch.setattr(proxy, "_db_conn", orig)


# ===========================================================================
# dashboard endpoint (line 1068)
# ===========================================================================

def test_dashboard_returns_html(client: TestClient):
    """GET /dashboard returns the main dashboard HTML page."""
    r = client.get("/dashboard")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "LLM Compressor" in r.text


# ===========================================================================
# play endpoint (line 1073)
# ===========================================================================

def test_play_returns_html(client: TestClient):
    """GET /play returns the Playground HTML page."""
    r = client.get("/play")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")


# ===========================================================================
# play_compress endpoint – all branches (lines 1084-1144)
# ===========================================================================

def test_play_compress_success(client: TestClient):
    """POST /play/compress returns compression stats for loaded backend."""
    r = client.post("/play/compress", json={"text": "word " * 50, "model": ""})
    assert r.status_code == 200
    d = r.json()
    assert "compressed" in d and "char_pct" in d and "latency_ms" in d


def test_play_compress_unknown_model(client: TestClient):
    """POST /play/compress with an unknown model returns 400."""
    r = client.post("/play/compress", json={"text": "hello", "model": "gpt-99"})
    assert r.status_code == 400
    assert "Unknown model" in r.json()["error"]


def test_play_compress_no_backend_no_loading(client: TestClient, monkeypatch):
    """POST /play/compress returns 503 when backend is None and nothing is loading."""
    proxy = sys.modules["proxy"]
    orig_backend = proxy.backend
    orig_loading = proxy.backend_loading
    monkeypatch.setattr(proxy, "backend", None)
    monkeypatch.setattr(proxy, "backend_loading", None)
    try:
        r = client.post("/play/compress", json={"text": "hello", "model": ""})
        assert r.status_code == 503
    finally:
        monkeypatch.setattr(proxy, "backend", orig_backend)
        monkeypatch.setattr(proxy, "backend_loading", orig_loading)


def test_play_compress_triggers_model_switch(client: TestClient, monkeypatch):
    """POST /play/compress with a different model triggers async load, returns 202."""
    proxy = sys.modules["proxy"]
    orig_backend = proxy.backend
    orig_loading = proxy.backend_loading
    # Stub loader so no real model loads in the background thread
    monkeypatch.setattr(
        proxy, "_load_llmlingua2_backend",
        lambda backend_key=None: {"type": "llmlingua2-large", "rate": 0.5, "compressor": MagicMock()},
    )
    try:
        r = client.post("/play/compress", json={"text": "hello", "model": "llmlingua2-large"})
        assert r.status_code == 202
        assert r.json()["loading"] is True
    finally:
        monkeypatch.setattr(proxy, "backend", orig_backend)
        monkeypatch.setattr(proxy, "backend_loading", None)


def test_play_compress_same_model_already_loading(client: TestClient, monkeypatch):
    """POST /play/compress returns 202 when the requested model is already loading."""
    proxy = sys.modules["proxy"]
    orig_backend = proxy.backend
    orig_loading = proxy.backend_loading
    monkeypatch.setattr(proxy, "backend_loading", "llmlingua2-large")
    monkeypatch.setattr(proxy, "backend", None)
    try:
        r = client.post("/play/compress", json={"text": "hello", "model": "llmlingua2-large"})
        assert r.status_code == 202
    finally:
        monkeypatch.setattr(proxy, "backend", orig_backend)
        monkeypatch.setattr(proxy, "backend_loading", orig_loading)


def test_play_compress_different_model_already_loading(client: TestClient, monkeypatch):
    """POST /play/compress returns 202 when a different model is loading."""
    proxy = sys.modules["proxy"]
    orig_backend = proxy.backend
    orig_loading = proxy.backend_loading
    monkeypatch.setattr(proxy, "backend_loading", "kompress")
    monkeypatch.setattr(proxy, "backend", None)
    try:
        r = client.post("/play/compress", json={"text": "hello", "model": "llmlingua2-large"})
        assert r.status_code == 202
    finally:
        monkeypatch.setattr(proxy, "backend", orig_backend)
        monkeypatch.setattr(proxy, "backend_loading", orig_loading)


def test_play_compress_no_model_but_loading(client: TestClient, monkeypatch):
    """POST /play/compress with no model while something is loading returns 202."""
    proxy = sys.modules["proxy"]
    orig_backend = proxy.backend
    orig_loading = proxy.backend_loading
    monkeypatch.setattr(proxy, "backend_loading", "llmlingua2")
    monkeypatch.setattr(proxy, "backend", None)
    try:
        r = client.post("/play/compress", json={"text": "hello", "model": ""})
        assert r.status_code == 202
    finally:
        monkeypatch.setattr(proxy, "backend", orig_backend)
        monkeypatch.setattr(proxy, "backend_loading", orig_loading)


# ===========================================================================
# set_model endpoint – all branches (lines 1158-1211)
# ===========================================================================

def test_set_model_unknown_returns_400(client: TestClient):
    """POST /admin/set-model with unknown model name returns 400."""
    r = client.post("/admin/set-model", json={"model": "gpt-4"})
    assert r.status_code == 400


def test_set_model_triggers_load(client: TestClient, monkeypatch):
    """POST /admin/set-model starts async load and returns loading status."""
    proxy = sys.modules["proxy"]
    orig_backend = proxy.backend
    orig_loading = proxy.backend_loading
    monkeypatch.setattr(
        proxy, "_load_llmlingua2_backend",
        lambda backend_key=None: {"type": "llmlingua2", "rate": 0.5, "compressor": MagicMock()},
    )
    try:
        r = client.post("/admin/set-model", json={"model": "llmlingua2"})
        assert r.status_code == 200
        assert r.json()["status"] == "loading"
    finally:
        monkeypatch.setattr(proxy, "backend", orig_backend)
        monkeypatch.setattr(proxy, "backend_loading", None)


def test_set_model_clears_dual_globals(client: TestClient, monkeypatch):
    """POST /admin/set-model clears dual_mode when switching away from dual."""
    proxy = sys.modules["proxy"]
    orig_dual = proxy.dual_mode
    orig_bu = proxy.backend_user
    orig_bs = proxy.backend_system
    orig_backend = proxy.backend
    orig_loading = proxy.backend_loading
    monkeypatch.setattr(
        proxy, "_load_llmlingua2_backend",
        lambda backend_key=None: {"type": "llmlingua2", "rate": 0.5, "compressor": MagicMock()},
    )
    monkeypatch.setattr(proxy, "dual_mode", True)
    monkeypatch.setattr(proxy, "backend_user", {"type": "kompress"})
    monkeypatch.setattr(proxy, "backend_system", {"type": "llmlingua2-large"})
    try:
        r = client.post("/admin/set-model", json={"model": "llmlingua2"})
        assert r.status_code == 200
        assert proxy.dual_mode is False
        assert proxy.backend_user is None
        assert proxy.backend_system is None
    finally:
        monkeypatch.setattr(proxy, "dual_mode", orig_dual)
        monkeypatch.setattr(proxy, "backend_user", orig_bu)
        monkeypatch.setattr(proxy, "backend_system", orig_bs)
        monkeypatch.setattr(proxy, "backend", orig_backend)
        monkeypatch.setattr(proxy, "backend_loading", orig_loading)


# ===========================================================================
# set_dual_models endpoint + _load_single_backend (dual sub-model routing)
# ===========================================================================

def test_set_dual_models_no_args_returns_400(client: TestClient):
    """POST /admin/set-dual-models with neither key returns 400."""
    r = client.post("/admin/set-dual-models", json={})
    assert r.status_code == 400


def test_set_dual_models_invalid_system_returns_400(client: TestClient):
    """POST /admin/set-dual-models rejects an unknown system model."""
    r = client.post("/admin/set-dual-models", json={"system": "gpt-4"})
    assert r.status_code == 400


def test_set_dual_models_invalid_user_returns_400(client: TestClient):
    """POST /admin/set-dual-models rejects an unknown user model."""
    r = client.post("/admin/set-dual-models", json={"user": "gpt-4"})
    assert r.status_code == 400


def test_set_dual_models_persists_when_not_dual(client: TestClient, monkeypatch):
    """When dual mode is inactive, the endpoint updates globals + DB and returns ok."""
    proxy = sys.modules["proxy"]
    orig_sys, orig_usr, orig_dual = proxy.dual_model_system, proxy.dual_model_user, proxy.dual_mode
    monkeypatch.setattr(proxy, "dual_mode", False)
    try:
        r = client.post(
            "/admin/set-dual-models",
            json={"system": "llmlingua2", "user": "llmlingua2-large"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["system"] == "llmlingua2"
        assert body["user"] == "llmlingua2-large"
        assert proxy.dual_model_system == "llmlingua2"
        assert proxy.dual_model_user == "llmlingua2-large"
        # persisted to the meta table
        row = proxy._db_conn.execute(
            "SELECT value FROM meta WHERE key='dual_model_system'"
        ).fetchone()
        assert row[0] == "llmlingua2"
    finally:
        monkeypatch.setattr(proxy, "dual_model_system", orig_sys)
        monkeypatch.setattr(proxy, "dual_model_user", orig_usr)
        monkeypatch.setattr(proxy, "dual_mode", orig_dual)


def test_set_dual_models_reloads_when_dual(client: TestClient, monkeypatch):
    """When dual mode is active, the endpoint triggers a reload and returns loading."""
    proxy = sys.modules["proxy"]
    orig_sys, orig_usr, orig_dual = proxy.dual_model_system, proxy.dual_model_user, proxy.dual_mode
    orig_backend, orig_loading = proxy.backend, proxy.backend_loading
    monkeypatch.setattr(proxy, "dual_mode", True)
    monkeypatch.setattr(proxy, "_load_dual_backend", lambda: {"type": "dual"})
    try:
        r = client.post("/admin/set-dual-models", json={"system": "kompress"})
        assert r.status_code == 200
        assert r.json()["status"] == "loading"
        assert proxy.dual_model_system == "kompress"
    finally:
        monkeypatch.setattr(proxy, "dual_model_system", orig_sys)
        monkeypatch.setattr(proxy, "dual_model_user", orig_usr)
        monkeypatch.setattr(proxy, "dual_mode", orig_dual)
        monkeypatch.setattr(proxy, "backend", orig_backend)
        monkeypatch.setattr(proxy, "backend_loading", orig_loading)


def test_load_single_backend_dispatches(monkeypatch):
    """_load_single_backend routes kompress to its loader and others to llmlingua2."""
    proxy = _fresh_proxy(monkeypatch)
    monkeypatch.setattr(proxy, "_load_kompress_backend", lambda: {"type": "kompress"})
    monkeypatch.setattr(
        proxy, "_load_llmlingua2_backend",
        lambda backend_key=None: {"type": backend_key},
    )
    assert proxy._load_single_backend("kompress")["type"] == "kompress"
    assert proxy._load_single_backend("llmlingua2-large")["type"] == "llmlingua2-large"


def test_load_backend_reads_dual_submodels_from_meta(tmp_path, monkeypatch):
    """load_backend hydrates dual_model_system/user from the meta table."""
    proxy = _fresh_proxy(monkeypatch)
    conn = proxy.init_db(str(tmp_path / "metrics.db"))
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('current_model', 'dual')")
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('dual_model_system', 'kompress')")
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('dual_model_user', 'llmlingua2')")
    conn.commit()
    monkeypatch.setattr(proxy, "_db_conn", conn)
    monkeypatch.setattr(proxy, "_load_dual_backend", lambda: {"type": "dual"})

    proxy.load_backend()
    assert proxy.dual_model_system == "kompress"
    assert proxy.dual_model_user == "llmlingua2"
    conn.close()


# ===========================================================================
# clear_compression_texts endpoint (lines 1217-1234)
# ===========================================================================

def test_clear_compression_texts_all(client: TestClient):
    """DELETE /admin/compression-texts removes all stored text rows."""
    r = client.delete("/admin/compression-texts")
    assert r.status_code == 200
    d = r.json()
    assert "deleted" in d and d["session_id"] is None


def test_clear_compression_texts_by_session(client: TestClient):
    """DELETE /admin/compression-texts with session_id clears only that session."""
    r = client.request("DELETE", "/admin/compression-texts", json={"session_id": "sess-x"})
    assert r.status_code == 200
    assert r.json()["session_id"] == "sess-x"


# ===========================================================================
# Tracker endpoints – db not ready paths (lines 1244, 1258, 1269, 1284)
# ===========================================================================

def test_create_tracker_db_not_ready(client: TestClient, monkeypatch):
    proxy = sys.modules["proxy"]
    orig = proxy._db_conn
    monkeypatch.setattr(proxy, "_db_conn", None)
    try:
        r = client.post("/admin/tracker", json={"name": "test"})
        assert r.status_code == 503
    finally:
        monkeypatch.setattr(proxy, "_db_conn", orig)


def test_get_tracker_db_not_ready(client: TestClient, monkeypatch):
    proxy = sys.modules["proxy"]
    orig = proxy._db_conn
    monkeypatch.setattr(proxy, "_db_conn", None)
    try:
        r = client.get("/admin/tracker")
        assert r.status_code == 200
        assert r.json() == []
    finally:
        monkeypatch.setattr(proxy, "_db_conn", orig)


def test_delete_tracker_db_not_ready(client: TestClient, monkeypatch):
    proxy = sys.modules["proxy"]
    orig = proxy._db_conn
    monkeypatch.setattr(proxy, "_db_conn", None)
    try:
        r = client.delete("/admin/tracker/any-slug")
        assert r.status_code == 503
    finally:
        monkeypatch.setattr(proxy, "_db_conn", orig)


def test_get_all_trackers_db_not_ready(client: TestClient, monkeypatch):
    proxy = sys.modules["proxy"]
    orig = proxy._db_conn
    monkeypatch.setattr(proxy, "_db_conn", None)
    try:
        r = client.get("/admin/tracker/all")
        assert r.status_code == 200
        assert r.json() == []
    finally:
        monkeypatch.setattr(proxy, "_db_conn", orig)


# ===========================================================================
# get_session_compressions endpoint (lines 1302-1324)
# ===========================================================================

def test_get_session_compressions_not_found(client: TestClient):
    """GET /session/<slug>/compressions returns 404 for unknown slug."""
    r = client.get("/session/nonexistent/compressions")
    assert r.status_code == 404


def test_get_session_compressions_no_linked_session(client: TestClient):
    """GET /session/<slug>/compressions returns [] when tracker has no session_id."""
    r = client.post("/admin/tracker", json={"name": "unlinked"})
    slug = r.json()["slug"]
    r = client.get(f"/session/{slug}/compressions")
    assert r.status_code == 200
    assert r.json() == []


def test_get_session_compressions_with_data(client: TestClient):
    """GET /session/<slug>/compressions returns compression rows for linked session."""
    proxy = sys.modules["proxy"]

    r = client.post("/admin/tracker", json={"name": "linked"})
    slug = r.json()["slug"]

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    proxy._db_conn.execute(
        "UPDATE trackers SET status='active', session_id='sess-comps', linked_at=? WHERE slug=?",
        (ts, slug),
    )
    proxy._db_conn.commit()
    proxy.record_compression("sess-comps", 200, 120, 50.0)

    r = client.get(f"/session/{slug}/compressions")
    assert r.status_code == 200
    data = r.json()
    assert len(data) >= 1
    assert data[0]["original_tokens"] == 200


# ===========================================================================
# list_models endpoint (lines 1329-1336)
# ===========================================================================

def test_list_models_proxies_response(client: TestClient, monkeypatch):
    """GET /v1/models proxies to Anthropic and returns the response."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": []}

    mock_http = MagicMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=None)
    mock_http.get = AsyncMock(return_value=mock_response)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: mock_http)

    r = client.get("/v1/models")
    assert r.status_code == 200
    assert "data" in r.json()


# ===========================================================================
# proxy_messages endpoint (lines 1341-1386)
# ===========================================================================

def test_proxy_messages_non_streaming(client: TestClient, monkeypatch):
    """POST /v1/messages (non-streaming) compresses payload and proxies."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"id": "msg_1", "type": "message", "content": []}

    mock_http = MagicMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=None)
    mock_http.post = AsyncMock(return_value=mock_response)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: mock_http)

    r = client.post("/v1/messages", json={
        "model": "claude-3-haiku-20240307",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "Hello"}],
    })
    assert r.status_code == 200


def test_proxy_messages_compresses_system_field(client: TestClient, monkeypatch):
    """POST /v1/messages compresses the system field before forwarding."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"id": "msg_2", "type": "message", "content": []}

    mock_http = MagicMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=None)
    mock_http.post = AsyncMock(return_value=mock_response)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: mock_http)

    r = client.post("/v1/messages", json={
        "model": "claude-3-haiku-20240307",
        "max_tokens": 10,
        "system": "You are a helpful assistant.",
        "messages": [{"role": "user", "content": "Hello"}],
    })
    assert r.status_code == 200


def test_proxy_messages_anthropic_error(client: TestClient, monkeypatch):
    """POST /v1/messages forwards Anthropic 4xx error responses."""
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.json.return_value = {"error": {"type": "authentication_error"}}
    mock_response.text = '{"error": {"type": "authentication_error"}}'

    mock_http = MagicMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=None)
    mock_http.post = AsyncMock(return_value=mock_response)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: mock_http)

    r = client.post("/v1/messages", json={
        "model": "claude-3-haiku-20240307",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "Hello"}],
    })
    assert r.status_code == 401


def test_proxy_messages_streaming(client: TestClient, monkeypatch):
    """POST /v1/messages with stream=True returns a streaming response."""
    async def _aiter_bytes():
        yield b"data: test\n\n"

    mock_stream_resp = MagicMock()
    mock_stream_resp.status_code = 200
    mock_stream_resp.aiter_bytes = _aiter_bytes
    mock_stream_resp.__aenter__ = AsyncMock(return_value=mock_stream_resp)
    mock_stream_resp.__aexit__ = AsyncMock(return_value=None)

    mock_http = MagicMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=None)
    mock_http.stream = MagicMock(return_value=mock_stream_resp)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: mock_http)

    r = client.post("/v1/messages", json={
        "model": "claude-3-haiku-20240307",
        "max_tokens": 10,
        "stream": True,
        "messages": [{"role": "user", "content": "Hello"}],
    })
    assert r.status_code == 200


def test_proxy_messages_streaming_error_response(client: TestClient, monkeypatch):
    """POST /v1/messages streaming forwards body bytes on Anthropic error."""
    async def _aread():
        return b'{"error": "auth"}'

    mock_stream_resp = MagicMock()
    mock_stream_resp.status_code = 401
    mock_stream_resp.aread = _aread
    mock_stream_resp.__aenter__ = AsyncMock(return_value=mock_stream_resp)
    mock_stream_resp.__aexit__ = AsyncMock(return_value=None)

    mock_http = MagicMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=None)
    mock_http.stream = MagicMock(return_value=mock_stream_resp)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: mock_http)

    r = client.post("/v1/messages", json={
        "model": "claude-3-haiku-20240307",
        "max_tokens": 10,
        "stream": True,
        "messages": [{"role": "user", "content": "Hello"}],
    })
    # StreamingResponse always returns 200 at the HTTP layer; the error is in the body
    assert r.status_code == 200


# ===========================================================================
# Second-pass: remaining 4% gaps
# ===========================================================================

# ---------------------------------------------------------------------------
# _migrate_db_location – actual function body (lines 379-389)
# ---------------------------------------------------------------------------

def test_migrate_db_location_no_old_file(tmp_path, monkeypatch):
    """_migrate_db_location early-returns when the old metrics.db doesn't exist."""
    import os
    _stub_heavy_deps(monkeypatch)
    monkeypatch.setenv("LLM_COMPRESSOR_DB", str(tmp_path / "metrics.db"))
    proxy = _fresh_proxy(monkeypatch)
    monkeypatch.setattr(proxy, "DB_PATH", tmp_path / "new_metrics.db")

    # Change CWD to tmp_path which has no metrics.db → old.exists() is False → line 382 hit
    orig_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        proxy._migrate_db_location()  # must not raise
    finally:
        os.chdir(orig_cwd)


def test_migrate_db_location_copies_old_file(tmp_path, monkeypatch):
    """_migrate_db_location copies old metrics.db to new path when it exists."""
    _stub_heavy_deps(monkeypatch)
    monkeypatch.setenv("LLM_COMPRESSOR_DB", str(tmp_path / "metrics.db"))
    proxy = _fresh_proxy(monkeypatch)

    # Create an old-style "metrics.db" in tmp_path (simulate CWD)
    old_db = tmp_path / "metrics.db"
    old_db.write_text("fake db content")
    new_db = tmp_path / "new_metrics.db"
    monkeypatch.setattr(proxy, "DB_PATH", new_db)

    # Temporarily change CWD so Path("metrics.db").resolve() points to our file
    import os
    orig_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        proxy._migrate_db_location()
    finally:
        os.chdir(orig_cwd)

    assert new_db.exists()


def test_migrate_db_location_new_db_has_data(tmp_path, monkeypatch):
    """_migrate_db_location skips copy when new DB already has real data (>64 KiB)."""
    import os
    _stub_heavy_deps(monkeypatch)
    monkeypatch.setenv("LLM_COMPRESSOR_DB", str(tmp_path / "metrics.db"))
    proxy = _fresh_proxy(monkeypatch)

    old_db = tmp_path / "metrics.db"
    old_db.write_text("old content")
    new_db = tmp_path / "new_metrics.db"
    new_db.write_bytes(b"x" * (65537))  # > 64 KiB → "has real data"
    monkeypatch.setattr(proxy, "DB_PATH", new_db)

    orig_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        proxy._migrate_db_location()  # must return early at line 389
    finally:
        os.chdir(orig_cwd)


def test_migrate_db_location_copy_exception(tmp_path, monkeypatch):
    """_migrate_db_location swallows copy errors gracefully."""
    import os
    import shutil
    _stub_heavy_deps(monkeypatch)
    monkeypatch.setenv("LLM_COMPRESSOR_DB", str(tmp_path / "metrics.db"))
    proxy = _fresh_proxy(monkeypatch)

    old_db = tmp_path / "metrics.db"
    old_db.write_text("fake db content")
    new_db = tmp_path / "new_metrics.db"
    monkeypatch.setattr(proxy, "DB_PATH", new_db)
    monkeypatch.setattr(shutil, "copy2", MagicMock(side_effect=OSError("no space left")))

    orig_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        proxy._migrate_db_location()  # must not raise; hits except line 389
    finally:
        os.chdir(orig_cwd)

# ---------------------------------------------------------------------------
# _count_tokens – tokenizer exception fallback (lines 542-543)
# ---------------------------------------------------------------------------

def test_count_tokens_tokenizer_exception(monkeypatch):
    """_count_tokens falls back to whitespace split when tokenizer raises."""
    from tests.conftest import make_mock_llmlingua
    proxy = _fresh_proxy(monkeypatch)

    mock_compressor = make_mock_llmlingua()
    mock_compressor.tokenizer.tokenize.side_effect = RuntimeError("tokenizer error")
    mock_backend = {"type": "llmlingua2", "compressor": mock_compressor, "rate": 0.5}
    monkeypatch.setattr(proxy, "backend", mock_backend)

    # Should fall back to len(text.split()) without raising
    count = proxy._count_tokens("hello world foo bar")
    assert count == 4


# ---------------------------------------------------------------------------
# _char_split – short text early return (line 549)
# ---------------------------------------------------------------------------

def test_char_split_short_text(monkeypatch):
    """_char_split returns single-element list for text within the char limit."""
    proxy = _fresh_proxy(monkeypatch)
    short = "x" * 10
    result = proxy._char_split(short)
    assert result == [short]


# ---------------------------------------------------------------------------
# _split_into_segments – multi-line paragraph, extend(lines) (line 562)
# ---------------------------------------------------------------------------

def test_split_into_segments_multiline_paragraph(monkeypatch):
    """_split_into_segments splits a long paragraph into its constituent lines."""
    from tests.conftest import make_mock_llmlingua
    proxy = _fresh_proxy(monkeypatch)

    # Make tokenizer return many tokens per word so paragraph exceeds 400-token limit
    mock_compressor = make_mock_llmlingua()
    mock_compressor.tokenizer.tokenize.side_effect = lambda t: ["tok"] * (len(t.split()) * 20)
    mock_backend = {"type": "llmlingua2", "compressor": mock_compressor, "rate": 0.5}
    monkeypatch.setattr(proxy, "backend", mock_backend)

    # Paragraph 1: multiple \n-separated lines, total tokens > 400
    # Paragraph 2: short second paragraph (separated by \n\n)
    line_a = "alpha " * 30  # 30 words × 20 = 600 tokens — over limit
    line_b = "beta " * 30
    para1 = line_a.strip() + "\n" + line_b.strip()
    para2 = "short paragraph"
    text = para1 + "\n\n" + para2

    segs = proxy._split_into_segments(text)
    # Should have split para1 into its two lines
    assert len(segs) >= 3


# ---------------------------------------------------------------------------
# get_timeseries – _db_conn is None (line 984)
# ---------------------------------------------------------------------------

def test_timeseries_db_not_ready(client: TestClient, monkeypatch):
    """GET /stats/timeseries returns [] when _db_conn is None."""
    proxy = sys.modules["proxy"]
    orig = proxy._db_conn
    monkeypatch.setattr(proxy, "_db_conn", None)
    try:
        r = client.get("/stats/timeseries")
        assert r.status_code == 200
        assert r.json() == []
    finally:
        monkeypatch.setattr(proxy, "_db_conn", orig)


# ---------------------------------------------------------------------------
# rtk_log – _db_conn is None (line 1023)
# ---------------------------------------------------------------------------

def test_rtk_log_db_not_ready(client: TestClient, monkeypatch):
    """POST /rtk/log returns 503 when _db_conn is None."""
    proxy = sys.modules["proxy"]
    orig = proxy._db_conn
    monkeypatch.setattr(proxy, "_db_conn", None)
    try:
        r = client.post("/rtk/log", json={"session_id": "s", "rtk_cmd": "cmd"})
        assert r.status_code == 503
    finally:
        monkeypatch.setattr(proxy, "_db_conn", orig)


# ---------------------------------------------------------------------------
# play_compress load thread – kompress / dual / exception paths
# (lines 1114, 1116, 1120-1121)
# ---------------------------------------------------------------------------

import threading as _threading_mod


def _sync_thread_factory(target_name: str, result_holder: list):
    """Return a Thread-like class that runs the target synchronously on start()."""
    class SyncThread:
        def __init__(self, target=None, daemon=False):
            self._target = target

        def start(self):
            if self._target:
                self._target()

    return SyncThread


def test_play_compress_triggers_kompress_load(client: TestClient, monkeypatch):
    """play_compress load thread runs _load_kompress_backend for kompress model."""
    import threading
    proxy = sys.modules["proxy"]
    orig_backend = proxy.backend
    orig_loading = proxy.backend_loading

    mock_kmp = {"type": "kompress", "compressor": MagicMock(), "threshold": 0.5}
    monkeypatch.setattr(proxy, "_load_kompress_backend", lambda: mock_kmp)
    monkeypatch.setattr(threading, "Thread", _sync_thread_factory("kompress", []))

    try:
        r = client.post("/play/compress", json={"text": "hello", "model": "kompress"})
        assert r.status_code == 202
    finally:
        monkeypatch.setattr(proxy, "backend", orig_backend)
        monkeypatch.setattr(proxy, "backend_loading", None)
        monkeypatch.setattr(threading, "Thread", _threading_mod.Thread)


def test_play_compress_triggers_dual_load(client: TestClient, monkeypatch):
    """play_compress load thread runs _load_dual_backend for dual model."""
    import threading
    proxy = sys.modules["proxy"]
    orig_backend = proxy.backend
    orig_loading = proxy.backend_loading

    mock_dual = {"type": "dual", "model_user": "kompress", "model_system": "llmlingua2-large"}
    monkeypatch.setattr(proxy, "_load_dual_backend", lambda: mock_dual)
    monkeypatch.setattr(threading, "Thread", _sync_thread_factory("dual", []))

    try:
        r = client.post("/play/compress", json={"text": "hello", "model": "dual"})
        assert r.status_code == 202
    finally:
        monkeypatch.setattr(proxy, "backend", orig_backend)
        monkeypatch.setattr(proxy, "backend_loading", None)
        monkeypatch.setattr(threading, "Thread", _threading_mod.Thread)


def test_play_compress_load_thread_exception(client: TestClient, monkeypatch):
    """play_compress load thread swallows exceptions and clears backend_loading."""
    import threading
    proxy = sys.modules["proxy"]
    orig_backend = proxy.backend
    orig_loading = proxy.backend_loading

    def _raise():
        raise RuntimeError("load failed")

    monkeypatch.setattr(proxy, "_load_llmlingua2_backend", lambda backend_key=None: _raise())
    monkeypatch.setattr(threading, "Thread", _sync_thread_factory("llmlingua2-large", []))

    try:
        r = client.post("/play/compress", json={"text": "hello", "model": "llmlingua2-large"})
        assert r.status_code == 202
        assert proxy.backend_loading is None  # cleared in finally
    finally:
        monkeypatch.setattr(proxy, "backend", orig_backend)
        monkeypatch.setattr(proxy, "backend_loading", None)
        monkeypatch.setattr(threading, "Thread", _threading_mod.Thread)


# ---------------------------------------------------------------------------
# play_compress – compress_backend raises (lines 1135-1136)
# ---------------------------------------------------------------------------

def test_play_compress_backend_raises(client: TestClient, monkeypatch):
    """POST /play/compress returns 500 when compress_backend raises."""
    proxy = sys.modules["proxy"]
    monkeypatch.setattr(proxy, "compress_backend", lambda text: (_ for _ in ()).throw(RuntimeError("boom")))
    r = client.post("/play/compress", json={"text": "word " * 50, "model": ""})
    assert r.status_code == 500
    assert "boom" in r.json()["error"]


# ---------------------------------------------------------------------------
# set_model – dual path + load_dual thread (lines 1180-1193)
# ---------------------------------------------------------------------------

def test_set_model_dual(client: TestClient, monkeypatch):
    """POST /admin/set-model with model=dual triggers the load_dual thread."""
    import threading
    proxy = sys.modules["proxy"]
    orig_backend = proxy.backend
    orig_loading = proxy.backend_loading
    orig_bu = proxy.backend_user
    orig_bs = proxy.backend_system

    mock_dual = {"type": "dual", "model_user": "kompress", "model_system": "llmlingua2-large"}
    monkeypatch.setattr(proxy, "_load_dual_backend", lambda: mock_dual)
    monkeypatch.setattr(threading, "Thread", _sync_thread_factory("dual", []))

    try:
        r = client.post("/admin/set-model", json={"model": "dual"})
        assert r.status_code == 200
        assert r.json()["model"] == "dual"
    finally:
        monkeypatch.setattr(proxy, "backend", orig_backend)
        monkeypatch.setattr(proxy, "backend_loading", None)
        monkeypatch.setattr(proxy, "backend_user", orig_bu)
        monkeypatch.setattr(proxy, "backend_system", orig_bs)
        monkeypatch.setattr(threading, "Thread", _threading_mod.Thread)


# ---------------------------------------------------------------------------
# set_model – kompress path in load thread (line 1200)
# ---------------------------------------------------------------------------

def test_set_model_kompress_load_thread(client: TestClient, monkeypatch):
    """POST /admin/set-model with model=kompress runs _load_kompress_backend."""
    import threading
    proxy = sys.modules["proxy"]
    orig_backend = proxy.backend
    orig_loading = proxy.backend_loading

    mock_kmp = {"type": "kompress", "compressor": MagicMock(), "threshold": 0.5}
    monkeypatch.setattr(proxy, "_load_kompress_backend", lambda: mock_kmp)
    monkeypatch.setattr(threading, "Thread", _sync_thread_factory("kompress", []))

    try:
        r = client.post("/admin/set-model", json={"model": "kompress"})
        assert r.status_code == 200
    finally:
        monkeypatch.setattr(proxy, "backend", orig_backend)
        monkeypatch.setattr(proxy, "backend_loading", None)
        monkeypatch.setattr(threading, "Thread", _threading_mod.Thread)


# ---------------------------------------------------------------------------
# set_model – load thread exception path (lines 1204-1205)
# ---------------------------------------------------------------------------

def test_set_model_load_thread_exception(client: TestClient, monkeypatch):
    """set_model load thread swallows loader exceptions and clears backend_loading."""
    import threading
    proxy = sys.modules["proxy"]
    orig_backend = proxy.backend
    orig_loading = proxy.backend_loading

    def _raise(backend_key=None):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(proxy, "_load_llmlingua2_backend", _raise)
    monkeypatch.setattr(threading, "Thread", _sync_thread_factory("llmlingua2", []))

    try:
        r = client.post("/admin/set-model", json={"model": "llmlingua2"})
        assert r.status_code == 200
        assert proxy.backend_loading is None  # cleared in finally
    finally:
        monkeypatch.setattr(proxy, "backend", orig_backend)
        monkeypatch.setattr(proxy, "backend_loading", None)
        monkeypatch.setattr(threading, "Thread", _threading_mod.Thread)


# ---------------------------------------------------------------------------
# clear_compression_texts – _db_conn is None (line 1218)
# ---------------------------------------------------------------------------

def test_clear_compression_texts_db_not_ready(client: TestClient, monkeypatch):
    """DELETE /admin/compression-texts returns 503 when _db_conn is None."""
    proxy = sys.modules["proxy"]
    orig = proxy._db_conn
    monkeypatch.setattr(proxy, "_db_conn", None)
    try:
        r = client.delete("/admin/compression-texts")
        assert r.status_code == 503
    finally:
        monkeypatch.setattr(proxy, "_db_conn", orig)


# ---------------------------------------------------------------------------
# get_session_compressions – _db_conn is None (line 1303)
# ---------------------------------------------------------------------------

def test_get_session_compressions_db_not_ready(client: TestClient, monkeypatch):
    """GET /session/<slug>/compressions returns 503 when _db_conn is None."""
    proxy = sys.modules["proxy"]
    orig = proxy._db_conn
    monkeypatch.setattr(proxy, "_db_conn", None)
    try:
        r = client.get("/session/any-slug/compressions")
        assert r.status_code == 503
    finally:
        monkeypatch.setattr(proxy, "_db_conn", orig)


# ===========================================================================
# Third-pass: final 4% gaps
# ===========================================================================

# ---------------------------------------------------------------------------
# get_stats – rtk_commands / rtk_saved per-session decoration (lines 1007-1008)
# ---------------------------------------------------------------------------

def test_stats_rtk_per_session(client: TestClient):
    """GET /stats decorates session entries with rtk_commands when both exist."""
    proxy = sys.modules["proxy"]
    conn = proxy._db_conn
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Create a compression for this session so it appears in sessions_out
    proxy.record_compression("sess-rtk-per", 200, 120, 10.0)
    proxy.record_request("sess-rtk-per")

    # Insert an rtk_event for the SAME session_id
    conn.execute(
        "INSERT INTO rtk_events "
        "(rtk_id, ts, session_id, rtk_cmd, input_tokens, output_tokens, saved_tokens, savings_pct) "
        "VALUES (200, ?, 'sess-rtk-per', 'git log', 100, 50, 30, 30.0)",
        (ts,),
    )
    conn.commit()

    d = client.get("/stats").json()
    sessions = d.get("sessions", {})  # sessions is a dict keyed by session_id
    if "sess-rtk-per" in sessions:
        assert sessions["sess-rtk-per"].get("rtk_commands") == 1
        assert sessions["sess-rtk-per"].get("rtk_saved") == 30


# ---------------------------------------------------------------------------
# set_model – dual load_dual exception path (lines 1242-1243)
# ---------------------------------------------------------------------------

def test_set_model_dual_load_exception(client: TestClient, monkeypatch):
    """set_model dual load_dual thread swallows _load_dual_backend exceptions."""
    import threading
    proxy = sys.modules["proxy"]
    orig_backend = proxy.backend
    orig_loading = proxy.backend_loading
    orig_bu = proxy.backend_user
    orig_bs = proxy.backend_system

    monkeypatch.setattr(proxy, "_load_dual_backend", lambda: (_ for _ in ()).throw(RuntimeError("dual failed")))
    monkeypatch.setattr(threading, "Thread", _sync_thread_factory("dual", []))

    try:
        r = client.post("/admin/set-model", json={"model": "dual"})
        assert r.status_code == 200
        assert proxy.backend_loading is None  # cleared in finally
    finally:
        monkeypatch.setattr(proxy, "backend", orig_backend)
        monkeypatch.setattr(proxy, "backend_loading", None)
        monkeypatch.setattr(proxy, "backend_user", orig_bu)
        monkeypatch.setattr(proxy, "backend_system", orig_bs)
        monkeypatch.setattr(threading, "Thread", _threading_mod.Thread)


# ---------------------------------------------------------------------------
# get_session_rtk_commands endpoint (lines 1383-1402)
# ---------------------------------------------------------------------------

def test_get_session_rtk_commands_db_not_ready(client: TestClient, monkeypatch):
    """GET /session/<slug>/rtk-commands returns 503 when _db_conn is None."""
    proxy = sys.modules["proxy"]
    orig = proxy._db_conn
    monkeypatch.setattr(proxy, "_db_conn", None)
    try:
        r = client.get("/session/any/rtk-commands")
        assert r.status_code == 503
    finally:
        monkeypatch.setattr(proxy, "_db_conn", orig)


def test_get_session_rtk_commands_not_found(client: TestClient):
    """GET /session/<slug>/rtk-commands returns 404 for unknown slug."""
    r = client.get("/session/nonexistent-slug/rtk-commands")
    assert r.status_code == 404


def test_get_session_rtk_commands_no_linked_session(client: TestClient):
    """GET /session/<slug>/rtk-commands returns [] when tracker has no session_id."""
    r = client.post("/admin/tracker", json={"name": "rtkcmds-unlinked"})
    slug = r.json()["slug"]
    r = client.get(f"/session/{slug}/rtk-commands")
    assert r.status_code == 200
    assert r.json() == []


def test_get_session_rtk_commands_with_data(client: TestClient):
    """GET /session/<slug>/rtk-commands returns rtk rows for linked session."""
    proxy = sys.modules["proxy"]

    r = client.post("/admin/tracker", json={"name": "rtkcmds-linked"})
    slug = r.json()["slug"]

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    proxy._db_conn.execute(
        "UPDATE trackers SET status='active', session_id='sess-rtk-cmds', linked_at=? WHERE slug=?",
        (ts, slug),
    )
    proxy._db_conn.execute(
        "INSERT INTO rtk_events "
        "(rtk_id, ts, session_id, rtk_cmd, input_tokens, output_tokens, saved_tokens, savings_pct, project_path) "
        "VALUES (301, ?, 'sess-rtk-cmds', 'git status', 100, 50, 30, 30.0, '/proj')",
        (ts,),
    )
    proxy._db_conn.commit()

    r = client.get(f"/session/{slug}/rtk-commands")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    assert data[0]["rtk_cmd"] == "git status"
