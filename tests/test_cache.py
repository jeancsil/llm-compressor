import sys
from unittest.mock import MagicMock


def _import_proxy(monkeypatch):
    for dep in ("llmlingua", "torch", "transformers"):
        if dep not in sys.modules:
            monkeypatch.setitem(sys.modules, dep, MagicMock())
    monkeypatch.delitem(sys.modules, "proxy", raising=False)
    import proxy
    return proxy


def test_cache_key_is_deterministic_and_model_scoped(monkeypatch):
    proxy = _import_proxy(monkeypatch)
    k1 = proxy._cache_key("hello world", "kompress", 0.5)
    k2 = proxy._cache_key("hello world", "kompress", 0.5)
    k3 = proxy._cache_key("hello world", "llmlingua2", 0.5)
    k4 = proxy._cache_key("hello world", "kompress", 0.6)
    assert k1 == k2                 # deterministic
    assert k1 != k3                 # model is part of key
    assert k1 != k4                 # rate is part of key
    assert k1.endswith("|kompress|0.5")
    assert len(k1.split("|")[0]) == 64   # sha256 hex


def test_init_db_creates_cache_schema(tmp_path, monkeypatch):
    proxy = _import_proxy(monkeypatch)
    conn = proxy.init_db(str(tmp_path / "m.db"))
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    assert "compression_cache" in tables
    cache_cols = [r[1] for r in conn.execute("PRAGMA table_info(compression_cache)")]
    for col in ("key", "model", "rate", "compressed_text", "original_tokens",
                "compressed_tokens", "created_at", "hit_count", "last_hit"):
        assert col in cache_cols
    comp_cols = [r[1] for r in conn.execute("PRAGMA table_info(compressions)")]
    assert "cache_hit" in comp_cols
    conn.close()


def test_cache_get_put_memory_and_disk(tmp_path, monkeypatch):
    proxy = _import_proxy(monkeypatch)
    conn = proxy.init_db(str(tmp_path / "m.db"))
    cache = proxy.CompressionCache(conn, max_mem=2, max_rows=50000)
    assert cache.get("k1") is None
    cache.put("k1", "COMP", 100, 60, "kompress", 0.5)
    assert cache.get("k1") == ("COMP", 100, 60)               # memory hit
    row = conn.execute("SELECT compressed_text, hit_count FROM compression_cache WHERE key='k1'").fetchone()
    assert row[0] == "COMP" and row[1] >= 1                   # persisted + hit counted
    conn.close()


def test_cache_memory_lru_evicts_but_disk_retains(tmp_path, monkeypatch):
    proxy = _import_proxy(monkeypatch)
    conn = proxy.init_db(str(tmp_path / "m.db"))
    cache = proxy.CompressionCache(conn, max_mem=2, max_rows=50000)
    cache.put("a", "A", 1, 1, "m", 0.5)
    cache.put("b", "B", 1, 1, "m", 0.5)
    cache.put("c", "C", 1, 1, "m", 0.5)   # evicts "a" from memory
    assert "a" not in cache._mem
    assert cache.get("a") == ("A", 1, 1)  # served from disk, re-promoted
    assert "a" in cache._mem


def test_cache_disk_row_cap_evicts_least_recently_hit(tmp_path, monkeypatch):
    proxy = _import_proxy(monkeypatch)
    conn = proxy.init_db(str(tmp_path / "m.db"))
    cache = proxy.CompressionCache(conn, max_mem=100, max_rows=2)
    cache.put("a", "A", 1, 1, "m", 0.5)
    cache.put("b", "B", 1, 1, "m", 0.5)
    cache.get("a")                         # bump a's last_hit so b is oldest
    cache.put("c", "C", 1, 1, "m", 0.5)    # over cap -> evict oldest last_hit
    keys = {r[0] for r in conn.execute("SELECT key FROM compression_cache")}
    assert len(keys) == 2 and "b" not in keys
    conn.close()


def test_record_compression_writes_cache_hit_flag(tmp_path, monkeypatch):
    proxy = _import_proxy(monkeypatch)
    conn = proxy.init_db(str(tmp_path / "m.db"))
    monkeypatch.setattr(proxy, "_db_conn", conn)
    monkeypatch.setattr(proxy, "backend", {"type": "kompress", "rate": 0.5})
    # hit path: no text passed -> no compression_texts row, cache_hit=1
    proxy.record_compression("sess", 100, 60, latency_ms=0.0, role="user", cache_hit=1)
    row = conn.execute("SELECT cache_hit FROM compressions ORDER BY id DESC LIMIT 1").fetchone()
    assert row[0] == 1
    texts = conn.execute("SELECT COUNT(*) FROM compression_texts").fetchone()[0]
    assert texts == 0
    conn.close()


def test_compress_text_caches_second_call(tmp_path, monkeypatch):
    proxy = _import_proxy(monkeypatch)
    conn = proxy.init_db(str(tmp_path / "m.db"))
    monkeypatch.setattr(proxy, "_db_conn", conn)
    monkeypatch.setattr(proxy, "_cache", proxy.CompressionCache(conn, max_mem=10, max_rows=100))

    calls = {"n": 0}
    def fake_compress(active, text):
        calls["n"] += 1
        return "COMPRESSED", 100, 60
    monkeypatch.setattr(proxy, "_compress_with", fake_compress)
    monkeypatch.setattr(proxy, "backend", {"type": "kompress", "rate": 0.5})
    monkeypatch.setattr(proxy, "dual_mode", False)

    text = "x" * 500   # > 200 so it is compressed
    assert proxy.compress_text(text, "sess") == "COMPRESSED"
    assert proxy.compress_text(text, "sess") == "COMPRESSED"
    assert calls["n"] == 1                                   # model ran only once
    hits = conn.execute("SELECT COUNT(*) FROM compressions WHERE cache_hit=1").fetchone()[0]
    assert hits == 1
    texts = conn.execute("SELECT COUNT(*) FROM compression_texts").fetchone()[0]
    assert texts == 1                                        # only the miss wrote text
    conn.close()
