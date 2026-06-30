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
    assert texts == 2                                        # miss + cache-hit both write text (commit 69e8372)
    conn.close()


def test_stats_endpoint_reports_cache(client):
    r = client.get("/stats")
    assert r.status_code == 200
    cache = r.json()["cache"]
    assert {"since_deploy", "last_24h"}.issubset(set(cache))
    for key in ("since_deploy", "last_24h"):
        window = cache[key]
        assert {"hits", "total", "hit_ratio"}.issubset(set(window))
        assert window["hit_ratio"] == 0.0   # fresh DB, no compressions yet


def test_cache_stats_windows_exclude_pre_deploy_and_old_rows(tmp_path, monkeypatch):
    from datetime import datetime, timezone, timedelta

    proxy = _import_proxy(monkeypatch)
    conn = proxy.init_db(str(tmp_path / "m.db"))
    monkeypatch.setattr(proxy, "_db_conn", conn)

    now = datetime.now(timezone.utc)
    # Pin the deploy marker 5 days back so we can place rows on either side.
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('cache_since', ?)",
        ((now - timedelta(days=5)).isoformat(timespec="seconds"),),
    )

    def ins(ts, cache_hit):
        conn.execute(
            "INSERT INTO compressions (ts, session_id, model, original_tokens, "
            "compressed_tokens, latency_ms, role, cache_hit) VALUES (?,?,?,?,?,?,?,?)",
            (ts.isoformat(timespec="seconds"), "s", "kompress", 100, 50, 0, "user", cache_hit),
        )

    # Pre-deploy backlog (10 days ago): all misses — excluded from both windows.
    for _ in range(8):
        ins(now - timedelta(days=10), 0)
    # After deploy, older than 24h (2 days ago): 1 hit — in since_deploy only.
    ins(now - timedelta(days=2), 1)
    # Within last 24h (1 hour ago): 2 hits + 1 miss — in both windows.
    ins(now - timedelta(hours=1), 1)
    ins(now - timedelta(hours=1), 1)
    ins(now - timedelta(hours=1), 0)
    conn.commit()

    stats = proxy._cache_stats()
    assert stats["since_deploy"] == {"hits": 3, "total": 4, "hit_ratio": round(3 / 4, 4)}
    assert stats["last_24h"] == {"hits": 2, "total": 3, "hit_ratio": round(2 / 3, 4)}
    conn.close()
