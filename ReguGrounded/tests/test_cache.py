# tests/test_cache.py
#
# Unit tests for utils/cache.py — LRU memory cache, SQLite disk cache,
# QueryCache, and ComponentCache.

import os
import sys
import tempfile
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.cache import _LRUCache, _SQLiteStore, QueryCache, ComponentCache, _make_key


# ---------------------------------------------------------------------------
# _make_key
# ---------------------------------------------------------------------------

class TestMakeKey:
    def test_deterministic(self):
        assert _make_key("query", "foo", 5) == _make_key("query", "foo", 5)

    def test_different_inputs_differ(self):
        assert _make_key("query", "foo", 5) != _make_key("query", "bar", 5)

    def test_case_sensitive(self):
        assert _make_key("query", "Foo", 5) != _make_key("query", "foo", 5)


# ---------------------------------------------------------------------------
# _LRUCache
# ---------------------------------------------------------------------------

class TestLRUCache:
    def test_basic_get_set(self):
        cache = _LRUCache(maxsize=10)
        cache.set("k1", {"data": 1})
        assert cache.get("k1") == {"data": 1}

    def test_miss_returns_none(self):
        cache = _LRUCache(maxsize=10)
        assert cache.get("nonexistent") is None

    def test_lru_eviction(self):
        cache = _LRUCache(maxsize=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        # Access "a" to make it recently used
        cache.get("a")
        # Add "d" — "b" should be evicted (LRU)
        cache.set("d", 4)
        assert cache.get("b") is None
        assert cache.get("a") == 1
        assert cache.get("c") == 3
        assert cache.get("d") == 4

    def test_ttl_expiry(self):
        cache = _LRUCache(maxsize=10, ttl=0.1)
        cache.set("k", "value")
        assert cache.get("k") == "value"
        time.sleep(0.15)
        assert cache.get("k") is None  # expired

    def test_evict_expired(self):
        cache = _LRUCache(maxsize=10, ttl=0.05)
        cache.set("k1", 1)
        cache.set("k2", 2)
        time.sleep(0.1)
        evicted = cache.evict_expired()
        assert evicted == 2
        assert cache.size() == 0

    def test_size(self):
        cache = _LRUCache(maxsize=10)
        assert cache.size() == 0
        cache.set("a", 1)
        cache.set("b", 2)
        assert cache.size() == 2

    def test_clear(self):
        cache = _LRUCache(maxsize=10)
        cache.set("k", "v")
        cache.clear()
        assert cache.size() == 0

    def test_overwrite(self):
        cache = _LRUCache(maxsize=10)
        cache.set("k", "old")
        cache.set("k", "new")
        assert cache.get("k") == "new"
        assert cache.size() == 1


# ---------------------------------------------------------------------------
# _SQLiteStore
# ---------------------------------------------------------------------------

class TestSQLiteStore:
    @pytest.fixture
    def store(self, tmp_path):
        db_path = str(tmp_path / "test_cache.db")
        return _SQLiteStore(db_path)

    def test_basic_get_set(self, store):
        store.set("k1", {"answer": "yes"})
        assert store.get("k1") == {"answer": "yes"}

    def test_miss_returns_none(self, store):
        assert store.get("missing") is None

    def test_ttl_expiry(self, store):
        store.set("k", "val", ttl=0.1)
        assert store.get("k") == "val"
        time.sleep(0.15)
        assert store.get("k") is None  # expired and deleted

    def test_clear(self, store):
        store.set("a", 1)
        store.set("b", 2)
        store.clear()
        assert store.get("a") is None
        assert store.get("b") is None

    def test_clear_expired(self, store):
        store.set("live", "yes", ttl=60)
        store.set("dead", "no", ttl=0.05)
        time.sleep(0.1)
        removed = store.clear_expired()
        assert removed == 1
        assert store.get("live") == "yes"
        assert store.get("dead") is None

    def test_stats(self, store):
        store.set("a", 1, ttl=60)
        store.set("b", 2, ttl=0.05)
        time.sleep(0.1)
        stats = store.stats()
        assert stats["total_entries"] == 2
        assert stats["expired_entries"] == 1

    def test_overwrite(self, store):
        store.set("k", "old")
        store.set("k", "new")
        assert store.get("k") == "new"


# ---------------------------------------------------------------------------
# QueryCache
# ---------------------------------------------------------------------------

class TestQueryCache:
    @pytest.fixture
    def cache(self, tmp_path):
        db_path = str(tmp_path / "qcache.db")
        return QueryCache(db_path=db_path, ttl=3600, maxsize=10)

    def test_miss_returns_none(self, cache):
        assert cache.get("What is Article 9?", top_k=5) is None

    def test_set_then_get_memory(self, cache):
        result = {"answer": "EU AI Act requires...", "citations": []}
        cache.set("What is Article 9?", top_k=5, result=result)
        assert cache.get("What is Article 9?", top_k=5) == result

    def test_case_insensitive_key(self, cache):
        result = {"answer": "test"}
        cache.set("What is Article 9?", top_k=5, result=result)
        # Same query, different casing — still a hit (key is lowercased)
        assert cache.get("what is article 9?", top_k=5) == result

    def test_disk_persistence(self, tmp_path):
        db_path = str(tmp_path / "persist.db")
        c1 = QueryCache(db_path=db_path, ttl=3600)
        c1.set("my query", top_k=5, result={"data": 42})

        # New cache instance — memory is cold, but disk has the entry
        c2 = QueryCache(db_path=db_path, ttl=3600)
        c2._memory.clear()  # force miss on layer 1
        assert c2.get("my query", top_k=5) == {"data": 42}

    def test_different_top_k_different_key(self, cache):
        result_5  = {"answer": "answer with top_k=5"}
        result_10 = {"answer": "answer with top_k=10"}
        cache.set("query", top_k=5,  result=result_5)
        cache.set("query", top_k=10, result=result_10)
        assert cache.get("query", top_k=5)  == result_5
        assert cache.get("query", top_k=10) == result_10

    def test_clear(self, cache):
        cache.set("q", top_k=5, result={"a": 1})
        cache.clear()
        assert cache.get("q", top_k=5) is None

    def test_stats_structure(self, cache):
        stats = cache.stats()
        assert "memory_entries" in stats
        assert "disk_entries"   in stats
        assert "ttl_seconds"    in stats


# ---------------------------------------------------------------------------
# ComponentCache
# ---------------------------------------------------------------------------

class TestComponentCache:
    @pytest.fixture
    def cache(self):
        return ComponentCache(maxsize=50, ttl=3600)

    def test_decomposition_miss(self, cache):
        assert cache.get_decomposition("What are bias audits?") is None

    def test_decomposition_set_get(self, cache):
        result = {"jurisdictions": ["NYC Local Law 144"], "sub_questions": []}
        cache.set_decomposition("What are bias audits?", result)
        assert cache.get_decomposition("What are bias audits?") == result

    def test_retrieval_miss(self, cache):
        assert cache.get_retrieval("sub q", "EU AI Act", 5) is None

    def test_retrieval_set_get(self, cache):
        chunks = [{"text": "Article 9...", "law_name": "EU AI Act"}]
        cache.set_retrieval("sub q", "EU AI Act", 5, chunks)
        assert cache.get_retrieval("sub q", "EU AI Act", 5) == chunks

    def test_retrieval_different_jurisdiction(self, cache):
        chunks_eu  = [{"text": "EU chunk"}]
        chunks_nyc = [{"text": "NYC chunk"}]
        cache.set_retrieval("sub q", "EU AI Act",        5, chunks_eu)
        cache.set_retrieval("sub q", "NYC Local Law 144", 5, chunks_nyc)
        assert cache.get_retrieval("sub q", "EU AI Act",        5) == chunks_eu
        assert cache.get_retrieval("sub q", "NYC Local Law 144", 5) == chunks_nyc

    def test_clear(self, cache):
        cache.set_decomposition("q", {"jurisdictions": []})
        cache.set_retrieval("sq", "EU AI Act", 5, [])
        cache.clear()
        assert cache.get_decomposition("q") is None
        assert cache.get_retrieval("sq", "EU AI Act", 5) is None

    def test_stats_structure(self, cache):
        stats = cache.stats()
        assert "decomposition_entries" in stats
        assert "retrieval_entries"     in stats


# ---------------------------------------------------------------------------
# Integration: QueryCache promotion from disk to memory
# ---------------------------------------------------------------------------

class TestQueryCachePromotion:
    def test_disk_hit_promotes_to_memory(self, tmp_path):
        db_path = str(tmp_path / "promo.db")
        cache = QueryCache(db_path=db_path, ttl=3600)
        result = {"answer": "promoted"}

        # Write to disk directly (bypass memory)
        from utils.cache import _make_key
        key = _make_key("query", "test query", 5)
        cache._disk.set(key, result, ttl=3600)

        # Memory should be cold
        assert cache._memory.get(key) is None

        # get() should find on disk and promote to memory
        fetched = cache.get("test query", top_k=5)
        assert fetched == result
        assert cache._memory.get(key) == result  # promoted
