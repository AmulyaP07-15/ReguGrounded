# utils/cache.py
#
# Multi-layer caching system for ReguGrounded.
#
#   QueryCache     — full query results (LRU in-memory + SQLite disk, TTL 1 hr)
#   ComponentCache — sub-operation results: decomposition, retrieval (LRU only)
#
# Both caches are thread-safe. Cache keys are deterministic SHA-256 hashes so
# slight whitespace differences in queries are still distinct entries.

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from collections import OrderedDict
from typing import Any, Optional

logger = logging.getLogger("cache")

_DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "cache", "query_cache.db"
)
_DEFAULT_TTL_SECONDS = 3600  # 1 hour


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_key(*parts: Any) -> str:
    """Deterministic SHA-256 key from any number of JSON-serialisable parts."""
    raw = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# LRU memory cache (used by both QueryCache and ComponentCache)
# ---------------------------------------------------------------------------

class _LRUCache:
    """Thread-safe LRU cache with optional per-entry TTL."""

    def __init__(self, maxsize: int = 128, ttl: Optional[float] = None):
        self._maxsize = maxsize
        self._ttl = ttl
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                return None
            value, expires_at = self._cache[key]
            if expires_at is not None and time.time() > expires_at:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            expires_at = (time.time() + self._ttl) if self._ttl else None
            self._cache[key] = (value, expires_at)
            self._cache.move_to_end(key)
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)  # evict LRU

    def delete(self, key: str) -> None:
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    def evict_expired(self) -> int:
        """Remove expired entries. Returns count evicted."""
        now = time.time()
        evicted = 0
        with self._lock:
            expired_keys = [
                k for k, (_, exp) in self._cache.items()
                if exp is not None and now > exp
            ]
            for k in expired_keys:
                del self._cache[k]
                evicted += 1
        return evicted


# ---------------------------------------------------------------------------
# SQLite persistence layer
# ---------------------------------------------------------------------------

class _SQLiteStore:
    """Simple SQLite-backed key/value store with TTL support."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path, check_same_thread=False)
        return self._local.conn

    def _init_db(self) -> None:
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                expires_at REAL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_expires ON cache(expires_at)")
        conn.commit()
        conn.close()

    def get(self, key: str) -> Optional[Any]:
        try:
            conn = self._conn()
            row = conn.execute(
                "SELECT value, expires_at FROM cache WHERE key = ?", (key,)
            ).fetchone()
            if not row:
                return None
            value_json, expires_at = row
            if expires_at is not None and time.time() > expires_at:
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                conn.commit()
                return None
            return json.loads(value_json)
        except Exception as e:
            logger.warning(f"SQLite get failed: {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        try:
            expires_at = (time.time() + ttl) if ttl else None
            conn = self._conn()
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, expires_at) VALUES (?, ?, ?)",
                (key, json.dumps(value, default=str), expires_at),
            )
            conn.commit()
        except Exception as e:
            logger.warning(f"SQLite set failed: {e}")

    def delete(self, key: str) -> None:
        try:
            conn = self._conn()
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()
        except Exception as e:
            logger.warning(f"SQLite delete failed: {e}")

    def clear(self) -> None:
        try:
            conn = self._conn()
            conn.execute("DELETE FROM cache")
            conn.commit()
        except Exception as e:
            logger.warning(f"SQLite clear failed: {e}")

    def clear_expired(self) -> int:
        """Delete expired entries and return count removed."""
        try:
            conn = self._conn()
            cur = conn.execute(
                "DELETE FROM cache WHERE expires_at IS NOT NULL AND expires_at < ?",
                (time.time(),),
            )
            conn.commit()
            return cur.rowcount
        except Exception as e:
            logger.warning(f"SQLite clear_expired failed: {e}")
            return 0

    def stats(self) -> dict:
        try:
            conn = self._conn()
            total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            expired = conn.execute(
                "SELECT COUNT(*) FROM cache WHERE expires_at IS NOT NULL AND expires_at < ?",
                (time.time(),),
            ).fetchone()[0]
            return {"total_entries": total, "expired_entries": expired}
        except Exception as e:
            logger.warning(f"SQLite stats failed: {e}")
            return {"total_entries": 0, "expired_entries": 0}


# ---------------------------------------------------------------------------
# Public: QueryCache
# ---------------------------------------------------------------------------

class QueryCache:
    """
    Two-layer cache for full pipeline results.

    Layer 1: LRU in-memory (fast, process-scoped, maxsize=256, TTL=1hr)
    Layer 2: SQLite on-disk  (survives restarts, same TTL)

    On a cache hit at layer 2, the result is promoted to layer 1.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        ttl: float = _DEFAULT_TTL_SECONDS,
        maxsize: int = 256,
    ):
        self._ttl = ttl
        self._memory = _LRUCache(maxsize=maxsize, ttl=ttl)
        self._disk   = _SQLiteStore(db_path or _DEFAULT_DB_PATH)

    def get(self, query: str, top_k: int = 5) -> Optional[dict]:
        key = _make_key("query", query.strip().lower(), top_k)

        # Layer 1: memory
        result = self._memory.get(key)
        if result is not None:
            logger.debug(f"Query cache HIT (memory): {query[:60]!r}")
            return result

        # Layer 2: disk
        result = self._disk.get(key)
        if result is not None:
            logger.debug(f"Query cache HIT (disk): {query[:60]!r}")
            self._memory.set(key, result)  # promote
            return result

        logger.debug(f"Query cache MISS: {query[:60]!r}")
        return None

    def set(self, query: str, top_k: int, result: dict) -> None:
        key = _make_key("query", query.strip().lower(), top_k)
        self._memory.set(key, result)
        self._disk.set(key, result, ttl=self._ttl)

    def clear(self) -> None:
        self._memory.clear()
        self._disk.clear()
        logger.info("Query cache cleared (memory + disk)")

    def clear_expired(self) -> int:
        mem_evicted  = self._memory.evict_expired()
        disk_removed = self._disk.clear_expired()
        total = mem_evicted + disk_removed
        logger.info(f"Cleared {total} expired entries ({mem_evicted} memory, {disk_removed} disk)")
        return total

    def stats(self) -> dict:
        disk_stats = self._disk.stats()
        return {
            "memory_entries": self._memory.size(),
            "disk_entries":   disk_stats["total_entries"],
            "disk_expired":   disk_stats["expired_entries"],
            "ttl_seconds":    self._ttl,
        }


# ---------------------------------------------------------------------------
# Public: ComponentCache
# ---------------------------------------------------------------------------

class ComponentCache:
    """
    Lightweight LRU-only cache for sub-pipeline components:
      - decomposition results from RLMEngine
      - retrieval results from ReasoningOrchestrator

    No disk persistence — these are cheaper to recompute than full queries.
    """

    def __init__(self, maxsize: int = 512, ttl: float = _DEFAULT_TTL_SECONDS):
        self._decomp   = _LRUCache(maxsize=maxsize, ttl=ttl)
        self._retrieval = _LRUCache(maxsize=maxsize, ttl=ttl)

    # ---- Decomposition ---------------------------------------------------

    def get_decomposition(self, query: str) -> Optional[dict]:
        key = _make_key("decomp", query.strip().lower())
        result = self._decomp.get(key)
        if result:
            logger.debug(f"Decomposition cache HIT: {query[:60]!r}")
        return result

    def set_decomposition(self, query: str, result: dict) -> None:
        key = _make_key("decomp", query.strip().lower())
        self._decomp.set(key, result)

    # ---- Retrieval -------------------------------------------------------

    def get_retrieval(self, sub_question: str, jurisdiction: Optional[str], top_k: int) -> Optional[list]:
        key = _make_key("retrieval", sub_question.strip().lower(), jurisdiction, top_k)
        result = self._retrieval.get(key)
        if result:
            logger.debug(f"Retrieval cache HIT: {sub_question[:60]!r}")
        return result

    def set_retrieval(self, sub_question: str, jurisdiction: Optional[str], top_k: int, result: list) -> None:
        key = _make_key("retrieval", sub_question.strip().lower(), jurisdiction, top_k)
        self._retrieval.set(key, result)

    # ---- Housekeeping ----------------------------------------------------

    def clear(self) -> None:
        self._decomp.clear()
        self._retrieval.clear()
        logger.info("Component cache cleared")

    def evict_expired(self) -> int:
        return self._decomp.evict_expired() + self._retrieval.evict_expired()

    def stats(self) -> dict:
        return {
            "decomposition_entries": self._decomp.size(),
            "retrieval_entries":     self._retrieval.size(),
        }
