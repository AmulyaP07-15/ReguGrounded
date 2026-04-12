# utils/metrics.py
#
# Lightweight in-process metrics tracker with JSON persistence.
# Tracks query success/failure, latency distribution, error types,
# and citation quality across the lifetime of the process.

import json
import logging
import os
import threading
import time
from typing import Dict, Optional

logger = logging.getLogger("metrics")

_DEFAULT_METRICS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "logs", "metrics.json"
)


class MetricsTracker:
    """
    Thread-safe metrics accumulator with JSON persistence.

    Tracks:
      - query counts (total / successful / failed)
      - latency (running sum + count for avg; min/max)
      - error type breakdown
      - citation accuracy (running sum for avg; hallucination rate)
      - cache hit/miss counts

    Call get_metrics() to get a snapshot dict at any time.
    Metrics are auto-saved after each record_* call when persist=True.
    """

    def __init__(
        self,
        metrics_path: Optional[str] = None,
        persist: bool = True,
    ):
        self._path = metrics_path or _DEFAULT_METRICS_PATH
        self._persist = persist
        self._lock = threading.Lock()
        self._data: Dict = self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_query(self, success: bool, latency_ms: int) -> None:
        """Record a completed query with its success status and latency."""
        with self._lock:
            q = self._data["queries"]
            q["total"] += 1
            if success:
                q["successful"] += 1
            else:
                q["failed"] += 1

            lat = self._data["latency"]
            lat["total_ms"] += latency_ms
            lat["count"] += 1
            if lat["min_ms"] is None or latency_ms < lat["min_ms"]:
                lat["min_ms"] = latency_ms
            if lat["max_ms"] is None or latency_ms > lat["max_ms"]:
                lat["max_ms"] = latency_ms

            self._maybe_save()

    def record_error(self, error_type: str) -> None:
        """Increment the counter for a specific error type."""
        with self._lock:
            errors = self._data["errors"]
            errors[error_type] = errors.get(error_type, 0) + 1
            self._maybe_save()

    def record_citation_issue(self, accuracy: float, hallucination_rate: float) -> None:
        """Record citation quality metrics for one query response."""
        with self._lock:
            cit = self._data["citations"]
            cit["total_responses"] += 1
            cit["accuracy_sum"] += accuracy
            cit["hallucination_sum"] += hallucination_rate
            self._maybe_save()

    def record_cache_hit(self, cache_layer: str = "query") -> None:
        """Record a cache hit for the given layer ('query', 'decomposition', 'retrieval')."""
        with self._lock:
            cache = self._data["cache"]
            layer_key = f"{cache_layer}_hits"
            cache[layer_key] = cache.get(layer_key, 0) + 1
            self._maybe_save()

    def record_cache_miss(self, cache_layer: str = "query") -> None:
        """Record a cache miss for the given layer."""
        with self._lock:
            cache = self._data["cache"]
            layer_key = f"{cache_layer}_misses"
            cache[layer_key] = cache.get(layer_key, 0) + 1
            self._maybe_save()

    def get_metrics(self) -> Dict:
        """
        Return a metrics snapshot with computed aggregates.

        Derived fields:
          - avg_latency_ms (float)
          - error_rate     (float, 0–1)
          - avg_citation_accuracy    (float, 0–1)
          - avg_hallucination_rate   (float, 0–1)
          - cache_hit_rate           (float per layer, 0–1)
        """
        with self._lock:
            d = self._data
            q  = d["queries"]
            lat = d["latency"]
            cit = d["citations"]

            avg_latency = (
                lat["total_ms"] / lat["count"] if lat["count"] > 0 else 0.0
            )
            error_rate = (
                q["failed"] / q["total"] if q["total"] > 0 else 0.0
            )
            avg_accuracy = (
                cit["accuracy_sum"] / cit["total_responses"]
                if cit["total_responses"] > 0 else 1.0
            )
            avg_hallucination = (
                cit["hallucination_sum"] / cit["total_responses"]
                if cit["total_responses"] > 0 else 0.0
            )

            cache = d["cache"]
            cache_hit_rates = {}
            for layer in ("query", "decomposition", "retrieval"):
                hits   = cache.get(f"{layer}_hits", 0)
                misses = cache.get(f"{layer}_misses", 0)
                total  = hits + misses
                cache_hit_rates[layer] = round(hits / total, 4) if total > 0 else 0.0

            return {
                "queries": {**q},
                "latency": {
                    **lat,
                    "avg_ms": round(avg_latency, 1),
                },
                "error_rate": round(error_rate, 4),
                "errors": {**d["errors"]},
                "citations": {
                    **cit,
                    "avg_accuracy": round(avg_accuracy, 4),
                    "avg_hallucination_rate": round(avg_hallucination, 4),
                },
                "cache": {
                    **cache,
                    "hit_rates": cache_hit_rates,
                },
            }

    def reset(self) -> None:
        """Reset all counters (useful between eval runs)."""
        with self._lock:
            self._data = self._default_data()
            self._maybe_save()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _default_data(self) -> Dict:
        return {
            "queries":   {"total": 0, "successful": 0, "failed": 0},
            "latency":   {"total_ms": 0, "count": 0, "min_ms": None, "max_ms": None},
            "errors":    {},
            "citations": {"total_responses": 0, "accuracy_sum": 0.0, "hallucination_sum": 0.0},
            "cache":     {},
        }

    def _load(self) -> Dict:
        if self._persist and os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, KeyError):
                logger.warning("metrics.json corrupt — resetting")
        return self._default_data()

    def _maybe_save(self) -> None:
        if not self._persist:
            return
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
        except OSError as e:
            logger.warning(f"Failed to persist metrics: {e}")
