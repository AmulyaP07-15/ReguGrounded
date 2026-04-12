# src/reranker.py
#
# Cross-encoder reranking for second-pass retrieval refinement.
#
# Stage 1 (broad): hybrid search retrieves top-20 candidates by combined score
# Stage 2 (precise): cross-encoder scores each (query, chunk_text) pair jointly,
#                    returning top-k by reranked relevance
#
# The cross-encoder reads both query and chunk together (not independently),
# making it much more sensitive to exact relevance than cosine similarity.

import hashlib
import logging
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("reranker")

# Lazy import — model is ~80 MB and takes ~2s to load; only loaded when reranking
_cross_encoder = None
_model_name    = None


def _get_cross_encoder(model_name: str):
    """Lazy singleton loader for the cross-encoder model."""
    global _cross_encoder, _model_name
    if _cross_encoder is None or _model_name != model_name:
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading cross-encoder: {model_name} ...")
        t0 = time.perf_counter()
        _cross_encoder = CrossEncoder(model_name)
        _model_name    = model_name
        logger.info(f"Cross-encoder loaded in {(time.perf_counter()-t0)*1000:.0f} ms")
    return _cross_encoder


# ---------------------------------------------------------------------------
# Result-level score cache (query_hash × chunk_id → score)
# ---------------------------------------------------------------------------

import time as _time

class _ScoreCache:
    """Simple TTL cache for cross-encoder scores."""

    def __init__(self, ttl: int = 3600):
        self._ttl  = ttl
        self._data: Dict[str, Tuple[float, float]] = {}  # key → (score, expires_at)

    def _key(self, query_hash: str, chunk_id: str) -> str:
        return f"{query_hash}::{chunk_id}"

    def get(self, query_hash: str, chunk_id: str) -> Optional[float]:
        k = self._key(query_hash, chunk_id)
        entry = self._data.get(k)
        if entry is None:
            return None
        score, expires_at = entry
        if _time.time() > expires_at:
            del self._data[k]
            return None
        return score

    def set(self, query_hash: str, chunk_id: str, score: float) -> None:
        k = self._key(query_hash, chunk_id)
        self._data[k] = (score, _time.time() + self._ttl)

    def evict_expired(self) -> int:
        now   = _time.time()
        stale = [k for k, (_, exp) in self._data.items() if now > exp]
        for k in stale:
            del self._data[k]
        return len(stale)


_score_cache = _ScoreCache()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rerank(
    query:        str,
    candidates:   List[Dict],
    final_k:      int  = 5,
    model_name:   str  = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    batch_size:   int  = 32,
    cache_ttl:    int  = 3600,
) -> List[Dict]:
    """
    Re-score `candidates` using a cross-encoder and return the top `final_k`.

    Each candidate must have:
      - "text"     : str — the chunk text to score against the query
      - "chunk_id" : str — unique ID (used for score caching)

    The input candidates are returned augmented with:
      - "rerank_score" : float — cross-encoder raw score (higher = more relevant)
      - "rerank_rank"  : int   — 1-based rank after reranking

    Reranking logs the position change of the top result and total latency.
    """
    if not candidates:
        return candidates

    ce     = _get_cross_encoder(model_name)
    q_hash = hashlib.sha256(query.encode()).hexdigest()[:16]

    # Separate cached and uncached pairs
    pairs_to_score: List[Tuple[int, str, str]] = []   # (original_idx, chunk_id, text)
    cached_scores:  Dict[int, float]           = {}

    for i, c in enumerate(candidates):
        chunk_id = c.get("chunk_id") or c.get("metadata", {}).get("chunk_id", f"chunk_{i}")
        cached   = _score_cache.get(q_hash, chunk_id)
        if cached is not None:
            cached_scores[i] = cached
        else:
            pairs_to_score.append((i, chunk_id, c.get("text", "")))

    # Batch-score uncached pairs
    if pairs_to_score:
        t0     = time.perf_counter()
        inputs = [(query, text) for _, _, text in pairs_to_score]
        raw    = ce.predict(inputs, batch_size=batch_size)
        scores = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            f"Cross-encoder scored {len(inputs)} pairs in {elapsed_ms:.0f} ms"
        )
        for (orig_idx, chunk_id, _), score in zip(pairs_to_score, scores):
            _score_cache.set(q_hash, chunk_id, score)
            cached_scores[orig_idx] = score
    else:
        logger.debug("All cross-encoder scores served from cache")

    # Attach scores to candidates
    for i, c in enumerate(candidates):
        c["rerank_score"] = round(cached_scores[i], 6)

    # Sort by rerank score
    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

    # Log position change of top result
    if candidates:
        original_top_id   = candidates[0].get("chunk_id") or candidates[0].get("metadata", {}).get("chunk_id")
        reranked_top_id   = reranked[0].get("chunk_id")   or reranked[0].get("metadata",  {}).get("chunk_id")
        if original_top_id != reranked_top_id:
            # Find where the original top ended up
            new_rank = next(
                (j + 1 for j, c in enumerate(reranked)
                 if (c.get("chunk_id") or c.get("metadata", {}).get("chunk_id")) == original_top_id),
                None,
            )
            logger.info(
                f"Reranking changed top result: "
                f"original top → rank {new_rank} after reranking"
            )
        else:
            logger.debug("Reranking confirmed original top result")

    # Annotate ranks
    for rank, c in enumerate(reranked, 1):
        c["rerank_rank"] = rank

    return reranked[:final_k]
