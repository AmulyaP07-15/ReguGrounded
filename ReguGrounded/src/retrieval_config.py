# src/retrieval_config.py
#
# Centralised configuration for the four retrieval improvements.
# All tuneable parameters live here so they can be changed without
# touching retrieval logic.

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HybridConfig:
    """BM25 + semantic fusion weights."""
    semantic_weight:  float = 0.6   # weight given to Pinecone cosine score
    keyword_weight:   float = 0.4   # weight given to BM25 score
    # When query is short (≤2 tokens) or contains explicit legal citations,
    # boost keyword weight to favour exact term matching.
    citation_boost_keyword: float = 0.5
    short_query_boost_keyword: float = 0.5
    short_query_threshold: int = 3  # token count at or below which to boost


@dataclass
class RerankerConfig:
    """Cross-encoder reranking settings."""
    model_name:   str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    first_pass_k: int = 20   # candidates retrieved before reranking
    final_k:      int = 5    # results kept after reranking
    cache_ttl:    int = 3600  # seconds to cache (query_hash, chunk_id) scores
    batch_size:   int = 32   # cross-encoder batch size


@dataclass
class ExpansionConfig:
    """Query expansion settings."""
    expansion_limit:   int  = 3     # max synonyms added per matched term
    skip_short_tokens: int  = 3     # don't expand tokens shorter than this
    # If True, average all expanded query embeddings; if False, use OR in BM25 only
    embed_expanded:    bool = True


@dataclass
class RetrievalConfig:
    """Master config passed to EnhancedRetriever."""
    # Feature toggles
    use_hybrid:    bool = True
    use_filtering: bool = True
    use_reranking: bool = True
    use_expansion: bool = True

    # Sub-configs
    hybrid:    HybridConfig    = field(default_factory=HybridConfig)
    reranker:  RerankerConfig  = field(default_factory=RerankerConfig)
    expansion: ExpansionConfig = field(default_factory=ExpansionConfig)

    # BM25 index cache path (None = memory only)
    bm25_cache_path: Optional[str] = "cache/bm25_index.pkl"


# Module-level default — import this for standard use
DEFAULT_CONFIG = RetrievalConfig()
