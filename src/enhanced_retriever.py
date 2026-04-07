# src/enhanced_retriever.py
#
# Enhanced retrieval pipeline with four production-grade improvements:
#
#   1. Hybrid search      — BM25 keyword + Pinecone semantic, score fusion
#   2. Jurisdiction filter — jurisdiction detection + Pinecone metadata filter
#   3. Cross-encoder rerank — two-stage: broad retrieval → precise reranking
#   4. Query expansion     — synonym expansion before embedding/BM25
#
# Drop-in replacement for the original Retriever:
#   old: retriever.retrieve(query, top_k=5, filter={"doc_id": "eu_ai_act"})
#   new: retriever.retrieve(query, top_k=5, filter={"doc_id": "eu_ai_act"})
#
# All four features are toggle-able; all off == original behaviour.

import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from src.bm25_index import BM25Index
from src.query_expander import QueryExpander
from src.reranker import rerank
from src.retrieval_config import RetrievalConfig, DEFAULT_CONFIG

load_dotenv(Path(__file__).parent.parent / ".env")

logger = logging.getLogger("enhanced_retriever")

EMBEDDING_MODEL = "all-mpnet-base-v2"

# ---------------------------------------------------------------------------
# Jurisdiction detection helpers
# ---------------------------------------------------------------------------

# Maps query signals → Pinecone doc_ids
_JURISDICTION_SIGNALS: Dict[str, List[str]] = {
    # Explicit regulation names
    "eu ai act":         ["eu_ai_act", "eu_ai_act_annex"],
    "european":          ["eu_ai_act", "eu_ai_act_annex"],
    "eu act":            ["eu_ai_act", "eu_ai_act_annex"],
    "nyc":               ["nyc_local_law_144"],
    "new york":          ["nyc_local_law_144"],
    "local law 144":     ["nyc_local_law_144"],
    "ll 144":            ["nyc_local_law_144"],
    "ll144":             ["nyc_local_law_144"],
    "§ 20-":             ["nyc_local_law_144"],
    "20-871":            ["nyc_local_law_144"],
    "colorado":          ["colorado_ai_act"],
    "sb 24-205":         ["colorado_ai_act"],
    "nist":              ["nist_ai_framework"],
    "ai rmf":            ["nist_ai_framework"],
    "risk management framework": ["nist_ai_framework"],
    # Concept → jurisdiction hints
    "bias audit":        ["nyc_local_law_144", "colorado_ai_act"],
    "aedt":              ["nyc_local_law_144"],
    "automated employ":  ["nyc_local_law_144"],
    "impact assessment": ["colorado_ai_act"],
    "conformity":        ["eu_ai_act"],
    "post-market":       ["eu_ai_act"],
    "general purpose ai": ["eu_ai_act"],
    "gpai":              ["eu_ai_act"],
    "govern":            ["nist_ai_framework"],
    "map function":      ["nist_ai_framework"],
    "measure function":  ["nist_ai_framework"],
}

# Terms that indicate a cross-jurisdiction / comparison query → don't filter
_COMPARISON_SIGNALS = ["compare", "vs", "versus", "differ", "difference", "contrast", "both", "all"]

# Legal citation pattern — keyword weight boost triggers on this too
_CITATION_PATTERN = re.compile(r"§\s*[\d][\d\w-]+|article\s+\d+|\bsection\s+\d+", re.I)


def _detect_jurisdictions(query: str) -> Optional[List[str]]:
    """
    Detect relevant doc_ids from query text.

    Returns:
        List of doc_id strings if specific jurisdiction(s) detected.
        None if the query is cross-jurisdiction or ambiguous (don't filter).
    """
    q = query.lower()

    # Comparison queries → no filter
    if any(sig in q for sig in _COMPARISON_SIGNALS):
        logger.debug("Comparison query detected — no jurisdiction filter")
        return None

    matched_doc_ids: set = set()
    for signal, doc_ids in _JURISDICTION_SIGNALS.items():
        if signal in q:
            matched_doc_ids.update(doc_ids)

    if not matched_doc_ids:
        return None  # ambiguous — retrieve from all

    result = sorted(matched_doc_ids)
    logger.debug(f"Jurisdiction filter: {result}")
    return result


def _keyword_weight(query: str, base_keyword_w: float, cfg) -> float:
    """
    Boost keyword weight when query contains legal citations or is very short.
    """
    tokens = query.split()
    if len(tokens) <= cfg.hybrid.short_query_threshold:
        return cfg.hybrid.short_query_boost_keyword
    if _CITATION_PATTERN.search(query):
        return cfg.hybrid.citation_boost_keyword
    return base_keyword_w


# ---------------------------------------------------------------------------
# EnhancedRetriever
# ---------------------------------------------------------------------------

class EnhancedRetriever:
    """
    Drop-in replacement for src.retriever.Retriever with four enhancements.

    All improvements are individually toggle-able via RetrievalConfig.
    When all features are disabled, behaviour is identical to the original.
    """

    def __init__(
        self,
        top_k:  int = 5,
        config: RetrievalConfig = DEFAULT_CONFIG,
    ):
        self.top_k  = top_k
        self.config = config

        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = self._init_pinecone()

        # BM25 index (only if hybrid enabled)
        self._bm25: Optional[BM25Index] = None
        if config.use_hybrid:
            self._bm25 = BM25Index.load_or_build(
                cache_path=config.bm25_cache_path,
            )

        # Query expander (only if expansion enabled)
        self._expander: Optional[QueryExpander] = None
        if config.use_expansion:
            self._expander = QueryExpander(
                expansion_limit=config.expansion.expansion_limit,
                skip_short_tokens=config.expansion.skip_short_tokens,
            )

        active = [
            feat for feat, on in [
                ("hybrid", config.use_hybrid),
                ("filtering", config.use_filtering),
                ("reranking", config.use_reranking),
                ("expansion", config.use_expansion),
            ] if on
        ]
        logger.info(f"EnhancedRetriever ready — active features: {active or ['baseline']}")

    # ------------------------------------------------------------------
    # Public API (matches original Retriever.retrieve signature)
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query:       str,
        top_k:       Optional[int] = None,
        filter:      Optional[Dict] = None,
        search_mode: str = "hybrid",   # "semantic" | "keyword" | "hybrid"
    ) -> List[Dict]:
        """
        Retrieve top-k regulatory chunks for a query.

        Args:
            query:       Natural language question.
            top_k:       Final number of results. Defaults to self.top_k.
            filter:      Explicit Pinecone filter (overrides auto-detection).
            search_mode: "hybrid" (default), "semantic", or "keyword".

        Returns:
            List of result dicts in the format expected by reasoning_orchestrator:
            [{"text", "metadata": {"law_name", "article_number", "section_number",
                                   "chunk_id", ...}, "score"}, ...]
        """
        final_k = top_k or self.top_k
        t0      = time.perf_counter()

        # ── 1. Query expansion ────────────────────────────────────────
        expanded_query    = query
        expansion_terms   = []
        if self.config.use_expansion and self._expander:
            expanded_query, expansion_terms = self._expander.expand(query)

        # ── 2. Jurisdiction detection / filter ────────────────────────
        pinecone_filter = filter   # caller-supplied filter takes precedence
        filter_doc_ids  = None

        if pinecone_filter is None and self.config.use_filtering:
            doc_ids = _detect_jurisdictions(query)
            if doc_ids:
                pinecone_filter = {"doc_id": {"$in": doc_ids}}
                filter_doc_ids  = doc_ids
                logger.info(f"Auto jurisdiction filter: {doc_ids}")
        elif pinecone_filter and "doc_id" in pinecone_filter:
            # Extract doc_ids from caller-supplied filter for BM25 filtering
            v = pinecone_filter["doc_id"]
            if isinstance(v, str):
                filter_doc_ids = [v]
            elif isinstance(v, dict) and "$in" in v:
                filter_doc_ids = v["$in"]

        # ── 3. Determine first-pass k ─────────────────────────────────
        first_pass_k = (
            self.config.reranker.first_pass_k
            if self.config.use_reranking
            else final_k
        )

        # ── 4. Retrieve candidates ────────────────────────────────────
        effective_mode = search_mode
        if not self.config.use_hybrid:
            effective_mode = "semantic"

        if effective_mode == "semantic":
            candidates = self._semantic_retrieve(
                expanded_query, first_pass_k, pinecone_filter
            )

        elif effective_mode == "keyword":
            candidates = self._keyword_retrieve(
                expanded_query, first_pass_k, filter_doc_ids
            )
            # Keyword-only still needs Pinecone metadata for law_name etc.
            candidates = self._enrich_from_bm25(candidates)

        else:  # hybrid
            candidates = self._hybrid_retrieve(
                query=query,
                expanded_query=expanded_query,
                top_k=first_pass_k,
                pinecone_filter=pinecone_filter,
                filter_doc_ids=filter_doc_ids,
            )

        # ── 5. Cross-encoder reranking ────────────────────────────────
        if self.config.use_reranking and len(candidates) > 1:
            candidates = rerank(
                query=query,
                candidates=candidates,
                final_k=final_k,
                model_name=self.config.reranker.model_name,
                batch_size=self.config.reranker.batch_size,
                cache_ttl=self.config.reranker.cache_ttl,
            )
        else:
            candidates = candidates[:final_k]

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Build log summary
        feature_notes = []
        if expansion_terms:
            feature_notes.append(f"expanded(+{len(expansion_terms)})")
        if filter_doc_ids:
            feature_notes.append(f"filtered({','.join(filter_doc_ids)})")
        if effective_mode == "hybrid":
            feature_notes.append("hybrid")
        if self.config.use_reranking:
            feature_notes.append("reranked")

        logger.info(
            f"Retrieved {len(candidates)} chunks in {elapsed_ms:.0f} ms"
            + (f" [{', '.join(feature_notes)}]" if feature_notes else "")
            + (f" — top score: {candidates[0]['score']:.3f}" if candidates else "")
        )

        return candidates

    # ------------------------------------------------------------------
    # Internal retrieval methods
    # ------------------------------------------------------------------

    def _semantic_retrieve(
        self,
        query:          str,
        top_k:          int,
        pinecone_filter: Optional[Dict],
    ) -> List[Dict]:
        """Pure semantic retrieval via Pinecone."""
        vec = self.model.encode(query, normalize_embeddings=True).tolist()
        resp = self.index.query(
            vector=vec,
            top_k=top_k,
            include_metadata=True,
            filter=pinecone_filter,
        )
        return self._format_pinecone_results(resp.matches, score_key="score")

    def _keyword_retrieve(
        self,
        query:          str,
        top_k:          int,
        filter_doc_ids: Optional[List[str]],
    ) -> List[Dict]:
        """Pure BM25 keyword retrieval."""
        if self._bm25 is None:
            logger.warning("BM25 index not available — falling back to semantic")
            return []
        return self._bm25.search(query, top_k=top_k, filter_doc_ids=filter_doc_ids)

    def _hybrid_retrieve(
        self,
        query:           str,
        expanded_query:  str,
        top_k:           int,
        pinecone_filter: Optional[Dict],
        filter_doc_ids:  Optional[List[str]],
    ) -> List[Dict]:
        """
        Hybrid retrieval: run semantic + keyword in parallel, fuse scores.
        """
        cfg = self.config.hybrid
        kw  = _keyword_weight(query, cfg.keyword_weight, self.config)
        sem = 1.0 - kw

        # Semantic pass
        sem_results = self._semantic_retrieve(expanded_query, top_k, pinecone_filter)

        # BM25 pass
        kw_results: List[Dict] = []
        if self._bm25:
            kw_results = self._bm25.search(
                expanded_query, top_k=top_k, filter_doc_ids=filter_doc_ids
            )

        if not kw_results:
            # BM25 unavailable — pure semantic
            logger.debug("Hybrid: BM25 unavailable, using semantic only")
            return sem_results

        # Build lookup maps: chunk_id → score
        sem_map: Dict[str, Dict] = {r["chunk_id"]: r for r in sem_results}
        kw_map:  Dict[str, Dict] = {r["chunk_id"]: r for r in kw_results}

        all_ids = set(sem_map) | set(kw_map)
        fused: List[Dict] = []

        for cid in all_ids:
            sem_score_norm = sem_map[cid]["score"] if cid in sem_map else 0.0
            kw_score_norm  = kw_map[cid]["bm25_score_norm"] if cid in kw_map else 0.0
            combined       = sem * sem_score_norm + kw * kw_score_norm

            # Start from semantic result if available, else enrich from BM25
            if cid in sem_map:
                base = dict(sem_map[cid])
            else:
                base = self._bm25_to_pinecone_format(kw_map[cid])

            base["sem_score"]      = round(sem_score_norm, 6)
            base["bm25_score"]     = round(kw_score_norm,  6)
            base["score"]          = round(combined,        6)
            base["search_mode"]    = "hybrid"
            fused.append(base)

        fused.sort(key=lambda x: x["score"], reverse=True)

        if fused:
            top = fused[0]
            logger.debug(
                f"Hybrid top: sem={top['sem_score']:.3f} "
                f"kw={top['bm25_score']:.3f} "
                f"combined={top['score']:.3f} "
                f"(sem_w={sem:.1f}, kw_w={kw:.1f})"
            )

        return fused[:top_k]

    # ------------------------------------------------------------------
    # Format helpers
    # ------------------------------------------------------------------

    def _format_pinecone_results(self, matches, score_key: str = "score") -> List[Dict]:
        """Convert Pinecone match objects to pipeline-compatible dicts."""
        from src.retriever import DOC_ID_TO_LAW_NAME
        results = []
        for m in matches:
            meta = m.metadata or {}
            doc_id  = meta.get("doc_id", "")
            section = meta.get("section", "")
            article_number, section_number = self._parse_section(section)
            results.append({
                "text":     meta.get("text", ""),
                "chunk_id": m.id,
                "metadata": {
                    "law_name":       DOC_ID_TO_LAW_NAME.get(doc_id, doc_id),
                    "article_number": article_number,
                    "section_number": section_number,
                    "chunk_id":       m.id,
                    "doc_id":         doc_id,
                    "section":        section,
                    "heading":        meta.get("heading", ""),
                    "source_file":    meta.get("source_file", ""),
                },
                "score":       round(float(m.score), 6),
                "search_mode": "semantic",
            })
        return results

    def _bm25_to_pinecone_format(self, bm25_result: Dict) -> Dict:
        """Convert a BM25 result dict to pipeline-compatible format."""
        from src.retriever import DOC_ID_TO_LAW_NAME
        doc_id  = bm25_result.get("doc_id", "")
        section = bm25_result.get("section", "")
        article_number, section_number = self._parse_section(section)
        return {
            "text":     bm25_result.get("text", ""),
            "chunk_id": bm25_result["chunk_id"],
            "metadata": {
                "law_name":       DOC_ID_TO_LAW_NAME.get(doc_id, doc_id),
                "article_number": article_number,
                "section_number": section_number,
                "chunk_id":       bm25_result["chunk_id"],
                "doc_id":         doc_id,
                "section":        section,
                "heading":        bm25_result.get("heading", ""),
                "source_file":    bm25_result.get("source_file", ""),
            },
            "score":       bm25_result.get("bm25_score_norm", 0.0),
            "search_mode": "keyword",
        }

    def _enrich_from_bm25(self, bm25_results: List[Dict]) -> List[Dict]:
        """Convert BM25-only results to pipeline format."""
        return [self._bm25_to_pinecone_format(r) for r in bm25_results]

    @staticmethod
    def _parse_section(section: str):
        """Mirror of Retriever._parse_section for article/section parsing."""
        if not section:
            return None, None
        m = re.match(r"Article\s+(\d+)", section, re.IGNORECASE)
        if m:
            return m.group(1), None
        m = re.match(r"§\s*([\d\w-]+)", section)
        if m:
            return None, m.group(1)
        m = re.match(r"((?:GOVERN|MAP|MEASURE|MANAGE)\s+[\d.]+)", section, re.IGNORECASE)
        if m:
            return None, m.group(1).strip()
        m = re.match(r"SECTION\s+(\d+)", section, re.IGNORECASE)
        if m:
            return None, m.group(1)
        m = re.match(r"Recital\s+\((\d+)\)", section, re.IGNORECASE)
        if m:
            return None, f"Recital {m.group(1)}"
        return None, None

    def _init_pinecone(self):
        from pinecone import Pinecone
        api_key    = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME", "regugrounded")
        if not api_key:
            raise EnvironmentError("PINECONE_API_KEY not set")
        pc = Pinecone(api_key=api_key)
        existing = [i.name for i in pc.list_indexes()]
        if index_name not in existing:
            raise RuntimeError(f"Pinecone index '{index_name}' not found")
        logger.info(f"Connected to Pinecone index '{index_name}'")
        return pc.Index(index_name)
