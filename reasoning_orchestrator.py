# reasoning_orchestrator.py

from typing import Callable, Dict, List, Optional

from utils.logger import setup_logger, log_function_call
from utils.cache import ComponentCache

logger = setup_logger("reasoning_orchestrator")
_component_cache = ComponentCache()

# Maps RLM jurisdiction names → Pinecone doc_id values for filtered retrieval
_DOC_ID_MAP: Dict[str, str] = {
    "EU AI Act":                          "eu_ai_act",
    "NYC Local Law 144":                  "nyc_local_law_144",
    "Colorado AI Act":                    "colorado_ai_act",
    "NIST AI Risk Management Framework":  "nist_ai_framework",
    "NIST AI RMF":                        "nist_ai_framework",
}


class ReasoningOrchestrator:
    """Runs retrieval for each sub-question and organizes results."""

    def __init__(self, retriever_func: Callable):
        self.retriever_func = retriever_func

    @log_function_call(logger)
    def run(self, decomposition: Dict, top_k: int = 5) -> List[Dict]:
        """
        Retrieve evidence for each sub-question.

        When a sub-question has a known jurisdiction, retrieval is filtered to
        that jurisdiction's doc_id so cross-jurisdiction noise is eliminated.

        Returns evidence_bundle: list of per-jurisdiction dicts, each with:
            jurisdiction, sub_question, retrieved_chunks (normalized, deduped)
        """
        evidence_bundle = []
        seen_chunk_ids: set = set()

        for item in decomposition["sub_questions"]:
            jurisdiction = item["jurisdiction"]
            sub_question = item["sub_question"]
            doc_filter   = self._jurisdiction_filter(jurisdiction)

            # Check retrieval cache
            cached_chunks = _component_cache.get_retrieval(sub_question, jurisdiction, top_k)
            if cached_chunks is not None:
                normalized = cached_chunks
                logger.debug(f"Retrieval cache hit for: {sub_question[:60]!r}")
            else:
                retrieval_ok = True
                try:
                    raw_chunks = self._retrieve(sub_question, top_k=top_k, filter=doc_filter)
                except Exception as e:
                    logger.error(f"Retrieval failed for '{sub_question[:60]}': {e}")
                    raw_chunks = []
                    retrieval_ok = False

                normalized = self._normalize_chunks(raw_chunks)
                # Only cache successful retrievals — don't persist empty error results
                if retrieval_ok:
                    _component_cache.set_retrieval(sub_question, jurisdiction, top_k, normalized)

            # Deduplicate across sub-questions (keep first occurrence)
            unique_chunks = []
            for chunk in normalized:
                cid = chunk.get("chunk_id") or chunk.get("text", "")[:80]
                if cid not in seen_chunk_ids:
                    seen_chunk_ids.add(cid)
                    unique_chunks.append(chunk)

            filter_note = f" [filtered: {doc_filter['doc_id']}]" if doc_filter else ""
            logger.info(
                f"Sub-question '{sub_question[:60]}'{filter_note} → "
                f"{len(normalized)} chunks, {len(unique_chunks)} new after dedup"
            )

            evidence_bundle.append({
                "jurisdiction":    jurisdiction,
                "sub_question":    sub_question,
                "retrieved_chunks": unique_chunks,
            })

        return evidence_bundle

    def get_all_chunks(self, evidence_bundle: List[Dict]) -> List[Dict]:
        """Flatten evidence bundle into a single list of chunks."""
        return [chunk for bundle in evidence_bundle for chunk in bundle["retrieved_chunks"]]

    def get_jurisdictions(self, evidence_bundle: List[Dict]) -> List[str]:
        """Return unique non-null jurisdictions from the bundle."""
        seen = set()
        result = []
        for bundle in evidence_bundle:
            j = bundle.get("jurisdiction")
            if j and j not in seen:
                seen.add(j)
                result.append(j)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _jurisdiction_filter(self, jurisdiction: Optional[str]) -> Optional[Dict]:
        """Return a Pinecone filter dict for a known jurisdiction, else None."""
        if not jurisdiction:
            return None
        doc_id = _DOC_ID_MAP.get(jurisdiction)
        return {"doc_id": doc_id} if doc_id else None

    def _retrieve(self, query: str, top_k: int, filter: Optional[Dict]) -> List[Dict]:
        """
        Call retriever_func with filter support.

        The real Retriever.retrieve() accepts a `filter` kwarg; mock retrievers
        in tests may not — fall back gracefully without the filter in that case.
        """
        try:
            return self.retriever_func(query, top_k=top_k, filter=filter)
        except TypeError:
            # Retriever doesn't accept filter (e.g. old mock) — call without
            return self.retriever_func(query, top_k=top_k)

    def _normalize_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Ensure all retrieved chunks follow one consistent format."""
        normalized = []
        for chunk in chunks:
            text     = chunk.get("text", "").strip()
            metadata = chunk.get("metadata", {})
            score    = chunk.get("score")

            normalized.append({
                "text":           text,
                "law_name":       metadata.get("law_name"),
                "article_number": metadata.get("article_number"),
                "section_number": metadata.get("section_number"),
                "chunk_id":       metadata.get("chunk_id"),
                "score":          score,
            })
        return normalized
