"""
retriever.py — ReguGrounded Retrieval Function
================================================
Embeds an incoming query and retrieves the top-k most relevant regulatory
chunks from Pinecone, returning structured output with text, metadata, and
similarity scores.

Pipeline position:
    [RLM Decomposition] → sub-questions → [Retriever] → evidence → [Answer Synthesis]

This module is the interface between the data layer (Amulya) and the
generation layer (partner). Each call returns a ranked list of chunks
that the synthesis layer uses to build a cited answer.

Output format per result:
    {
        "chunk_id":    "eu_ai_act_chunk_042",
        "doc_id":      "eu_ai_act",
        "section":     "Article 9",
        "heading":     "Risk management system",
        "text":        "Article 9 requires providers of high-risk AI...",
        "score":       0.87,
        "char_count":  1423,
        "source_file": "data/processed/eu_ai_act.txt"
    }

Usage:
    retriever = Retriever()

    # Simple query
    results = retriever.retrieve("What are the bias audit requirements?")

    # Filter to a specific document
    results = retriever.retrieve(
        "What are the bias audit requirements?",
        filter={"doc_id": "nyc_local_law_144"}
    )

    # Retrieve for multiple sub-questions at once (for RLM pipeline)
    results = retriever.retrieve_multi(
        ["What is a bias audit?", "Who must conduct a bias audit?"],
        top_k=3
    )
"""

import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("retriever")

EMBEDDING_MODEL = "all-mpnet-base-v2"
DEFAULT_TOP_K   = 5

# Valid doc_id values for filter validation
VALID_DOC_IDS = {
    "eu_ai_act",
    "eu_ai_act_annex",
    "nyc_local_law_144",
    "nist_ai_framework",
    "colorado_ai_act",
}

# Maps internal doc_id → human-readable law name used in citations
DOC_ID_TO_LAW_NAME = {
    "eu_ai_act":          "EU AI Act",
    "eu_ai_act_annex":    "EU AI Act (Annex)",
    "nyc_local_law_144":  "NYC Local Law 144",
    "nist_ai_framework":  "NIST AI Risk Management Framework",
    "colorado_ai_act":    "Colorado AI Act",
}


class Retriever:
    """
    Embeds queries and retrieves relevant regulatory chunks from Pinecone.

    Uses the same embedding model as the ingestion pipeline (all-mpnet-base-v2)
    so query vectors are in the same space as the stored chunk vectors.

    Args:
        top_k: Default number of results to return per query. Can be
               overridden per call. Typical values: 3 (focused) to 10 (broad).
    """

    def __init__(self, top_k: int = DEFAULT_TOP_K):
        self.top_k = top_k

        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        self.index = self._init_pinecone()
        logger.info("Retriever ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter: dict | None = None,
    ) -> list[dict]:
        """
        Embed a query and return the top-k most relevant chunks.

        Args:
            query:  The natural language question to search for.
            top_k:  Number of results to return. Defaults to self.top_k.
            filter: Optional Pinecone metadata filter dict. Examples:
                      {"doc_id": "eu_ai_act"}
                      {"doc_id": {"$in": ["eu_ai_act", "nyc_local_law_144"]}}

        Returns:
            List of result dicts, ranked by similarity score (highest first).
            Each dict contains: chunk_id, doc_id, section, heading, text,
            score, char_count, source_file.

        Raises:
            ValueError: If filter contains an unrecognized doc_id.
        """
        k = top_k or self.top_k

        if filter:
            self._validate_filter(filter)

        # Embed the query with the same model used during ingestion
        query_vector = self.model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        # Query Pinecone
        response = self.index.query(
            vector=query_vector,
            top_k=k,
            include_metadata=True,
            filter=filter,
        )

        results = self._format_results(response.matches)

        logger.info(
            f"Query: '{query[:60]}...' → {len(results)} results "
            f"(top score: {results[0]['score']:.3f})" if results else
            f"Query: '{query[:60]}' → 0 results"
        )

        return results

    def retrieve_multi(
        self,
        queries: list[str],
        top_k: int | None = None,
        filter: dict | None = None,
        deduplicate: bool = True,
    ) -> list[dict]:
        """
        Retrieve results for multiple sub-questions and merge into one list.

        Designed for the RLM pipeline where a user question is decomposed
        into sub-questions. Each sub-question is retrieved independently,
        then results are merged, deduplicated, and re-ranked by score.

        Args:
            queries:      List of sub-questions from RLM decomposition.
            top_k:        Results per sub-question before deduplication.
            filter:       Metadata filter applied to all sub-questions.
            deduplicate:  If True (default), remove duplicate chunk_ids,
                          keeping the highest score for each.

        Returns:
            Merged, deduplicated list of results sorted by score descending.
        """
        k = top_k or self.top_k
        all_results = []

        for query in queries:
            results = self.retrieve(query, top_k=k, filter=filter)
            all_results.extend(results)

        if deduplicate:
            all_results = self._deduplicate(all_results)

        # Re-sort by score after merging
        all_results.sort(key=lambda r: r["score"], reverse=True)

        logger.info(
            f"retrieve_multi: {len(queries)} queries → "
            f"{len(all_results)} unique results after deduplication"
        )

        return all_results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_pinecone(self):
        """Connect to Pinecone and return the index object."""
        from pinecone import Pinecone

        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "PINECONE_API_KEY not set. Add it to your .env file."
            )

        index_name = os.getenv("PINECONE_INDEX_NAME", "regugrounded")
        pc = Pinecone(api_key=api_key)

        existing = [i.name for i in pc.list_indexes()]
        if index_name not in existing:
            raise RuntimeError(
                f"Pinecone index '{index_name}' not found. "
                "Run embedder.py first to create and populate the index."
            )

        logger.info(f"Connected to Pinecone index '{index_name}'")
        return pc.Index(index_name)

    def _format_results(self, matches) -> list[dict]:
        """
        Convert raw Pinecone matches into the format expected by the
        reasoning pipeline (reasoning_orchestrator.py).

        Output shape per result:
            {
                "text":     "regulatory snippet...",
                "metadata": {
                    "law_name":       "EU AI Act",
                    "article_number": "9",          # or None
                    "section_number": "20-871",     # or None
                    "chunk_id":       "eu_ai_act_chunk_042"
                },
                "score": 0.87
            }
        """
        results = []
        for match in matches:
            meta = match.metadata or {}
            doc_id  = meta.get("doc_id", "")
            section = meta.get("section", "")

            article_number, section_number = self._parse_section(section)

            results.append({
                "text": meta.get("text", ""),
                "metadata": {
                    "law_name":       DOC_ID_TO_LAW_NAME.get(doc_id, doc_id),
                    "article_number": article_number,
                    "section_number": section_number,
                    "chunk_id":       match.id,
                    # extra fields kept for debugging / future use
                    "doc_id":         doc_id,
                    "section":        section,
                    "heading":        meta.get("heading", ""),
                    "source_file":    meta.get("source_file", ""),
                },
                "score": round(float(match.score), 4),
            })
        return results

    def _parse_section(self, section: str) -> tuple[str | None, str | None]:
        """
        Parse a section label into (article_number, section_number).

        Examples:
            "Article 9"    → ("9",      None)
            "§ 20-871"     → (None,     "20-871")
            "GOVERN 1.3"   → (None,     "GOVERN 1.3")
            "SECTION 2"    → (None,     "2")
            "Recital (32)" → (None,     "Recital 32")
            "Preamble"     → (None,     None)
        """
        if not section:
            return None, None

        # Article N
        m = re.match(r"Article\s+(\d+)", section, re.IGNORECASE)
        if m:
            return m.group(1), None

        # § 20-871
        m = re.match(r"§\s*([\d\w-]+)", section)
        if m:
            return None, m.group(1)

        # GOVERN 1.3 / MAP 2.1 / MEASURE / MANAGE
        m = re.match(r"((?:GOVERN|MAP|MEASURE|MANAGE)\s+[\d.]+)", section, re.IGNORECASE)
        if m:
            return None, m.group(1).strip()

        # SECTION 2
        m = re.match(r"SECTION\s+(\d+)", section, re.IGNORECASE)
        if m:
            return None, m.group(1)

        # Recital (32)
        m = re.match(r"Recital\s+\((\d+)\)", section, re.IGNORECASE)
        if m:
            return None, f"Recital {m.group(1)}"

        return None, None

    def _deduplicate(self, results: list[dict]) -> list[dict]:
        """
        Remove duplicate chunk_ids, keeping the highest score for each.
        Preserves order of first occurrence for equal scores.
        """
        seen = {}
        for r in results:
            cid = r["chunk_id"]
            if cid not in seen or r["score"] > seen[cid]["score"]:
                seen[cid] = r
        return list(seen.values())

    def _validate_filter(self, filter: dict) -> None:
        """Warn if a doc_id filter references an unknown document."""
        doc_id = filter.get("doc_id")
        if isinstance(doc_id, str) and doc_id not in VALID_DOC_IDS:
            logger.warning(
                f"Unknown doc_id in filter: '{doc_id}'. "
                f"Valid options: {sorted(VALID_DOC_IDS)}"
            )
        elif isinstance(doc_id, dict):
            for val in doc_id.get("$in", []):
                if val not in VALID_DOC_IDS:
                    logger.warning(f"Unknown doc_id in filter: '{val}'")


# ---------------------------------------------------------------------------
# Module-level retrieve() function
# ---------------------------------------------------------------------------
# Partner's query_interface.py does: from retriever import retrieve
# This wraps the Retriever class so it can be imported as a plain function.

_retriever_instance: Retriever | None = None

def retrieve(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Module-level retrieval function for use by the reasoning pipeline.

    Lazily initializes the Retriever on first call so imports are fast.
    Returns results in the format expected by reasoning_orchestrator.py:
        [{"text": ..., "metadata": {...}, "score": ...}, ...]
    """
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = Retriever(top_k=top_k)
    return _retriever_instance.retrieve(query, top_k=top_k)


# ---------------------------------------------------------------------------
# Quick test — run directly to verify retrieval is working
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    retriever = Retriever(top_k=5)

    test_queries = [
        ("What are the bias audit requirements?",         None),
        ("What is a high-risk AI system?",                {"doc_id": "eu_ai_act"}),
        ("What governance practices does NIST recommend?",{"doc_id": "nist_ai_framework"}),
        ("What notice must employers give employees?",    {"doc_id": "nyc_local_law_144"}),
    ]

    for query, filter in test_queries:
        print(f"\n{'='*65}")
        print(f"Query:  {query}")
        if filter:
            print(f"Filter: {filter}")
        print(f"{'='*65}")

        results = retriever.retrieve(query, top_k=3, filter=filter)

        for i, r in enumerate(results, 1):
            m = r["metadata"]
            loc = f"Article {m['article_number']}" if m["article_number"] else m["section_number"] or "—"
            print(f"\n  [{i}] score={r['score']:.3f}  |  {m['law_name']}  |  {loc}")
            print(f"      {m['heading'][:70]}")
            print(f"      {r['text'][:200].replace(chr(10), ' ')}...")