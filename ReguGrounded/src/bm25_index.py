# src/bm25_index.py
#
# BM25 keyword search index over all regulatory chunks.
#
# Loads all chunk JSON files, tokenises the text, builds a BM25Okapi index,
# and exposes a search() method that returns normalised scores aligned with
# Pinecone chunk IDs so the hybrid retriever can merge them.
#
# Index is optionally persisted to disk (pickle) so it doesn't need to be
# rebuilt on every process start.

import json
import logging
import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

logger = logging.getLogger("bm25_index")

# Path patterns for chunk files
_CHUNKS_DIR = Path(__file__).parent.parent / "data" / "chunks"

# Legal citation tokens to treat as atomic units (don't split on hyphen)
_CITATION_RE = re.compile(r"§\s*[\d][\d\w-]*|article\s+\d+|\d+-\d+\w*", re.I)


def _tokenize(text: str) -> List[str]:
    """
    Tokenise regulatory text for BM25.

    Strategy:
      1. Preserve legal citations as atomic tokens (§ 20-871, Article 9)
      2. Lowercase everything
      3. Split on whitespace / punctuation
      4. Drop stop-words and very short tokens
    """
    # Normalise citations first so they survive splitting
    text = text.lower()

    # Replace whitespace in known multi-word citations with underscore so
    # they stay as one token: "article 9" → "article_9"
    text = re.sub(r"article\s+(\d+)", r"article_\1", text)
    text = re.sub(r"§\s*([\d][\d\w-]*)", r"section_\1", text)

    # Tokenise
    tokens = re.findall(r"[a-z0-9][a-z0-9_\-]*", text)

    # Remove single-char tokens and common stop-words
    stop_words = {
        "the", "a", "an", "and", "or", "of", "to", "in", "for",
        "is", "are", "be", "by", "as", "at", "on", "with", "that",
        "this", "it", "its", "from", "not", "no", "shall", "may",
        "must", "will", "has", "have", "had", "such", "any", "all",
        "each", "which", "who", "whom", "their", "they", "them",
    }
    return [t for t in tokens if len(t) > 1 and t not in stop_words]


class BM25Index:
    """
    In-memory BM25 index over all regulatory chunks.

    Attributes:
        chunk_ids: list of chunk IDs in index order
        doc_ids:   list of doc_ids (jurisdiction) per chunk
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1         = k1
        self.b          = b
        self._bm25:     Optional[BM25Okapi] = None
        self.chunk_ids: List[str]  = []
        self.doc_ids:   List[str]  = []
        self._chunks:   List[Dict] = []   # full chunk dicts for result assembly

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, chunks_dir: Path = _CHUNKS_DIR) -> "BM25Index":
        """
        Load all chunk JSON files from chunks_dir and build the BM25 index.
        Returns self for chaining.
        """
        all_chunks = self._load_all_chunks(chunks_dir)
        self._chunks   = all_chunks
        self.chunk_ids = [c["chunk_id"] for c in all_chunks]
        self.doc_ids   = [c["doc_id"]   for c in all_chunks]

        corpus = [_tokenize(c["text"]) for c in all_chunks]
        self._bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)

        logger.info(
            f"BM25 index built: {len(all_chunks)} chunks, "
            f"vocab size ≈ {len(set(t for doc in corpus for t in doc)):,}"
        )
        return self

    def _load_all_chunks(self, chunks_dir: Path) -> List[Dict]:
        """Load and merge all *_chunks.json files."""
        all_chunks = []
        for path in sorted(chunks_dir.glob("*_chunks.json")):
            with open(path, encoding="utf-8") as f:
                chunks = json.load(f)
            for c in chunks:
                # Ensure text field exists (some chunks store body separately)
                if "text" not in c:
                    c["text"] = c.get("body", "")
            all_chunks.extend(chunks)
            logger.debug(f"Loaded {len(chunks)} chunks from {path.name}")
        logger.info(f"Total chunks loaded: {len(all_chunks)}")
        return all_chunks

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 20,
        filter_doc_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        BM25 keyword search.

        Args:
            query:          User query string (tokenised internally).
            top_k:          Number of results to return.
            filter_doc_ids: If given, only return chunks from these doc_ids.

        Returns:
            List of dicts: {"chunk_id", "doc_id", "bm25_score", "bm25_score_norm",
                            "text", "section", "heading", "source_file"}
            Sorted by bm25_score descending.
        """
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built — call .build() first")

        tokens = _tokenize(query)
        if not tokens:
            return []

        raw_scores: List[float] = self._bm25.get_scores(tokens).tolist()

        # Pair with metadata
        paired: List[Tuple[float, int]] = [
            (score, i)
            for i, score in enumerate(raw_scores)
            if (filter_doc_ids is None or self.doc_ids[i] in filter_doc_ids)
        ]

        # Sort descending
        paired.sort(key=lambda x: x[0], reverse=True)

        # Normalise scores to [0, 1]
        max_score = paired[0][0] if paired and paired[0][0] > 0 else 1.0
        min_score = 0.0

        results = []
        for score, idx in paired[:top_k]:
            chunk = self._chunks[idx]
            norm  = (score - min_score) / (max_score - min_score) if max_score > 0 else 0.0
            results.append({
                "chunk_id":        chunk["chunk_id"],
                "doc_id":          chunk["doc_id"],
                "bm25_score":      round(score, 6),
                "bm25_score_norm": round(norm, 6),
                "text":            chunk["text"],
                "section":         chunk.get("section", ""),
                "heading":         chunk.get("heading", ""),
                "source_file":     chunk.get("source_file", ""),
            })

        return results

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """Return the raw chunk dict for a given chunk_id."""
        for c in self._chunks:
            if c["chunk_id"] == chunk_id:
                return c
        return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Pickle the index to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "bm25":      self._bm25,
                "chunk_ids": self.chunk_ids,
                "doc_ids":   self.doc_ids,
                "chunks":    self._chunks,
                "k1":        self.k1,
                "b":         self.b,
            }, f)
        logger.info(f"BM25 index saved → {path}")

    def load(self, path: str) -> "BM25Index":
        """Load a pickled index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._bm25     = data["bm25"]
        self.chunk_ids = data["chunk_ids"]
        self.doc_ids   = data["doc_ids"]
        self._chunks   = data["chunks"]
        self.k1        = data.get("k1", self.k1)
        self.b         = data.get("b",  self.b)
        logger.info(f"BM25 index loaded from {path} ({len(self.chunk_ids)} chunks)")
        return self

    @classmethod
    def load_or_build(
        cls,
        cache_path: Optional[str] = None,
        chunks_dir: Path = _CHUNKS_DIR,
        k1: float = 1.5,
        b:  float = 0.75,
    ) -> "BM25Index":
        """
        Load from cache if available and fresh, otherwise build and save.
        """
        idx = cls(k1=k1, b=b)

        if cache_path and os.path.exists(cache_path):
            # Check if any chunk file is newer than the cached index
            cache_mtime = os.path.getmtime(cache_path)
            stale = any(
                os.path.getmtime(p) > cache_mtime
                for p in chunks_dir.glob("*_chunks.json")
            )
            if not stale:
                idx.load(cache_path)
                return idx
            logger.info("Chunk files updated — rebuilding BM25 index")

        idx.build(chunks_dir)
        if cache_path:
            idx.save(cache_path)
        return idx
