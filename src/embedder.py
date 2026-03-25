"""
embedder.py — ReguGrounded Embedding & Pinecone Upload
=======================================================
Converts regulatory text chunks into vector embeddings using a local
sentence-transformers model, then upserts them into Pinecone with full
metadata so the RAG layer can filter by document, section, and heading.

Pipeline position:
    Chunking → [Embedding & Upload] → Pinecone → Agentic Retrieval

Embedding model: all-mpnet-base-v2
    - 768 dimensions
    - Runs fully locally, no API key or cost
    - Strong semantic similarity for general + regulatory text

Pinecone index: regugrounded
    - Each vector ID = chunk_id  (e.g. "eu_ai_act_chunk_042")
    - Metadata stored per vector:
        doc_id, section, heading, text, char_count, token_est, source_file
    - Metadata enables filtered search: "only search EU AI Act articles"

Usage:
    python src/embedder.py                    # embed & upload all docs
    python src/embedder.py --doc eu_ai_act    # single doc
    python src/embedder.py --dry-run          # embed only, skip Pinecone
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("embedder")

# Load .env from project root (one level up from src/)
load_dotenv(Path(__file__).parent.parent / ".env")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EMBEDDING_MODEL  = "all-mpnet-base-v2"   # 768-dim, local, free
PINECONE_DIMS    = 768
PINECONE_METRIC  = "cosine"
BATCH_SIZE       = 64    # vectors per upsert call (Pinecone max is 100)
CHUNKS_DIR       = Path("data/chunks")


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class Embedder:
    """
    Loads chunk JSON files, generates embeddings locally, and upserts to
    Pinecone. Designed to be idempotent — re-running overwrites existing
    vectors with the same chunk_id rather than duplicating them.

    Args:
        dry_run: If True, embed but don't touch Pinecone. Useful for
                 testing embedding quality before committing to upload.
    """

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run

        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"  Model loaded — output dims: {self.model.get_sentence_embedding_dimension()}")

        if not dry_run:
            self.index = self._init_pinecone()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_and_upload(self, chunks_path: str | Path) -> dict:
        """
        Embed all chunks in a JSON file and upsert to Pinecone.

        Args:
            chunks_path: Path to a *_chunks.json file from chunker.py.

        Returns:
            Summary dict: {doc_id, total_chunks, uploaded, duration_sec}
        """
        chunks_path = Path(chunks_path)
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        doc_id = chunks[0]["doc_id"] if chunks else chunks_path.stem

        logger.info(f"Embedding {len(chunks)} chunks from {chunks_path.name}...")
        start = time.time()

        # Extract texts and generate all embeddings in one batched pass
        texts = [c["text"] for c in chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,   # cosine similarity works on normalized vecs
        )

        if self.dry_run:
            logger.info(f"  [dry-run] Skipping Pinecone upload for {doc_id}")
            duration = time.time() - start
            return {"doc_id": doc_id, "total_chunks": len(chunks), "uploaded": 0, "duration_sec": duration}

        # Upsert in batches
        uploaded = self._upsert_batches(chunks, embeddings)

        duration = time.time() - start
        logger.info(f"  Done: {uploaded} vectors upserted in {duration:.1f}s")

        return {
            "doc_id":       doc_id,
            "total_chunks": len(chunks),
            "uploaded":     uploaded,
            "duration_sec": duration,
        }

    def embed_all(self, chunks_dir: str | Path = CHUNKS_DIR) -> list[dict]:
        """
        Embed and upload every *_chunks.json file in chunks_dir.
        Returns a list of summary dicts, one per document.
        """
        chunks_dir = Path(chunks_dir)
        chunk_files = sorted(chunks_dir.glob("*_chunks.json"))

        if not chunk_files:
            logger.warning(f"No chunk files found in {chunks_dir}")
            return []

        results = []
        for f in chunk_files:
            try:
                result = self.embed_and_upload(f)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed on {f.name}: {e}")
                results.append({"doc_id": f.stem, "error": str(e)})

        return results

    # ------------------------------------------------------------------
    # Pinecone setup
    # ------------------------------------------------------------------

    def _init_pinecone(self):
        """
        Connect to Pinecone and create the index if it doesn't exist yet.
        Returns the Index object ready for upserts.
        """
        from pinecone import Pinecone, ServerlessSpec

        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "PINECONE_API_KEY not set. Add it to your .env file."
            )

        index_name = os.getenv("PINECONE_INDEX_NAME", "regugrounded")

        pc = Pinecone(api_key=api_key)

        existing = [i.name for i in pc.list_indexes()]
        if index_name not in existing:
            logger.info(f"Creating Pinecone index '{index_name}' ({PINECONE_DIMS} dims, {PINECONE_METRIC})...")
            pc.create_index(
                name=index_name,
                dimension=PINECONE_DIMS,
                metric=PINECONE_METRIC,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            # Wait for index to be ready
            while not pc.describe_index(index_name).status["ready"]:
                logger.info("  Waiting for index to initialize...")
                time.sleep(2)
            logger.info(f"  Index '{index_name}' ready.")
        else:
            logger.info(f"Using existing Pinecone index '{index_name}'")

        return pc.Index(index_name)

    # ------------------------------------------------------------------
    # Upsert helpers
    # ------------------------------------------------------------------

    def _upsert_batches(self, chunks: list[dict], embeddings) -> int:
        """
        Upsert chunks + embeddings to Pinecone in batches of BATCH_SIZE.

        Each Pinecone vector:
            id:       chunk_id  (unique, stable — re-runs overwrite cleanly)
            values:   embedding (list of 768 floats)
            metadata: doc_id, section, heading, text, char_count, token_est, source_file

        Pinecone metadata note: text is stored so we can retrieve the full
        chunk text without a secondary database lookup during RAG.
        """
        total_uploaded = 0

        for i in range(0, len(chunks), BATCH_SIZE):
            batch_chunks = chunks[i : i + BATCH_SIZE]
            batch_embeds = embeddings[i : i + BATCH_SIZE]

            vectors = []
            for chunk, embedding in zip(batch_chunks, batch_embeds):
                vectors.append({
                    "id":     chunk["chunk_id"],
                    "values": embedding.tolist(),
                    "metadata": {
                        "doc_id":      chunk["doc_id"],
                        "section":     chunk["section"],
                        "heading":     chunk["heading"],
                        "text":        chunk["text"],        # stored for retrieval
                        "char_count":  chunk["char_count"],
                        "token_est":   chunk["token_est"],
                        "source_file": chunk["source_file"],
                    },
                })

            self.index.upsert(vectors=vectors)
            total_uploaded += len(vectors)
            logger.info(f"  Upserted batch {i // BATCH_SIZE + 1} ({total_uploaded}/{len(chunks)} vectors)")

        return total_uploaded


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Embed regulatory chunks and upload to Pinecone.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/embedder.py                        # embed & upload all docs
  python src/embedder.py --doc eu_ai_act        # single doc only
  python src/embedder.py --dry-run              # test embeddings, skip Pinecone
        """,
    )
    parser.add_argument(
        "--doc",
        type=str,
        help="Process a single doc by name (e.g. eu_ai_act, nist_ai_framework).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate embeddings but skip Pinecone upload.",
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=CHUNKS_DIR,
        help=f"Directory containing *_chunks.json files (default: {CHUNKS_DIR})",
    )
    args = parser.parse_args()

    embedder = Embedder(dry_run=args.dry_run)

    if args.doc:
        chunk_file = args.chunks_dir / f"{args.doc}_chunks.json"
        if not chunk_file.exists():
            print(f"Error: {chunk_file} not found.")
            sys.exit(1)
        results = [embedder.embed_and_upload(chunk_file)]
    else:
        results = embedder.embed_all(args.chunks_dir)

    # Summary
    print(f"\n{'='*60}")
    print("  Embedding & Upload Summary")
    print(f"{'='*60}")
    total_vecs = 0
    for r in results:
        if "error" in r:
            print(f"  {r['doc_id']:<35} ERROR: {r['error']}")
        else:
            status = "dry-run" if args.dry_run else f"{r['uploaded']} vectors"
            print(f"  {r['doc_id']:<35} {r['total_chunks']:>4} chunks  {status}  ({r['duration_sec']:.1f}s)")
            total_vecs += r.get("uploaded", r.get("total_chunks", 0))
    print(f"{'='*60}")
    if not args.dry_run:
        print(f"  Total vectors in Pinecone: {total_vecs}")
    print()


if __name__ == "__main__":
    main()