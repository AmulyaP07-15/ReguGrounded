"""
chunker.py — ReguGrounded Intelligent Chunking
================================================
Splits cleaned regulatory text files into semantically meaningful chunks,
preserving legal structure so each chunk can be cited accurately.

Pipeline position:
    PDF Ingestion → [Intelligent Chunking] → Embedding → Pinecone → RAG

Each output chunk is a dict with metadata so Pinecone can store and filter:
    {
        "chunk_id":    "eu_ai_act_article_003",
        "doc_id":      "eu_ai_act",
        "section":     "Article 3",
        "heading":     "Definitions",
        "text":        "For the purpose of this Regulation...",
        "char_count":  1842,
        "token_est":   460,          # rough estimate: chars / 4
        "source_file": "data/processed/eu_ai_act.txt"
    }

Chunking strategy per document:
    - NYC Local Law 144  → split on § 20-870, § 20-871, ...
    - EU AI Act          → split on Article N
    - Colorado AI Act    → split on SECTION N
    - NIST AI RMF        → split on GOVERN/MAP/MEASURE/MANAGE subcategories
                           + numbered sections (1., 1.1, ...)
    - EU AI Act Annex    → split on Annex headings

Long sections (> max_chunk_chars) are sub-split with overlap so no
embedding loses context at a boundary.

Usage:
    chunker = DocumentChunker()
    chunks = chunker.chunk_file("data/processed/eu_ai_act.txt")
    chunker.save_chunks(chunks, "data/chunks/")
"""

import json
import re
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Iterator

logger = logging.getLogger("chunker")

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """One retrievable unit of a regulatory document."""
    chunk_id:    str
    doc_id:      str
    section:     str        # e.g. "Article 3" or "§ 20-871" or "GOVERN 1.1"
    heading:     str        # e.g. "Definitions" or "Requirements for automated..."
    text:        str        # full text of this chunk (includes section + heading)
    char_count:  int = field(init=False)
    token_est:   int = field(init=False)
    source_file: str = ""

    def __post_init__(self):
        self.char_count = len(self.text)
        self.token_est  = self.char_count // 4   # rough 4 chars/token heuristic

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Per-document split patterns
# ---------------------------------------------------------------------------

# Each entry: (doc_id_substring, compiled regex that marks the START of a new chunk)
# The regex must match the first line of a section (the "§ 20-870" line, "Article 3" line, etc.)
# Flags: re.MULTILINE so ^ matches start of each line, not just start of string.

SPLIT_PATTERNS = {
    "nyc_local_law_144": re.compile(
        r"^(§\s*20-\d+[^\n]*)",
        re.MULTILINE,
    ),
    "eu_ai_act": re.compile(
        r"^(Article\s+\d+\b[^\n]*|Recital\s+\(\d+\)[^\n]*)",
        re.MULTILINE | re.IGNORECASE,
    ),
    "eu_ai_act_annex": re.compile(
        r"^(ANNEX\s+[IVX\d]+[^\n]*|LIST\s+OF\s+[A-Z]+[^\n]*)",
        re.MULTILINE | re.IGNORECASE,
    ),
    "colorado_ai_act": re.compile(
        r"^(SECTION\s+\d+[^\n]*)",
        re.MULTILINE | re.IGNORECASE,
    ),
    "nist_ai_framework": re.compile(
        # GOVERN 1.1:, MAP 2.3:, MEASURE 1.1:, MANAGE 4.2:
        # Also top-level numbered sections: "1 Framing Risk", "1.1 Understanding..."
        r"^((?:GOVERN|MAP|MEASURE|MANAGE)\s+\d+(?:\.\d+)?[^\n]*"
        r"|^\d+(?:\.\d+)*\s+[A-Z][^\n]{5,})",
        re.MULTILINE,
    ),
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DocumentChunker:
    """
    Splits a processed regulatory text file into chunks aligned with the
    document's legal structure (articles, sections, subcategories).

    Args:
        max_chunk_chars: Chunks longer than this are sub-split with overlap.
                         Default 2000 chars ≈ 500 tokens — fits most embedding
                         models (Ada-002 limit is 8191 tokens).
        overlap_chars:   Characters of overlap between sub-chunks so context
                         isn't lost at artificial split points. Default 200.
    """

    def __init__(
        self,
        max_chunk_chars: int = 2000,
        overlap_chars:   int = 200,
        min_chunk_chars: int = 300,
    ):
        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars   = overlap_chars
        self.min_chunk_chars = min_chunk_chars  # merge chunks below this size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_file(self, txt_path: str | Path) -> list[Chunk]:
        """
        Load a processed .txt file and return a list of Chunk objects.

        The doc_id is inferred from the filename stem, which must match one
        of the keys in SPLIT_PATTERNS (e.g. "eu_ai_act", "nyc_local_law_144").

        Raises:
            FileNotFoundError: if the file doesn't exist.
            ValueError: if no split pattern exists for this document.
        """
        txt_path = Path(txt_path)
        if not txt_path.exists():
            raise FileNotFoundError(f"Processed file not found: {txt_path}")

        doc_id = txt_path.stem
        pattern = self._get_pattern(doc_id)

        text = txt_path.read_text(encoding="utf-8")

        # EU AI Act: secondary split of preamble (Recitals) before Article 1
        if "eu_ai_act" in doc_id and "annex" not in doc_id:
            text = self._label_eu_recitals(text)

        raw_chunks = self._split_by_pattern(text, pattern)

        chunks: list[Chunk] = []
        for idx, (section_line, body) in enumerate(raw_chunks, start=1):
            section, heading = self._parse_section_heading(section_line)
            full_text = f"{section_line}\n{body}".strip()

            # Sub-split if the chunk exceeds the size limit
            sub_texts = self._maybe_split_long(full_text)

            for sub_idx, sub_text in enumerate(sub_texts):
                suffix = f"_{sub_idx + 1:02d}" if len(sub_texts) > 1 else ""
                chunk_id = f"{doc_id}_chunk_{idx:03d}{suffix}"

                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    section=section,
                    heading=heading,
                    text=sub_text,
                    source_file=str(txt_path),
                ))

        # Merge chunks that are too small (e.g. TOC lines from NIST) into successor
        chunks = self._merge_tiny_chunks(chunks)

        logger.info(
            f"{doc_id}: {len(chunks)} chunks "
            f"(avg {sum(c.char_count for c in chunks) // max(len(chunks),1)} chars)"
        )
        return chunks

    def chunk_all(self, processed_dir: str | Path) -> dict[str, list[Chunk]]:
        """
        Chunk every .txt file in processed_dir. Returns a dict mapping
        doc_id → list of Chunk objects.

        Files whose doc_id has no registered split pattern are skipped with
        a warning rather than raising an exception.
        """
        processed_dir = Path(processed_dir)
        results = {}
        for txt_file in sorted(processed_dir.glob("*.txt")):
            doc_id = txt_file.stem
            if not self._has_pattern(doc_id):
                logger.warning(f"No split pattern for '{doc_id}' — skipping")
                continue
            try:
                results[doc_id] = self.chunk_file(txt_file)
            except Exception as e:
                logger.error(f"Failed to chunk {txt_file.name}: {e}")
        return results

    def save_chunks(
        self,
        chunks: list[Chunk] | dict[str, list[Chunk]],
        output_dir: str | Path,
    ) -> list[Path]:
        """
        Save chunks to JSON files in output_dir.

        Accepts either:
          - a list[Chunk] for a single document → one JSON file
          - a dict[doc_id, list[Chunk]] for multiple docs → one file per doc

        Returns list of written file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written = []

        if isinstance(chunks, dict):
            for doc_id, doc_chunks in chunks.items():
                path = self._write_json(doc_chunks, output_dir / f"{doc_id}_chunks.json")
                written.append(path)
        else:
            # Infer doc_id from first chunk
            doc_id = chunks[0].doc_id if chunks else "unknown"
            path = self._write_json(chunks, output_dir / f"{doc_id}_chunks.json")
            written.append(path)

        return written

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_pattern(self, doc_id: str) -> re.Pattern:
        """Look up the split pattern for a doc_id, with a helpful error."""
        for key, pattern in SPLIT_PATTERNS.items():
            if key in doc_id:
                return pattern
        raise ValueError(
            f"No split pattern registered for doc_id='{doc_id}'. "
            f"Available: {list(SPLIT_PATTERNS.keys())}"
        )

    def _has_pattern(self, doc_id: str) -> bool:
        return any(key in doc_id for key in SPLIT_PATTERNS)

    def _split_by_pattern(
        self, text: str, pattern: re.Pattern
    ) -> list[tuple[str, str]]:
        """
        Split text on every line matching the pattern.

        Returns list of (section_line, body) tuples. Text before the first
        match is kept as a "preamble" chunk under section "Preamble".
        """
        matches = list(pattern.finditer(text))

        if not matches:
            # No structure found — return whole text as one chunk
            logger.warning("No section markers found; returning document as single chunk")
            return [("Preamble", text)]

        results: list[tuple[str, str]] = []

        # Preamble: everything before the first match
        preamble = text[: matches[0].start()].strip()
        if preamble:
            results.append(("Preamble", preamble))

        # Each section: from its match to the next match (or end of text)
        for i, match in enumerate(matches):
            section_line = match.group(1).strip()
            start = match.end()
            end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body  = text[start:end].strip()
            results.append((section_line, body))

        return results

    def _parse_section_heading(self, section_line: str) -> tuple[str, str]:
        """
        Split a section line into (section_id, heading_title).

        Examples:
            "§ 20-870 Definitions."           → ("§ 20-870", "Definitions.")
            "Article 3"                        → ("Article 3", "")
            "GOVERN 1.1: Legal and regulatory" → ("GOVERN 1.1", "Legal and regulatory")
            "SECTION 1. In Colorado..."        → ("SECTION 1", "In Colorado...")
        """
        # Pattern: identifier (§ / Article / GOVERN etc.) + optional heading
        m = re.match(
            r"^((?:§\s*[\d\w-]+|Article\s+\d+|SECTION\s+\d+|ANNEX\s+[\w]+|"
            r"(?:GOVERN|MAP|MEASURE|MANAGE)\s+\d+(?:\.\d+)?|\d+(?:\.\d+)*))"
            r"[:\.\s]*(.*)$",
            section_line,
            re.IGNORECASE,
        )
        if m:
            section = m.group(1).strip()
            heading = m.group(2).strip().rstrip(".")
            return section, heading

        # Fallback: use the whole line as section, no heading
        return section_line.strip(), ""

    def _maybe_split_long(self, text: str) -> list[str]:
        """
        If text exceeds max_chunk_chars, split it into overlapping sub-chunks.

        Splits at paragraph boundaries (double newline) where possible so we
        don't cut mid-sentence.
        """
        if len(text) <= self.max_chunk_chars:
            return [text]

        # Find paragraph boundaries
        paragraphs = re.split(r"\n\n+", text)
        sub_chunks = []
        current    = ""

        for para in paragraphs:
            candidate = (current + "\n\n" + para).strip() if current else para

            if len(candidate) <= self.max_chunk_chars:
                current = candidate
            else:
                if current:
                    sub_chunks.append(current)
                # If a single paragraph is itself too long, hard-split by chars
                if len(para) > self.max_chunk_chars:
                    for hard_chunk in self._hard_split(para):
                        sub_chunks.append(hard_chunk)
                    current = ""
                else:
                    # Start new chunk, carry overlap from previous
                    overlap_text = current[-self.overlap_chars:] if current else ""
                    current = (overlap_text + "\n\n" + para).strip()

        if current:
            sub_chunks.append(current)

        return sub_chunks if sub_chunks else [text]

    def _hard_split(self, text: str) -> list[str]:
        """
        Hard-split a single oversized paragraph into overlapping char-level chunks.
        Used only when a paragraph itself exceeds max_chunk_chars.
        """
        chunks = []
        step   = self.max_chunk_chars - self.overlap_chars
        for i in range(0, len(text), step):
            chunks.append(text[i : i + self.max_chunk_chars])
        return chunks

    def _label_eu_recitals(self, text: str) -> str:
        """
        Pre-process the EU AI Act text to insert section markers for each
        Recital before Article 1, so they become individually retrievable chunks.

        Recitals look like:  (1) The purpose of this Regulation...
        We rewrite them as:  Recital (1)\nThe purpose of this Regulation...
        so the existing Article split pattern doesn't need to change.

        Only touches text before the first "^Article" line — definition lists
        inside Articles also use (N) numbering and must not be affected.
        """
        # Find where Article 1 begins (the authoritative text, not a reference)
        article1_match = re.search(r"^Article\s+1\s*$", text, re.MULTILINE)
        if not article1_match:
            return text  # can't find boundary; leave unchanged

        preamble = text[: article1_match.start()]
        rest     = text[article1_match.start():]

        # Recital pattern: (N) followed by capital letter, not a single quote
        # This matches recitals but not definition list items like (1) 'AI system'
        recital_re = re.compile(r"^\((\d+)\)\s+([A-Z][^'])", re.MULTILINE)
        labeled_preamble = recital_re.sub(r"Recital (\1)\n\2", preamble)

        return labeled_preamble + rest

    def _merge_tiny_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Merge chunks below min_chunk_chars into their successor.

        This cleans up table-of-contents lines that the NIST pattern accidentally
        captures as standalone chunks (e.g. "1.2.1 RiskMeasurement 5" = 26 chars).
        Tiny chunks are folded into the next chunk's text; if there is no successor,
        they are appended to the predecessor instead.
        """
        if not chunks:
            return chunks

        merged: list[Chunk] = []
        pending_text = ""  # text from tiny chunks waiting to be attached

        for chunk in chunks:
            if pending_text:
                # Prepend accumulated tiny text to this chunk
                new_text = (pending_text + "\n\n" + chunk.text).strip()
                pending_text = ""
                chunk = Chunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    section=chunk.section,
                    heading=chunk.heading,
                    text=new_text,
                    source_file=chunk.source_file,
                )

            if chunk.char_count < self.min_chunk_chars:
                # Hold this chunk; try to attach to the next one
                pending_text = (pending_text + "\n\n" + chunk.text).strip() if pending_text else chunk.text
            else:
                merged.append(chunk)

        # Any remaining tiny text appended to last chunk
        if pending_text and merged:
            last = merged[-1]
            new_text = (last.text + "\n\n" + pending_text).strip()
            merged[-1] = Chunk(
                chunk_id=last.chunk_id,
                doc_id=last.doc_id,
                section=last.section,
                heading=last.heading,
                text=new_text,
                source_file=last.source_file,
            )
        elif pending_text:
            # Edge case: all chunks were tiny — keep as single chunk
            merged.append(Chunk(
                chunk_id=f"{chunks[0].doc_id}_chunk_merged",
                doc_id=chunks[0].doc_id,
                section="Preamble",
                heading="",
                text=pending_text,
                source_file=chunks[0].source_file,
            ))

        return merged

    def _write_json(self, chunks: list[Chunk], path: Path) -> Path:
        """Serialize chunks to a JSON file and return the path."""
        data = [c.to_dict() for c in chunks]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Saved {len(chunks)} chunks → {path}")
        return path


# ---------------------------------------------------------------------------
# Example usage (run directly for a quick test)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    processed_dir = Path("data/processed")
    chunks_dir    = Path("data/chunks")

    chunker = DocumentChunker(max_chunk_chars=2000, overlap_chars=200)

    if len(sys.argv) > 1:
        # Single file mode: python src/chunker.py data/processed/eu_ai_act.txt
        txt_path = Path(sys.argv[1])
        chunks = chunker.chunk_file(txt_path)
        written = chunker.save_chunks(chunks, chunks_dir)
        doc_id = txt_path.stem
    else:
        # Chunk all documents
        all_chunks = chunker.chunk_all(processed_dir)
        written    = chunker.save_chunks(all_chunks, chunks_dir)

        print(f"\n{'='*60}")
        print("  Chunking Summary")
        print(f"{'='*60}")
        total = 0
        for doc_id, chunks in all_chunks.items():
            avg = sum(c.char_count for c in chunks) // max(len(chunks), 1)
            print(f"  {doc_id:<30} {len(chunks):>4} chunks  (avg {avg} chars)")
            total += len(chunks)
        print(f"  {'TOTAL':<30} {total:>4} chunks")
        print(f"{'='*60}")
        print(f"\nOutput: {chunks_dir.resolve()}")
        for p in written:
            print(f"  {p.name}")
        sys.exit(0)

    # Preview a few chunks from single-file mode
    print(f"\n{len(chunks)} chunks from {txt_path.name}")
    print(f"Output: {written[0]}\n")
    for i, chunk in enumerate(chunks[:3]):
        print(f"--- Chunk {i+1}: {chunk.section} | {chunk.heading} ---")
        print(chunk.text[:300])
        print(f"[{chunk.char_count} chars / ~{chunk.token_est} tokens]\n")