"""
pdf_ingestion.py — ReguGrounded Data & Retrieval Layer
=======================================================
Extracts and cleans text from regulatory PDF documents while preserving
legal structure (article numbers, section headings, § markers, etc.).

This is the first stage in the ReguGrounded pipeline:
  PDF Ingestion → Intelligent Chunking → Embedding → Pinecone → RAG

Supports: EU AI Act, NYC Local Law 144, NIST AI RMF, Colorado AI Act

Usage:
    ingestion = PDFIngestion()
    text = ingestion.process_pdf("data/raw /Local Law 144.pdf")
    # or save directly:
    ingestion.process_and_save("data/raw /Local Law 144.pdf", "data/processed/")
"""

import re
import logging
from pathlib import Path

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

# Configure logging so pipeline stages can trace extraction issues
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pdf_ingestion")


class PDFIngestion:
    """
    Extracts clean, structure-preserving text from regulatory PDF documents.

    Extraction strategy:
    1. Try pdfplumber (better layout awareness, handles complex PDFs)
    2. Fall back to PyPDF2 if pdfplumber fails or returns empty text

    Cleaning strategy:
    - Remove page numbers and running headers/footers
    - Rejoin words hyphenated across line breaks
    - Collapse excessive whitespace while keeping paragraph breaks
    - Preserve legal structure markers: §, Article, Section, Annex, GOVERN-x.x
    """

    # Patterns that identify lines which are pure page numbers or short headers
    # to be stripped. These are intentionally conservative — we only remove
    # things we're very confident are artifacts, not content.
    PAGE_NUMBER_PATTERNS = [
        re.compile(r"^\s*\d{1,4}\s*$"),                    # standalone digit(s)
        re.compile(r"^\s*-\s*\d{1,4}\s*-\s*$"),            # - 12 -  style
        re.compile(r"^\s*Page\s+\d+\s*(of\s+\d+)?\s*$", re.IGNORECASE),
    ]

    # Hyphenated line-break: word ending in hyphen at end of line, continued
    # on the next line with a lowercase letter (almost always a split word).
    HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")

    # Regulatory structure markers we must NOT remove during cleaning.
    # Used to verify preserved structure during post-processing.
    STRUCTURE_MARKERS = re.compile(
        r"(§\s*\d|Article\s+\d|Section\s+\d|GOVERN-|MANAGE-|MAP-|MEASURE-"
        r"|Annex\s+[IVX\d]|Recital\s+\d|\(\d+\))",
        re.IGNORECASE,
    )

    def extract_with_pdfplumber(self, pdf_path: Path) -> str:
        """
        Extract text page-by-page using pdfplumber.

        pdfplumber preserves layout better than PyPDF2 for complex documents
        like the EU AI Act with multi-column footnotes or NIST's structured
        grid sections. Each page is extracted separately so we can insert a
        page separator comment that downstream chunking can strip or keep.

        Returns empty string if extraction yields no usable text.
        """
        if not PDFPLUMBER_AVAILABLE:
            raise ImportError("pdfplumber is not installed. Run: pip install pdfplumber")

        pages_text = []
        with pdfplumber.open(pdf_path) as pdf:
            logger.info(f"  pdfplumber: {len(pdf.pages)} pages found in {pdf_path.name}")
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    pages_text.append(text)
                else:
                    logger.debug(f"  Page {page_num}: no text extracted (possibly image-only)")

        return "\n".join(pages_text)

    def extract_with_pypdf2(self, pdf_path: Path) -> str:
        """
        Extract text using PyPDF2 as a fallback extractor.

        PyPDF2 is less accurate with layout but handles a wider range of PDF
        encodings. Used when pdfplumber fails or returns empty output.
        """
        if not PYPDF2_AVAILABLE:
            raise ImportError("PyPDF2 is not installed. Run: pip install PyPDF2")

        pages_text = []
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            logger.info(f"  PyPDF2: {len(reader.pages)} pages found in {pdf_path.name}")
            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text()
                    if text:
                        pages_text.append(text)
                except Exception as e:
                    logger.warning(f"  Page {page_num}: PyPDF2 extraction error — {e}")

        return "\n".join(pages_text)

    def extract_raw_text(self, pdf_path: Path) -> str:
        """
        Extract raw text from a PDF, trying pdfplumber first then PyPDF2.

        Returns the raw (uncleaned) text. Raises RuntimeError if both
        extractors fail or produce empty output.
        """
        raw_text = ""

        # --- Attempt 1: pdfplumber ---
        if PDFPLUMBER_AVAILABLE:
            try:
                logger.info(f"Attempting pdfplumber extraction...")
                raw_text = self.extract_with_pdfplumber(pdf_path)
                if raw_text.strip():
                    logger.info(f"  pdfplumber succeeded ({len(raw_text):,} chars)")
                    return raw_text
                else:
                    logger.warning("  pdfplumber returned empty text, falling back to PyPDF2")
            except Exception as e:
                logger.warning(f"  pdfplumber failed: {e} — falling back to PyPDF2")

        # --- Attempt 2: PyPDF2 ---
        if PYPDF2_AVAILABLE:
            try:
                logger.info(f"Attempting PyPDF2 extraction...")
                raw_text = self.extract_with_pypdf2(pdf_path)
                if raw_text.strip():
                    logger.info(f"  PyPDF2 succeeded ({len(raw_text):,} chars)")
                    return raw_text
                else:
                    logger.warning("  PyPDF2 also returned empty text")
            except Exception as e:
                logger.error(f"  PyPDF2 failed: {e}")

        if not raw_text.strip():
            raise RuntimeError(
                f"Both extractors failed to produce text from {pdf_path.name}. "
                "The PDF may be image-only (scanned). Consider running OCR first."
            )
        return raw_text

    def clean_text(self, raw_text: str) -> str:
        """
        Clean extracted text while preserving regulatory document structure.

        Cleaning steps applied in order:
        1. Fix encoding artifacts (ligatures, smart quotes, common mojibake)
        2. Rejoin hyphenated line breaks (word- \\n break → wordbbreak)
        3. Remove standalone page numbers and short page headers
        4. Normalize whitespace (max 2 consecutive newlines, single spaces)
        5. Strip leading/trailing whitespace per line
        """
        text = raw_text

        # Step 1: Fix common encoding artifacts
        text = self._fix_encoding(text)

        # Step 2: Rejoin hyphenated line breaks
        # "compli-\nance" → "compliance"
        text = self.HYPHEN_BREAK.sub(r"\1\2", text)

        # Step 3: Remove page numbers and running headers/footers line by line
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            if self._is_artifact_line(line):
                logger.debug(f"  Removed artifact line: {repr(line.strip())}")
                continue
            cleaned_lines.append(line)
        text = "\n".join(cleaned_lines)

        # Step 4: Collapse runs of 3+ blank lines down to 2 (paragraph breaks)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Step 5: Collapse multiple spaces/tabs to a single space,
        # but preserve leading indentation for structured sections.
        lines = text.split("\n")
        cleaned_lines = [re.sub(r"[ \t]{2,}", " ", line).rstrip() for line in lines]
        text = "\n".join(cleaned_lines)

        return text.strip()

    def _fix_encoding(self, text: str) -> str:
        """
        Fix common PDF encoding artifacts that appear in regulatory documents.

        These patterns are sourced from observed issues in EU AI Act and NIST
        PDFs extracted with pdfplumber/PyPDF2.
        """
        replacements = {
            # PDF ligatures that don't decode properly
            "\ufb01": "fi",   # ﬁ → fi
            "\ufb02": "fl",   # ﬂ → fl
            "\ufb03": "ffi",  # ﬃ → ffi
            "\ufb04": "ffl",  # ﬄ → ffl
            "\u2019": "'",    # right single quotation mark
            "\u2018": "'",    # left single quotation mark
            "\u201c": '"',    # left double quotation mark
            "\u201d": '"',    # right double quotation mark
            "\u2013": "-",    # en dash
            "\u2014": "--",   # em dash
            "\u00a0": " ",    # non-breaking space → regular space
            "\u00ad": "",     # soft hyphen → remove
        }
        for bad, good in replacements.items():
            text = text.replace(bad, good)
        return text

    def _is_artifact_line(self, line: str) -> bool:
        """
        Return True if this line is a PDF artifact (page number, empty header)
        that should be removed.

        Conservative: only removes lines we're confident are not content.
        A line with any regulatory structure marker is always kept.
        """
        stripped = line.strip()

        # Never remove lines containing regulatory structure markers
        if self.STRUCTURE_MARKERS.search(stripped):
            return False

        # Check against page number patterns
        for pattern in self.PAGE_NUMBER_PATTERNS:
            if pattern.match(stripped):
                return True

        # Remove very short lines that are just punctuation or whitespace
        # (common header/footer remnants) but not numbered items like "(a)"
        if len(stripped) <= 3 and not re.match(r"^\(?[a-zA-Z0-9]+\)?\.?$", stripped):
            return True

        return False

    def process_pdf(self, pdf_path: str | Path) -> str:
        """
        Full pipeline: extract raw text from PDF then clean it.

        This is the main entry point for processing a single PDF.

        Args:
            pdf_path: Path to the PDF file (string or Path object).

        Returns:
            Cleaned text with regulatory structure preserved.

        Raises:
            FileNotFoundError: If the PDF doesn't exist at the given path.
            RuntimeError: If both extractors fail (e.g., scanned/image PDF).
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Processing: {pdf_path.name}")
        logger.info(f"  File size: {pdf_path.stat().st_size / 1024:.1f} KB")

        raw_text = self.extract_raw_text(pdf_path)
        cleaned_text = self.clean_text(raw_text)

        # Log structure preservation stats
        structure_hits = len(self.STRUCTURE_MARKERS.findall(cleaned_text))
        logger.info(
            f"  Done: {len(cleaned_text):,} chars, "
            f"{cleaned_text.count(chr(10)):,} lines, "
            f"{structure_hits} structure markers preserved"
        )

        return cleaned_text

    def process_and_save(
        self,
        pdf_path: str | Path,
        output_dir: str | Path,
        output_stem: str | None = None,
    ) -> Path:
        """
        Process a PDF and save the cleaned text to a .txt file.

        The output filename defaults to the PDF's stem (filename without
        extension), so "Local Law 144.pdf" → "Local Law 144.txt".
        Pass output_stem to override (e.g., "nyc_local_law_144").

        Args:
            pdf_path:    Path to the source PDF.
            output_dir:  Directory to write the cleaned .txt file into.
            output_stem: Override the output filename stem (no extension).

        Returns:
            Path to the written output file.
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = output_stem or pdf_path.stem
        output_path = output_dir / f"{stem}.txt"

        cleaned_text = self.process_pdf(pdf_path)

        output_path.write_text(cleaned_text, encoding="utf-8")
        logger.info(f"  Saved → {output_path}")

        return output_path


# ---------------------------------------------------------------------------
# Example usage (run this file directly to test a single PDF)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    # Default to NYC Local Law 144 for quick testing
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")

    test_pdf = raw_dir / "Local Law 144.pdf"
    if len(sys.argv) > 1:
        test_pdf = Path(sys.argv[1])

    ingestion = PDFIngestion()

    print(f"\nTesting with: {test_pdf}")
    print("=" * 60)

    output_path = ingestion.process_and_save(
        pdf_path=test_pdf,
        output_dir=processed_dir,
        output_stem="nyc_local_law_144",
    )

    # Print a preview of the output
    text = output_path.read_text(encoding="utf-8")
    print(f"\nOutput file: {output_path}")
    print(f"Total characters: {len(text):,}")
    print(f"Total lines: {text.count(chr(10)):,}")
    print("\n--- First 1000 characters ---")
    print(text[:1000])
    print("\n--- Last 500 characters ---")
    print(text[-500:])