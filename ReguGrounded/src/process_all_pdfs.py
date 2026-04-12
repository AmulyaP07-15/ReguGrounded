"""
process_all_pdfs.py — ReguGrounded Batch PDF Processor
=======================================================
Processes all regulatory PDFs in data/raw/ and writes clean text files
to data/processed/. Designed to be run once to prepare the corpus for
the next pipeline stage (intelligent chunking).

Documents processed:
    - Local Law 144.pdf        → nyc_local_law_144.txt
    - nist_ai_framework.pdf    → nist_ai_framework.txt
    - eu_act                   → eu_ai_act.txt          (PDF without extension)
    - eu_act_annex.pdf         → eu_ai_act_annex.txt
    - colorado_act.pdf         → colorado_ai_act.txt

Usage:
    python src/process_all_pdfs.py

    # Verbose output:
    python src/process_all_pdfs.py --verbose

    # Process a specific file only:
    python src/process_all_pdfs.py --file "data/raw /eu_act"
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Allow running from the project root or from src/
sys.path.insert(0, str(Path(__file__).parent))
from pdf_ingestion import PDFIngestion

logger = logging.getLogger("process_all_pdfs")


@dataclass
class ProcessingResult:
    """Holds the outcome of processing a single PDF."""
    pdf_path: Path
    output_path: Path | None = None
    success: bool = False
    error: str = ""
    char_count: int = 0
    duration_sec: float = 0.0


# ---------------------------------------------------------------------------
# Document registry
# ---------------------------------------------------------------------------
# Maps each source PDF to the output stem we want. This makes filenames
# predictable for downstream pipeline stages regardless of the source name.
# The raw/ dir has a trailing space in its name — this is intentional here.

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

DOCUMENTS = [
    {
        "pdf":  RAW_DIR / "Local Law 144.pdf",
        "stem": "nyc_local_law_144",
        "description": "NYC Local Law 144 — Automated Employment Decision Tools",
    },
    {
        "pdf":  RAW_DIR / "nist_ai_framework.pdf",
        "stem": "nist_ai_framework",
        "description": "NIST AI Risk Management Framework (AI RMF 1.0)",
    },
    {
        "pdf":  RAW_DIR / "eu_act",             # PDF stored without .pdf extension
        "stem": "eu_ai_act",
        "description": "EU AI Act (Regulation 2024/1689)",
    },
    {
        "pdf":  RAW_DIR / "eu_act_annex.pdf",
        "stem": "eu_ai_act_annex",
        "description": "EU AI Act — Annexes",
    },
    {
        "pdf":  RAW_DIR / "colorado_act.pdf",
        "stem": "colorado_ai_act",
        "description": "Colorado AI Act (SB 205)",
    },
]


def process_all(
    documents: list[dict] | None = None,
    output_dir: Path = PROCESSED_DIR,
    verbose: bool = False,
) -> list[ProcessingResult]:
    """
    Process all documents in the registry (or a custom list) and return results.

    Args:
        documents:  Override the default DOCUMENTS list. Each entry must be a
                    dict with keys: "pdf" (Path), "stem" (str), "description" (str).
        output_dir: Where to write cleaned .txt files.
        verbose:    If True, logs every cleaning step at DEBUG level.

    Returns:
        List of ProcessingResult objects, one per document.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    docs = documents or DOCUMENTS
    ingestion = PDFIngestion()
    results = []

    print(f"\n{'='*65}")
    print(f"  ReguGrounded — Batch PDF Processor")
    print(f"  Output directory: {output_dir.resolve()}")
    print(f"  Documents to process: {len(docs)}")
    print(f"{'='*65}\n")

    for i, doc in enumerate(docs, start=1):
        pdf_path = Path(doc["pdf"])
        stem = doc["stem"]
        description = doc["description"]

        print(f"[{i}/{len(docs)}] {description}")
        print(f"       Source: {pdf_path}")

        result = ProcessingResult(pdf_path=pdf_path)
        start = time.time()

        # Skip missing files with a clear message instead of crashing the batch
        if not pdf_path.exists():
            result.error = f"File not found: {pdf_path}"
            result.duration_sec = time.time() - start
            logger.warning(f"  SKIPPED — {result.error}")
            print(f"       Status: SKIPPED (file not found)\n")
            results.append(result)
            continue

        try:
            output_path = ingestion.process_and_save(
                pdf_path=pdf_path,
                output_dir=output_dir,
                output_stem=stem,
            )
            text = output_path.read_text(encoding="utf-8")

            result.success = True
            result.output_path = output_path
            result.char_count = len(text)
            result.duration_sec = time.time() - start

            print(f"       Status: OK  →  {output_path.name}")
            print(f"       Size:   {result.char_count:,} chars  |  {result.duration_sec:.1f}s\n")

        except Exception as e:
            result.error = str(e)
            result.duration_sec = time.time() - start
            logger.error(f"  FAILED — {e}")
            print(f"       Status: FAILED\n       Error:  {e}\n")

        results.append(result)

    _print_summary(results)
    return results


def _print_summary(results: list[ProcessingResult]) -> None:
    """Print a final summary table of all processing results."""
    successes = [r for r in results if r.success]
    failures  = [r for r in results if not r.success and r.error and "not found" not in r.error]
    skipped   = [r for r in results if not r.success and "not found" in r.error]

    print(f"{'='*65}")
    print(f"  Summary")
    print(f"{'='*65}")
    print(f"  Succeeded : {len(successes)}/{len(results)}")
    if skipped:
        print(f"  Skipped   : {len(skipped)} (files not found)")
    if failures:
        print(f"  Failed    : {len(failures)}")
        for r in failures:
            print(f"    - {r.pdf_path.name}: {r.error}")

    if successes:
        total_chars = sum(r.char_count for r in successes)
        total_time  = sum(r.duration_sec for r in results)
        print(f"\n  Total text extracted : {total_chars:,} characters")
        print(f"  Total time           : {total_time:.1f}s")
        print(f"\n  Output files:")
        for r in successes:
            print(f"    {r.output_path}")
    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Process all regulatory PDFs for the ReguGrounded pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all documents
  python src/process_all_pdfs.py

  # Process a single file
  python src/process_all_pdfs.py --file "data/raw /Local Law 144.pdf" --stem nyc_local_law_144

  # Verbose output (shows every cleaning decision)
  python src/process_all_pdfs.py --verbose
        """,
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Process a single PDF instead of the full batch.",
    )
    parser.add_argument(
        "--stem",
        type=str,
        help="Output filename stem when using --file (e.g. nyc_local_law_144).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DIR,
        help=f"Output directory (default: {PROCESSED_DIR})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging to see every cleaning step.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.file:
        # Single-file mode
        stem = args.stem or Path(args.file).stem
        docs = [{
            "pdf": args.file,
            "stem": stem,
            "description": f"Single file: {Path(args.file).name}",
        }]
    else:
        docs = None  # use default DOCUMENTS registry

    results = process_all(
        documents=docs,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )

    # Exit with non-zero code if any extraction (not just skipped) failed
    hard_failures = [r for r in results if not r.success and "not found" not in r.error]
    sys.exit(1 if hard_failures else 0)


if __name__ == "__main__":
    main()