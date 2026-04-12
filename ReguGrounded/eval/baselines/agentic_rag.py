"""
baselines/agentic_rag.py — Agentic RAG Baseline
=================================================
Thin wrapper around ReguGrounded's full pipeline.  Used as the "upper bound"
comparison point in eval — shows what the complete agentic system achieves.

Provides the same interface as standard_rag.process_query so eval scripts can
swap between baselines with a single import change.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from query_interface import process_query as _process_query  # noqa: E402


def process_query(question: str, top_k: int = 5) -> dict:
    """
    Full ReguGrounded agentic pipeline.

    Decomposition → multi-retrieval → synthesis → citation validation.
    Adds a 'baseline' tag to metadata so reports can identify the source.
    """
    result = _process_query(question, top_k=top_k)
    result["metadata"]["baseline"] = "agentic_rag"
    return result
