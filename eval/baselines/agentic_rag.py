"""
baselines/agentic_rag.py — Agentic RAG Baseline
=================================================
Wrapper around the AgenticRAGPipeline — the truly agentic pipeline that is
distinct from the standard RLM pipeline.

Differences from standard_rag and the RLM pipeline:
  - Dynamic tool selection per sub-question (semantic / keyword / hybrid)
  - LLM-based evidence reflection after each retrieval round
  - Iterative re-retrieval with query reformulation if evidence is weak
  - Uses EnhancedRetriever (BM25 + semantic + reranking) instead of plain Pinecone
  - Returns agent_trace in metadata showing every decision the agent made

Provides the same interface as standard_rag.process_query so eval scripts can
swap between baselines with a single import change.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agentic_rag_pipeline import process_query as _agentic_process_query  # noqa: E402


def process_query(question: str, top_k: int = 5) -> dict:
    """
    Truly agentic RAG pipeline.

    Tool selection → iterative retrieval + reflection → synthesis → validation.
    Adds a 'baseline' tag to metadata so reports can identify the source.
    """
    result = _agentic_process_query(question, top_k=top_k)
    result["metadata"]["baseline"] = "agentic_rag"
    return result
