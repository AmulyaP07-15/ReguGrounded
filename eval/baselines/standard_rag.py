"""
baselines/standard_rag.py — Standard RAG Baseline
===================================================
Simple retrieve-then-read pipeline with NO query decomposition and NO
citation validation.  Used as a baseline to demonstrate the value added
by the ReguGrounded agentic approach.

Flow: Query → retrieve(top_k=5) → single LLM call → answer
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.retriever import retrieve  # noqa: E402

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def process_query(question: str, top_k: int = 5) -> dict:
    """
    Standard RAG pipeline (baseline).

    Does NOT decompose the question or validate citations.

    Returns the same top-level keys as ReguGrounded's process_query so
    the two can be compared directly in eval scripts.
    """
    import time
    t0 = time.time()

    chunks = retrieve(question, top_k=top_k)

    context = "\n\n".join(
        f"[{i+1}] {c['metadata'].get('law_name','')} "
        f"{c['metadata'].get('article_number') or c['metadata'].get('section_number','')}: "
        f"{c['text'][:400]}"
        for i, c in enumerate(chunks)
    )

    prompt = (
        f"Answer the following regulatory compliance question using only the provided context.\n"
        f"Include relevant article/section citations in your answer.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\nAnswer:"
    )

    try:
        response = _get_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"LLM call failed: {e}"

    latency_ms = int((time.time() - t0) * 1000)

    return {
        "question":     question,
        "sub_questions":[question],  # no decomposition
        "answer":       answer,
        "citations":    [],          # no structured citations
        "validation": {
            "total_citations":    0,
            "valid_citations":    0,
            "invalid_citations":  0,
            "accuracy":           None,
            "hallucination_rate": None,
            "details":            {"valid": [], "invalid": []},
        },
        "metadata": {
            "jurisdictions_searched": [],
            "total_chunks_retrieved": len(chunks),
            "latency_ms":             latency_ms,
            "baseline":               "standard_rag",
        },
    }
