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

load_dotenv(Path(__file__).parent.parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.retriever import retrieve          # noqa: E402
from citation_validator import validate_citations  # noqa: E402
from utils.llm_client import get_llm_client  # noqa: E402


def _get_client():
    return get_llm_client()


def process_query(question: str, top_k: int = 5) -> dict:
    """
    Standard RAG pipeline (baseline).

    Single retrieve → single LLM call, with NO query decomposition.
    Citations in the answer ARE validated against retrieved chunks so
    hallucination_rate is directly comparable to the ReguGrounded pipeline.

    Returns the same top-level keys as ReguGrounded's process_query so
    the two can be compared directly in eval scripts.
    """
    import time
    t0 = time.time()

    chunks = retrieve(question, top_k=top_k)

    # Flatten chunk metadata to the shape citation_validator expects
    flat_chunks = [
        {
            "law_name":       c["metadata"].get("law_name", ""),
            "article_number": c["metadata"].get("article_number", ""),
            "section_number": c["metadata"].get("section_number", ""),
            "text":           c.get("text", ""),
        }
        for c in chunks
    ]

    context = "\n\n".join(
        f"[{i+1}] {c['metadata'].get('law_name','')} "
        f"{c['metadata'].get('article_number') or c['metadata'].get('section_number','')}: "
        f"{c['text'][:400]}"
        for i, c in enumerate(chunks)
    )

    prompt = (
        f"Answer the following regulatory compliance question using only the provided context.\n"
        f"Always cite the FULL law name and article/section together "
        f"(e.g. 'EU AI Act Article 9', 'NYC Local Law 144 § 20-871').\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\nAnswer:"
    )

    llm_failed = False
    try:
        response = _get_client().chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer     = f"LLM call failed: {e}"
        llm_failed = True

    # Validate citations post-hoc for fair comparison.
    # Standard RAG does not include validation as a system component — this is
    # applied externally so hallucination_rate is comparable across all systems.
    validation = validate_citations(answer, flat_chunks)
    validation["note"] = (
        "Post-hoc validation for comparison. "
        "Standard RAG does not include validation as a system component."
    )

    latency_ms = int((time.time() - t0) * 1000)

    return {
        "question":         question,
        "sub_questions":    [question],  # no decomposition
        "answer":           answer,
        "citations":        [],          # no structured citation list (single-pass)
        "validation":       validation,
        "retrieved_chunks": flat_chunks, # exposed for consistency / eval script access
        "metadata": {
            "jurisdictions_searched":  [],
            "total_chunks_retrieved":  len(chunks),
            "latency_ms":              latency_ms,
            "baseline":                "standard_rag",
            "has_built_in_validation": False,
            "validation_method":       "post_hoc_for_comparison",
            "llm_synthesis_failed":    llm_failed,
        },
    }
