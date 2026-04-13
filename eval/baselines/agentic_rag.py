"""
baselines/agentic_rag.py — Agentic RAG Baseline
=================================================
Middle-ground pipeline between Standard RAG and CAG with RLM.

    Standard RAG     — single retrieve → single LLM call. No decomposition.

    ARAG (this file) — LLM-based RLM decomposition into sub-questions
                       → parallel multi-retrieval per jurisdiction
                       → LLM synthesis grounded in retrieved chunks
                       NO citation validation. NO guardrails.

    CAG with RLM     — LLM decomposition + multi-retrieval
                       + citation validation + input/output guardrails.

The "agentic" distinction is that the LLM actively reasons about how to
decompose the query and which jurisdictions to search — same as CAG but
without the validation and safety layers on top.
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from rlm_engine             import RLMEngine              # noqa: E402
from reasoning_orchestrator import ReasoningOrchestrator  # noqa: E402
from answer_synthesizer     import AnswerSynthesizer      # noqa: E402
from citation_validator     import validate_citations     # noqa: E402
from src.retriever          import Retriever              # noqa: E402

_retriever_instance = None
_rlm_instance       = None
_orchestrator       = None
_synthesizer        = None


def _get_pipeline():
    global _retriever_instance, _rlm_instance, _orchestrator, _synthesizer
    if _retriever_instance is None:
        _retriever_instance = Retriever()
        _rlm_instance       = RLMEngine()
        _orchestrator       = ReasoningOrchestrator(retriever_func=_retriever_instance.retrieve)
        _synthesizer        = AnswerSynthesizer()
    return _rlm_instance, _orchestrator, _synthesizer


def process_query(question: str, top_k: int = 5) -> dict:
    """
    Agentic RAG pipeline.

    LLM-based RLM decomposition → parallel multi-retrieval → LLM synthesis.
    No citation validation. No guardrails.
    """
    t0 = time.time()

    rlm, orchestrator, synthesizer = _get_pipeline()

    # 1. LLM decomposition (same as CAG)
    try:
        decomposition = rlm.decompose(question)
    except Exception:
        decomposition = {
            "original_query": question,
            "jurisdictions":  [],
            "sub_questions":  [{"jurisdiction": None, "sub_question": question}],
        }

    # 2. Multi-retrieval per sub-question (same as CAG)
    evidence_bundle = orchestrator.run(decomposition, top_k=top_k)
    jurisdictions   = orchestrator.get_jurisdictions(evidence_bundle)

    # 3. Synthesis (same as CAG)
    synthesis = synthesizer.synthesize(
        original_query=question,
        decomposition=decomposition,
        evidence_bundle=evidence_bundle,
    )

    all_chunks = orchestrator.get_all_chunks(evidence_bundle)

    # Validate citations post-hoc for fair comparison.
    # A-RAG does not include validation as a system component — this is
    # applied externally so hallucination_rate is comparable across all systems.
    # Chunks from the orchestrator are already normalized to flat format.
    validation = validate_citations(synthesis["answer"], all_chunks)
    validation["note"] = (
        "Post-hoc validation for comparison. "
        "A-RAG does not include validation as a system component."
    )

    latency_ms = int((time.time() - t0) * 1000)

    return {
        "question":      question,
        "sub_questions": synthesis["sub_questions"],
        "answer":        synthesis["answer"],
        "citations":     synthesis["citations"],
        "validation":    validation,
        "metadata": {
            "jurisdictions_searched":  jurisdictions,
            "total_chunks_retrieved":  len(all_chunks),
            "latency_ms":              latency_ms,
            "baseline":                "agentic_rag",
            "has_built_in_validation": False,
            "validation_method":       "post_hoc_for_comparison",
        },
    }
