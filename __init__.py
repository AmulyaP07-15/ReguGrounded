"""
ReguGrounded — Corrective Agentic Generation (CAG) for Regulatory Compliance Q&A
=================================================================================
A multi-stage RAG pipeline that answers regulatory compliance questions grounded
in retrieved evidence, with built-in citation validation and a corrective retry loop.

Core pipeline (in execution order):
    QueryInterface        — entry point; runs the full pipeline
    RLMEngine             — decomposes queries into jurisdiction-specific sub-questions
    ReasoningOrchestrator — parallel retrieval across all sub-questions
    AnswerSynthesizer     — LLM synthesis with optional corrective constraints
    CitationValidator     — validates citations against retrieved chunks
    QueryClarifier        — optional pre-pipeline clarification (interactive mode)

Utilities:
    utils.llm_client      — unified LLM client (Anthropic / Groq / Gemini, with key rotation)
    utils.guardrails      — input/output guardrails
    utils.cache           — SQLite query cache and in-memory component cache
    utils.rate_limiter    — token-per-minute rate limiter
    utils.metrics         — latency and accuracy tracking
    utils.logger          — structured logging

Evaluation:
    eval.compare_models   — side-by-side comparison of Standard RAG, ARAG, and CAG
    eval.evaluate_answers — LLM-as-judge answer correctness scoring
    eval.ground_truth     — 53 curated questions across EU AI Act, NYC LL144,
                            Colorado AI Act, NIST AI RMF, and cross-jurisdictional

Quick start:
    from query_interface import process_query
    result = process_query("What are the bias audit requirements under NYC Local Law 144?")
    print(result["answer"])
"""

from query_interface import process_query, QueryInterface

__all__ = ["process_query", "QueryInterface"]
