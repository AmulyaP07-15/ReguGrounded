# tests/test_integration.py
#
# Integration tests for the ReguGrounded reasoning pipeline.
# Run with:  pytest tests/test_integration.py -v
#
# Tests 1-3 use a mock retriever so they run without Pinecone credentials.
# Tests 4-5 (live tests) hit the real Pinecone index — skip with:
#   pytest tests/test_integration.py -v -m "not live"

import json
import sys
import os
import time
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from citation_validator import extract_citations, validate_citations
from reasoning_orchestrator import ReasoningOrchestrator
from answer_synthesizer import AnswerSynthesizer
from query_interface import QueryInterface
from query_clarifier import QueryClarifier

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

NYC_CHUNK = {
    "text": (
        "Section 20-871 of the NYC Administrative Code requires that any employer using "
        "an automated employment decision tool must ensure it has been subject to a bias "
        "audit conducted no more than one year prior to each use of the tool."
    ),
    "law_name": "NYC Local Law 144",
    "article_number": None,
    "section_number": "20-871",
    "chunk_id": "nyc_chunk_001",
    "score": 0.92,
}

NYC_CHUNK_2 = {
    "text": (
        "Section 20-870 defines 'automated employment decision tool' as any computational "
        "process derived from machine learning that issues simplified output used to filter "
        "candidates for employment."
    ),
    "law_name": "NYC Local Law 144",
    "article_number": None,
    "section_number": "20-870",
    "chunk_id": "nyc_chunk_002",
    "score": 0.88,
}

EU_CHUNK = {
    "text": (
        "Article 9 of the EU AI Act requires that high-risk AI systems be subject to a "
        "risk management system consisting of a continuous iterative process run throughout "
        "the entire lifecycle of the high-risk AI system."
    ),
    "law_name": "EU AI Act",
    "article_number": "9",
    "section_number": None,
    "chunk_id": "eu_chunk_009",
    "score": 0.89,
}

COLORADO_CHUNK = {
    "text": (
        "Section 6-1-1707 of the Colorado AI Act requires developers of high-risk artificial "
        "intelligence systems to conduct impact assessments that include testing for algorithmic "
        "discrimination. Assessments must be updated annually."
    ),
    "law_name": "Colorado AI Act",
    "article_number": None,
    "section_number": "6-1-1707",
    "chunk_id": "co_chunk_1707",
    "score": 0.85,
}

NIST_CHUNK = {
    "text": (
        "GOVERN 1.1 — Policies, processes, procedures, and practices across the organization "
        "related to the mapping, measuring, and managing of AI risks are in place, transparent, "
        "and implemented effectively."
    ),
    "law_name": "NIST AI Risk Management Framework",
    "article_number": None,
    "section_number": "GOVERN 1.1",
    "chunk_id": "nist_chunk_govern1",
    "score": 0.83,
}

ALL_CHUNKS = [NYC_CHUNK, NYC_CHUNK_2, EU_CHUNK, COLORADO_CHUNK, NIST_CHUNK]


def mock_retriever(query: str, top_k: int = 5, filter: dict = None):
    """Returns a deterministic set of chunks regardless of query (for unit tests)."""
    chunks = ALL_CHUNKS
    # Honour doc_id filter so jurisdiction-filtered tests work correctly
    if filter and "doc_id" in filter:
        doc_id = filter["doc_id"]
        doc_to_law = {
            "nyc_local_law_144": "NYC Local Law 144",
            "eu_ai_act":         "EU AI Act",
            "colorado_ai_act":   "Colorado AI Act",
            "nist_ai_framework": "NIST AI Risk Management Framework",
        }
        law = doc_to_law.get(doc_id)
        if law:
            chunks = [c for c in ALL_CHUNKS if c["law_name"] == law]
    return [
        {
            "text": c["text"],
            "metadata": {
                "law_name":       c["law_name"],
                "article_number": c["article_number"],
                "section_number": c["section_number"],
                "chunk_id":       c["chunk_id"],
            },
            "score": c["score"],
        }
        for c in chunks[:top_k]
    ]


# ---------------------------------------------------------------------------
# Test 1 — Single-jurisdiction query (NYC)
# ---------------------------------------------------------------------------

def test_single_jurisdiction_nyc():
    """Query about NYC alone should return NYC chunks and valid citations."""
    interface = QueryInterface(retriever_func=mock_retriever)
    result = interface.run("What does NYC Local Law 144 require for bias audits?", top_k=3)

    assert result["question"]
    assert len(result["sub_questions"]) >= 1
    assert result["answer"]
    assert isinstance(result["citations"], list)

    # Metadata sanity
    assert result["metadata"]["total_chunks_retrieved"] > 0
    assert result["metadata"]["latency_ms"] >= 0

    # Validation accuracy should be high (mock retriever always returns matching chunks)
    v = result["validation"]
    assert isinstance(v["accuracy"], float)
    assert 0.0 <= v["accuracy"] <= 1.0

    print(f"\n[test_single_jurisdiction_nyc] answer preview: {result['answer'][:120]}")
    print(f"  citations: {[c['source']+' '+c['article'] for c in result['citations']]}")
    print(f"  validation accuracy: {v['accuracy']*100:.1f}%")


# ---------------------------------------------------------------------------
# Test 2 — Multi-jurisdiction query (EU + NYC)
# ---------------------------------------------------------------------------

def test_multi_jurisdiction():
    """Query spanning EU AI Act and NYC should decompose into ≥2 sub-questions."""
    interface = QueryInterface(retriever_func=mock_retriever)
    result = interface.run(
        "Compare EU AI Act and NYC Local Law 144 transparency requirements.", top_k=3
    )

    assert len(result["sub_questions"]) >= 2, (
        f"Expected ≥2 sub-questions, got {len(result['sub_questions'])}"
    )
    assert result["answer"]
    assert result["metadata"]["total_chunks_retrieved"] > 0

    print(f"\n[test_multi_jurisdiction] sub-questions: {result['sub_questions']}")
    print(f"  jurisdictions: {result['metadata']['jurisdictions_searched']}")


# ---------------------------------------------------------------------------
# Test 3 — Citation validator unit tests
# ---------------------------------------------------------------------------

def test_citation_validator_valid():
    """Citations that exist in the retrieved chunks should be marked valid."""
    answer = (
        "NYC Local Law 144 § 20-871 requires a bias audit. "
        "The EU AI Act Article 9 mandates a risk management system."
    )
    validation = validate_citations(answer, ALL_CHUNKS)

    assert validation["total_citations"] == 2
    assert validation["valid_count"] == 2
    assert validation["invalid_count"] == 0
    assert validation["accuracy"] == 1.0

    print(f"\n[test_citation_validator_valid] {validation}")


def test_citation_validator_hallucination():
    """Citations not in retrieved chunks should be flagged as invalid."""
    answer = (
        "According to NYC Local Law 144 § 20-999 the employer must do X. "  # hallucinated section
        "EU AI Act Article 99 also applies."                                  # hallucinated article
    )
    validation = validate_citations(answer, ALL_CHUNKS)

    assert validation["total_citations"] == 2
    # Neither § 20-999 nor Article 99 exists in ALL_CHUNKS
    assert validation["invalid_count"] == 2
    assert validation["accuracy"] == 0.0

    print(f"\n[test_citation_validator_hallucination] {validation}")


# ---------------------------------------------------------------------------
# Test 4 — Empty retrieval edge case
# ---------------------------------------------------------------------------

def test_empty_retrieval():
    """System should handle empty retrieval gracefully without crashing."""
    def empty_retriever(query: str, top_k: int = 5):
        return []

    interface = QueryInterface(retriever_func=empty_retriever)
    result = interface.run("What are the bias audit requirements?", top_k=3)

    assert result["question"]
    assert isinstance(result["answer"], str)   # may say "no evidence" — that's fine
    assert result["metadata"]["total_chunks_retrieved"] == 0
    # No citations extracted → accuracy should default to 1.0 (nothing to validate)
    assert result["validation"]["total_citations"] == 0
    assert result["validation"]["accuracy"] == 1.0

    print(f"\n[test_empty_retrieval] answer: {result['answer'][:120]}")


# ---------------------------------------------------------------------------
# Test 5 — Live end-to-end (requires Pinecone + OpenAI credentials)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper: mock a successful OpenAI synthesis response
# ---------------------------------------------------------------------------

def _mock_openai(answer_text: str):
    """Context manager that patches answer_synthesizer.OpenAI with a fixed response."""
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = answer_text
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp
    return patch("answer_synthesizer.OpenAI", return_value=mock_client)


# ---------------------------------------------------------------------------
# LATENCY TESTS
# ---------------------------------------------------------------------------

def test_orchestrator_retrieval_latency():
    """Orchestrator with mock retriever should complete in < 500 ms for 3 sub-questions."""
    orch = ReasoningOrchestrator(retriever_func=mock_retriever)
    decomp = {
        "original_query": "bias audit requirements",
        "jurisdictions": ["NYC Local Law 144", "EU AI Act", "Colorado AI Act"],
        "sub_questions": [
            {"jurisdiction": "NYC Local Law 144", "sub_question": "NYC bias audit"},
            {"jurisdiction": "EU AI Act",         "sub_question": "EU bias audit"},
            {"jurisdiction": "Colorado AI Act",   "sub_question": "Colorado bias audit"},
        ],
    }
    t0 = time.time()
    bundle = orch.run(decomp, top_k=5)
    elapsed_ms = (time.time() - t0) * 1000

    assert elapsed_ms < 500, (
        f"Orchestrator took {elapsed_ms:.0f} ms for 3 sub-questions (limit: 500 ms)"
    )
    assert len(bundle) == 3
    print(f"\n[test_orchestrator_retrieval_latency] {elapsed_ms:.1f} ms for 3 sub-questions")


def test_pipeline_latency_mocked_llm():
    """Full pipeline with mocked OpenAI call should complete in < 1 second."""
    answer_text = (
        "NYC Local Law 144 § 20-871 requires a bias audit. "
        "Colorado AI Act § 6-1-1707 requires impact assessments."
    )
    with _mock_openai(answer_text):
        interface = QueryInterface(retriever_func=mock_retriever)
        t0 = time.time()
        result = interface.run(
            "What do NYC Local Law 144 and Colorado AI Act require for bias audits?",
            top_k=3,
        )
        elapsed_ms = (time.time() - t0) * 1000

    assert elapsed_ms < 1000, (
        f"Pipeline took {elapsed_ms:.0f} ms with mocked LLM (limit: 1 000 ms)"
    )
    assert result["metadata"]["latency_ms"] < 1000
    print(
        f"\n[test_pipeline_latency_mocked_llm] "
        f"wall: {elapsed_ms:.1f} ms | recorded: {result['metadata']['latency_ms']} ms"
    )


def test_latency_scales_linearly():
    """
    Orchestrator latency for 4 sub-questions should be < 10x the latency for 1
    (i.e., no exponential blowup — confirms no accidental O(n²) behaviour).
    """
    def timed_orch(n_subqs: int) -> float:
        orch = ReasoningOrchestrator(retriever_func=mock_retriever)
        decomp = {
            "original_query": "test",
            "jurisdictions": [],
            "sub_questions": [
                {"jurisdiction": None, "sub_question": f"sub question {i}"}
                for i in range(n_subqs)
            ],
        }
        t0 = time.time()
        orch.run(decomp, top_k=3)
        return (time.time() - t0) * 1000

    t1 = timed_orch(1)
    t4 = timed_orch(4)

    ratio = t4 / max(t1, 0.5)  # floor avoids division-by-near-zero on fast machines
    assert ratio < 10, (
        f"4x sub-questions was {ratio:.1f}x slower than 1x — possible non-linear scaling"
    )
    print(
        f"\n[test_latency_scales_linearly] "
        f"1 sub-q: {t1:.1f} ms | 4 sub-q: {t4:.1f} ms | ratio: {ratio:.1f}x"
    )


# ---------------------------------------------------------------------------
# NUANCED CITATION TESTS
# ---------------------------------------------------------------------------

def test_citation_abbreviated_eu_format():
    """'EU AI Act Art. 9' (abbreviated) should extract and validate correctly."""
    answer = "The EU AI Act Art. 9 requires risk management for high-risk systems."
    validation = validate_citations(answer, [EU_CHUNK])

    assert validation["total_citations"] == 1
    assert validation["valid_count"] == 1, (
        f"Expected abbreviated 'Art. 9' to match chunk with article_number='9'; got {validation}"
    )
    print(f"\n[test_citation_abbreviated_eu_format] {validation}")


def test_citation_nist_dash_vs_space():
    """
    NIST citations use 'GOVERN-1.1' (dash) but Pinecone stores 'GOVERN 1.1' (space).
    The validator should normalise the separator and still match.
    """
    answer = "NIST AI RMF GOVERN-1.1 outlines governance policy requirements."
    validation = validate_citations(answer, [NIST_CHUNK])

    assert validation["total_citations"] == 1
    assert validation["valid_count"] == 1, (
        f"NIST ref normalisation failed — 'GOVERN-1.1' should match stored 'GOVERN 1.1'; "
        f"got {validation}"
    )
    print(f"\n[test_citation_nist_dash_vs_space] {validation}")


def test_citation_colorado_pattern():
    """Colorado AI Act § 6-1-1707 should extract and validate correctly."""
    answer = "Colorado AI Act § 6-1-1707 mandates annual impact assessments."
    validation = validate_citations(answer, [COLORADO_CHUNK])

    assert validation["total_citations"] == 1
    assert validation["valid_count"] == 1
    print(f"\n[test_citation_colorado_pattern] {validation}")


def test_citation_mixed_accuracy():
    """2 valid citations + 1 hallucinated should yield ~66.7% accuracy."""
    answer = (
        "NYC Local Law 144 § 20-871 requires a bias audit. "
        "NYC Local Law 144 § 20-870 defines the tool. "
        "NYC Local Law 144 § 20-999 imposes an additional obligation."  # hallucinated
    )
    validation = validate_citations(answer, [NYC_CHUNK, NYC_CHUNK_2])

    assert validation["total_citations"] == 3
    assert validation["valid_count"] == 2
    assert validation["invalid_count"] == 1
    assert abs(validation["accuracy"] - 2 / 3) < 0.01, (
        f"Expected ~66.7%, got {validation['accuracy']*100:.1f}%"
    )
    print(
        f"\n[test_citation_mixed_accuracy] "
        f"accuracy: {validation['accuracy']*100:.1f}% | invalid: {validation['invalid_citations']}"
    )


def test_citation_repeated_in_answer():
    """The same valid citation repeated twice should count twice (both valid)."""
    answer = (
        "As stated in NYC Local Law 144 § 20-871, a bias audit is required. "
        "The one-year window in NYC Local Law 144 § 20-871 applies to each deployment."
    )
    validation = validate_citations(answer, [NYC_CHUNK])

    assert validation["total_citations"] == 2
    assert validation["valid_count"] == 2
    assert validation["invalid_count"] == 0
    print(f"\n[test_citation_repeated_in_answer] {validation}")


def test_citation_none_in_answer():
    """Answer with no citations at all should have accuracy 1.0 (vacuously valid)."""
    answer = "Bias audits are important for ensuring fair and transparent hiring practices."
    validation = validate_citations(answer, ALL_CHUNKS)

    assert validation["total_citations"] == 0
    assert validation["accuracy"] == 1.0
    print(f"\n[test_citation_none_in_answer] {validation}")


# ---------------------------------------------------------------------------
# EDGE / ROBUSTNESS TESTS
# ---------------------------------------------------------------------------

def test_retriever_failure_is_graceful():
    """
    If the retriever raises on the first sub-question, the pipeline should still
    complete using results from the remaining sub-questions.
    """
    call_count = {"n": 0}

    def flaky_retriever(query: str, top_k: int = 5):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("Simulated Pinecone timeout")
        return mock_retriever(query, top_k)

    answer_text = "NYC Local Law 144 § 20-871 requires a bias audit."
    with _mock_openai(answer_text):
        interface = QueryInterface(retriever_func=flaky_retriever)
        result = interface.run(
            "What do NYC Local Law 144 and EU AI Act require?", top_k=3
        )

    assert result["question"]
    assert isinstance(result["answer"], str)
    # First sub-question returned nothing; second succeeded — total > 0
    assert result["metadata"]["total_chunks_retrieved"] > 0, (
        "Expected chunks from the non-failing sub-question"
    )
    print(
        f"\n[test_retriever_failure_is_graceful] "
        f"chunks: {result['metadata']['total_chunks_retrieved']} | "
        f"answer: {result['answer'][:80]}"
    )


def test_cross_document_three_jurisdictions():
    """
    Query spanning 3 known jurisdictions should produce ≥3 sub-questions and
    call the retriever ≥3 times.
    """
    retriever_calls: list = []

    def tracking_retriever(query: str, top_k: int = 5):
        retriever_calls.append(query)
        return mock_retriever(query, top_k)

    answer_text = (
        "NYC Local Law 144 § 20-871, EU AI Act Article 9, and "
        "Colorado AI Act § 6-1-1707 all impose obligations on hiring AI."
    )
    with _mock_openai(answer_text):
        interface = QueryInterface(retriever_func=tracking_retriever)
        result = interface.run(
            "What do NYC Local Law 144, EU AI Act, and Colorado AI Act require "
            "for high-risk AI systems?",
            top_k=5,  # must be ≥4 so mock_retriever includes the Colorado chunk
        )

    assert len(result["sub_questions"]) >= 3, (
        f"Expected ≥3 sub-questions for 3 jurisdictions, got {len(result['sub_questions'])}"
    )
    assert len(retriever_calls) >= 3, (
        f"Expected ≥3 retriever calls, got {len(retriever_calls)}"
    )
    assert result["validation"]["total_citations"] == 3
    assert result["validation"]["valid_citations"] == 3
    print(
        f"\n[test_cross_document_three_jurisdictions] "
        f"sub-questions: {len(result['sub_questions'])} | "
        f"retriever calls: {len(retriever_calls)} | "
        f"validation: {result['validation']['accuracy']*100:.0f}%"
    )


# ---------------------------------------------------------------------------
# CLARIFIER TESTS
# ---------------------------------------------------------------------------

def _mock_clarifier_openai(assess_payload: dict, refine_text: str = ""):
    """
    Patch query_clarifier.OpenAI so LLM calls return controlled responses.
    assess_payload: dict returned by the assess call (json_object)
    refine_text:    string returned by the refine call
    """
    def _side_effect(**kwargs):
        resp = MagicMock()
        fmt = (kwargs.get("response_format") or {}).get("type")
        if fmt == "json_object":
            resp.choices[0].message.content = json.dumps(assess_payload)
        else:
            resp.choices[0].message.content = refine_text
        return resp

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _side_effect
    return patch("query_clarifier.OpenAI", return_value=mock_client)


def test_clarifier_needs_clarification():
    """Vague query with no jurisdiction should trigger clarification via LLM."""
    payload = {
        "needs_clarification": True,
        "questions": [
            "Which regulation should I focus on? (EU AI Act, NYC LL144, Colorado, or all?)",
            "Are you asking as an employer, developer, or for general research?",
        ],
        "reasoning": "Jurisdiction and role are unspecified.",
    }
    with _mock_clarifier_openai(payload, refine_text="What bias audit requirements apply to employers under NYC Local Law 144?"):
        clarifier = QueryClarifier()
        # This query has no law name → fast path should NOT fire → LLM reached
        assessment = clarifier.assess("What are the bias audit requirements?")

    assert assessment["needs_clarification"] is True
    assert len(assessment["questions"]) == 2
    assert assessment.get("fast_path") is False   # LLM path was taken
    print(f"\n[test_clarifier_needs_clarification] questions: {assessment['questions']}")


# ── Fast-path tests (no LLM call needed) ────────────────────────────────────

def test_fast_path_named_regulation():
    """Query that names a regulation should skip clarification without any LLM call."""
    clarifier = QueryClarifier.__new__(QueryClarifier)   # no OpenAI client needed
    for query in [
        "What does NYC Local Law 144 require for bias audits?",
        "What are the EU AI Act transparency requirements?",
        "What does the Colorado AI Act say about developers?",
        "What does NIST AI RMF recommend for governance?",
    ]:
        result = clarifier.assess(query)
        assert result["needs_clarification"] is False, f"Should not clarify: '{query}'"
        assert result["fast_path"] is True, f"Should use fast path: '{query}'"
        print(f"\n  fast_path OK: '{query[:55]}' → {result['reasoning']}")


def test_fast_path_section_reference():
    """Query with a § or Article reference should skip clarification without LLM call."""
    clarifier = QueryClarifier.__new__(QueryClarifier)
    for query in [
        "What does § 20-871 require?",
        "Explain Article 9 requirements.",
        "What are the obligations under § 6-1-1707?",
    ]:
        result = clarifier.assess(query)
        assert result["needs_clarification"] is False, f"Should not clarify: '{query}'"
        assert result["fast_path"] is True
    print(f"\n[test_fast_path_section_reference] all passed")


def test_fast_path_comparison_query():
    """Comparison queries skip clarification without LLM call."""
    clarifier = QueryClarifier.__new__(QueryClarifier)
    for query in [
        "How do bias audit requirements differ between NYC and Colorado?",
        "Compare EU AI Act vs Colorado AI Act for high-risk AI.",
        "What is the difference between NYC and EU transparency requirements?",
    ]:
        result = clarifier.assess(query)
        assert result["needs_clarification"] is False, f"Should not clarify: '{query}'"
        assert result["fast_path"] is True
    print(f"\n[test_fast_path_comparison_query] all passed")


def test_fast_path_does_not_fire_for_vague_queries():
    """
    Truly vague queries should NOT be caught by the fast path —
    they should reach the LLM for assessment.
    """
    clarifier = QueryClarifier.__new__(QueryClarifier)
    vague_queries = [
        "What are the requirements?",
        "What do I need to do about AI?",
        "What notice requirements exist?",
        "What is high-risk AI?",
    ]
    for query in vague_queries:
        reason = clarifier._fast_path_reason(query)
        assert reason is None, (
            f"Fast path fired for vague query '{query}' with reason: {reason}"
        )
    print(f"\n[test_fast_path_does_not_fire_for_vague_queries] all {len(vague_queries)} vague queries correctly reach LLM")


def test_clarifier_no_clarification_for_specific_query():
    """Specific query with law + section should be caught by fast path (no LLM)."""
    clarifier = QueryClarifier.__new__(QueryClarifier)
    assessment = clarifier.assess("What does NYC Local Law 144 § 20-871 require?")

    assert assessment["needs_clarification"] is False
    assert assessment["questions"] == []
    assert assessment["fast_path"] is True   # fast path, not LLM
    print(f"\n[test_clarifier_no_clarification_for_specific_query] fast_path={assessment['fast_path']}, reasoning={assessment['reasoning']}")


def test_clarifier_refine_incorporates_answers():
    """refine() should produce a query that includes jurisdiction and role from answers."""
    refined_text = "What bias audit requirements apply to employers using automated hiring tools under NYC Local Law 144?"
    with _mock_clarifier_openai({}, refine_text=refined_text):
        clarifier = QueryClarifier()
        result = clarifier.refine(
            original_query="What are bias audit requirements?",
            qa_pairs={
                "Which regulation?": "NYC Local Law 144",
                "What is your role?": "Employer",
            },
        )

    assert "NYC Local Law 144" in result
    assert result == refined_text
    print(f"\n[test_clarifier_refine_incorporates_answers] refined: {result}")


def test_clarifier_run_interactive_with_answers():
    """
    run_interactive() should ask questions, collect answers, and return refined query.
    """
    payload = {
        "needs_clarification": True,
        "questions": ["Which jurisdiction?", "What is your role?"],
        "reasoning": "Missing jurisdiction and role.",
    }
    refined = "What bias audit requirements apply to NYC employers using AI hiring tools?"
    answers = iter(["NYC Local Law 144", "Employer"])

    with _mock_clarifier_openai(payload, refine_text=refined):
        clarifier = QueryClarifier()
        result_query, was_clarified = clarifier.run_interactive(
            "What are the bias audit requirements?",
            input_fn=lambda _: next(answers),
        )

    assert was_clarified is True
    assert result_query == refined
    print(f"\n[test_clarifier_run_interactive_with_answers] refined: {result_query}")


def test_clarifier_run_interactive_skipped_answers():
    """If user skips all answers (empty strings), return original query unchanged."""
    payload = {
        "needs_clarification": True,
        "questions": ["Which jurisdiction?"],
        "reasoning": "Jurisdiction unclear.",
    }
    with _mock_clarifier_openai(payload):
        clarifier = QueryClarifier()
        result_query, was_clarified = clarifier.run_interactive(
            "What are the requirements?",
            input_fn=lambda _: "",   # user hits enter without answering
        )

    assert was_clarified is False
    assert result_query == "What are the requirements?"
    print(f"\n[test_clarifier_run_interactive_skipped_answers] returned original query")


def test_clarifier_llm_failure_fallback():
    """When OpenAI is unavailable, assess() returns needs_clarification=False gracefully."""
    with patch("query_clarifier.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("quota exceeded")
        mock_cls.return_value = mock_client

        clarifier  = QueryClarifier()
        assessment = clarifier.assess("What are the bias audit requirements?")

    assert assessment["needs_clarification"] is False
    assert assessment["questions"] == []
    print(f"\n[test_clarifier_llm_failure_fallback] gracefully skipped: {assessment['reasoning']}")


def test_clarifier_enforces_max_two_questions():
    """Even if LLM returns 3 questions, assess() should cap at 2."""
    payload = {
        "needs_clarification": True,
        "questions": ["Q1?", "Q2?", "Q3?"],
        "reasoning": "Very vague query.",
    }
    with _mock_clarifier_openai(payload):
        clarifier  = QueryClarifier()
        assessment = clarifier.assess("Tell me about AI regulations.")

    assert len(assessment["questions"]) <= 2
    print(f"\n[test_clarifier_enforces_max_two_questions] capped at {len(assessment['questions'])} questions")


@pytest.mark.live
def test_live_end_to_end():
    """
    Full pipeline test hitting real Pinecone and OpenAI.
    Run with:  pytest tests/test_integration.py -v -m live
    """
    from src.retriever import retrieve

    interface = QueryInterface(retriever_func=retrieve)
    result = interface.run(
        "What bias audit requirements apply to hiring AI in NYC and Colorado?",
        top_k=5,
    )

    assert result["answer"], "Expected a non-empty answer"
    assert "(LLM synthesis unavailable)" not in result["answer"], (
        "Answer fell back to raw text — LLM call failed. Check OpenAI API key/quota."
    )

    # Must have named citations with a non-empty article label
    named_citations = [c for c in result["citations"] if c["article"].strip()]
    assert len(named_citations) > 0, (
        f"Expected at least one citation with an article/section label. "
        f"Got: {result['citations']}"
    )

    # Validator must have actually checked something (not vacuous 0/0 pass)
    v = result["validation"]
    assert v["total_citations"] > 0, (
        "Validator found 0 inline citations in the answer — "
        "LLM is not writing citations in the expected format."
    )
    assert v["accuracy"] >= 0.9, (
        f"Citation accuracy below 90%: {v['accuracy']:.1%}  "
        f"Invalid: {v['details']['invalid']}"
    )

    assert result["metadata"]["latency_ms"] < 30_000, "Latency exceeded 30s"

    print(f"\n[test_live_end_to_end]")
    print(f"  Answer preview: {result['answer'][:200]}")
    print(f"  Citations ({len(named_citations)}): "
          f"{[c['source']+' '+c['article'] for c in named_citations]}")
    print(f"  Validation: {v['accuracy']*100:.1f}% "
          f"({v['valid_citations']}/{v['total_citations']})")
    if v["details"]["invalid"]:
        print(f"  Hallucinated: {v['details']['invalid']}")
    print(f"  Latency: {result['metadata']['latency_ms']} ms")
