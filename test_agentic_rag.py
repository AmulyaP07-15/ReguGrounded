#!/usr/bin/env python3
"""
test_agentic_rag.py  —  Agentic RAG End-to-End Demo & Evaluation
=================================================================
Runs ReguGrounded's full agentic RAG pipeline on a set of regulatory
compliance questions and reports per-query traces + aggregate metrics.

What this script does
---------------------
Each query goes through four stages of the agentic RAG pipeline, and
this script prints exactly what happens at every stage so you can see
the system "thinking":

  Stage 1 — Query Decomposition
    RLMEngine detects which regulations are relevant and breaks the query
    into one focused sub-question per jurisdiction.

  Stage 2 — Parallel Evidence Retrieval
    ReasoningOrchestrator fires one retrieval request per sub-question
    concurrently. Each request uses hybrid BM25 + semantic search with
    a jurisdiction-scoped Pinecone filter so cross-regulation noise is
    eliminated. Results are deduplicated across sub-questions.

  Stage 3 — Grounded Answer Synthesis
    AnswerSynthesizer builds a structured prompt from the retrieved chunks
    and calls gemini-2.0-flash with a strict grounding rule: the answer must
    cite only what was retrieved — no prior knowledge allowed.

  Stage 4 — Citation Validation
    validate_citations compares every citation in the generated answer
    against the actual chunk metadata. Any citation that doesn't match
    a retrieved chunk is flagged as a hallucination.

Aggregate metrics reported at the end
--------------------------------------
  - Success rate          (% of queries that completed without error)
  - Latency p50/p90/p99   (response time percentiles in ms)
  - Citation accuracy     (% of citations matched to real chunks)
  - Hallucination rate    (% of citations with no matching chunk)
  - Avg chunks retrieved  (per query)
  - Avg sub-questions     (per query)

Usage
-----
    python test_agentic_rag.py                      # 3-question quick mode
    python test_agentic_rag.py --mode full           # all 18 ground-truth questions
    python test_agentic_rag.py --verbose             # print per-step trace for each query
    python test_agentic_rag.py --question "..."      # run a single custom question
    python test_agentic_rag.py --mode full --verbose # both

Requirements
------------
    GEMINI_API_KEY and PINECONE_API_KEY must be set in .env
    Pinecone index must already be populated (run: python src/process_all_pdfs.py)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from query_interface import QueryInterface                     # noqa: E402
from rlm_engine import RLMEngine                              # noqa: E402
from reasoning_orchestrator import ReasoningOrchestrator      # noqa: E402
from answer_synthesizer import AnswerSynthesizer              # noqa: E402
from citation_validator import validate_citations             # noqa: E402

GROUND_TRUTH_PATH = ROOT / "eval" / "ground_truth.json"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_retriever():
    """Lazy-load the retriever singleton (downloads embedding model on first call)."""
    from src.retriever import Retriever
    return Retriever()


def _percentile(values: list, p: int) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = (p / 100) * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return round(sorted_vals[lo] + (idx - lo) * (sorted_vals[hi] - sorted_vals[lo]), 1)


def _divider(char="─", width=70):
    print(char * width)


# ── Per-query verbose trace ───────────────────────────────────────────────────

def run_single_query_verbose(
    question: str,
    rlm: RLMEngine,
    orchestrator: ReasoningOrchestrator,
    synthesizer: AnswerSynthesizer,
    top_k: int = 5,
) -> dict:
    """
    Run one query through the full pipeline, printing each stage.

    Returns the same result dict as QueryInterface.run() so batch
    evaluation can reuse these results without re-running anything.
    """
    _divider("═")
    print(f"  QUERY: {question}")
    _divider("═")

    t_start = time.time()

    # ── Stage 1: Query Decomposition ─────────────────────────────────────────
    print("\n  [Stage 1] Query Decomposition")
    _divider()
    decomposition = rlm.decompose(question)
    jurisdictions = decomposition["jurisdictions"]
    sub_questions = decomposition["sub_questions"]

    if jurisdictions:
        print(f"  Jurisdictions detected: {', '.join(jurisdictions)}")
    else:
        print("  No specific jurisdiction detected — querying all regulations")

    print(f"  Sub-questions ({len(sub_questions)}):")
    for i, sq in enumerate(sub_questions, 1):
        j = sq.get("jurisdiction") or "general"
        print(f"    {i}. [{j}] {sq['sub_question']}")

    # ── Stage 2: Parallel Evidence Retrieval ─────────────────────────────────
    print("\n  [Stage 2] Parallel Evidence Retrieval")
    _divider()
    evidence_bundle = orchestrator.run(decomposition, top_k=top_k)
    all_chunks = orchestrator.get_all_chunks(evidence_bundle)

    for bundle in evidence_bundle:
        j = bundle.get("jurisdiction") or "general"
        n = len(bundle["retrieved_chunks"])
        print(f"  [{j}] → {n} chunk(s) retrieved")
        if n > 0:
            top = bundle["retrieved_chunks"][0]
            law = top.get("law_name") or "unknown"
            art = top.get("article_number")
            score = top.get("score")
            ref = f"Article {art}" if art else ""
            score_str = f"  score={score:.3f}" if score is not None else ""
            print(f"    Top chunk: {law} {ref}{score_str}")
            print(f"    \"{top['text'][:120].strip()}...\"")

    print(f"\n  Total chunks after deduplication: {len(all_chunks)}")

    # ── Stage 3: Grounded Answer Synthesis ───────────────────────────────────
    print("\n  [Stage 3] Grounded Answer Synthesis")
    _divider()
    synthesis = synthesizer.synthesize(
        original_query=question,
        decomposition=decomposition,
        evidence_bundle=evidence_bundle,
    )

    answer = synthesis["answer"]
    citations = synthesis["citations"]
    print(f"  Answer ({len(answer)} chars):")
    words = answer.split()
    line, lines = [], []
    for word in words:

        line.append(word)
        if len(" ".join(line)) > 66:
            lines.append("  " + " ".join(line))
            line = []
    if line:
        lines.append("  " + " ".join(line))
    print("\n".join(lines))

    print(f"\n  Citations ({len(citations)}):")
    for c in citations:
        src = c.get("source", "")
        art = c.get("article", "")
        print(f"    - {src} {art}".rstrip())

    # ── Stage 4: Citation Validation ─────────────────────────────────────────
    print("\n  [Stage 4] Citation Validation")
    _divider()
    validation = validate_citations(answer, all_chunks)

    total = validation["total_citations"]
    valid = validation["valid_count"]
    invalid = validation["invalid_count"]
    accuracy = validation["accuracy"]
    hallucination_rate = validation["hallucination_rate"]

    if total == 0:
        print("  No citations found in answer — nothing to validate")
    else:
        status = "PASS" if hallucination_rate <= 0.10 else "WARN"
        print(f"  [{status}] {valid}/{total} citations matched to retrieved chunks")
        print(f"  Citation accuracy:   {accuracy*100:.1f}%")
        print(f"  Hallucination rate:  {hallucination_rate*100:.1f}%  (target: ≤10%)")
        if invalid > 0:
            print(f"  Unmatched citations:")
            for d in validation.get("details", []):
                if not d.get("valid"):
                    print(f"    ! {d.get('citation', '')}")

    latency_ms = int((time.time() - t_start) * 1000)
    print(f"\n  Latency: {latency_ms} ms")

    return {
        "question":      question,
        "sub_questions": [sq["sub_question"] for sq in sub_questions],
        "answer":        answer,
        "citations":     citations,
        "validation": {
            "total_citations":    total,
            "valid_citations":    valid,
            "invalid_citations":  invalid,
            "accuracy":           accuracy,
            "hallucination_rate": hallucination_rate,
            "details":            validation.get("details", []),
        },
        "metadata": {
            "jurisdictions_searched": jurisdictions,
            "total_chunks_retrieved": len(all_chunks),
            "latency_ms":             latency_ms,
            "cache_hit":              False,
            "guardrail_warnings":     [],
        },
    }


# ── Batch evaluation ──────────────────────────────────────────────────────────

def run_batch(
    questions: list,
    verbose: bool = False,
    top_k: int = 5,
) -> dict:
    """
    Run every question through the pipeline and collect aggregate metrics.

    When verbose=True, each query gets a full stage-by-stage trace.
    When verbose=False, a compact one-line summary is printed per query.

    Returns a metrics dict with success_rate, latency percentiles,
    citation accuracy, hallucination rate, and per-question results.
    """
    print("\nLoading retriever (embedding model)...")
    retriever = _load_retriever()

    rlm          = RLMEngine()
    orchestrator = ReasoningOrchestrator(retriever_func=retriever.retrieve)
    synthesizer  = AnswerSynthesizer()

    per_question = []
    latencies: list = []
    accuracies: list = []
    hallucination_rates: list = []
    chunk_counts: list = []
    errors: list = []

    print(f"\nRunning {len(questions)} question(s)...\n")

    for i, item in enumerate(questions, 1):
        question = item["question"] if isinstance(item, dict) else item
        qid      = item.get("id", i) if isinstance(item, dict) else i

        if verbose:
            try:
                result = run_single_query_verbose(
                    question=question,
                    rlm=rlm,
                    orchestrator=orchestrator,
                    synthesizer=synthesizer,
                    top_k=top_k,
                )
                success = True
                error   = None
            except Exception as e:
                result  = None
                success = False
                error   = str(e)
                print(f"\n  ERROR: {e}")
        else:
            interface = QueryInterface(
                retriever_func=retriever.retrieve,
                use_cache=False,
            )
            t0 = time.time()
            try:
                result  = interface.run(user_query=question, top_k=top_k)
                success = True
                error   = None
            except Exception as e:
                result  = None
                success = False
                error   = str(e)
            latency_override = int((time.time() - t0) * 1000)

        if success and result:
            latency_ms = result["metadata"]["latency_ms"]
            chunks     = result["metadata"]["total_chunks_retrieved"]
            accuracy   = result["validation"]["accuracy"]
            h_rate     = result["validation"]["hallucination_rate"]

            latencies.append(latency_ms)
            chunk_counts.append(chunks)
            accuracies.append(accuracy)
            hallucination_rates.append(h_rate)

            flag  = "OK  " if h_rate <= 0.10 else "WARN"
            ncit  = result["validation"]["total_citations"]
            nvalid= result["validation"]["valid_citations"]

            if not verbose:
                print(
                    f"  [{qid:02d}] {flag}  {latency_ms:6d}ms  "
                    f"chunks={chunks:2d}  "
                    f"cit={nvalid}/{ncit}  "
                    f"| {question[:55]}"
                )

            per_question.append({
                "id":               qid,
                "question":         question,
                "success":          True,
                "latency_ms":       latency_ms,
                "chunks":           chunks,
                "citation_accuracy":accuracy,
                "hallucination_rate":h_rate,
            })

        else:
            err_msg = error or "unknown error"
            errors.append(f"Q{qid}: {err_msg}")
            if not verbose:
                print(f"  [{qid:02d}] FAIL  --- ms  | {question[:55]}")
                print(f"        ERROR: {err_msg}")
            per_question.append({
                "id":               qid,
                "question":         question,
                "success":          False,
                "latency_ms":       0,
                "chunks":           0,
                "citation_accuracy":0.0,
                "hallucination_rate":1.0,
                "error":            err_msg,
            })

    n_total   = len(per_question)
    n_success = sum(1 for r in per_question if r["success"])

    return {
        "total":              n_total,
        "successful":         n_success,
        "failed":             n_total - n_success,
        "success_rate":       round(n_success / n_total, 4) if n_total else 0.0,
        "latency_p50":        _percentile(latencies, 50),
        "latency_p90":        _percentile(latencies, 90),
        "latency_p99":        _percentile(latencies, 99),
        "avg_latency":        round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        "avg_chunks":         round(sum(chunk_counts) / len(chunk_counts), 1) if chunk_counts else 0.0,
        "avg_citation_accuracy":  round(sum(accuracies) / len(accuracies), 4) if accuracies else 0.0,
        "avg_hallucination_rate": round(sum(hallucination_rates) / len(hallucination_rates), 4) if hallucination_rates else 0.0,
        "errors":             errors,
        "per_question":       per_question,
    }


# ── Summary display ───────────────────────────────────────────────────────────

def print_summary(metrics: dict):
    """Print aggregate metrics with pass/fail against targets."""

    def _fmt(val, target, fmt, pass_fn):
        v_str = format(val, fmt)
        t_str = format(target, fmt)
        status = "PASS" if pass_fn(val, target) else "FAIL"
        return f"{v_str}  (target: {t_str})  [{status}]"

    print("\n")
    _divider("═")
    print("  AGGREGATE METRICS")
    _divider("═")
    print(f"  Questions run:     {metrics['total']}")
    print(f"  Successful:        {metrics['successful']}")
    print(f"  Failed:            {metrics['failed']}")
    print()
    print(f"  Success rate:      {_fmt(metrics['success_rate'],           0.95, '.1%', lambda v,t: v>=t)}")
    print(f"  Citation accuracy: {_fmt(metrics['avg_citation_accuracy'],  0.90, '.1%', lambda v,t: v>=t)}")
    print(f"  Hallucination rate:{_fmt(metrics['avg_hallucination_rate'], 0.10, '.1%', lambda v,t: v<=t)}")
    print(f"  Avg chunks/query:  {metrics['avg_chunks']:.1f}")
    print()
    print(f"  Latency p50:       {metrics['latency_p50']:.0f} ms")
    print(f"  Latency p90:       {metrics['latency_p90']:.0f} ms  (target: ≤10,000 ms)")
    print(f"  Latency p99:       {metrics['latency_p99']:.0f} ms")
    print(f"  Avg latency:       {metrics['avg_latency']:.0f} ms")

    if metrics["errors"]:
        print(f"\n  Errors ({len(metrics['errors'])}):")
        for e in metrics["errors"]:
            print(f"    {e}")

    _divider("═")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Agentic RAG end-to-end demo and evaluation for ReguGrounded",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["quick", "full"], default="quick",
        help="quick=3 questions from ground truth, full=all 18 (default: quick)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-stage trace for every query (decomposition → retrieval → synthesis → validation)",
    )
    parser.add_argument(
        "--question", type=str, default=None,
        help="Run a single custom question instead of the ground-truth set",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of chunks to retrieve per sub-question (default: 5)",
    )
    args = parser.parse_args()

    print("\nReguGrounded — Agentic RAG Test")
    print("=" * 70)
    print("Pipeline: Decompose → Parallel Retrieve → Synthesize → Validate")
    print("=" * 70)

    if args.question:
        questions = [{"id": 1, "question": args.question}]
        print(f"\nMode: single custom question")
    else:
        with open(GROUND_TRUTH_PATH) as f:
            all_questions = json.load(f)
        questions = all_questions[:3] if args.mode == "quick" else all_questions
        print(f"\nMode: {args.mode} ({len(questions)} questions from ground_truth.json)")

    print(f"Verbose: {'yes — showing per-stage trace' if args.verbose else 'no — compact output'}")
    print(f"top_k:   {args.top_k} chunks per sub-question\n")

    metrics = run_batch(questions, verbose=args.verbose, top_k=args.top_k)
    print_summary(metrics)


if __name__ == "__main__":
    main()
