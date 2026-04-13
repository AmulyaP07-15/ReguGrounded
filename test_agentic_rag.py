#!/usr/bin/env python3
"""
test_agentic_rag.py  —  Agentic RAG End-to-End Demo & Evaluation
=================================================================
Runs the AgenticRAGPipeline on regulatory compliance questions and
prints exactly what the agent does at every stage.

This pipeline is DIFFERENT from the standard RLM pipeline:
  RLM:     decompose → retrieve (fixed, one pass) → synthesize → validate
  Agentic: decompose → [plan tool → retrieve → reflect → loop?] → synthesize → validate

Per-query trace (--verbose mode)
---------------------------------
  Stage 1 — Query Decomposition
    Same as RLM: RLMEngine breaks the query into jurisdiction sub-questions.

  Stage 2 — Agent Loop (NEW — not in RLM)
    For each jurisdiction, the agent runs up to 3 retrieval iterations:
      a. PLAN:    ToolSelector picks semantic / keyword / hybrid based on query
      b. RETRIEVE: EnhancedRetriever fetches chunks with the chosen mode
      c. REFLECT:  LLM judges whether evidence is sufficient
      d. ITERATE:  If not sufficient, reformulate query + switch tool, go to (a)

  Stage 3 — Grounded Answer Synthesis
    AnswerSynthesizer builds answer from ALL accumulated chunks (across iterations).

  Stage 4 — Citation Validation
    validate_citations checks every citation against retrieved chunks.

Aggregate metrics
-----------------
  Success rate, latency p50/p90/p99, citation accuracy, hallucination rate,
  avg chunks/query, avg iterations/jurisdiction.

Usage
-----
    python test_agentic_rag.py                      # 3-question quick mode
    python test_agentic_rag.py --mode full           # all 18 ground-truth questions
    python test_agentic_rag.py --verbose             # full per-stage + agent trace
    python test_agentic_rag.py --question "..."      # single custom question
    python test_agentic_rag.py --mode full --verbose # both

Requirements
------------
    OPENAI_API_KEY and PINECONE_API_KEY must be set in .env
    Pinecone index must already be populated (run: python src/process_all_pdfs.py)
"""

import argparse
import json
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from agentic_rag_pipeline import AgenticRAGPipeline           # noqa: E402

GROUND_TRUTH_PATH = ROOT / "eval" / "ground_truth.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_pipeline() -> AgenticRAGPipeline:
    """Load EnhancedRetriever + AgenticRAGPipeline (downloads embedding model once)."""
    from src.enhanced_retriever import EnhancedRetriever
    from src.retrieval_config import DEFAULT_CONFIG
    retriever = EnhancedRetriever(config=DEFAULT_CONFIG)
    return AgenticRAGPipeline(retriever=retriever)


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


def _wrap(text: str, indent: str = "  ", width: int = 66) -> str:
    words = text.split()
    line, lines = [], []
    for word in words:
        line.append(word)
        if len(" ".join(line)) > width:
            lines.append(indent + " ".join(line))
            line = []
    if line:
        lines.append(indent + " ".join(line))
    return "\n".join(lines)


# ── Per-query verbose trace ───────────────────────────────────────────────────

def run_single_query_verbose(
    question: str,
    pipeline: AgenticRAGPipeline,
    top_k: int = 5,
) -> dict:
    """
    Run one query through the AgenticRAGPipeline, printing every stage
    including the full agent loop trace (tool choices, reflections, iterations).
    """
    _divider("═")
    print(f"  QUERY: {question}")
    _divider("═")

    # Run the full agentic pipeline
    result = pipeline.run(question, top_k=top_k)

    latency_ms    = result["metadata"]["latency_ms"]
    agent_trace   = result["metadata"].get("agent_trace", [])
    jurisdictions = result["metadata"]["jurisdictions_searched"]

    # ── Stage 1: Query Decomposition ─────────────────────────────────────────
    print("\n  [Stage 1] Query Decomposition")
    _divider()
    if jurisdictions:
        print(f"  Jurisdictions detected: {', '.join(jurisdictions)}")
    else:
        print("  No specific jurisdiction detected — querying all regulations")

    sub_questions = result["sub_questions"]
    print(f"  Sub-questions ({len(sub_questions)}):")
    for i, sq in enumerate(sub_questions, 1):
        j = agent_trace[i - 1]["jurisdiction"] if i - 1 < len(agent_trace) else "general"
        print(f"    {i}. [{j}] {sq}")

    # ── Stage 2: Agent Loop (per jurisdiction) ────────────────────────────────
    print("\n  [Stage 2] Agent Loop  ← NEW vs RLM pipeline")
    _divider()
    for trace in agent_trace:
        j          = trace["jurisdiction"]
        n_iters    = trace["iterations"]
        tools      = trace["tools_used"]
        reasons    = trace["tool_reasons"]
        queries    = trace["queries_issued"]
        new_chunks = trace["new_per_iter"]
        reflections= trace["reflections"]
        final_n    = trace["final_chunks"]
        final_avg  = trace["final_avg_score"]

        print(f"\n  [{j}]  {n_iters} iteration(s)  →  {final_n} chunks  avg_score={final_avg}")

        for it_idx in range(n_iters):
            tool    = tools[it_idx]    if it_idx < len(tools)    else "?"
            reason  = reasons[it_idx]  if it_idx < len(reasons)  else ""
            query   = queries[it_idx]  if it_idx < len(queries)  else ""
            new_c   = new_chunks[it_idx] if it_idx < len(new_chunks) else 0
            refl    = reflections[it_idx] if it_idx < len(reflections) else {}

            suf     = refl.get("sufficient", True)
            r_text  = refl.get("reasoning", "")
            gap     = refl.get("gap", "")

            print(f"\n    iter {it_idx + 1}")
            print(f"      tool:      {tool}  ({reason})")
            print(f"      query:     {query[:80]}")
            print(f"      new chunks: {new_c}")
            print(f"      reflect:   sufficient={suf} | {r_text}")
            if not suf and gap:
                print(f"      gap:       {gap}")

    total_chunks = result["metadata"]["total_chunks_retrieved"]
    print(f"\n  Total chunks accumulated: {total_chunks}")

    # ── Stage 3: Grounded Answer Synthesis ───────────────────────────────────
    print("\n  [Stage 3] Grounded Answer Synthesis")
    _divider()
    answer    = result["answer"]
    citations = result["citations"]

    print(f"  Answer ({len(answer)} chars):")
    print(_wrap(answer))

    print(f"\n  Citations ({len(citations)}):")
    for c in citations:
        print(f"    - {c.get('source','')} {c.get('article','')}".rstrip())

    # ── Stage 4: Citation Validation ─────────────────────────────────────────
    print("\n  [Stage 4] Citation Validation")
    _divider()
    v      = result["validation"]
    total  = v["total_citations"]
    valid  = v["valid_citations"]
    inv    = v["invalid_citations"]
    acc    = v["accuracy"]
    h_rate = v["hallucination_rate"]

    if total == 0:
        print("  No citations found in answer — nothing to validate")
    else:
        status = "PASS" if h_rate <= 0.10 else "WARN"
        print(f"  [{status}] {valid}/{total} citations matched to retrieved chunks")
        print(f"  Citation accuracy:   {acc*100:.1f}%")
        print(f"  Hallucination rate:  {h_rate*100:.1f}%  (target: ≤10%)")
        if inv > 0:
            for d in v.get("details", {}).get("invalid", []):
                print(f"    ! {d}")

    print(f"\n  Latency: {latency_ms} ms")

    return result


# ── Batch evaluation ──────────────────────────────────────────────────────────

def run_batch(
    questions: list,
    verbose: bool = False,
    top_k: int = 5,
) -> dict:
    """
    Run every question through AgenticRAGPipeline and collect aggregate metrics.

    verbose=True  → full per-stage agent trace per query
    verbose=False → compact one-line summary per query

    Extra metric vs RLM test: avg_iterations (how many retrieval rounds the
    agent needed on average per jurisdiction).
    """
    print("\nLoading EnhancedRetriever + AgenticRAGPipeline...")
    pipeline = _load_pipeline()

    per_question:        list = []
    latencies:           list = []
    accuracies:          list = []
    hallucination_rates: list = []
    chunk_counts:        list = []
    iteration_counts:    list = []
    errors:              list = []

    print(f"\nRunning {len(questions)} question(s)...\n")

    for i, item in enumerate(questions, 1):
        question = item["question"] if isinstance(item, dict) else item
        qid      = item.get("id", i) if isinstance(item, dict) else i

        try:
            if verbose:
                result = run_single_query_verbose(question, pipeline, top_k=top_k)
            else:
                result = pipeline.run(question, top_k=top_k)
            success = True
            error   = None
        except Exception as e:
            result  = None
            success = False
            error   = str(e)
            if verbose:
                print(f"\n  ERROR: {e}")

        if success and result:
            latency_ms = result["metadata"]["latency_ms"]
            chunks     = result["metadata"]["total_chunks_retrieved"]
            accuracy   = result["validation"]["accuracy"]
            h_rate     = result["validation"]["hallucination_rate"]

            # Average iterations across all jurisdictions in this query
            agent_trace = result["metadata"].get("agent_trace", [])
            avg_iters   = (
                round(sum(t["iterations"] for t in agent_trace) / len(agent_trace), 1)
                if agent_trace else 1.0
            )

            latencies.append(latency_ms)
            chunk_counts.append(chunks)
            accuracies.append(accuracy)
            hallucination_rates.append(h_rate)
            iteration_counts.append(avg_iters)

            flag   = "OK  " if h_rate <= 0.10 else "WARN"
            ncit   = result["validation"]["total_citations"]
            nvalid = result["validation"]["valid_citations"]

            if not verbose:
                tools_used = list({
                    t for trace in agent_trace for t in trace.get("tools_used", [])
                })
                print(
                    f"  [{qid:02d}] {flag}  {latency_ms:6d}ms  "
                    f"chunks={chunks:2d}  iters={avg_iters:.1f}  "
                    f"tools={','.join(tools_used) or '?'}  "
                    f"cit={nvalid}/{ncit}  "
                    f"| {question[:45]}"
                )

            per_question.append({
                "id":                qid,
                "question":          question,
                "success":           True,
                "latency_ms":        latency_ms,
                "chunks":            chunks,
                "avg_iterations":    avg_iters,
                "citation_accuracy": accuracy,
                "hallucination_rate":h_rate,
            })

        else:
            err_msg = error or "unknown error"
            errors.append(f"Q{qid}: {err_msg}")
            if not verbose:
                print(f"  [{qid:02d}] FAIL  --- ms  | {question[:55]}")
                print(f"        ERROR: {err_msg}")
            per_question.append({
                "id":                qid,
                "question":          question,
                "success":           False,
                "latency_ms":        0,
                "chunks":            0,
                "avg_iterations":    0,
                "citation_accuracy": 0.0,
                "hallucination_rate":1.0,
                "error":             err_msg,
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
        "avg_iterations":     round(sum(iteration_counts) / len(iteration_counts), 2) if iteration_counts else 0.0,
        "avg_citation_accuracy":  round(sum(accuracies) / len(accuracies), 4) if accuracies else 0.0,
        "avg_hallucination_rate": round(sum(hallucination_rates) / len(hallucination_rates), 4) if hallucination_rates else 0.0,
        "errors":             errors,
        "per_question":       per_question,
    }


# ── Summary display ───────────────────────────────────────────────────────────

def print_summary(metrics: dict):
    """Print aggregate metrics with pass/fail against targets."""

    def _fmt(val, target, fmt, pass_fn):
        v_str  = format(val, fmt)
        t_str  = format(target, fmt)
        status = "PASS" if pass_fn(val, target) else "FAIL"
        return f"{v_str}  (target: {t_str})  [{status}]"

    print("\n")
    _divider("═")
    print("  AGGREGATE METRICS  (AgenticRAGPipeline)")
    _divider("═")
    print(f"  Questions run:       {metrics['total']}")
    print(f"  Successful:          {metrics['successful']}")
    print(f"  Failed:              {metrics['failed']}")
    print()
    print(f"  Success rate:        {_fmt(metrics['success_rate'],           0.95, '.1%', lambda v,t: v>=t)}")
    print(f"  Citation accuracy:   {_fmt(metrics['avg_citation_accuracy'],  0.90, '.1%', lambda v,t: v>=t)}")
    print(f"  Hallucination rate:  {_fmt(metrics['avg_hallucination_rate'], 0.10, '.1%', lambda v,t: v<=t)}")
    print(f"  Avg chunks/query:    {metrics['avg_chunks']:.1f}")
    print(f"  Avg iterations/jur:  {metrics['avg_iterations']:.2f}  ← agentic loop depth")
    print()
    print(f"  Latency p50:         {metrics['latency_p50']:.0f} ms")
    print(f"  Latency p90:         {metrics['latency_p90']:.0f} ms  (target: ≤10,000 ms)")
    print(f"  Latency p99:         {metrics['latency_p99']:.0f} ms")
    print(f"  Avg latency:         {metrics['avg_latency']:.0f} ms")

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
        help="Print full per-stage trace including agent loop (tool choices, reflections)",
    )
    parser.add_argument(
        "--question", type=str, default=None,
        help="Run a single custom question instead of the ground-truth set",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of chunks to retrieve per sub-question per iteration (default: 5)",
    )
    args = parser.parse_args()

    print("\nReguGrounded — Agentic RAG Test")
    print("=" * 70)
    print("Pipeline: Decompose → [Plan Tool → Retrieve → Reflect → Loop?] → Synthesize → Validate")
    print("=" * 70)

    if args.question:
        questions = [{"id": 1, "question": args.question}]
        print(f"\nMode: single custom question")
    else:
        with open(GROUND_TRUTH_PATH) as f:
            all_questions = json.load(f)
        questions = all_questions[:3] if args.mode == "quick" else all_questions
        print(f"\nMode: {args.mode} ({len(questions)} questions from ground_truth.json)")

    print(f"Verbose: {'yes — full agent trace per query' if args.verbose else 'no — compact output'}")
    print(f"top_k:   {args.top_k} chunks per sub-question per iteration\n")

    metrics = run_batch(questions, verbose=args.verbose, top_k=args.top_k)
    print_summary(metrics)


if __name__ == "__main__":
    main()
