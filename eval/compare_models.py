#!/usr/bin/env python3
"""
compare_models.py — Side-by-side comparison of all three models
================================================================
Runs Standard RAG, Agentic RAG, and CAG with RLM on the same questions
and prints a performance comparison table.

Metrics per model:
    Success Rate       — % of queries that completed without error
    Avg Latency        — mean response time in ms
    Citation Accuracy  — % of citations matched to retrieved chunks
    Hallucination Rate — % of citations with no matching chunk
    Avg Chunks         — average chunks retrieved per query

Usage:
    python eval/compare_models.py               # quick mode (3 questions)
    python eval/compare_models.py --mode full   # all 18 ground-truth questions
    python eval/compare_models.py --verbose     # print per-question details
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from eval.baselines.standard_rag import process_query as standard_rag_query  # noqa: E402
from eval.baselines.agentic_rag  import process_query as agentic_rag_query   # noqa: E402
from query_interface             import process_query as cag_rlm_query        # noqa: E402

GROUND_TRUTH_PATH = Path(__file__).parent / "ground_truth.json"

MODELS = [
    ("Standard RAG",  standard_rag_query),
    ("ARAG",          agentic_rag_query),
    ("CAG with RLM",  cag_rlm_query),
]


def _apply_posthoc_validation(result: dict) -> dict:
    """
    Apply post-hoc citation validation to any result that doesn't have built-in
    validation (validation key is None or missing).

    Uses retrieved_chunks from the result dict — both Standard RAG and A-RAG
    expose this key so the eval script can measure them fairly.
    """
    from citation_validator import validate_citations  # local import — avoid circular

    if result.get("validation") is not None:
        return result   # already validated (e.g. Standard RAG, CAG)

    chunks = result.get("retrieved_chunks", [])
    answer = result.get("answer", "")
    validation = validate_citations(answer, chunks)
    validation["note"] = "Post-hoc validation applied by eval script for comparison."
    result["validation"] = validation
    return result


def run_model(name: str, query_fn, questions: list, verbose: bool) -> dict:
    """Run one model over all questions, apply post-hoc validation where needed,
    and return aggregate metrics."""
    latencies, accuracies, hallucination_rates, chunk_counts = [], [], [], []
    errors        = []
    llm_failures  = 0
    retry_total   = 0
    validation_method = "unknown"

    print(f"\n  Running {name}...")

    for item in questions:
        question = item["question"]
        qid = item.get("id", "?")
        t0 = time.time()

        try:
            result = query_fn(question)

            # Apply post-hoc validation for systems that don't have it built-in
            result = _apply_posthoc_validation(result)

            latency_ms = int((time.time() - t0) * 1000)
            meta = result.get("metadata", {})
            v    = result.get("validation", {}) or {}

            # Capture metadata on first successful result
            if validation_method == "unknown" and "validation_method" in meta:
                validation_method = meta["validation_method"]
            if meta.get("llm_synthesis_failed"):
                llm_failures += 1
            if meta.get("retry_attempts", 1) > 1:
                retry_total += 1

            accuracy = v.get("accuracy", 1.0)
            h_rate   = v.get("hallucination_rate", 0.0)
            chunks   = meta.get("total_chunks_retrieved", 0)

            latencies.append(latency_ms)
            accuracies.append(accuracy)
            hallucination_rates.append(h_rate)
            chunk_counts.append(chunks)

            if verbose:
                llm_note = " [FALLBACK]" if meta.get("llm_synthesis_failed") else ""
                retry_note = f" [retry×{meta.get('retry_attempts',1)}]" if meta.get("corrective_action_taken") else ""
                flag = "OK  " if h_rate <= 0.10 else "WARN"
                print(f"    [{qid:02d}] {flag}  {latency_ms:6d}ms  "
                      f"cit_acc={accuracy:.0%}  h_rate={h_rate:.0%}  chunks={chunks}"
                      f"{llm_note}{retry_note}  | {question[:45]}")

        except Exception as e:
            latency_ms = int((time.time() - t0) * 1000)
            errors.append(f"Q{qid}: {e}")
            latencies.append(latency_ms)
            accuracies.append(0.0)
            hallucination_rates.append(1.0)
            chunk_counts.append(0)
            if verbose:
                print(f"    [{qid:02d}] FAIL  {latency_ms:6d}ms  | {question[:50]}")
                print(f"          ERROR: {e}")

    n    = len(questions)
    n_ok = n - len(errors)

    return {
        "name":              name,
        "success_rate":      round(n_ok / n, 4) if n else 0.0,
        "avg_latency_ms":    round(sum(latencies) / len(latencies)) if latencies else 0,
        "avg_cit_accuracy":  round(sum(accuracies) / len(accuracies), 4) if accuracies else 0.0,
        "avg_hallucination": round(sum(hallucination_rates) / len(hallucination_rates), 4) if hallucination_rates else 0.0,
        "avg_chunks":        round(sum(chunk_counts) / len(chunk_counts), 1) if chunk_counts else 0.0,
        "errors":            errors,
        "validation_method": validation_method,
        "llm_failures":      llm_failures,
        "retry_total":       retry_total,
        "n_questions":       n,
    }


def print_methodology():
    """Explain the three distinct systems and how validation differs."""
    print("\n" + "=" * 70)
    print("  THREE-SYSTEM COMPARISON METHODOLOGY")
    print("=" * 70)
    print()
    print("  Standard RAG:")
    print("    - Single retrieve → single LLM call")
    print("    - No query decomposition")
    print("    - No built-in validation  (measured post-hoc by this script)")
    print()
    print("  A-RAG (Agentic RAG):")
    print("    - RLM query decomposition → sub-questions per jurisdiction")
    print("    - Parallel multi-retrieval")
    print("    - LLM synthesis from all evidence")
    print("    - No built-in validation  (measured post-hoc by this script)")
    print()
    print("  CAG / ReguGrounded:")
    print("    - RLM query decomposition into jurisdiction-specific sub-questions")
    print("    - Full regulatory corpus (~130k tokens) preloaded into LLM context")
    print("      → no Pinecone retrieval; all 395 chunks always present")
    print("      → Anthropic: corpus cached in system message (KV reuse across queries)")
    print("    - Built-in citation validation with corrective retry loop")
    print("      → on hallucination, re-synthesizes with invalid citations forbidden")
    print("    - Input / output guardrails")
    print()
    print("  All systems measured with the same validator for fair comparison.")
    print("  NOTE: [FALLBACK] means Gemini quota was exhausted — LLM call failed.")
    print("        Fallback answers contain no citations, so scores are trivially perfect.")
    print("        Metrics marked [FALLBACK] do not reflect real system behaviour.")
    print("=" * 70)


_VALIDATION_LABELS = {
    "post_hoc_for_comparison":      "Post-hoc (external)",
    "built_in_grounded_generation": "Built-in + retry",
    "none":                         "None (post-hoc by eval)",
    "unknown":                      "Unknown",
}


def print_comparison(results: list[dict]):
    """Print a formatted side-by-side comparison table."""

    def _bar(val, target, higher_is_better=True):
        ok = val >= target if higher_is_better else val <= target
        return "PASS" if ok else "FAIL"

    col     = 24
    divider = "─" * (col + 26 * len(results))

    print(f"\n\n{'═' * len(divider)}")
    print("  MODEL COMPARISON")
    print(f"{'═' * len(divider)}")

    # Header
    header = f"  {'Metric':<{col}}"
    for r in results:
        header += f"  {r['name']:>24}"
    print(header)
    print(f"  {divider}")

    def row(label, key, fmt, target, higher_is_better=True):
        line = f"  {label:<{col}}"
        for r in results:
            val    = r[key]
            v_str  = format(val, fmt)
            status = _bar(val, target, higher_is_better)
            line  += f"  {v_str:>12} [{status}]  "
        print(line)

    def text_row(label, values):
        line = f"  {label:<{col}}"
        for v in values:
            line += f"  {v:>24}"
        print(line)

    row("Success Rate",       "success_rate",      ".1%", 0.95)
    row("Avg Latency (ms)",   "avg_latency_ms",    "d",   10000, higher_is_better=False)
    row("Citation Accuracy",  "avg_cit_accuracy",  ".1%", 0.90)
    row("Hallucination Rate", "avg_hallucination", ".1%", 0.10,  higher_is_better=False)
    row("Avg Chunks",         "avg_chunks",        ".1f", 3.0)

    print(f"  {divider}")

    text_row("Validation",
             [_VALIDATION_LABELS.get(r.get("validation_method", "unknown"), "?") for r in results])

    # LLM fallback warning — only shown if any failures occurred
    any_fallbacks = any(r.get("llm_failures", 0) > 0 for r in results)
    if any_fallbacks:
        text_row("LLM Failures (fallback)",
                 [f"{r.get('llm_failures',0)}/{r.get('n_questions',0)}" for r in results])
        print(f"\n  *** WARNING: [FALLBACK] answers have no citations → trivially perfect")
        print(f"      scores.  Results reflect API fallback behaviour, not real LLM output.")

    # CAG-specific: corrective retries
    cag = next((r for r in results if "CAG" in r["name"]), None)
    if cag and cag.get("retry_total", 0) > 0:
        print(f"\n  CAG corrective retries triggered: {cag['retry_total']} / {cag['n_questions']} questions")

    print(f"\n  {divider}")

    for r in results:
        if r["errors"]:
            print(f"\n  {r['name']} errors:")
            for e in r["errors"]:
                print(f"    {e}")

    print(f"{'═' * len(divider)}\n")


def main():
    parser = argparse.ArgumentParser(description="Compare all three RAG models side by side")
    parser.add_argument("--mode",    choices=["quick", "full"], default="quick",
                        help="quick=10 curated questions (3 hard multi + 7 single-jurisdiction), full=all 50 (default: quick)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-question results for each model")
    args = parser.parse_args()

    with open(GROUND_TRUTH_PATH) as f:
        all_questions = json.load(f)

    if args.mode == "quick":
        # 10 questions: 3 hard cross-jurisdictional + 2 hard single-jurisdiction
        # + 1 medium from each of the 4 frameworks (EU, NYC, Colorado, NIST)
        # Chosen to stress-test retrieval depth, multi-hop reasoning, and citation grounding.
        TARGET_IDS = {
            16, 52, 53,   # exhaustive multi-article (EU lifecycle Q16, multi deployer Q52, multi transparency Q53)
            46, 49,       # hard multi-framework (EU+NYC hiring Q46, bias audit comparison Q49)
            4,  35,       # hard single-jurisdiction (EU deployer→provider Q4, CO deployer→developer Q35)
            2,            # medium EU (biometric ID exceptions — specific Article 5 details)
            28,           # medium Colorado (deployer obligations Q28)
            38,           # medium NIST (supply chain / MAP Q38)
        }
        id_map    = {q["id"]: q for q in all_questions}
        questions = [id_map[i] for i in sorted(TARGET_IDS) if i in id_map]
    else:
        questions = all_questions

    print(f"\nReguGrounded — Model Comparison")
    print(f"{'=' * 60}")
    print(f"Mode: {args.mode} | Questions: {len(questions)}")
    print(f"Models: {', '.join(name for name, _ in MODELS)}")

    results = []
    for name, query_fn in MODELS:
        result = run_model(name, query_fn, questions, verbose=args.verbose)
        results.append(result)

    print_methodology()
    print_comparison(results)


if __name__ == "__main__":
    main()
