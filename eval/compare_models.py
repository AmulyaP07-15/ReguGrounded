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


def run_model(name: str, query_fn, questions: list, verbose: bool) -> dict:
    """Run one model over all questions and return aggregate metrics."""
    latencies, accuracies, hallucination_rates, chunk_counts = [], [], [], []
    errors = []
    validation_method = "unknown"

    print(f"\n  Running {name}...")

    for item in questions:
        question = item["question"]
        qid = item.get("id", "?")
        t0 = time.time()

        try:
            result = query_fn(question)
            latency_ms = int((time.time() - t0) * 1000)

            meta = result.get("metadata", {})
            v = result.get("validation", {})
            accuracy   = v.get("accuracy",         v.get("hallucination_rate", None))
            h_rate     = v.get("hallucination_rate", None)
            chunks     = meta.get("total_chunks_retrieved", 0)

            # Capture validation_method from first successful result
            if validation_method == "unknown" and "validation_method" in meta:
                validation_method = meta["validation_method"]

            # standard_rag stores accuracy differently
            if accuracy is None:
                valid   = v.get("valid_count",   0)
                total   = v.get("total_citations", 0)
                accuracy = (valid / total) if total else 1.0
            if h_rate is None:
                invalid = v.get("invalid_count", 0)
                total   = v.get("total_citations", 0)
                h_rate  = (invalid / total) if total else 0.0

            latencies.append(latency_ms)
            accuracies.append(accuracy)
            hallucination_rates.append(h_rate)
            chunk_counts.append(chunks)

            if verbose:
                flag = "OK  " if h_rate <= 0.10 else "WARN"
                print(f"    [{qid:02d}] {flag}  {latency_ms:6d}ms  "
                      f"cit_acc={accuracy:.0%}  chunks={chunks}  "
                      f"| {question[:50]}")

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

    n = len(questions)
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
    }


def print_validation_methodology():
    """Print explanation of citation validation methodology."""
    print("\n" + "=" * 70)
    print("  CITATION VALIDATION METHODOLOGY")
    print("=" * 70)
    print("  All systems prompt the LLM to generate citations in answers.")
    print("  All systems measured using the same citation validator for comparison.")
    print()
    print("  - Standard RAG:  Post-hoc validation (not part of baseline system)")
    print("  - A-RAG:         Post-hoc validation (not part of baseline system)")
    print("  - CAG/ReguGrounded: Built-in validation (core system component)")
    print()
    print("  Post-hoc validation measures actual hallucination rates in baselines.")
    print("  CAG's built-in validation actively prevents hallucinations from reaching users.")
    print("=" * 70)


_VALIDATION_METHOD_LABELS = {
    "post_hoc_for_comparison":       "Post-hoc (external)",
    "built_in_grounded_generation":  "Built-in (system feature)",
    "unknown":                       "Unknown",
}


def print_comparison(results: list[dict]):
    """Print a formatted side-by-side comparison table."""

    def _bar(val, target, higher_is_better=True):
        ok = val >= target if higher_is_better else val <= target
        return "PASS" if ok else "FAIL"

    col = 22
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
            val = r[key]
            v_str = format(val, fmt)
            status = _bar(val, target, higher_is_better)
            line += f"  {v_str:>12} [{status}]  "
        print(line)

    def text_row(label, values):
        """Row for non-numeric values (no PASS/FAIL bracket)."""
        line = f"  {label:<{col}}"
        for v in values:
            line += f"  {v:>24}"
        print(line)

    row("Success Rate",       "success_rate",      ".1%",  0.95)
    row("Avg Latency (ms)",   "avg_latency_ms",    "d",    10000, higher_is_better=False)
    row("Citation Accuracy",  "avg_cit_accuracy",  ".1%",  0.90)
    row("Hallucination Rate", "avg_hallucination", ".1%",  0.10, higher_is_better=False)
    row("Avg Chunks",         "avg_chunks",        ".1f",  3.0)

    print(f"  {divider}")

    validation_labels = [
        _VALIDATION_METHOD_LABELS.get(r.get("validation_method", "unknown"), "Unknown")
        for r in results
    ]
    text_row("Validation Method", validation_labels)

    print(f"  {divider}")

    # Errors
    for r in results:
        if r["errors"]:
            print(f"\n  {r['name']} errors:")
            for e in r["errors"]:
                print(f"    {e}")

    print(f"{'═' * len(divider)}\n")


def main():
    parser = argparse.ArgumentParser(description="Compare all three RAG models side by side")
    parser.add_argument("--mode",    choices=["quick", "full"], default="quick",
                        help="quick=3 questions, full=all 18 (default: quick)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-question results for each model")
    args = parser.parse_args()

    with open(GROUND_TRUTH_PATH) as f:
        all_questions = json.load(f)
    questions = all_questions[:3] if args.mode == "quick" else all_questions

    print(f"\nReguGrounded — Model Comparison")
    print(f"{'=' * 60}")
    print(f"Mode: {args.mode} | Questions: {len(questions)}")
    print(f"Models: {', '.join(name for name, _ in MODELS)}")

    results = []
    for name, query_fn in MODELS:
        result = run_model(name, query_fn, questions, verbose=args.verbose)
        results.append(result)

    print_validation_methodology()
    print_comparison(results)


if __name__ == "__main__":
    main()
