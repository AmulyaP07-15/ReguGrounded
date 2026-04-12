"""
evaluate_e2e.py — ReguGrounded End-to-End Performance Evaluation
=================================================================
Runs the full pipeline on every test question and measures reliability,
latency percentiles, and throughput.

Metrics:
    Success Rate  - % of queries that complete without errors
    Latency p50/p90/p99 - Response time percentiles (ms)
    Avg Latency   - Mean response time (ms)
    Retrieval Coverage - Avg chunks retrieved per query

Usage:
    python eval/evaluate_e2e.py
    python eval/evaluate_e2e.py --mode quick --verbose
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from query_interface import process_query  # noqa: E402

GROUND_TRUTH_PATH = Path(__file__).parent / "ground_truth.json"


def _percentile(values: list[float], p: int) -> float:
    """Compute the p-th percentile of a sorted list."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = (p / 100) * (len(sorted_vals) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return round(sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo]), 1)


def evaluate_e2e(test_data: list[dict], verbose: bool = False) -> dict:
    """
    Run the full pipeline on all questions and measure performance.

    Args:
        test_data: List of ground-truth dicts (from ground_truth.json).
        verbose:   Print per-question timings.

    Returns:
        {
            "success_rate":       float,
            "total_questions":    int,
            "successful":         int,
            "failed":             int,
            "latency_p50":        float,
            "latency_p90":        float,
            "latency_p99":        float,
            "avg_latency":        float,
            "avg_chunks":         float,
            "errors":             [str],
            "per_question":       [...]
        }
    """
    per_question = []
    latencies: list[float] = []
    chunk_counts: list[int] = []
    errors: list[str] = []

    for item in test_data:
        question = item["question"]
        t0 = time.time()

        try:
            result = process_query(question)
            latency_ms = (time.time() - t0) * 1000

            latencies.append(latency_ms)
            chunks = result["metadata"]["total_chunks_retrieved"]
            chunk_counts.append(chunks)

            entry = {
                "id":           item["id"],
                "question":     question,
                "jurisdiction": item.get("jurisdiction", ""),
                "difficulty":   item.get("difficulty", ""),
                "success":      True,
                "latency_ms":   round(latency_ms, 1),
                "chunks":       chunks,
                "sub_questions":len(result["sub_questions"]),
                "citations":    len(result["citations"]),
                "error":        None,
            }
        except Exception as e:
            latency_ms = (time.time() - t0) * 1000
            err_msg = str(e)
            errors.append(f"Q{item['id']}: {err_msg}")
            entry = {
                "id":           item["id"],
                "question":     question,
                "jurisdiction": item.get("jurisdiction", ""),
                "difficulty":   item.get("difficulty", ""),
                "success":      False,
                "latency_ms":   round(latency_ms, 1),
                "chunks":       0,
                "sub_questions":0,
                "citations":    0,
                "error":        err_msg,
            }

        per_question.append(entry)

        if verbose:
            flag = "OK  " if entry["success"] else "FAIL"
            print(f"  [{item['id']:02d}] {flag}  {entry['latency_ms']:7.0f}ms  "
                  f"chunks={entry['chunks']}  sub_q={entry['sub_questions']}  "
                  f"| {question[:50]}")
            if not entry["success"]:
                print(f"        ERROR: {entry['error']}")

    successful = [r for r in per_question if r["success"]]
    failed     = [r for r in per_question if not r["success"]]
    n          = len(per_question)

    return {
        "success_rate":    round(len(successful) / n, 4) if n > 0 else 0.0,
        "total_questions": n,
        "successful":      len(successful),
        "failed":          len(failed),
        "latency_p50":     _percentile(latencies, 50),
        "latency_p90":     _percentile(latencies, 90),
        "latency_p99":     _percentile(latencies, 99),
        "avg_latency":     round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        "avg_chunks":      round(sum(chunk_counts) / len(chunk_counts), 1) if chunk_counts else 0.0,
        "errors":          errors,
        "per_question":    per_question,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate end-to-end pipeline performance")
    parser.add_argument("--mode",    choices=["quick", "full"], default="full")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(GROUND_TRUTH_PATH) as f:
        test_data = json.load(f)
    if args.mode == "quick":
        test_data = test_data[:5]

    print(f"\nEnd-to-End Evaluation — {len(test_data)} questions ({args.mode} mode)\n")
    results = evaluate_e2e(test_data, verbose=args.verbose)

    print(f"  Success Rate: {results['success_rate']:.1%}")
    print(f"  p50 Latency:  {results['latency_p50']:.0f} ms")
    print(f"  p90 Latency:  {results['latency_p90']:.0f} ms")
    print(f"  p99 Latency:  {results['latency_p99']:.0f} ms")
    print(f"  Avg Latency:  {results['avg_latency']:.0f} ms")
    print(f"  Avg Chunks:   {results['avg_chunks']:.1f}")
    if results["errors"]:
        print(f"\n  Errors ({len(results['errors'])}):")
        for e in results["errors"]:
            print(f"    {e}")


if __name__ == "__main__":
    main()
