"""
benchmark_hallucination.py — Hallucination Rate Benchmark
==========================================================
Runs Standard RAG and ReguGrounded (Agentic RAG) side-by-side on the same
ground-truth questions and compares citation hallucination rates.

Hallucination rate is defined as:
    % of citations in the generated answer that are NOT present in the
    retrieved evidence chunks (measured by citation_validator.py).

Metrics reported per system and per question:
    - Hallucination Rate   : fraction of generated citations that are fabricated
    - Citation Precision   : fraction of generated citations that are grounded
    - Total / Valid / Invalid citation counts

Usage:
    python eval/benchmark_hallucination.py
    python eval/benchmark_hallucination.py --mode quick
    python eval/benchmark_hallucination.py --mode full --verbose --save
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.baselines.standard_rag import process_query as standard_rag_query  # noqa: E402
from eval.baselines.agentic_rag  import process_query as agentic_rag_query   # noqa: E402

GROUND_TRUTH_PATH = Path(__file__).parent / "ground_truth.json"
RESULTS_DIR       = Path(__file__).parent / "results"

SYSTEMS = {
    "standard_rag":  standard_rag_query,
    "agentic_rag":   agentic_rag_query,
}


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------

def _run_system(name: str, fn, questions: list[dict], verbose: bool) -> dict:
    """Run one system on all questions. Returns aggregated + per-question results."""
    per_question = []

    print(f"\n  Running {name} on {len(questions)} questions...")

    for item in questions:
        q = item["question"]
        t0 = time.time()
        try:
            result     = fn(q)
            latency_ms = (time.time() - t0) * 1000
            val        = result["validation"]

            entry = {
                "id":                item["id"],
                "question":          q,
                "jurisdiction":      item.get("jurisdiction", ""),
                "difficulty":        item.get("difficulty", ""),
                "hallucination_rate":val["hallucination_rate"],
                "precision":         val["accuracy"],
                "total_citations":   val["total_citations"],
                "valid_citations":   val["valid_count"],
                "invalid_citations": val["invalid_count"],
                "hallucinated":      val["details"]["invalid"],
                "latency_ms":        round(latency_ms, 1),
                "error":             None,
            }
        except Exception as e:
            latency_ms = (time.time() - t0) * 1000
            entry = {
                "id":                item["id"],
                "question":          q,
                "jurisdiction":      item.get("jurisdiction", ""),
                "difficulty":        item.get("difficulty", ""),
                "hallucination_rate":1.0,
                "precision":         0.0,
                "total_citations":   0,
                "valid_citations":   0,
                "invalid_citations": 0,
                "hallucinated":      [],
                "latency_ms":        round(latency_ms, 1),
                "error":             str(e),
            }

        per_question.append(entry)

        if verbose:
            hr  = entry["hallucination_rate"]
            flag = "HALL" if hr > 0 else "  OK"
            err  = f"  ERROR: {entry['error']}" if entry["error"] else ""
            print(
                f"    [{item['id']:02d}] {flag}  hall={hr:.0%}  "
                f"cites={entry['total_citations']} (valid={entry['valid_citations']} "
                f"invalid={entry['invalid_citations']})  "
                f"{entry['latency_ms']:.0f}ms  | {q[:45]}"
                + err
            )
            if entry["hallucinated"]:
                print(f"           Hallucinated: {entry['hallucinated']}")

    n = len(per_question)
    questions_with_citations = [r for r in per_question if r["total_citations"] > 0]
    nc = len(questions_with_citations)

    # Aggregate metrics
    avg_hall = (
        sum(r["hallucination_rate"] for r in per_question) / n
        if n > 0 else 1.0
    )
    avg_precision = (
        sum(r["precision"] for r in per_question) / n
        if n > 0 else 0.0
    )
    total_cites   = sum(r["total_citations"]   for r in per_question)
    valid_cites   = sum(r["valid_citations"]   for r in per_question)
    invalid_cites = sum(r["invalid_citations"] for r in per_question)

    # Hallucination rate only over questions that generated at least one citation
    hall_over_cited = (
        sum(r["hallucination_rate"] for r in questions_with_citations) / nc
        if nc > 0 else 0.0
    )

    # Breakdown by difficulty
    by_difficulty: dict[str, list] = {}
    for r in per_question:
        d = r.get("difficulty", "unknown")
        by_difficulty.setdefault(d, []).append(r["hallucination_rate"])
    diff_summary = {
        d: round(sum(v) / len(v), 4)
        for d, v in by_difficulty.items()
    }

    # Breakdown by jurisdiction
    by_jurisdiction: dict[str, list] = {}
    for r in per_question:
        j = r.get("jurisdiction", "unknown")
        by_jurisdiction.setdefault(j, []).append(r["hallucination_rate"])
    juris_summary = {
        j: round(sum(v) / len(v), 4)
        for j, v in by_jurisdiction.items()
    }

    return {
        "system":                    name,
        "total_questions":           n,
        "avg_hallucination_rate":    round(avg_hall, 4),
        "hall_rate_cited_only":      round(hall_over_cited, 4),
        "avg_citation_precision":    round(avg_precision, 4),
        "total_citations_generated": total_cites,
        "valid_citations":           valid_cites,
        "invalid_citations":         invalid_cites,
        "by_difficulty":             diff_summary,
        "by_jurisdiction":           juris_summary,
        "per_question":              per_question,
    }


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def _print_report(results: dict[str, dict]) -> None:
    systems = list(results.keys())

    print("\n" + "=" * 65)
    print("  HALLUCINATION BENCHMARK RESULTS")
    print("=" * 65)

    # Summary table
    col_w = 22
    header = f"{'Metric':<30}" + "".join(f"{s:>{col_w}}" for s in systems)
    print(f"\n{header}")
    print("-" * (30 + col_w * len(systems)))

    def row(label, key, fmt=".1%"):
        vals = "".join(f"{results[s][key]:>{col_w}{fmt}}" for s in systems)
        print(f"  {label:<28}{vals}")

    row("Hallucination Rate (all)",   "avg_hallucination_rate")
    row("Hallucination Rate (cited)", "hall_rate_cited_only")
    row("Citation Precision",         "avg_citation_precision")

    def row_int(label, key):
        vals = "".join(f"{results[s][key]:>{col_w}d}" for s in systems)
        print(f"  {label:<28}{vals}")

    row_int("Total Citations Generated", "total_citations_generated")
    row_int("  Valid",                   "valid_citations")
    row_int("  Hallucinated",            "invalid_citations")

    # Per-difficulty
    all_diffs = sorted({d for s in systems for d in results[s]["by_difficulty"]})
    if all_diffs:
        print(f"\n  By Difficulty:")
        for d in all_diffs:
            vals = "".join(
                f"{results[s]['by_difficulty'].get(d, 0):>{col_w}.1%}"
                for s in systems
            )
            print(f"    {d:<26}{vals}")

    # Per-jurisdiction
    all_juris = sorted({j for s in systems for j in results[s]["by_jurisdiction"]})
    if all_juris:
        print(f"\n  By Jurisdiction:")
        for j in all_juris:
            vals = "".join(
                f"{results[s]['by_jurisdiction'].get(j, 0):>{col_w}.1%}"
                for s in systems
            )
            print(f"    {j:<26}{vals}")

    # Delta summary
    if len(systems) == 2:
        s1, s2 = systems[0], systems[1]
        delta = (results[s1]["avg_hallucination_rate"]
                 - results[s2]["avg_hallucination_rate"])
        winner = s2 if delta > 0 else s1
        print(f"\n  {winner} hallucination rate is "
              f"{abs(delta):.1%} lower than "
              f"{s1 if winner == s2 else s2}.")

    print("=" * 65)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Hallucination rate benchmark: Standard RAG vs ReguGrounded"
    )
    parser.add_argument("--mode",    choices=["quick", "full"], default="full",
                        help="quick = first 5 questions only")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-question breakdown")
    parser.add_argument("--save",    action="store_true",
                        help="Save results JSON to eval/results/")
    args = parser.parse_args()

    with open(GROUND_TRUTH_PATH) as f:
        test_data = json.load(f)
    if args.mode == "quick":
        test_data = test_data[:5]

    print(f"\nHallucination Benchmark — {len(test_data)} questions ({args.mode} mode)")

    results: dict[str, dict] = {}
    for name, fn in SYSTEMS.items():
        results[name] = _run_system(name, fn, test_data, verbose=args.verbose)

    _print_report(results)

    if args.save:
        RESULTS_DIR.mkdir(exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = RESULTS_DIR / f"hallucination_benchmark_{args.mode}_{ts}.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved → {path}")


if __name__ == "__main__":
    main()
