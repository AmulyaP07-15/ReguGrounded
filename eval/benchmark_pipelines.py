"""
eval/benchmark_pipelines.py — Pipeline Comparison Benchmark
=============================================================
Runs three pipelines side-by-side on the same ground-truth questions
and produces a comprehensive comparison report.

Pipelines compared
------------------
  standard_rag   — single retrieve → synthesize (no decomposition, no iteration)
  rlm            — decompose → parallel retrieve → synthesize → validate
  agentic_rag    — decompose → [plan tool → retrieve → reflect → loop?] → synthesize → validate

Metrics
-------
  Hallucination rate     — % citations not found in retrieved chunks
  Citation accuracy      — % citations matched to retrieved chunks
  Success rate           — % queries that completed without error
  Avg latency            — mean response time in ms
  Latency p50 / p90      — response time percentiles
  Avg chunks / query     — mean chunks retrieved per query
  Avg iterations / juris — mean agent loop depth (agentic_rag only)

Breakdowns
----------
  By jurisdiction  — hallucination rate per regulation
  By difficulty    — hallucination rate per difficulty tier

Usage
-----
    python eval/benchmark_pipelines.py                      # full, 18 questions
    python eval/benchmark_pipelines.py --mode quick         # first 5 questions
    python eval/benchmark_pipelines.py --verbose            # per-question detail
    python eval/benchmark_pipelines.py --save               # write JSON report
    python eval/benchmark_pipelines.py --systems rlm agentic_rag   # subset
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

GROUND_TRUTH_PATH = Path(__file__).parent / "ground_truth.json"
RESULTS_DIR       = Path(__file__).parent / "results"

# Lazy imports — pipelines are only loaded when selected, so you don't pay the
# model-loading cost for systems you didn't ask for.
def _load_standard_rag():
    from eval.baselines.standard_rag import process_query
    return process_query

def _load_rlm():
    from query_interface import process_query
    return process_query

def _load_agentic_rag():
    from agentic_rag_pipeline import process_query
    return process_query

AVAILABLE_SYSTEMS = {
    "standard_rag": _load_standard_rag,
    "rlm":          _load_rlm,
    "agentic_rag":  _load_agentic_rag,
}

def _as_int(v) -> int:
    """Normalise citation counts — some pipelines return lists, others return ints."""
    return len(v) if isinstance(v, list) else int(v)


# ── Per-question runner ───────────────────────────────────────────────────────

def _run_one(name: str, fn: Callable, item: dict, verbose: bool) -> dict:
    q  = item["question"]
    t0 = time.time()
    try:
        result     = fn(q)
        latency_ms = (time.time() - t0) * 1000
        val        = result["validation"]
        meta       = result.get("metadata", {})

        # Agentic-specific: average iterations per jurisdiction
        agent_trace = meta.get("agent_trace", [])
        avg_iters   = (
            round(sum(t["iterations"] for t in agent_trace) / len(agent_trace), 2)
            if agent_trace else None
        )
        tools_used = list({t for trace in agent_trace for t in trace.get("tools_used", [])})

        entry = {
            "id":                item["id"],
            "question":          q,
            "jurisdiction":      item.get("jurisdiction", "unknown"),
            "difficulty":        item.get("difficulty", "unknown"),
            "success":           True,
            "hallucination_rate":val["hallucination_rate"],
            "citation_accuracy": val["accuracy"],
            "total_citations":   val["total_citations"],
            "valid_citations":   _as_int(val.get("valid_citations", val.get("valid_count", 0))),
            "invalid_citations": _as_int(val.get("invalid_citations", val.get("invalid_count", 0))),
            "hallucinated":      val.get("details", {}).get("invalid", []),
            "chunks_retrieved":  meta.get("total_chunks_retrieved", 0),
            "latency_ms":        round(latency_ms, 1),
            "avg_iterations":    avg_iters,
            "tools_used":        tools_used,
            "error":             None,
        }
    except Exception as e:
        latency_ms = (time.time() - t0) * 1000
        entry = {
            "id":                item["id"],
            "question":          q,
            "jurisdiction":      item.get("jurisdiction", "unknown"),
            "difficulty":        item.get("difficulty", "unknown"),
            "success":           False,
            "hallucination_rate":1.0,
            "citation_accuracy": 0.0,
            "total_citations":   0,
            "valid_citations":   0,
            "invalid_citations": 0,
            "hallucinated":      [],
            "chunks_retrieved":  0,
            "latency_ms":        round(latency_ms, 1),
            "avg_iterations":    None,
            "tools_used":        [],
            "error":             str(e),
        }

    if verbose:
        hr   = entry["hallucination_rate"]
        flag = "HALL" if hr > 0 else "  OK"
        err  = f"  ERROR: {entry['error']}" if entry["error"] else ""
        iter_str = f"  iters={entry['avg_iterations']:.1f}" if entry["avg_iterations"] else ""
        print(
            f"    [{item['id']:02d}] {flag}  hall={hr:.0%}  "
            f"acc={entry['citation_accuracy']:.0%}  "
            f"chunks={entry['chunks_retrieved']:2d}{iter_str}  "
            f"{entry['latency_ms']:.0f}ms  | {q[:40]}"
            + err
        )
        if entry["hallucinated"]:
            print(f"           Hallucinated: {entry['hallucinated']}")

    return entry


# ── System runner ─────────────────────────────────────────────────────────────

def _run_system(name: str, fn: Callable, questions: List[dict], verbose: bool) -> dict:
    """Run one pipeline on all questions, return aggregated + per-question results."""
    per_question = []
    print(f"\n  [{name}]  running {len(questions)} question(s)...")

    for item in questions:
        entry = _run_one(name, fn, item, verbose)
        per_question.append(entry)

    n         = len(per_question)
    success_n = sum(1 for r in per_question if r["success"])
    cited     = [r for r in per_question if r["total_citations"] > 0]

    latencies     = [r["latency_ms"]        for r in per_question]
    hall_rates    = [r["hallucination_rate"] for r in per_question]
    accuracies    = [r["citation_accuracy"]  for r in per_question]
    chunk_counts  = [r["chunks_retrieved"]   for r in per_question]
    iter_vals     = [r["avg_iterations"] for r in per_question if r["avg_iterations"] is not None]

    def _pct(vals, p):
        if not vals:
            return 0.0
        sv  = sorted(vals)
        idx = (p / 100) * (len(sv) - 1)
        lo  = int(idx)
        hi  = min(lo + 1, len(sv) - 1)
        return round(sv[lo] + (idx - lo) * (sv[hi] - sv[lo]), 1)

    # Breakdowns
    by_jurisdiction: Dict[str, list] = {}
    by_difficulty:   Dict[str, list] = {}
    for r in per_question:
        by_jurisdiction.setdefault(r["jurisdiction"], []).append(r["hallucination_rate"])
        by_difficulty.setdefault(r["difficulty"], []).append(r["hallucination_rate"])

    return {
        "system":               name,
        "total_questions":      n,
        "success_rate":         round(success_n / n, 4) if n else 0.0,
        "avg_hallucination_rate": round(sum(hall_rates) / n, 4) if n else 1.0,
        "hall_rate_cited_only": round(
            sum(r["hallucination_rate"] for r in cited) / len(cited), 4
        ) if cited else 0.0,
        "avg_citation_accuracy": round(sum(accuracies) / n, 4) if n else 0.0,
        "total_citations":      sum(r["total_citations"]   for r in per_question),
        "valid_citations":      sum(r["valid_citations"]   for r in per_question),
        "invalid_citations":    sum(r["invalid_citations"] for r in per_question),
        "avg_chunks":           round(sum(chunk_counts) / n, 1) if n else 0.0,
        "avg_iterations":       round(sum(iter_vals) / len(iter_vals), 2) if iter_vals else None,
        "avg_latency_ms":       round(sum(latencies) / n, 1) if n else 0.0,
        "latency_p50":          _pct(latencies, 50),
        "latency_p90":          _pct(latencies, 90),
        "latency_p99":          _pct(latencies, 99),
        "by_jurisdiction":      {j: round(sum(v)/len(v), 4) for j, v in by_jurisdiction.items()},
        "by_difficulty":        {d: round(sum(v)/len(v), 4) for d, v in by_difficulty.items()},
        "per_question":         per_question,
    }


# ── Report printer ────────────────────────────────────────────────────────────

def _print_report(results: Dict[str, dict]) -> None:
    systems = list(results.keys())
    W       = 20   # column width

    def divider(c="─", n=70): print(c * n)
    def header_row(label, *vals):
        print(f"  {label:<30}" + "".join(f"{v:>{W}}" for v in vals))
    def pct_row(label, key):
        header_row(label, *[f"{results[s][key]:.1%}" for s in systems])
    def ms_row(label, key):
        header_row(label, *[f"{results[s][key]:.0f} ms" for s in systems])
    def float_row(label, key, fmt=".2f"):
        vals = []
        for s in systems:
            v = results[s].get(key)
            vals.append(f"{v:{fmt}}" if v is not None else "  N/A")
        header_row(label, *vals)
    def int_row(label, key):
        header_row(label, *[str(results[s][key]) for s in systems])

    print("\n")
    divider("═", 70)
    print("  PIPELINE COMPARISON BENCHMARK")
    divider("═", 70)

    # Column headers
    print(f"\n  {'Metric':<30}" + "".join(f"{s:>{W}}" for s in systems))
    divider()

    # Core metrics
    print("  Quality")
    pct_row("  Hallucination rate (all)",   "avg_hallucination_rate")
    pct_row("  Hallucination rate (cited)",  "hall_rate_cited_only")
    pct_row("  Citation accuracy",           "avg_citation_accuracy")
    pct_row("  Success rate",                "success_rate")

    print()
    print("  Citations")
    int_row("  Total generated",             "total_citations")
    int_row("    Grounded",                  "valid_citations")
    int_row("    Hallucinated",              "invalid_citations")

    print()
    print("  Retrieval")
    float_row("  Avg chunks / query",        "avg_chunks", ".1f")
    float_row("  Avg iterations / juris",    "avg_iterations", ".2f")

    print()
    print("  Latency")
    ms_row("  p50",                          "latency_p50")
    ms_row("  p90",                          "latency_p90")
    ms_row("  p99",                          "latency_p99")
    ms_row("  Mean",                         "avg_latency_ms")

    # Per-jurisdiction
    all_j = sorted({j for s in systems for j in results[s]["by_jurisdiction"]})
    if all_j:
        print(f"\n  Hallucination Rate by Jurisdiction")
        divider()
        for j in all_j:
            vals = [f"{results[s]['by_jurisdiction'].get(j, 0):.1%}" for s in systems]
            header_row(f"  {j}", *vals)

    # Per-difficulty
    all_d = sorted({d for s in systems for d in results[s]["by_difficulty"]})
    if all_d:
        print(f"\n  Hallucination Rate by Difficulty")
        divider()
        for d in all_d:
            vals = [f"{results[s]['by_difficulty'].get(d, 0):.1%}" for s in systems]
            header_row(f"  {d}", *vals)

    # Delta summary vs standard_rag baseline
    if "standard_rag" in results:
        base_hall = results["standard_rag"]["avg_hallucination_rate"]
        print(f"\n  Improvement over standard_rag (hallucination rate)")
        divider()
        for s in systems:
            if s == "standard_rag":
                continue
            delta  = base_hall - results[s]["avg_hallucination_rate"]
            sign   = "↓" if delta > 0 else "↑"
            colour = "better" if delta > 0 else "worse"
            print(f"    {s:<28}  {sign} {abs(delta):.1%}  ({colour})")

    divider("═", 70)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare standard_rag, RLM, and AgenticRAG pipelines side-by-side"
    )
    parser.add_argument(
        "--mode", choices=["quick", "full"], default="full",
        help="quick = first 5 questions, full = all 18 (default: full)"
    )
    parser.add_argument(
        "--systems", nargs="+",
        choices=list(AVAILABLE_SYSTEMS.keys()),
        default=list(AVAILABLE_SYSTEMS.keys()),
        help="Which pipelines to run (default: all three)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-question breakdown for each system"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save full results JSON to eval/results/"
    )
    args = parser.parse_args()

    with open(GROUND_TRUTH_PATH) as f:
        all_questions = json.load(f)
    questions = all_questions[:5] if args.mode == "quick" else all_questions

    print(f"\nPipeline Benchmark — {len(questions)} questions ({args.mode} mode)")
    print(f"Systems: {', '.join(args.systems)}")
    print(f"\nNote: first run downloads models (~30s). Subsequent runs are faster.\n")

    results: Dict[str, dict] = {}
    for name in args.systems:
        print(f"Loading {name}...")
        fn = AVAILABLE_SYSTEMS[name]()
        results[name] = _run_system(name, fn, questions, verbose=args.verbose)

    _print_report(results)

    if args.save:
        RESULTS_DIR.mkdir(exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = RESULTS_DIR / f"pipeline_benchmark_{args.mode}_{ts}.json"
        # Remove per_question details for compact save (keep aggregates)
        save_data = {
            s: {k: v for k, v in r.items() if k != "per_question"}
            for s, r in results.items()
        }
        with open(path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\n  Saved → {path}")


if __name__ == "__main__":
    main()
