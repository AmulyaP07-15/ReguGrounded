#!/usr/bin/env python3
"""
run_all_evals.py — ReguGrounded Master Evaluation Script
=========================================================
Runs all four evaluation suites in sequence and saves a timestamped JSON report.

Usage:
    python eval/run_all_evals.py
    python eval/run_all_evals.py --mode quick
    python eval/run_all_evals.py --mode full --verbose
    python eval/run_all_evals.py --mode full --skip retrieval
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Make project root and eval/ importable
_ROOT = Path(__file__).parent.parent
_EVAL = Path(__file__).parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_EVAL))

from evaluate_citations import evaluate_citations          # noqa: E402
from evaluate_answers   import evaluate_answers            # noqa: E402
from evaluate_e2e       import evaluate_e2e                # noqa: E402

GROUND_TRUTH_PATH = _EVAL / "ground_truth.json"
RESULTS_DIR       = _EVAL / "results"


def _load_test_data(mode: str) -> list[dict]:
    with open(GROUND_TRUTH_PATH) as f:
        data = json.load(f)
    return data[:5] if mode == "quick" else data


def _run_retrieval_eval(verbose: bool) -> dict:
    """Run the existing retrieval eval and normalise its output."""
    sys.path.insert(0, str(_ROOT / "src"))
    from evaluate_retrieval import run_eval  # lives in eval/
    avg_precision, avg_recall, avg_f1 = run_eval(top_k=5, verbose=verbose)
    return {
        "precision_5": round(avg_precision, 4),
        "recall_5":    round(avg_recall, 4),
        "f1_5":        round(avg_f1, 4),
    }


def run_all_evaluations(mode: str = "full", verbose: bool = False, skip: list = None):
    skip = skip or []

    print("=" * 70)
    print("  ReguGrounded — Full Evaluation Suite")
    print("=" * 70)

    print()
    print("  CITATION VALIDATION METHODOLOGY")
    print("  " + "-" * 66)
    print("  All systems measured using the same citation validator for fair comparison.")
    print("  - Standard RAG:   Post-hoc validation (not part of baseline system)")
    print("  - A-RAG:          Post-hoc validation (not part of baseline system)")
    print("  - ReguGrounded:   Built-in validation (core system component)")
    print("  " + "-" * 66)

    test_data = _load_test_data(mode)
    print(f"\n  Mode: {mode} | Questions: {len(test_data)}\n")

    report = {
        "timestamp":    datetime.now().isoformat(),
        "mode":         mode,
        "num_questions":len(test_data),
    }

    # ── 1. Retrieval ─────────────────────────────────────────────────────────
    if "retrieval" not in skip:
        print("[1/4] Retrieval Quality")
        print("-" * 40)
        retrieval = _run_retrieval_eval(verbose=verbose)
        report["retrieval"] = retrieval
        print(f"  Recall@5:    {retrieval['recall_5']:.1%}")
        print(f"  Precision@5: {retrieval['precision_5']:.1%}")
        print(f"  F1@5:        {retrieval['f1_5']:.1%}")
    else:
        print("[1/4] Retrieval Quality  (skipped)")
        report["retrieval"] = None

    # ── 2. Citation Accuracy ─────────────────────────────────────────────────
    if "citations" not in skip:
        print("\n[2/4] Citation Accuracy")
        print("-" * 40)
        citations = evaluate_citations(test_data, verbose=verbose)
        report["citations"] = citations
        print(f"  Precision:         {citations['precision']:.1%}")
        print(f"  Recall:            {citations['recall']:.1%}")
        print(f"  F1:                {citations['f1']:.1%}")
        print(f"  Hallucination Rate:{citations['hallucination_rate']:.1%}")
    else:
        print("\n[2/4] Citation Accuracy  (skipped)")
        report["citations"] = None

    # ── 3. Answer Correctness ────────────────────────────────────────────────
    if "answers" not in skip:
        print("\n[3/4] Answer Correctness (LLM judge)")
        print("-" * 40)
        answers = evaluate_answers(test_data, verbose=verbose)
        report["answers"] = answers
        print(f"  Avg Relevance:    {answers['avg_relevance']:.2f}/5.0")
        print(f"  Avg Accuracy:     {answers['avg_accuracy']:.2f}/5.0")
        print(f"  Avg Completeness: {answers['avg_completeness']:.2f}/5.0")
        print(f"  Avg Overall:      {answers['avg_overall']:.2f}/5.0")
    else:
        print("\n[3/4] Answer Correctness  (skipped)")
        report["answers"] = None

    # ── 4. End-to-End Performance ────────────────────────────────────────────
    if "e2e" not in skip:
        print("\n[4/4] End-to-End Performance")
        print("-" * 40)
        e2e = evaluate_e2e(test_data, verbose=verbose)
        report["e2e"] = e2e
        print(f"  Success Rate: {e2e['success_rate']:.1%}")
        print(f"  p50 Latency:  {e2e['latency_p50']:.0f} ms")
        print(f"  p90 Latency:  {e2e['latency_p90']:.0f} ms")
        print(f"  p99 Latency:  {e2e['latency_p99']:.0f} ms")
        if e2e["errors"]:
            print(f"  Errors: {len(e2e['errors'])}")
    else:
        print("\n[4/4] End-to-End Performance  (skipped)")
        report["e2e"] = None

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  SUMMARY vs Targets")
    print(f"{'=' * 70}")

    def _fmt(val, target, fmt=".1%", pass_fn=lambda v, t: v >= t):
        if val is None:
            return "skipped"
        v_str = format(val, fmt)
        t_str = format(target, fmt)
        status = "PASS" if pass_fn(val, target) else "FAIL"
        return f"{v_str}  (target: {t_str})  [{status}]"

    if report.get("retrieval"):
        r = report["retrieval"]
        print(f"  Retrieval Recall@5:      {_fmt(r['recall_5'],    0.90)}")
        print(f"  Retrieval Precision@5:   {_fmt(r['precision_5'], 0.80)}")

    if report.get("citations"):
        c = report["citations"]
        print(f"  Citation Precision:      {_fmt(c['precision'],          0.90)}")
        print(f"  Citation Recall:         {_fmt(c['recall'],             0.50)}")
        print(f"  Hallucination Rate:      {_fmt(c['hallucination_rate'], 0.10, pass_fn=lambda v,t: v<=t)}")

    if report.get("answers"):
        a = report["answers"]
        print(f"  Answer Quality (overall):{_fmt(a['avg_overall'], 4.0, fmt='.2f', pass_fn=lambda v,t: v>=t)}")

    if report.get("e2e"):
        e = report["e2e"]
        print(f"  Success Rate:            {_fmt(e['success_rate'],  0.95)}")
        print(f"  p90 Latency:             {_fmt(e['latency_p90'], 10000, fmt='.0f', pass_fn=lambda v,t: v<=t)} ms")

    # ── Save report ───────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"eval_{mode}_{ts}.json"
    # Strip per_question arrays from saved report to keep it small unless full mode
    if mode == "quick":
        report_to_save = report
    else:
        report_to_save = json.loads(json.dumps(report))  # deep copy
        for key in ("citations", "answers", "e2e"):
            if report_to_save.get(key) and "per_question" in report_to_save[key]:
                del report_to_save[key]["per_question"]

    with open(output_path, "w") as f:
        json.dump(report_to_save, f, indent=2)

    print(f"\n  Report saved → {output_path.relative_to(_ROOT)}")
    print("=" * 70)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all ReguGrounded evaluations")
    parser.add_argument("--mode",    choices=["quick", "full"], default="full",
                        help="quick=5 questions, full=18 questions")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-question details for each eval")
    parser.add_argument("--skip",    nargs="*",
                        choices=["retrieval", "citations", "answers", "e2e"],
                        default=[],
                        help="Skip one or more evaluation stages")
    args = parser.parse_args()
    run_all_evaluations(mode=args.mode, verbose=args.verbose, skip=args.skip)
