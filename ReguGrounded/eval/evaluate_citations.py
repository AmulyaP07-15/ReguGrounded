"""
evaluate_citations.py — ReguGrounded Citation Accuracy Evaluation
==================================================================
Measures citation precision, recall, F1, and hallucination rate across all
ground-truth test questions.

Metrics:
    Citation Precision  - % of generated citations that exist in retrieved chunks
    Citation Recall     - % of expected citations that were actually included
    Citation F1         - Harmonic mean of precision and recall
    Hallucination Rate  - % of generated citations not in evidence

Usage:
    python eval/evaluate_citations.py
    python eval/evaluate_citations.py --mode quick
"""

import argparse
import json
import sys
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from query_interface import process_query  # noqa: E402

GROUND_TRUTH_PATH = Path(__file__).parent / "ground_truth.json"


def _normalize_citation(ref: str) -> str:
    """Strip whitespace and lower-case for loose matching."""
    return ref.strip().lower().replace("  ", " ")


def _citation_in_set(expected_ref: str, generated_articles: list[str]) -> bool:
    """Return True if expected_ref is contained in any of the generated article strings."""
    norm_expected = _normalize_citation(expected_ref)
    for gen in generated_articles:
        if norm_expected in _normalize_citation(gen):
            return True
    return False


def evaluate_citations(test_data: list[dict], verbose: bool = False) -> dict:
    """
    Run citation evaluation across all test questions.

    Args:
        test_data: List of ground-truth dicts (from ground_truth.json).
        verbose:   Print per-question details.

    Returns:
        {
            "precision":         float,
            "recall":            float,
            "f1":                float,
            "hallucination_rate":float,
            "total_questions":   int,
            "per_question":      [...]
        }
    """
    per_question = []

    for item in test_data:
        question = item["question"]
        expected = item.get("expected_citations", [])

        try:
            result = process_query(question)
        except Exception as e:
            per_question.append({
                "id":                item["id"],
                "question":          question,
                "precision":         0.0,
                "recall":            0.0,
                "f1":                0.0,
                "hallucination_rate":1.0,
                "hallucinated":      [],
                "error":             str(e),
            })
            continue

        validation = result["validation"]
        total_gen  = validation["total_citations"]
        valid_gen  = validation["valid_citations"]

        # Citation precision: % of generated citations backed by retrieved evidence
        precision = valid_gen / total_gen if total_gen > 0 else 1.0

        # Citation recall: % of expected citations present in generated answer
        generated_articles = [c["article"] for c in result["citations"]]
        if expected:
            hits = sum(1 for exp in expected if _citation_in_set(exp, generated_articles))
            recall = hits / len(expected)
        else:
            recall = 1.0  # no expected citations → nothing to miss

        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        entry = {
            "id":                item["id"],
            "question":          question,
            "jurisdiction":      item.get("jurisdiction", ""),
            "difficulty":        item.get("difficulty", ""),
            "precision":         round(precision, 4),
            "recall":            round(recall, 4),
            "f1":                round(f1, 4),
            "hallucination_rate":validation["hallucination_rate"],
            "hallucinated":      validation["details"]["invalid"],
            "expected":          expected,
            "generated_articles":generated_articles,
        }
        per_question.append(entry)

        if verbose:
            status = "OK" if precision >= 0.9 and recall >= 0.5 else "WARN"
            print(f"  [{item['id']:02d}] {status}  P={precision:.2f}  R={recall:.2f}  "
                  f"F1={f1:.2f}  hall={validation['hallucination_rate']:.2f}  "
                  f"| {question[:55]}")
            if validation["details"]["invalid"]:
                print(f"        Hallucinated: {validation['details']['invalid']}")

    if not per_question:
        return {
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "hallucination_rate": 1.0, "total_questions": 0, "per_question": [],
        }

    n = len(per_question)
    return {
        "precision":          round(sum(r["precision"]          for r in per_question) / n, 4),
        "recall":             round(sum(r["recall"]             for r in per_question) / n, 4),
        "f1":                 round(sum(r["f1"]                 for r in per_question) / n, 4),
        "hallucination_rate": round(sum(r["hallucination_rate"] for r in per_question) / n, 4),
        "total_questions":    n,
        "per_question":       per_question,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate citation accuracy")
    parser.add_argument("--mode",    choices=["quick", "full"], default="full")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(GROUND_TRUTH_PATH) as f:
        test_data = json.load(f)
    if args.mode == "quick":
        test_data = test_data[:5]

    print(f"\nCitation Evaluation — {len(test_data)} questions ({args.mode} mode)\n")
    results = evaluate_citations(test_data, verbose=args.verbose)

    print(f"  Precision:         {results['precision']:.1%}")
    print(f"  Recall:            {results['recall']:.1%}")
    print(f"  F1:                {results['f1']:.1%}")
    print(f"  Hallucination Rate:{results['hallucination_rate']:.1%}")


if __name__ == "__main__":
    main()
