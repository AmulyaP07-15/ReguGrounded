"""
evaluate_answers.py — ReguGrounded Answer Correctness Evaluation (LLM-as-judge)
================================================================================
Uses GPT-4o-mini to score each generated answer against a reference answer on
three criteria: relevance, accuracy, and completeness.

Metrics (all 1-5 scale):
    Relevance     - Does the answer address the question asked?
    Accuracy      - Are the regulatory facts correct vs the reference?
    Completeness  - Does it cover the key points from the reference?
    Overall       - Mean of the three scores

Usage:
    python eval/evaluate_answers.py
    python eval/evaluate_answers.py --mode quick --verbose
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent / ".env")

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from query_interface import process_query  # noqa: E402

GROUND_TRUTH_PATH = Path(__file__).parent / "ground_truth.json"

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    return _client


_JUDGE_PROMPT = """\
You are an expert evaluator of regulatory compliance answers.

QUESTION:
{question}

REFERENCE ANSWER (ground truth):
{reference}

GENERATED ANSWER (to evaluate):
{generated}

Rate the generated answer on a scale of 1-5 for each criterion:

1. Relevance: Does it directly address the question?
   1 = Off-topic or missing the point entirely
   3 = Partially addresses the question
   5 = Directly and fully answers what was asked

2. Accuracy: Are the regulatory facts and citations correct compared to the reference?
   1 = Contains significant factual errors or unsupported claims
   3 = Mostly correct with minor errors or omissions
   5 = Factually accurate and consistent with the reference

3. Completeness: Does it cover the key points mentioned in the reference?
   1 = Misses most key points
   3 = Covers some key points but omits important details
   5 = Covers all or nearly all key points from the reference

Return ONLY valid JSON — no extra text, no markdown:
{{"relevance": <1-5>, "accuracy": <1-5>, "completeness": <1-5>, "explanation": "<one sentence>"}}"""


def llm_judge(question: str, generated: str, reference: str) -> dict:
    """
    Use an LLM to score one generated answer against the reference.

    Returns:
        {"relevance": int, "accuracy": int, "completeness": int,
         "overall": float, "explanation": str}
    """
    prompt = _JUDGE_PROMPT.format(
        question=question, reference=reference, generated=generated
    )
    try:
        response = _get_client().chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        scores = json.loads(raw)
        scores["overall"] = round(
            (scores["relevance"] + scores["accuracy"] + scores["completeness"]) / 3, 3
        )
        return scores
    except Exception as e:
        return {
            "relevance": 0, "accuracy": 0, "completeness": 0,
            "overall": 0.0, "explanation": f"Judge error: {e}",
        }


def evaluate_answers(test_data: list[dict], verbose: bool = False) -> dict:
    """
    Evaluate answer quality across all test questions.

    Args:
        test_data: List of ground-truth dicts (from ground_truth.json).
        verbose:   Print per-question scores.

    Returns:
        {
            "avg_relevance":    float,
            "avg_accuracy":     float,
            "avg_completeness": float,
            "avg_overall":      float,
            "total_questions":  int,
            "per_question":     [...]
        }
    """
    per_question = []

    for item in test_data:
        question  = item["question"]
        reference = item["ground_truth_answer"]

        try:
            result = process_query(question)
            generated = result["answer"]
        except Exception as e:
            per_question.append({
                "id": item["id"], "question": question,
                "relevance": 0, "accuracy": 0, "completeness": 0,
                "overall": 0.0, "explanation": f"Pipeline error: {e}",
            })
            continue

        scores = llm_judge(question, generated, reference)
        entry = {
            "id":           item["id"],
            "question":     question,
            "jurisdiction": item.get("jurisdiction", ""),
            "difficulty":   item.get("difficulty", ""),
            **scores,
        }
        per_question.append(entry)

        if verbose:
            flag = "OK" if scores["overall"] >= 4.0 else "WARN"
            print(f"  [{item['id']:02d}] {flag}  "
                  f"R={scores['relevance']}  A={scores['accuracy']}  "
                  f"C={scores['completeness']}  overall={scores['overall']:.2f}  "
                  f"| {question[:50]}")
            print(f"        {scores.get('explanation','')}")

    if not per_question:
        return {
            "avg_relevance": 0.0, "avg_accuracy": 0.0,
            "avg_completeness": 0.0, "avg_overall": 0.0,
            "total_questions": 0, "per_question": [],
        }

    n = len(per_question)
    return {
        "avg_relevance":    round(sum(r["relevance"]    for r in per_question) / n, 3),
        "avg_accuracy":     round(sum(r["accuracy"]     for r in per_question) / n, 3),
        "avg_completeness": round(sum(r["completeness"] for r in per_question) / n, 3),
        "avg_overall":      round(sum(r["overall"]      for r in per_question) / n, 3),
        "total_questions":  n,
        "per_question":     per_question,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate answer correctness (LLM judge)")
    parser.add_argument("--mode",    choices=["quick", "full"], default="full")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(GROUND_TRUTH_PATH) as f:
        test_data = json.load(f)
    if args.mode == "quick":
        test_data = test_data[:5]

    print(f"\nAnswer Evaluation — {len(test_data)} questions ({args.mode} mode)\n")
    results = evaluate_answers(test_data, verbose=args.verbose)

    print(f"  Avg Relevance:    {results['avg_relevance']:.2f}/5.0")
    print(f"  Avg Accuracy:     {results['avg_accuracy']:.2f}/5.0")
    print(f"  Avg Completeness: {results['avg_completeness']:.2f}/5.0")
    print(f"  Avg Overall:      {results['avg_overall']:.2f}/5.0")


if __name__ == "__main__":
    main()
