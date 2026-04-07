"""
evaluate_retrieval.py — ReguGrounded Retrieval Accuracy Evaluation
===================================================================
Measures how well the retriever surfaces relevant regulatory chunks
for a curated set of test questions with known expected sources.

Metrics computed:
    - Recall@k    : fraction of relevant docs that appeared in top-k results
    - Precision@k : fraction of top-k results that were actually relevant
    - F1@k        : harmonic mean of Precision@k and Recall@k
    - MRR         : Mean Reciprocal Rank (how high was the first hit?)
    - Avg top score : mean similarity score of the #1 result

Usage:
    python eval/evaluate_retrieval.py
    python eval/evaluate_retrieval.py --top-k 3
    python eval/evaluate_retrieval.py --top-k 10 --verbose
"""

import argparse
import sys
from pathlib import Path

# Make src/ importable from anywhere
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from retriever import Retriever  # noqa: E402

ALL_DOC_IDS = {
    "eu_ai_act",
    "eu_ai_act_annex",
    "nyc_local_law_144",
    "nist_ai_framework",
    "colorado_ai_act",
}

# ---------------------------------------------------------------------------
# Evaluation dataset
# Each entry has:
#   query        : natural language question
#   relevant_docs: doc_ids that SHOULD appear in results
#   negative_docs: doc_ids that should NOT appear in top results
#                  (automatically inferred as all docs not in relevant_docs,
#                   but listed explicitly for clarity and override)
#   relevant_kws : keywords expected in the top result's text (secondary signal)
# ---------------------------------------------------------------------------
EVAL_QUESTIONS = [
    # ── EU AI Act ────────────────────────────────────────────────────────────
    {
        "query": "What are the requirements for high-risk AI systems under the EU AI Act?",
        "relevant_docs": ["eu_ai_act"],
        "negative_docs": ["nyc_local_law_144", "nist_ai_framework", "colorado_ai_act"],
        "relevant_kws": ["high-risk", "Article"],
    },
    {
        "query": "What obligations do providers of high-risk AI systems have?",
        "relevant_docs": ["eu_ai_act"],
        "negative_docs": ["nyc_local_law_144", "nist_ai_framework", "colorado_ai_act"],
        "relevant_kws": ["provider", "obligation"],
    },
    {
        "query": "What is required in an EU AI Act conformity assessment?",
        "relevant_docs": ["eu_ai_act", "eu_ai_act_annex"],
        "negative_docs": ["nyc_local_law_144", "nist_ai_framework", "colorado_ai_act"],
        "relevant_kws": ["conformity", "assessment"],
    },
    {
        "query": "What transparency requirements apply to AI systems interacting with humans?",
        "relevant_docs": ["eu_ai_act"],
        "negative_docs": ["nyc_local_law_144", "nist_ai_framework", "colorado_ai_act"],
        "relevant_kws": ["transparency", "natural person"],
    },
    {
        "query": "What is the EU AI Act definition of a general purpose AI model?",
        "relevant_docs": ["eu_ai_act"],
        "negative_docs": ["nyc_local_law_144", "nist_ai_framework", "colorado_ai_act"],
        "relevant_kws": ["general-purpose", "model"],
    },
    {
        "query": "What are prohibited AI practices under the EU AI Act?",
        "relevant_docs": ["eu_ai_act"],
        "negative_docs": ["nyc_local_law_144", "nist_ai_framework", "colorado_ai_act"],
        "relevant_kws": ["prohibited", "manipulation"],
    },
    {
        "query": "What post-market monitoring obligations exist under the EU AI Act?",
        "relevant_docs": ["eu_ai_act"],
        "negative_docs": ["nyc_local_law_144", "nist_ai_framework", "colorado_ai_act"],
        "relevant_kws": ["post-market", "monitoring"],
    },

    # ── NYC Local Law 144 ────────────────────────────────────────────────────
    {
        "query": "What bias audit requirements does NYC Local Law 144 impose?",
        "relevant_docs": ["nyc_local_law_144"],
        "negative_docs": ["eu_ai_act", "eu_ai_act_annex", "nist_ai_framework"],
        "relevant_kws": ["bias audit", "automated employment"],
    },
    {
        "query": "What notice must employers give candidates about automated hiring tools?",
        "relevant_docs": ["nyc_local_law_144"],
        "negative_docs": ["eu_ai_act", "eu_ai_act_annex", "nist_ai_framework"],
        "relevant_kws": ["notice", "candidate", "employer"],
    },
    {
        "query": "Who must conduct a bias audit under NYC Local Law 144?",
        "relevant_docs": ["nyc_local_law_144"],
        "negative_docs": ["eu_ai_act", "eu_ai_act_annex", "nist_ai_framework"],
        "relevant_kws": ["independent auditor", "audit"],
    },

    # ── NIST AI Risk Management Framework ───────────────────────────────────
    {
        "query": "What governance practices does NIST recommend for AI risk management?",
        "relevant_docs": ["nist_ai_framework"],
        "negative_docs": ["eu_ai_act", "eu_ai_act_annex", "nyc_local_law_144", "colorado_ai_act"],
        "relevant_kws": ["governance", "GOVERN"],
    },
    {
        "query": "How does NIST define trustworthy AI characteristics?",
        "relevant_docs": ["nist_ai_framework"],
        "negative_docs": ["eu_ai_act", "eu_ai_act_annex", "nyc_local_law_144", "colorado_ai_act"],
        "relevant_kws": ["trustworthy", "reliable"],
    },
    {
        "query": "What does the NIST AI RMF say about AI risk measurement?",
        "relevant_docs": ["nist_ai_framework"],
        "negative_docs": ["eu_ai_act", "eu_ai_act_annex", "nyc_local_law_144", "colorado_ai_act"],
        "relevant_kws": ["MEASURE", "risk"],
    },
    {
        "query": "What is the NIST MAP function in the AI Risk Management Framework?",
        "relevant_docs": ["nist_ai_framework"],
        "negative_docs": ["eu_ai_act", "eu_ai_act_annex", "nyc_local_law_144", "colorado_ai_act"],
        "relevant_kws": ["MAP", "context"],
    },

    # ── Colorado AI Act ──────────────────────────────────────────────────────
    {
        "query": "What does the Colorado AI Act require from developers of high-risk AI?",
        "relevant_docs": ["colorado_ai_act"],
        "negative_docs": ["eu_ai_act", "eu_ai_act_annex", "nyc_local_law_144", "nist_ai_framework"],
        "relevant_kws": ["developer", "high-risk"],
    },
    {
        "query": "What consumer protections does the Colorado AI Act provide?",
        "relevant_docs": ["colorado_ai_act"],
        "negative_docs": ["eu_ai_act", "eu_ai_act_annex", "nyc_local_law_144", "nist_ai_framework"],
        "relevant_kws": ["consumer", "protection"],
    },

    # ── Cross-jurisdictional ─────────────────────────────────────────────────
    {
        "query": "How do bias audit requirements differ between NYC and Colorado?",
        "relevant_docs": ["nyc_local_law_144", "colorado_ai_act"],
        "negative_docs": ["nist_ai_framework"],
        "relevant_kws": ["audit", "bias"],
    },
    {
        "query": "What risk management obligations apply to AI systems across regulations?",
        "relevant_docs": ["eu_ai_act", "nist_ai_framework", "colorado_ai_act"],
        "negative_docs": [],
        "relevant_kws": ["risk", "management"],
    },
]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def recall_at_k(results: list[dict], relevant_docs: list[str]) -> float:
    """
    Fraction of relevant docs that appeared in the top-k results.
    e.g. if 2 docs are relevant and 1 appears in results → 0.5
    """
    if not relevant_docs:
        return 0.0
    result_docs = {r["metadata"]["doc_id"] for r in results}
    hits = len(set(relevant_docs) & result_docs)
    return hits / len(relevant_docs)


def precision_at_k(results: list[dict], relevant_docs: list[str]) -> float:
    """
    Fraction of the top-k results that came from a relevant doc.
    e.g. if top-5 has 3 results from relevant docs → 0.6
    """
    if not results:
        return 0.0
    relevant_set = set(relevant_docs)
    relevant_hits = sum(1 for r in results if r["metadata"]["doc_id"] in relevant_set)
    return relevant_hits / len(results)


def f1_at_k(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def reciprocal_rank(results: list[dict], relevant_docs: list[str]) -> float:
    """1/rank of the first relevant result, or 0 if none found."""
    for rank, r in enumerate(results, start=1):
        if r["metadata"]["doc_id"] in relevant_docs:
            return 1.0 / rank
    return 0.0


def keyword_hit(results: list[dict], keywords: list[str]) -> bool:
    """True if any keyword appears (case-insensitive) in the top result's text."""
    if not results:
        return False
    top_text = results[0]["text"].lower()
    return any(kw.lower() in top_text for kw in keywords)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_eval(top_k: int = 5, verbose: bool = False):
    print(f"\n{'='*65}")
    print(f"  ReguGrounded — Retrieval Evaluation  (top_k={top_k})")
    print(f"{'='*65}\n")

    retriever = Retriever(top_k=top_k)

    recall_scores = []
    precision_scores = []
    f1_scores = []
    rr_results = []
    top_scores = []
    kw_hits = []

    for i, item in enumerate(EVAL_QUESTIONS, start=1):
        query        = item["query"]
        relevant_docs = item["relevant_docs"]
        relevant_kws  = item["relevant_kws"]

        results = retriever.retrieve(query, top_k=top_k)

        rec   = recall_at_k(results, relevant_docs)
        prec  = precision_at_k(results, relevant_docs)
        f1    = f1_at_k(prec, rec)
        rr    = reciprocal_rank(results, relevant_docs)
        kw    = keyword_hit(results, relevant_kws)
        top_score = results[0]["score"] if results else 0.0

        recall_scores.append(rec)
        precision_scores.append(prec)
        f1_scores.append(f1)
        rr_results.append(rr)
        top_scores.append(top_score)
        kw_hits.append(kw)

        # PASS = at least one relevant doc appeared
        status = "PASS" if rec > 0 else "FAIL"

        if verbose or rec == 0:
            print(f"[{i:02d}] {status}  |  P={prec:.2f}  R={rec:.2f}  F1={f1:.2f}  RR={rr:.2f}  score={top_score:.3f}")
            print(f"      Query: {query}")
            if results:
                top = results[0]["metadata"]
                loc = f"Article {top['article_number']}" if top.get("article_number") else top.get("section_number") or "—"
                print(f"      Top result: {top['law_name']} | {loc} | {top.get('heading','')[:55]}")
            if rec == 0:
                print(f"      Expected: {relevant_docs}")
                got_docs = [r["metadata"]["doc_id"] for r in results]
                print(f"      Got:      {got_docs}")
            print()
        else:
            print(f"[{i:02d}] {status}  |  P={prec:.2f}  R={rec:.2f}  F1={f1:.2f}  RR={rr:.2f}  score={top_score:.3f}  |  {query[:50]}")

    # ── Overall summary ───────────────────────────────────────────────────────
    n = len(EVAL_QUESTIONS)
    avg_recall    = sum(recall_scores) / n
    avg_precision = sum(precision_scores) / n
    avg_f1        = sum(f1_scores) / n
    avg_mrr       = sum(rr_results) / n
    avg_score     = sum(top_scores) / n
    kw_acc        = sum(kw_hits) / n

    print(f"\n{'='*65}")
    print(f"  RESULTS  ({n} questions, top_k={top_k})")
    print(f"{'='*65}")
    print(f"  Precision@{top_k}      : {avg_precision:.1%}  (avg fraction of top-{top_k} results that were relevant)")
    print(f"  Recall@{top_k}         : {avg_recall:.1%}  (avg fraction of relevant docs found in top-{top_k})")
    print(f"  F1@{top_k}             : {avg_f1:.1%}  (harmonic mean of precision and recall)")
    print(f"  MRR               : {avg_mrr:.3f}  (avg reciprocal rank of first relevant result)")
    print(f"  Avg top score     : {avg_score:.3f}  (mean cosine similarity of #1 result)")
    print(f"  Keyword hit rate  : {kw_acc:.1%}  (relevant keyword in top result text)")
    print(f"{'='*65}\n")

    # ── Per-source breakdown ──────────────────────────────────────────────────
    doc_ids = ["eu_ai_act", "nyc_local_law_144", "nist_ai_framework", "colorado_ai_act"]
    print("  Per-source breakdown:")
    print(f"  {'Source':<28} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Queries':>8}")
    print(f"  {'-'*64}")
    for doc_id in doc_ids:
        indices = [idx for idx, item in enumerate(EVAL_QUESTIONS) if doc_id in item["relevant_docs"]]
        if not indices:
            continue
        p = sum(precision_scores[i] for i in indices) / len(indices)
        r = sum(recall_scores[i]    for i in indices) / len(indices)
        f = sum(f1_scores[i]        for i in indices) / len(indices)
        print(f"  {doc_id:<28} {p:>9.1%} {r:>8.1%} {f:>8.1%} {len(indices):>7}")
    print()

    return avg_precision, avg_recall, avg_f1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ReguGrounded retrieval accuracy")
    parser.add_argument("--top-k",   type=int,  default=5,     help="Results to retrieve per query (default: 5)")
    parser.add_argument("--verbose", action="store_true",       help="Print details for every query, not just failures")
    args = parser.parse_args()

    run_eval(top_k=args.top_k, verbose=args.verbose)