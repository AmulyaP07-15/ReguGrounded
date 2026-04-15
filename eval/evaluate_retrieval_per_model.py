"""
eval/evaluate_retrieval_per_model.py — Per-model retrieval evaluation
======================================================================
Measures Precision@k, Recall@k, F1@k, and MRR for each model's full
retrieval path against ground_truth.json expected citations.

Unlike evaluate_retrieval.py (which tests the retriever in isolation),
this script runs each model's *actual* retrieval strategy:

  Standard RAG  — single retrieve(query, top_k=5), no decomposition
  ARAG          — RLM decompose(query) → parallel retrieve per sub-question
  CAG           — same retrieval path as ARAG (corrective loop is post-retrieval)

Metrics are article-level (not doc-level):
  Precision@k  — fraction of retrieved chunks whose article/section matches
                 an expected citation
  Recall@k     — fraction of expected citations covered by at least one
                 retrieved chunk
  F1@k         — harmonic mean
  MRR          — reciprocal rank of first chunk matching any expected citation

Usage:
    python eval/evaluate_retrieval_per_model.py
    python eval/evaluate_retrieval_per_model.py --mode quick   # 10 questions
    python eval/evaluate_retrieval_per_model.py --mode full    # all 53
    python eval/evaluate_retrieval_per_model.py --verbose
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.retriever          import retrieve as base_retrieve   # noqa: E402
from src.enhanced_retriever import EnhancedRetriever           # noqa: E402
from rlm_engine             import RLMEngine                   # noqa: E402

GROUND_TRUTH_PATH = Path(__file__).parent / "ground_truth.json"

# Questions used in quick mode — same 10 as compare_models.py
QUICK_IDS = {2, 4, 16, 28, 35, 38, 46, 49, 52, 53}


# ---------------------------------------------------------------------------
# Citation matching helpers
# ---------------------------------------------------------------------------

def _normalize(ref: str) -> str:
    return ref.strip().lower().replace("  ", " ")


def _chunk_matches_citation(chunk: dict, expected_ref: str) -> bool:
    """
    Return True if the chunk's article/section metadata matches expected_ref.

    expected_ref examples: "Article 9", "§ 20-871", "§ 6-1-1703", "GOVERN 1.1"
    chunk metadata keys: article_number, section_number
    """
    meta = chunk.get("metadata", chunk)  # works for both raw and processed dicts
    article = meta.get("article_number") or ""
    section = meta.get("section_number") or ""
    text    = chunk.get("text", "")

    norm_exp = _normalize(expected_ref)

    # "Article 9" → article_number == "9"
    m = re.match(r"article\s+(\S+)", norm_exp)
    if m and _normalize(article) == _normalize(m.group(1)):
        return True

    # "§ 20-871" or "§ 6-1-1703" → section_number contains the ref
    m = re.match(r"§\s*(.+)", norm_exp)
    if m:
        ref_part = _normalize(m.group(1))
        if ref_part in _normalize(section) or _normalize(section) in ref_part:
            return True

    # NIST subcategory e.g. "GOVERN 1.1", "MAP 1.1"
    if norm_exp in _normalize(section):
        return True

    # Last resort: check raw text (catches CRS refs embedded in chunk text)
    if norm_exp in _normalize(text[:500]):
        return True

    return False


def _hits(chunks: list, expected: list) -> list:
    """Return list of booleans: whether each expected citation is covered."""
    return [any(_chunk_matches_citation(c, e) for c in chunks) for e in expected]


# ---------------------------------------------------------------------------
# Retrieval strategies
# ---------------------------------------------------------------------------

def retrieve_standard_rag(query: str, top_k: int = 5) -> list:
    """Single retrieve, no decomposition."""
    return base_retrieve(query, top_k=top_k)


def retrieve_arag_cag(query: str, rlm: RLMEngine,
                      retriever: EnhancedRetriever, top_k: int = 5) -> list:
    """
    RLM decompose → parallel retrieve per sub-question → deduplicated pool.
    This is the shared retrieval path for both ARAG and CAG.
    """
    decomp       = rlm.decompose(query)
    sub_questions = decomp.get("sub_questions", [])

    if not sub_questions:
        sub_questions = [{"sub_question": query, "jurisdiction": None}]

    seen_ids = set()
    all_chunks = []

    for sq in sub_questions:
        sub_q  = sq.get("sub_question", query)
        juris  = sq.get("jurisdiction")
        filter_ = {"doc_id": _jurisdiction_to_doc_id(juris)} if juris else None
        chunks = retriever.retrieve(sub_q, top_k=top_k, filter=filter_)
        for c in chunks:
            cid = c.get("metadata", {}).get("chunk_id") or c.get("chunk_id", "")
            if cid not in seen_ids:
                seen_ids.add(cid)
                all_chunks.append(c)

    return all_chunks


_JURIS_TO_DOC = {
    "EU AI Act":       "eu_ai_act",
    "NYC Local Law 144": "nyc_local_law_144",
    "Colorado AI Act": "colorado_ai_act",
    "NIST AI RMF":     "nist_ai_framework",
}

def _jurisdiction_to_doc_id(juris: str | None) -> str | None:
    if not juris:
        return None
    for key, val in _JURIS_TO_DOC.items():
        if key.lower() in juris.lower():
            return val
    return None


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(chunks: list, expected: list) -> dict:
    if not expected:
        return {"recall": 1.0, "coverage": 1.0, "mrr": 1.0,
                "n_retrieved": len(chunks), "n_expected": 0}

    hit_flags = _hits(chunks, expected)
    recall    = sum(hit_flags) / len(expected)

    # Coverage: fraction of expected citations covered by at least one chunk
    # (same as recall — kept separately for clarity in the table)
    coverage = recall

    # MRR: reciprocal rank of first chunk matching any expected citation
    mrr = 0.0
    for rank, c in enumerate(chunks, 1):
        if any(_chunk_matches_citation(c, e) for e in expected):
            mrr = 1.0 / rank
            break

    return {
        "recall":      round(recall,   4),
        "coverage":    round(coverage, 4),
        "mrr":         round(mrr,      4),
        "n_retrieved": len(chunks),
        "n_expected":  len(expected),
    }


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_eval(questions: list, top_k: int = 5, verbose: bool = False):
    print(f"\nInitialising retrievers...")
    enhanced = EnhancedRetriever()
    rlm      = RLMEngine()

    models = [
        ("Standard RAG",  lambda q: retrieve_standard_rag(q, top_k=top_k)),
        ("ARAG",          lambda q: retrieve_arag_cag(q, rlm, enhanced, top_k=top_k)),
        ("CAG + RLM",     lambda q: retrieve_arag_cag(q, rlm, enhanced, top_k=top_k)),
    ]

    all_results = {}

    for name, retrieve_fn in models:
        print(f"\n  Running {name}...")
        R, MRR, N = [], [], []

        for q in questions:
            expected = q.get("expected_citations", [])
            try:
                chunks = retrieve_fn(q["question"])
            except Exception as e:
                print(f"    ERROR Q{q['id']}: {e}")
                R.append(0); MRR.append(0); N.append(0)
                continue

            m = compute_metrics(chunks, expected)
            R.append(m["recall"]); MRR.append(m["mrr"]); N.append(m["n_retrieved"])

            if verbose:
                status = "OK" if m["recall"] >= 0.5 else "WARN"
                print(f"    [{q['id']:02d}] {status}  "
                      f"R={m['recall']:.2f}  MRR={m['mrr']:.2f}  "
                      f"chunks={m['n_retrieved']}  exp={m['n_expected']}  "
                      f"| {q['question'][:55]}")

        n = len(R)
        all_results[name] = {
            "recall":     round(sum(R)/n,   4),
            "mrr":        round(sum(MRR)/n, 4),
            "avg_chunks": round(sum(N)/n,   1),
        }

    _print_table(all_results, top_k)
    return all_results


def _print_table(results: dict, top_k: int):
    names = list(results.keys())
    col   = 18

    print(f"\n\n{'='*75}")
    print(f"  RETRIEVAL EVALUATION — Recall@k  (top_k={top_k} per sub-question)")
    print(f"{'='*75}")

    header = f"  {'Metric':<{col}}"
    for n in names:
        header += f"  {n:>18}"
    print(header)
    print(f"  {'-'*73}")

    rows = [
        ("Recall@k",    "recall",      "%"),
        ("MRR",         "mrr",         "%"),
        ("Avg Chunks",  "avg_chunks",  "f"),
    ]
    for label, key, fmt in rows:
        line = f"  {label:<{col}}"
        for n in names:
            v = results[n][key]
            line += f"  {v:>17.1%}" if fmt == "%" else f"  {v:>17.1f}"
        print(line)

    print(f"  {'-'*73}")
    print(f"{'='*75}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",    choices=["quick", "full"], default="quick")
    parser.add_argument("--top-k",   type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(GROUND_TRUTH_PATH) as f:
        all_q = json.load(f)

    if args.mode == "quick":
        questions = [q for q in all_q if q["id"] in QUICK_IDS]
    else:
        questions = all_q

    print(f"\nPer-model retrieval evaluation")
    print(f"Mode: {args.mode} | Questions: {len(questions)} | top_k: {args.top_k}")

    run_eval(questions, top_k=args.top_k, verbose=args.verbose)


if __name__ == "__main__":
    main()
