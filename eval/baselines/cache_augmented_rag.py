"""
eval/baselines/cache_augmented_rag.py — Cache-Augmented Generation Baseline
=============================================================================
Cache-Augmented Generation (CAG): the ENTIRE regulatory corpus is loaded into
the LLM's context window at startup. No vector retrieval at query time.

    Standard RAG  — Pinecone retrieval (top-5 chunks) → single LLM call
    ARAG          — decomposition → multi-retrieval → LLM
    CAG (corrective) — decomposition → multi-retrieval → validate → retry
    CAG (cache)   — full corpus in context → LLM  ← this file

Design:
  - All 395 regulatory chunks (~130k tokens) are preloaded from data/chunks/
  - If ANTHROPIC_API_KEY is set: corpus is placed in a cached system message
    (cache_control={"type": "ephemeral"}) so the KV state is reused across all
    queries — only the question differs after the first call.
  - Other providers (Gemini, Groq): corpus passed in context each call.

Tradeoffs vs. retrieval-based systems:
  + No retrieval errors — all relevant chunks always present
  + Eliminates missed citations on exhaustive multi-article questions
  + No Pinecone / embedding dependency
  - Requires a large context window (~130k tokens; Claude 200k, Gemini 1M)
  - Higher first-call latency; subsequent calls benefit from KV caching
  - Less focused: LLM attends to all 395 chunks, not a curated top-k
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

# ---------------------------------------------------------------------------
# doc_id → canonical law name (matches citation_validator._LAW_ALIASES)
# ---------------------------------------------------------------------------

_DOC_ID_TO_LAW = {
    "eu_ai_act":         "EU AI Act",
    "eu_ai_act_annex":   "EU AI Act",
    "colorado_ai_act":   "Colorado AI Act",
    "nyc_local_law_144": "NYC Local Law 144",
    "nist_ai_framework": "NIST AI Risk Management Framework",
}

_CHUNKS_DIR = ROOT / "data" / "chunks"

# Module-level singletons — loaded once per process
_corpus_text:  Optional[str]  = None
_flat_chunks:  Optional[list] = None


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def _section_to_fields(doc_id: str, section: str) -> dict:
    """
    Convert a raw section string (from chunk JSON) to the law_name /
    article_number / section_number shape that citation_validator expects.
    """
    s = section.strip() if section else ""

    if not s or s.lower() == "preamble":
        return {"article_number": "", "section_number": ""}

    if doc_id in ("eu_ai_act", "eu_ai_act_annex"):
        # "Article 9" → article_number = "9"
        m = re.match(r"Article\s+(\d+)", s, re.IGNORECASE)
        return {"article_number": m.group(1) if m else "", "section_number": ""}

    if doc_id in ("nyc_local_law_144", "colorado_ai_act"):
        # "§ 20-871" → section_number = "20-871"
        m = re.match(r"§\s*([\d][\d\w-]*)", s)
        return {"article_number": "", "section_number": m.group(1) if m else ""}

    if doc_id == "nist_ai_framework":
        # "GOVERN 1.2", "1.1" etc. → store as article_number
        # citation_validator normalizes "GOVERN-1.2" → "GOVERN 1.2" for matching
        return {"article_number": s, "section_number": ""}

    return {"article_number": "", "section_number": ""}


def _load_corpus() -> tuple:
    """
    Load all regulatory chunks from data/chunks/ once.

    Returns:
        corpus_text  — formatted string of the full regulatory corpus
        flat_chunks  — list of dicts with law_name/article_number/section_number/text
                       (same shape citation_validator expects)
    """
    global _corpus_text, _flat_chunks
    if _corpus_text is not None:
        return _corpus_text, _flat_chunks

    chunks_by_law: dict = {}
    flat_chunks:   list = []

    for json_file in sorted(_CHUNKS_DIR.glob("*.json")):
        with open(json_file) as fh:
            data = json.load(fh)
        raw_chunks = data if isinstance(data, list) else data.get("chunks", [])

        for chunk in raw_chunks:
            doc_id   = chunk.get("doc_id", "")
            section  = chunk.get("section", "")
            text     = chunk.get("text", "").strip()
            law_name = _DOC_ID_TO_LAW.get(doc_id, doc_id)

            fields = _section_to_fields(doc_id, section)

            flat_chunk = {
                "law_name":       law_name,
                "article_number": fields["article_number"],
                "section_number": fields["section_number"],
                "text":           text,
            }
            flat_chunks.append(flat_chunk)
            chunks_by_law.setdefault(law_name, []).append((section, text))

    # Build formatted corpus string
    lines = ["REGULATORY CORPUS", "=" * 60]
    for law, entries in chunks_by_law.items():
        lines.append(f"\n{'─' * 60}")
        lines.append(f"LAW: {law}")
        lines.append("─" * 60)
        for section, text in entries:
            label = f"{law} — {section}" if section and section.lower() != "preamble" else law
            lines.append(f"\n[{label}]")
            lines.append(text)

    _corpus_text = "\n".join(lines)
    _flat_chunks = flat_chunks
    return _corpus_text, flat_chunks


# ---------------------------------------------------------------------------
# LLM call helpers
# ---------------------------------------------------------------------------

_SYSTEM_INSTRUCTIONS = (
    "You are a regulatory compliance expert. Answer questions using ONLY "
    "the regulatory corpus provided below.\n\n"
    "Rules:\n"
    "- Always cite the FULL law name + section/article inline.\n"
    "  CORRECT:   'NYC Local Law 144 § 20-871 requires a bias audit...'\n"
    "  CORRECT:   'EU AI Act Article 9 establishes a risk management system...'\n"
    "  INCORRECT: '§ 20-871 requires...'  (missing law name)\n"
    "  INCORRECT: 'Article 9 establishes...'  (missing law name)\n"
    "- Only cite sources that appear in the corpus. Do not invent citations.\n"
    "- If multiple jurisdictions apply, address each one in turn.\n"
    "- Be concise and precise. If the corpus does not cover the topic, say so.\n\n"
)


def _call_anthropic_cached(corpus: str, question: str) -> str:
    """
    Call Claude with the corpus in a cached system message.
    The system message is marked cache_control=ephemeral so Anthropic caches
    the KV state after the first call — subsequent queries only encode the question.
    """
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    model   = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001").strip()
    client  = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=[
            {
                "type": "text",
                "text": _SYSTEM_INSTRUCTIONS + corpus,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": question}],
        temperature=0.1,
    )
    return response.content[0].text.strip()


def _call_regular_client(corpus: str, question: str) -> str:
    """
    Call LLM (Gemini/Groq) with corpus inline in the user message.
    No native KV caching — the full corpus is encoded each call.
    """
    from utils.llm_client import get_llm_client

    client = get_llm_client()
    prompt = (
        f"{_SYSTEM_INSTRUCTIONS}"
        f"{corpus}\n\n"
        f"Question: {question}"
    )
    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=2048,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def process_query(question: str, top_k: int = None) -> dict:
    """
    Cache-Augmented Generation pipeline.

    Loads the entire regulatory corpus into LLM context. No vector retrieval.
    When ANTHROPIC_API_KEY is available the corpus is prompt-cached so only
    the first call pays the full 130k-token encoding cost.

    top_k is accepted but ignored — CAG always uses the full corpus.
    """
    t0 = time.time()

    corpus, flat_chunks = _load_corpus()

    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    cache_used    = bool(anthropic_key)
    llm_failed    = False

    try:
        if anthropic_key:
            answer = _call_anthropic_cached(corpus, question)
        else:
            answer = _call_regular_client(corpus, question)
    except Exception as exc:
        answer     = f"LLM call failed: {exc}"
        llm_failed = True

    latency_ms = int((time.time() - t0) * 1000)

    return {
        "question":         question,
        "sub_questions":    [question],   # no decomposition
        "answer":           answer,
        "citations":        [],           # post-hoc validation applied by eval script
        "validation":       None,         # post-hoc
        "retrieved_chunks": flat_chunks,  # all chunks exposed for post-hoc validation
        "metadata": {
            "jurisdictions_searched": list(_DOC_ID_TO_LAW.values()),
            "total_chunks_retrieved": len(flat_chunks),
            "latency_ms":             latency_ms,
            "baseline":               "cache_augmented_rag",
            "has_built_in_validation": False,
            "validation_method":       "none",
            "cache_used":              cache_used,
            "llm_synthesis_failed":    llm_failed,
        },
    }