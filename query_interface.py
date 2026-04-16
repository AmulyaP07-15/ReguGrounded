# query_interface.py

import json
import re
import time
from pathlib import Path
from typing import Optional

from rlm_engine import RLMEngine
from answer_synthesizer import AnswerSynthesizer
from citation_validator import validate_citations
from query_clarifier import QueryClarifier
from utils.logger import setup_logger, log_function_call
from utils.metrics import MetricsTracker
from utils.cache import QueryCache
from utils.guardrails import InputGuard, OutputGuard, GuardrailBlockedError

logger        = setup_logger("query_interface")
metrics       = MetricsTracker()
_input_guard  = InputGuard()
_output_guard = OutputGuard()

# ---------------------------------------------------------------------------
# Corpus loading (Cache-Augmented Generation)
# ---------------------------------------------------------------------------

_DOC_ID_TO_LAW = {
    "eu_ai_act":         "EU AI Act",
    "eu_ai_act_annex":   "EU AI Act",
    "colorado_ai_act":   "Colorado AI Act",
    "nyc_local_law_144": "NYC Local Law 144",
    "nist_ai_framework": "NIST AI Risk Management Framework",
}

_corpus_text:   Optional[str]  = None
_corpus_chunks: Optional[list] = None


def _load_corpus() -> tuple:
    """
    Load all regulatory chunks from data/chunks/ once per process.

    Returns:
        corpus_text  — formatted string of the full regulatory corpus (~134k tokens)
        flat_chunks  — list of dicts with law_name/article_number/section_number/text
                       (shape expected by citation_validator)
    """
    global _corpus_text, _corpus_chunks
    if _corpus_text is not None:
        return _corpus_text, _corpus_chunks

    chunks_dir   = Path(__file__).parent / "data" / "chunks"
    flat_chunks  = []
    chunks_by_law: dict = {}

    for json_file in sorted(chunks_dir.glob("*.json")):
        with open(json_file) as fh:
            data = json.load(fh)
        raw_chunks = data if isinstance(data, list) else data.get("chunks", [])

        for chunk in raw_chunks:
            doc_id   = chunk.get("doc_id", "")
            section  = chunk.get("section", "").strip()
            text     = chunk.get("text", "").strip()
            law_name = _DOC_ID_TO_LAW.get(doc_id, doc_id)

            article_number = ""
            section_number = ""
            if section and section.lower() != "preamble":
                if doc_id in ("eu_ai_act", "eu_ai_act_annex"):
                    m = re.match(r"Article\s+(\d+)", section, re.IGNORECASE)
                    if m:
                        article_number = m.group(1)
                elif doc_id in ("nyc_local_law_144", "colorado_ai_act"):
                    m = re.match(r"§\s*([\d][\d\w-]*)", section)
                    if m:
                        section_number = m.group(1)
                elif doc_id == "nist_ai_framework":
                    article_number = section  # e.g. "GOVERN 1.2"

            flat_chunks.append({
                "law_name":       law_name,
                "article_number": article_number,
                "section_number": section_number,
                "chunk_id":       chunk.get("chunk_id", ""),
                "text":           text,
            })
            chunks_by_law.setdefault(law_name, []).append((section, text))

    lines = ["REGULATORY CORPUS", "=" * 60]
    for law, entries in chunks_by_law.items():
        lines.append(f"\n{'─' * 60}")
        lines.append(f"LAW: {law}")
        lines.append("─" * 60)
        for section, text in entries:
            label = f"{law} — {section}" if section and section.lower() != "preamble" else law
            lines.append(f"\n[{label}]")
            lines.append(text)

    _corpus_text   = "\n".join(lines)
    _corpus_chunks = flat_chunks
    return _corpus_text, flat_chunks


def _build_corpus_bundle(decomposition: dict, flat_chunks: list) -> list:
    """
    Build an evidence_bundle from the full corpus grouped by sub-question jurisdiction.
    Used instead of Pinecone retrieval in the cache-augmented pipeline.
    """
    chunks_by_law: dict = {}
    for chunk in flat_chunks:
        chunks_by_law.setdefault(chunk.get("law_name", ""), []).append(chunk)

    evidence_bundle = []
    for item in decomposition["sub_questions"]:
        jurisdiction = item.get("jurisdiction")
        sub_question = item.get("sub_question", "")
        chunks = chunks_by_law.get(jurisdiction, flat_chunks) if jurisdiction else flat_chunks
        evidence_bundle.append({
            "jurisdiction":     jurisdiction,
            "sub_question":     sub_question,
            "retrieved_chunks": chunks,
        })
    return evidence_bundle


# ---------------------------------------------------------------------------
# QueryInterface
# ---------------------------------------------------------------------------

class QueryInterface:
    """Accepts a user query and runs the full reasoning pipeline."""

    def __init__(self, retriever_func=None, use_cache: bool = False):
        self.rlm_engine  = RLMEngine()
        self.synthesizer = AnswerSynthesizer()
        self.use_cache   = use_cache
        self._cache      = QueryCache() if use_cache else None

        # Retriever is optional — when provided (e.g. tests), uses Pinecone.
        # When None (production), uses the cached full corpus.
        if retriever_func is not None:
            from reasoning_orchestrator import ReasoningOrchestrator
            self.orchestrator = ReasoningOrchestrator(retriever_func=retriever_func)
        else:
            self.orchestrator = None

    @staticmethod
    def _adaptive_top_k(decomposition: dict, base_top_k: int) -> int:
        """Scale top_k with the number of jurisdictions (retrieval mode only)."""
        n = len([s for s in decomposition["sub_questions"] if s.get("jurisdiction")])
        if n >= 3:
            return max(base_top_k, 10)
        if n == 2:
            return max(base_top_k, 7)
        return base_top_k

    @log_function_call(logger)
    def run(self, user_query: str, top_k: int = 5) -> dict:
        """
        Run the full pipeline from query to structured answer.

        Returns:
            {
                "question": str,
                "sub_questions": [str],
                "answer": str,
                "citations": [{"source", "article", "excerpt", "jurisdiction"}],
                "validation": {"total_citations", "valid_citations", "invalid_citations",
                               "accuracy", "hallucination_rate", "details"},
                "metadata": {"jurisdictions_searched", "total_chunks_retrieved",
                             "latency_ms", "cache_hit", "guardrail_warnings"}
            }

        Raises:
            GuardrailBlockedError — if the input fails a hard guardrail check
        """
        t_start = time.time()

        # ── Input guardrails ───────────────────────────────────────────
        input_result = _input_guard.run(user_query)
        if input_result.blocks():
            blocking_violation = next(
                v for v in input_result.violations if v.severity == "block"
            )
            metrics.record_error(f"guardrail_{blocking_violation.category.lower()}")
            raise GuardrailBlockedError(blocking_violation)

        input_warnings = input_result.warning_strings()

        # ── Cache check ───────────────────────────────────────────────
        if self.use_cache and self._cache is not None:
            cached = self._cache.get(user_query, top_k)
            if cached is not None:
                metrics.record_cache_hit("query")
                logger.info(f"Cache hit for query: {user_query[:80]!r}")
                cached["metadata"]["cache_hit"] = True
                return cached
            metrics.record_cache_miss("query")
        cacheable = self.use_cache and self._cache is not None

        try:
            result = self._run_pipeline(user_query, top_k, t_start)

            # ── Output guardrails ─────────────────────────────────────
            output_result = _output_guard.run(result)
            all_warnings  = input_warnings + output_result.warning_strings()
            result["metadata"]["guardrail_warnings"] = all_warnings

            metrics.record_query(success=True, latency_ms=result["metadata"]["latency_ms"])
            metrics.record_citation_issue(
                accuracy=result["validation"]["accuracy"],
                hallucination_rate=result["validation"]["hallucination_rate"],
            )
            if cacheable:
                self._cache.set(user_query, top_k, result)
            return result

        except GuardrailBlockedError:
            raise
        except Exception as exc:
            latency_ms = int((time.time() - t_start) * 1000)
            metrics.record_query(success=False, latency_ms=latency_ms)
            metrics.record_error(type(exc).__name__)
            logger.error(f"Pipeline failed for query {user_query[:80]!r}: {exc}", exc_info=True)
            raise

    def _run_pipeline(self, user_query: str, top_k: int, t_start: float) -> dict:
        # 1. Decompose
        try:
            decomposition = self.rlm_engine.decompose(user_query)
        except Exception as e:
            logger.warning(f"Query decomposition failed ({e}); using fallback decomposition")
            metrics.record_error("decomposition_failure")
            decomposition = {
                "original_query": user_query,
                "jurisdictions":  [],
                "sub_questions":  [{"jurisdiction": None, "sub_question": user_query}],
            }

        # 2. Evidence — corpus (production) or Pinecone (test/legacy)
        if self.orchestrator is None:
            # Cache-Augmented Generation: full corpus in context, no retrieval
            corpus_text, flat_chunks = _load_corpus()
            evidence_bundle = _build_corpus_bundle(decomposition, flat_chunks)
            all_chunks      = flat_chunks
            jurisdictions   = list(dict.fromkeys(
                item.get("jurisdiction") for item in decomposition["sub_questions"]
                if item.get("jurisdiction")
            ))
            logger.info(
                f"Corpus mode: {len(all_chunks)} chunks loaded across "
                f"{len(jurisdictions)} jurisdiction(s): {jurisdictions}"
            )
        else:
            # Pinecone retrieval (used when retriever_func injected, e.g. tests)
            corpus_text     = None
            effective_top_k = self._adaptive_top_k(decomposition, top_k)
            evidence_bundle = self.orchestrator.run(decomposition, top_k=effective_top_k)
            all_chunks      = self.orchestrator.get_all_chunks(evidence_bundle)
            jurisdictions   = self.orchestrator.get_jurisdictions(evidence_bundle)
            logger.info(
                f"Retrieval mode: {len(all_chunks)} chunks across "
                f"{len(jurisdictions)} jurisdiction(s): {jurisdictions}"
            )

        # 3 + 4. Synthesize with corrective retry loop.
        # CAG's key distinction: when citations are not grounded in the corpus
        # the synthesizer is called again with the bad citations explicitly
        # forbidden, actively preventing hallucinations.
        _MAX_RETRIES         = 2
        _HALLUCINATION_LIMIT = 0.10

        synthesis      = None
        validation_raw = None
        invalid_cites  = None

        for attempt in range(_MAX_RETRIES):
            synthesis = self.synthesizer.synthesize(
                original_query=user_query,
                decomposition=decomposition,
                evidence_bundle=evidence_bundle,
                invalid_citations=invalid_cites,
                corpus_text=corpus_text,
            )
            validation_raw = validate_citations(synthesis["answer"], all_chunks)

            h_rate = validation_raw["hallucination_rate"]

            if h_rate <= _HALLUCINATION_LIMIT:
                if attempt > 0:
                    logger.info(
                        f"Corrective retry succeeded on attempt {attempt + 1}: "
                        f"hallucination rate now {h_rate:.1%}"
                    )
                break

            invalid_cites = validation_raw["invalid_citations"]
            if attempt < _MAX_RETRIES - 1:
                logger.warning(
                    f"Attempt {attempt + 1}: hallucination rate {h_rate:.1%} "
                    f"({validation_raw['invalid_count']} invalid citations). "
                    f"Retrying with corrective constraints on: {invalid_cites}"
                )
            else:
                logger.error(
                    f"Max retries reached. Final hallucination rate: {h_rate:.1%}. "
                    f"Returning best available answer."
                )

        corrective_action_taken = attempt > 0
        llm_failed              = synthesis.get("llm_synthesis_failed", False)

        latency_ms = int((time.time() - t_start) * 1000)
        logger.info(f"Pipeline completed in {latency_ms} ms (attempts: {attempt + 1})")

        return {
            "question":      user_query,
            "sub_questions": synthesis["sub_questions"],
            "answer":        synthesis["answer"],
            "citations":     synthesis["citations"],
            "validation": {
                "total_citations":    validation_raw["total_citations"],
                "valid_citations":    validation_raw["valid_count"],
                "invalid_citations":  validation_raw["invalid_count"],
                "accuracy":           validation_raw["accuracy"],
                "hallucination_rate": validation_raw["hallucination_rate"],
                "details":            validation_raw["details"],
            },
            "metadata": {
                "jurisdictions_searched":  jurisdictions,
                "total_chunks_retrieved":  len(all_chunks),
                "latency_ms":              latency_ms,
                "cache_hit":               False,
                "guardrail_warnings":      [],
                "has_built_in_validation": True,
                "validation_method":       "built_in_grounded_generation",
                "retry_attempts":          attempt + 1,
                "corrective_action_taken": corrective_action_taken,
                "llm_synthesis_failed":    llm_failed,
            },
        }


def process_query(user_query: str, top_k: int = 5, use_cache: bool = True) -> dict:
    """Module-level convenience function. Uses corpus mode (no Pinecone)."""
    interface = QueryInterface(use_cache=use_cache)
    return interface.run(user_query=user_query, top_k=top_k)


def main():
    """Simple CLI entry point for testing."""
    interface = QueryInterface()
    clarifier = QueryClarifier()

    print("\nReguGrounded — Regulatory Compliance Q&A")
    print("=" * 50)
    user_query = input("Enter compliance question: ").strip()
    if not user_query:
        print("No query provided.")
        return

    try:
        refined_query, was_clarified = clarifier.run_interactive(user_query)
    except Exception:
        refined_query, was_clarified = user_query, False

    if was_clarified:
        print(f"\nRefined question: {refined_query}")

    print("\nProcessing...\n")
    try:
        result = interface.run(user_query=refined_query, top_k=5)
    except GuardrailBlockedError as e:
        print(f"\nQuery rejected: {e}")
        return

    if was_clarified:
        print(f"Original question:  {user_query}")
        print(f"Refined question:   {refined_query}")
    else:
        print(f"Question: {result['question']}")

    print(f"\nSub-questions:")
    for i, sq in enumerate(result["sub_questions"], 1):
        print(f"  {i}. {sq}")

    print(f"\nAnswer:\n{result['answer']}")

    print(f"\nCitations ({len(result['citations'])}):")
    for c in result["citations"]:
        print(f"  - {c['source']} {c['article']}")

    v = result["validation"]
    print(f"\nValidation: {v['accuracy']*100:.1f}% citations valid "
          f"({v['valid_citations']}/{v['total_citations']})")
    if v["invalid_citations"] > 0:
        print(f"  WARNING: {v['invalid_citations']} citation(s) not found in retrieved evidence")

    m = result["metadata"]
    print(f"\nMetadata:")
    print(f"  Jurisdictions: {', '.join(m['jurisdictions_searched']) or 'N/A'}")
    print(f"  Chunks retrieved: {m['total_chunks_retrieved']}")
    print(f"  Latency: {m['latency_ms']} ms")
    if m.get("cache_hit"):
        print("  (served from cache)")
    for w in m.get("guardrail_warnings", []):
        print(f"  [GUARDRAIL WARNING] {w}")


if __name__ == "__main__":
    main()
