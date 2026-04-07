# query_interface.py

import time

from rlm_engine import RLMEngine
from reasoning_orchestrator import ReasoningOrchestrator
from answer_synthesizer import AnswerSynthesizer
from citation_validator import validate_citations
from query_clarifier import QueryClarifier
from utils.logger import setup_logger, log_function_call
from utils.metrics import MetricsTracker
from utils.cache import QueryCache
from utils.guardrails import InputGuard, OutputGuard, GuardrailBlockedError

logger       = setup_logger("query_interface")
metrics      = MetricsTracker()
_input_guard  = InputGuard()
_output_guard = OutputGuard()


class QueryInterface:
    """Accepts a user query and runs the full reasoning pipeline."""

    def __init__(self, retriever_func, use_cache: bool = False):
        self.rlm_engine   = RLMEngine()
        self.orchestrator = ReasoningOrchestrator(retriever_func=retriever_func)
        self.synthesizer  = AnswerSynthesizer()
        self.use_cache    = use_cache
        # Per-instance cache — keeps test runs isolated from each other
        self._cache       = QueryCache() if use_cache else None

    @staticmethod
    def _adaptive_top_k(decomposition: dict, base_top_k: int) -> int:
        """Scale top_k with the number of jurisdictions to ensure enough evidence per source."""
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
                                    (injection attempt, query too long/short, etc.)
        """
        t_start = time.time()

        # ── Input guardrails (runs before any LLM call) ───────────────
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
                # Stamp the live cache_hit flag — the stored value is always False
                cached["metadata"]["cache_hit"] = True
                return cached
            metrics.record_cache_miss("query")
        cacheable = self.use_cache and self._cache is not None

        try:
            result = self._run_pipeline(user_query, top_k, t_start)

            # ── Output guardrails (runs after synthesis) ──────────────
            output_result   = _output_guard.run(result)
            all_warnings    = input_warnings + output_result.warning_strings()
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

        # 2. Retrieve — scale top_k for multi-jurisdiction queries
        effective_top_k  = self._adaptive_top_k(decomposition, top_k)
        evidence_bundle  = self.orchestrator.run(decomposition, top_k=effective_top_k)
        all_chunks       = self.orchestrator.get_all_chunks(evidence_bundle)
        jurisdictions    = self.orchestrator.get_jurisdictions(evidence_bundle)

        logger.info(
            f"Retrieved {len(all_chunks)} chunks across "
            f"{len(jurisdictions)} jurisdiction(s): {jurisdictions}"
        )

        # 3. Synthesize
        synthesis = self.synthesizer.synthesize(
            original_query=user_query,
            decomposition=decomposition,
            evidence_bundle=evidence_bundle,
        )

        # 4. Validate citations
        validation_raw = validate_citations(synthesis["answer"], all_chunks)

        if validation_raw["hallucination_rate"] > 0:
            logger.warning(
                f"Citation issues detected: {validation_raw['invalid_count']} invalid "
                f"({validation_raw['hallucination_rate']*100:.1f}% hallucination rate)"
            )

        latency_ms = int((time.time() - t_start) * 1000)
        logger.info(f"Pipeline completed in {latency_ms} ms")

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
                "jurisdictions_searched": jurisdictions,
                "total_chunks_retrieved": len(all_chunks),
                "latency_ms":             latency_ms,
                "cache_hit":              False,
                "guardrail_warnings":     [],   # populated after output guard runs
            },
        }


_retriever_instance = None


def _get_retriever():
    """Lazy singleton — loads the embedding model once per process."""
    global _retriever_instance
    if _retriever_instance is None:
        from src.retriever import Retriever
        _retriever_instance = Retriever()
    return _retriever_instance


def process_query(user_query: str, top_k: int = 5, use_cache: bool = True) -> dict:
    """Module-level convenience function matching the example usage in the spec."""
    interface = QueryInterface(
        retriever_func=_get_retriever().retrieve,
        use_cache=use_cache,
    )
    return interface.run(user_query=user_query, top_k=top_k)


def main():
    """Simple CLI entry point for testing."""
    from src.retriever import retrieve

    interface = QueryInterface(retriever_func=_get_retriever().retrieve)
    clarifier = QueryClarifier()

    print("\nReguGrounded — Regulatory Compliance Q&A")
    print("=" * 50)
    user_query = input("Enter compliance question: ").strip()
    if not user_query:
        print("No query provided.")
        return

    # Clarification layer — ask up to 2 questions to sharpen the query
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

    # Show original alongside refined so the user can see what changed
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
