# agentic_rag_pipeline.py
"""
Truly Agentic RAG Pipeline
==========================
Unlike the RLM pipeline (one fixed decompose→retrieve→synthesize pass),
this pipeline runs a multi-iteration agent loop per jurisdiction:

  PLAN     → agent selects retrieval tool per sub-question
             (semantic / keyword / hybrid) based on query characteristics
  RETRIEVE → execute chosen tool against EnhancedRetriever
  REFLECT  → LLM judges whether evidence is sufficient to answer the question
  ITERATE  → if insufficient: reformulate query, switch tools, retrieve again
             (up to MAX_ITERATIONS rounds)
  SYNTHESIZE → generate grounded answer from all accumulated evidence
  VALIDATE   → check every citation against retrieved chunks

How this differs from the RLM pipeline
---------------------------------------
RLM:      decompose → retrieve (fixed, one pass) → synthesize → validate
Agentic:  decompose → [plan tool → retrieve → reflect → loop?] → synthesize → validate

Key additions:
  1. Dynamic tool selection  — agent picks semantic / keyword / hybrid per query
  2. Evidence reflection     — LLM decides "do I have enough to answer this?"
  3. Iterative re-retrieval  — reformulated query + tool switch if evidence is weak
  4. Agent trace             — full transparency: which tools, how many rounds,
                               what was missing, final evidence quality
  5. EnhancedRetriever       — BM25 + semantic + reranking vs plain Pinecone
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent / ".env")

from answer_synthesizer import AnswerSynthesizer          # noqa: E402
from citation_validator import validate_citations          # noqa: E402
from rlm_engine import RLMEngine                          # noqa: E402
from utils.logger import setup_logger                     # noqa: E402
from utils.rate_limiter import rate_limiter               # noqa: E402

logger = setup_logger("agentic_rag_pipeline")

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_ITERATIONS  = 3     # max retrieval rounds per jurisdiction before giving up
SCORE_THRESHOLD = 0.65  # avg chunk score below this → try again
MIN_CHUNKS      = 2     # fewer chunks than this → definitely try again

_DOC_ID_MAP: Dict[str, str] = {
    "EU AI Act":                          "eu_ai_act",
    "NYC Local Law 144":                  "nyc_local_law_144",
    "Colorado AI Act":                    "colorado_ai_act",
    "NIST AI Risk Management Framework":  "nist_ai_framework",
    "NIST AI RMF":                        "nist_ai_framework",
}


# ── Data classes ──────────────────────────────────────────────────────────────

class RetrievalTool(str, Enum):
    SEMANTIC = "semantic"   # pure vector similarity — best for conceptual questions
    KEYWORD  = "keyword"    # BM25 keyword match — best for exact citations
    HYBRID   = "hybrid"     # fusion of both — best for complex / mixed queries


@dataclass
class AgentIteration:
    iteration:  int
    tool:       RetrievalTool
    tool_reason: str
    query:      str
    chunks_retrieved: int
    new_chunks: int
    avg_score:  float


@dataclass
class ReflectionResult:
    sufficient:         bool
    reasoning:          str
    gap_description:    str
    reformulated_query: str
    suggested_tool:     Optional[RetrievalTool] = None


@dataclass
class AgentState:
    """Tracks the full history of one jurisdiction's agent loop."""
    original_question:  str
    jurisdiction:       str
    current_query:      str
    iterations:         List[AgentIteration]   = field(default_factory=list)
    reflections:        List[ReflectionResult] = field(default_factory=list)
    seen_chunk_ids:     set                    = field(default_factory=set)
    accumulated_chunks: List[Dict]             = field(default_factory=list)

    def add_chunks(self, chunks: List[Dict]) -> int:
        """Deduplicate by chunk_id and accumulate. Returns count of new chunks."""
        added = 0
        for chunk in chunks:
            cid = chunk.get("chunk_id") or chunk.get("text", "")[:80]
            if cid not in self.seen_chunk_ids:
                self.seen_chunk_ids.add(cid)
                self.accumulated_chunks.append(chunk)
                added += 1
        return added

    def avg_score(self) -> float:
        scores = [c.get("score", 0) for c in self.accumulated_chunks if c.get("score")]
        return round(sum(scores) / len(scores), 3) if scores else 0.0

    def to_trace(self) -> Dict:
        return {
            "jurisdiction":     self.jurisdiction or "general",
            "iterations":       len(self.iterations),
            "tools_used":       [it.tool.value for it in self.iterations],
            "tool_reasons":     [it.tool_reason for it in self.iterations],
            "queries_issued":   [it.query for it in self.iterations],
            "chunks_per_iter":  [it.chunks_retrieved for it in self.iterations],
            "new_per_iter":     [it.new_chunks for it in self.iterations],
            "final_chunks":     len(self.accumulated_chunks),
            "final_avg_score":  self.avg_score(),
            "reflections": [
                {
                    "sufficient":      r.sufficient,
                    "reasoning":       r.reasoning,
                    "gap":             r.gap_description,
                    "reformulated_to": r.reformulated_query if not r.sufficient else "",
                }
                for r in self.reflections
            ],
        }


# ── Tool selector ─────────────────────────────────────────────────────────────

class ToolSelector:
    """
    Decides which retrieval tool to use for a given query.

    Rules (in priority order, no LLM needed):
      1. Contains § N or Article N  → KEYWORD  (exact citation lookup)
      2. Comparison / cross-reg query→ HYBRID   (needs broad coverage)
      3. Very short query (≤ 4 words)→ HYBRID   (under-specified, cast wide net)
      4. Reformulated query          → alternate tool from previous iteration
      5. Default                     → SEMANTIC  (conceptual understanding)
    """

    _CITATION_RE   = re.compile(r"§\s*[\d]|article\s+\d+|section\s+\d+", re.IGNORECASE)
    _COMPARISON_KW = {"compare", "vs", "versus", "differ", "contrast", "both", "across"}

    @classmethod
    def select(cls, query: str, previous_tool: Optional[RetrievalTool] = None) -> Tuple[RetrievalTool, str]:
        """
        Returns (tool, reason_string).
        If previous_tool is provided (i.e. this is a retry), we switch tools.
        """
        # On retry, switch to a different tool to get different evidence
        if previous_tool is not None:
            if previous_tool == RetrievalTool.SEMANTIC:
                return RetrievalTool.HYBRID, "retry: switching from semantic to hybrid"
            if previous_tool == RetrievalTool.HYBRID:
                return RetrievalTool.KEYWORD, "retry: switching from hybrid to keyword"
            return RetrievalTool.HYBRID, "retry: switching from keyword to hybrid"

        q = query.lower()

        if cls._CITATION_RE.search(query):
            return RetrievalTool.KEYWORD, "citation pattern detected"

        if any(w in q for w in cls._COMPARISON_KW):
            return RetrievalTool.HYBRID, "comparison/cross-jurisdiction query"

        if len(query.split()) <= 4:
            return RetrievalTool.HYBRID, "short query — cast wide net"

        return RetrievalTool.SEMANTIC, "conceptual question — semantic search"


# ── Evidence reflector ────────────────────────────────────────────────────────

class EvidenceReflector:
    """
    LLM-based judge: is the retrieved evidence sufficient to answer the question?

    Two-tier approach:
      Fast-path: if chunks < MIN_CHUNKS or avg_score < SCORE_THRESHOLD,
                 skip LLM and return insufficient immediately.
      LLM-path:  ask gpt-4o-mini to evaluate evidence quality and identify gaps.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model  = model

    def reflect(
        self,
        original_question: str,
        jurisdiction:      str,
        chunks:            List[Dict],
        iteration:         int,
    ) -> ReflectionResult:
        # Fast-path: no chunks
        if not chunks:
            return ReflectionResult(
                sufficient=False,
                reasoning="No chunks retrieved",
                gap_description="No evidence found — try broader query",
                reformulated_query=f"{original_question} requirements obligations compliance",
            )

        scores = [c.get("score", 0) for c in chunks if c.get("score")]
        avg    = sum(scores) / len(scores) if scores else 0.0

        # Fast-path: low quantity or quality
        if len(chunks) < MIN_CHUNKS or avg < SCORE_THRESHOLD:
            return ReflectionResult(
                sufficient=False,
                reasoning=f"Weak evidence: {len(chunks)} chunks, avg_score={avg:.2f} (threshold={SCORE_THRESHOLD})",
                gap_description="Evidence quality too low — broadening search",
                reformulated_query=self._broaden(original_question, jurisdiction),
            )

        # Last iteration: accept whatever we have
        if iteration >= MAX_ITERATIONS - 1:
            return ReflectionResult(
                sufficient=True,
                reasoning="Max iterations reached — proceeding with available evidence",
                gap_description="",
                reformulated_query="",
            )

        # LLM-path: substantive quality check
        return self._llm_reflect(original_question, jurisdiction, chunks)

    def _llm_reflect(
        self,
        question:     str,
        jurisdiction: str,
        chunks:       List[Dict],
    ) -> ReflectionResult:
        summary = "\n".join(
            f"  [{i+1}] score={c.get('score', 0):.2f} | "
            f"{c.get('law_name', '')} | {c.get('text', '')[:200]}"
            for i, c in enumerate(chunks[:5])
        )

        prompt = f"""You are evaluating retrieved regulatory evidence for a compliance question.

Question: {question}
Target jurisdiction: {jurisdiction or "any regulation"}

Retrieved evidence ({len(chunks)} chunks, showing top 5):
{summary}

Evaluate whether this evidence is sufficient to write a complete, accurate answer.

Return JSON with these exact keys:
{{
  "sufficient": true or false,
  "reasoning": "1-2 sentences explaining your judgment",
  "gap_description": "what specific information is missing, or 'none' if sufficient",
  "reformulated_query": "a better search query targeting the gap, or empty string if sufficient",
  "suggested_tool": "semantic" or "keyword" or "hybrid" — which retrieval mode would best fill the gap
}}

Be strict: only mark sufficient=true if the evidence directly addresses the question with specific provisions."""

        try:
            rate_limiter.acquire(estimated_tokens=700)
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            tool_str = data.get("suggested_tool", "")
            try:
                suggested = RetrievalTool(tool_str) if tool_str else None
            except ValueError:
                suggested = None
            return ReflectionResult(
                sufficient=bool(data.get("sufficient", False)),
                reasoning=data.get("reasoning", ""),
                gap_description=data.get("gap_description", ""),
                reformulated_query=data.get("reformulated_query", ""),
                suggested_tool=suggested,
            )
        except Exception as e:
            logger.warning(f"Reflection LLM call failed ({e}); treating as sufficient")
            return ReflectionResult(
                sufficient=True,
                reasoning="Reflection unavailable — proceeding with retrieved evidence",
                gap_description="",
                reformulated_query="",
            )

    @staticmethod
    def _broaden(question: str, jurisdiction: str) -> str:
        base   = question.rstrip(".")
        suffix = f" {jurisdiction}" if jurisdiction else ""
        return f"{base} requirements obligations compliance{suffix}"


# ── Main pipeline ─────────────────────────────────────────────────────────────

class AgenticRAGPipeline:
    """
    Truly agentic RAG pipeline.

    Each jurisdiction gets its own independent agent loop running in parallel:

      for each sub-question (in parallel):
          iteration 1..MAX_ITERATIONS:
              tool = ToolSelector.select(current_query, previous_tool)
              chunks = EnhancedRetriever.retrieve(current_query, mode=tool)
              reflection = EvidenceReflector.reflect(question, jurisdiction, chunks)
              if reflection.sufficient: break
              current_query = reflection.reformulated_query

      synthesize(all accumulated chunks)
      validate_citations(answer, all_chunks)
    """

    def __init__(self, retriever):
        self.retriever   = retriever
        self.rlm         = RLMEngine()
        self.synthesizer = AnswerSynthesizer()
        self.reflector   = EvidenceReflector()

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, user_query: str, top_k: int = 5) -> Dict:
        """
        Run the full agentic pipeline. Returns the same schema as
        QueryInterface.run() plus an 'agent_trace' field in metadata.
        """
        t_start = time.time()

        # 1. Decompose (shared with RLM — jurisdiction detection is the same)
        try:
            decomposition = self.rlm.decompose(user_query)
        except Exception as e:
            logger.warning(f"Decomposition failed ({e}); using fallback")
            decomposition = {
                "original_query": user_query,
                "jurisdictions":  [],
                "sub_questions":  [{"jurisdiction": None, "sub_question": user_query}],
            }

        sub_questions = decomposition["sub_questions"]
        jurisdictions = decomposition["jurisdictions"]

        # 2. Agent loops — one per sub-question, all in parallel
        agent_states: Dict[int, AgentState] = {}

        def _loop(idx: int, item: Dict) -> Tuple[int, AgentState]:
            return idx, self._agent_loop(
                original_question=item["sub_question"],
                jurisdiction=item.get("jurisdiction"),
                top_k=top_k,
            )

        with ThreadPoolExecutor(max_workers=max(len(sub_questions), 1)) as pool:
            futures = {pool.submit(_loop, i, item): i for i, item in enumerate(sub_questions)}
            for future in as_completed(futures):
                idx, state = future.result()
                agent_states[idx] = state

        # 3. Build evidence bundle
        evidence_bundle = []
        for i, item in enumerate(sub_questions):
            state = agent_states[i]
            evidence_bundle.append({
                "jurisdiction":      item.get("jurisdiction"),
                "sub_question":      item["sub_question"],
                "retrieved_chunks":  state.accumulated_chunks,
            })

        all_chunks = [c for b in evidence_bundle for c in b["retrieved_chunks"]]

        logger.info(
            f"Agentic pipeline: {len(sub_questions)} jurisdiction(s), "
            f"{sum(len(s.iterations) for s in agent_states.values())} total retrieval rounds, "
            f"{len(all_chunks)} total chunks"
        )

        # 4. Synthesize
        synthesis = self.synthesizer.synthesize(
            original_query=user_query,
            decomposition=decomposition,
            evidence_bundle=evidence_bundle,
        )

        # 5. Validate citations
        validation_raw = validate_citations(synthesis["answer"], all_chunks)

        latency_ms = int((time.time() - t_start) * 1000)

        return {
            "question":      user_query,
            "sub_questions": [item["sub_question"] for item in sub_questions],
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
                "pipeline":                "agentic_rag",
                # Agent trace — not present in RLM pipeline output
                "agent_trace": [agent_states[i].to_trace() for i in range(len(sub_questions))],
            },
        }

    # ── Agent loop (per jurisdiction) ─────────────────────────────────────────

    def _agent_loop(
        self,
        original_question: str,
        jurisdiction:      Optional[str],
        top_k:             int,
    ) -> AgentState:
        """
        Retrieval-reflection loop for one jurisdiction.

        Each iteration:
          1. Select tool (switches on retry to get different evidence)
          2. Retrieve
          3. Reflect
          4. If sufficient → stop. If not → reformulate, loop.
        """
        state      = AgentState(
            original_question=original_question,
            jurisdiction=jurisdiction or "",
            current_query=original_question,
        )
        doc_filter   = self._jurisdiction_filter(jurisdiction)
        previous_tool: Optional[RetrievalTool] = None

        for iteration in range(MAX_ITERATIONS):
            # 1. Select tool
            tool, tool_reason = ToolSelector.select(
                state.current_query,
                previous_tool=previous_tool if iteration > 0 else None,
            )
            previous_tool = tool

            logger.info(
                f"[{jurisdiction or 'general'}] iter={iteration+1}/{MAX_ITERATIONS} "
                f"tool={tool.value} ({tool_reason}) "
                f"query={state.current_query[:60]!r}"
            )

            # 2. Retrieve
            raw_chunks = self._retrieve(state.current_query, tool, top_k, doc_filter)
            normalized = self._normalize(raw_chunks)
            new_count  = state.add_chunks(normalized)

            scores  = [c.get("score", 0) for c in normalized if c.get("score")]
            avg     = round(sum(scores) / len(scores), 3) if scores else 0.0

            state.iterations.append(AgentIteration(
                iteration=iteration + 1,
                tool=tool,
                tool_reason=tool_reason,
                query=state.current_query,
                chunks_retrieved=len(normalized),
                new_chunks=new_count,
                avg_score=avg,
            ))

            logger.info(
                f"[{jurisdiction or 'general'}] iter={iteration+1} → "
                f"{len(normalized)} chunks ({new_count} new), avg_score={avg}"
            )

            # 3. Reflect
            reflection = self.reflector.reflect(
                original_question=original_question,
                jurisdiction=jurisdiction or "",
                chunks=state.accumulated_chunks,
                iteration=iteration,
            )
            state.reflections.append(reflection)

            logger.info(
                f"[{jurisdiction or 'general'}] reflection: sufficient={reflection.sufficient} | "
                f"{reflection.reasoning}"
            )

            # 4. Decide whether to continue
            if reflection.sufficient or iteration == MAX_ITERATIONS - 1:
                if not reflection.sufficient:
                    logger.warning(
                        f"[{jurisdiction or 'general'}] Max iterations reached — "
                        f"proceeding with {len(state.accumulated_chunks)} chunks"
                    )
                break

            # Use reflector's suggested tool if available
            if reflection.suggested_tool:
                previous_tool = reflection.suggested_tool

            new_query = reflection.reformulated_query or state.current_query
            if new_query == state.current_query:
                logger.debug(f"[{jurisdiction or 'general'}] Query unchanged — stopping early")
                break
            state.current_query = new_query

        return state

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _retrieve(
        self,
        query:      str,
        tool:       RetrievalTool,
        top_k:      int,
        doc_filter: Optional[Dict],
    ) -> List[Dict]:
        """Call EnhancedRetriever with the selected search_mode."""
        try:
            return self.retriever.retrieve(
                query,
                top_k=top_k,
                filter=doc_filter,
                search_mode=tool.value,
            )
        except TypeError:
            # Fallback for retrievers that don't accept search_mode
            return self.retriever.retrieve(query, top_k=top_k, filter=doc_filter)
        except Exception as e:
            logger.error(f"Retrieval error (tool={tool.value}): {e}")
            return []

    @staticmethod
    def _jurisdiction_filter(jurisdiction: Optional[str]) -> Optional[Dict]:
        if not jurisdiction:
            return None
        doc_id = _DOC_ID_MAP.get(jurisdiction)
        return {"doc_id": doc_id} if doc_id else None

    @staticmethod
    def _normalize(chunks: List[Dict]) -> List[Dict]:
        """Normalise EnhancedRetriever output to the pipeline's standard format."""
        result = []
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            result.append({
                "text":           chunk.get("text", "").strip(),
                "law_name":       meta.get("law_name")       or chunk.get("law_name"),
                "article_number": meta.get("article_number") or chunk.get("article_number"),
                "section_number": meta.get("section_number") or chunk.get("section_number"),
                "chunk_id":       meta.get("chunk_id")       or chunk.get("chunk_id"),
                "score":          chunk.get("score"),
                "search_mode":    chunk.get("search_mode", "unknown"),
            })
        return result


# ── Module-level API ──────────────────────────────────────────────────────────

_pipeline_instance: Optional[AgenticRAGPipeline] = None


def _get_pipeline() -> AgenticRAGPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        from src.enhanced_retriever import EnhancedRetriever
        from src.retrieval_config import DEFAULT_CONFIG
        retriever = EnhancedRetriever(config=DEFAULT_CONFIG)
        _pipeline_instance = AgenticRAGPipeline(retriever=retriever)
    return _pipeline_instance


def process_query(question: str, top_k: int = 5) -> Dict:
    """
    Module-level entry point.

    Drop-in replacement for query_interface.process_query — same return schema,
    plus 'agent_trace' in metadata showing every tool choice, iteration, and
    reflection the agent made.
    """
    return _get_pipeline().run(question, top_k=top_k)
