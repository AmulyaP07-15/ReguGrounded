# answer_synthesizer.py

import os
from typing import Dict, List

from openai import OpenAI
from pathlib import Path

from dotenv import load_dotenv

from utils.logger import setup_logger, log_function_call, log_errors
from utils.rate_limiter import rate_limiter

load_dotenv(Path(__file__).parent / ".env")

logger = setup_logger("answer_synthesizer")


class AnswerSynthesizer:
    """Uses an LLM to generate a grounded answer with inline citations from retrieved evidence."""

    def __init__(self, model: str = "gemini-2.0-flash"):
        self.client = OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        self.model = model

    @log_function_call(logger)
    def synthesize(
        self,
        original_query: str,
        decomposition: Dict,
        evidence_bundle: List[Dict],
    ) -> Dict:
        """
        Generate a natural-language answer with inline citations.

        Returns dict with keys: question, sub_questions, answer, citations.
        """
        all_chunks = []
        for bundle in evidence_bundle:
            all_chunks.extend(bundle["retrieved_chunks"])

        citations = self._build_citations(all_chunks)
        logger.info(f"Built {len(citations)} citations from {len(all_chunks)} chunks")
        answer = self._llm_synthesize(original_query, evidence_bundle, citations)

        sub_questions = [item["sub_question"] for item in decomposition["sub_questions"]]

        return {
            "question": original_query,
            "sub_questions": sub_questions,
            "answer": answer,
            "citations": citations,
            "all_chunks": all_chunks,  # passed through for validator; stripped in query_interface
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_citations(self, chunks: List[Dict]) -> List[Dict]:
        """Build deduplicated structured citation list from chunk metadata."""
        seen = set()
        citations = []

        for chunk in chunks:
            law_name = chunk.get("law_name", "")
            article = chunk.get("article_number")
            section = chunk.get("section_number")
            if not law_name:
                continue

            ref = article or section or ""
            key = (law_name, ref)
            if key in seen:
                continue
            seen.add(key)

            article_label = ""
            if article:
                article_label = f"Article {article}"
            elif section:
                article_label = f"§ {section}"

            # Skip chunks that have no identifiable article or section —
            # they produce empty citation labels the LLM can't use.
            if not article_label:
                continue

            text = chunk.get("text", "")
            excerpt = text[:250].strip() + ("..." if len(text) > 250 else "")

            citations.append({
                "source": law_name,
                "article": article_label,
                "excerpt": excerpt,
                "jurisdiction": law_name,
            })

        return citations

    def _llm_synthesize(
        self,
        original_query: str,
        evidence_bundle: List[Dict],
        citations: List[Dict],
    ) -> str:
        """Call the LLM to write the final answer grounded in retrieved evidence."""
        evidence_text = self._format_evidence_for_prompt(evidence_bundle)
        citation_refs = self._format_citation_refs(citations)

        prompt = f"""You are a regulatory compliance expert. Answer the user's question using ONLY the provided regulatory evidence below.

Rules:
- Always cite the FULL law name + section/article together inline.
  CORRECT:   "NYC Local Law 144 § 20-871 requires a bias audit..."
  CORRECT:   "EU AI Act Article 9 establishes a risk management system..."
  INCORRECT: "§ 20-871 requires..."   ← missing law name
  INCORRECT: "Article 9 establishes..." ← missing law name
- Only use citations from the Available Citations list below. Do not invent citations.
- If multiple jurisdictions apply, address each one in turn.
- Be concise and precise. If the evidence is insufficient, say so clearly.

Examples of correct inline citation style:
  "NYC Local Law 144 § 20-871 requires employers to conduct a bias audit no more than one year prior to use."
  "EU AI Act Article 9 mandates a risk management system throughout the AI lifecycle."
  "Colorado AI Act § 6-1-1707 requires developers to conduct annual impact assessments."

User Question:
{original_query}

Available Citations:
{citation_refs}

Retrieved Evidence:
{evidence_text}

Answer:"""

        try:
            rate_limiter.acquire(estimated_tokens=1500)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")
            return self._fallback_answer(original_query, evidence_bundle, citations)

    def _format_evidence_for_prompt(self, evidence_bundle: List[Dict]) -> str:
        lines = []
        for bundle in evidence_bundle:
            juris = bundle.get("jurisdiction") or "General"
            lines.append(f"\n[{juris}]")
            for i, chunk in enumerate(bundle["retrieved_chunks"], 1):
                law = chunk.get("law_name", "")
                article = chunk.get("article_number")
                section = chunk.get("section_number")
                loc = f"Article {article}" if article else (f"§ {section}" if section else "")
                ref = f"{law} {loc}".strip()
                lines.append(f"  ({i}) {ref}: {chunk['text'][:400]}")
        return "\n".join(lines)

    def _fallback_answer(
        self,
        original_query: str,
        evidence_bundle: List[Dict],
        citations: List[Dict],
    ) -> str:
        """
        Structured fallback when the LLM is unavailable.
        Produces a readable summary instead of raw chunk text.
        """
        if not any(b["retrieved_chunks"] for b in evidence_bundle):
            return "No relevant evidence was retrieved for this question."

        parts = [f"Based on retrieved regulatory evidence for: '{original_query}'\n"]

        for bundle in evidence_bundle:
            juris = bundle.get("jurisdiction") or "General"
            chunks = bundle["retrieved_chunks"]
            if not chunks:
                parts.append(f"{juris}: No relevant chunks retrieved.")
                continue

            best = chunks[0]
            law   = best.get("law_name", juris)
            art   = best.get("article_number")
            sec   = best.get("section_number")
            loc   = f"Article {art}" if art else (f"§ {sec}" if sec else "")
            ref   = f"{law} {loc}".strip()
            text  = best["text"][:400].strip()
            parts.append(f"{ref}:\n{text}")

        if citations:
            cite_labels = ", ".join(
                f"{c['source']} {c['article']}".strip() for c in citations
            )
            parts.append(f"\nRelevant sources: {cite_labels}")

        parts.append("\n(Note: answer generated from retrieved text — LLM synthesis unavailable.)")
        return "\n\n".join(parts)

    def _format_citation_refs(self, citations: List[Dict]) -> str:
        if not citations:
            return "  (none)"
        lines = []
        for c in citations:
            label = f"{c['source']} {c['article']}".strip()
            lines.append(f"  - {label}")
        return "\n".join(lines)
