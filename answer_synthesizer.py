# answer_synthesizer.py

from typing import Dict, List, Set


class AnswerSynthesizer:
    """Builds a structured final response from retrieved evidence."""

    def synthesize(
        self,
        original_query: str,
        decomposition: Dict,
        evidence_bundle: List[Dict]
    ) -> Dict:
        """Return final answer, citations, and evidence snippets."""
        final_answer_parts = []
        cited_regulations: Set[str] = set()
        supporting_evidence = []

        for bundle in evidence_bundle:
            jurisdiction = bundle["jurisdiction"]
            sub_question = bundle["sub_question"]
            chunks = bundle["retrieved_chunks"]

            section_answer = self._build_section_answer(
                jurisdiction=jurisdiction,
                sub_question=sub_question,
                chunks=chunks
            )
            final_answer_parts.append(section_answer)

            for chunk in chunks:
                citation = self._format_citation(chunk)
                if citation:
                    cited_regulations.add(citation)

                snippet = self._format_snippet(chunk)
                if snippet:
                    supporting_evidence.append(snippet)

        return {
            "query": original_query,
            "final_answer": "\n\n".join(final_answer_parts).strip(),
            "cited_regulations": sorted(cited_regulations),
            "supporting_evidence": supporting_evidence[:5]
        }

    def _build_section_answer(self, jurisdiction: str, sub_question: str, chunks: List[Dict]) -> str:
        """Create one answer block for one jurisdiction."""
        if not chunks:
            if jurisdiction:
                return f"{jurisdiction}: No supporting evidence was retrieved."
            return "No supporting evidence was retrieved."

        best_chunk = chunks[0]
        best_text = best_chunk["text"]

        if jurisdiction:
            return f"{jurisdiction}: {best_text}"
        return best_text

    def _format_citation(self, chunk: Dict) -> str:
        """Format citation using metadata."""
        law_name = chunk.get("law_name")
        article_number = chunk.get("article_number")
        section_number = chunk.get("section_number")

        if not law_name:
            return ""

        parts = [law_name]

        if article_number:
            parts.append(f"Article {article_number}")

        if section_number:
            parts.append(f"Section {section_number}")

        return " — ".join(parts)

    def _format_snippet(self, chunk: Dict) -> str:
        """Return a short evidence snippet."""
        text = chunk.get("text", "").strip()
        if not text:
            return ""

        if len(text) > 300:
            return text[:300] + "..."
        return text