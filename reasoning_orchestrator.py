# reasoning_orchestrator.py

from typing import Dict, List


class ReasoningOrchestrator:
    """Runs retrieval for each sub-question and organizes results."""

    def __init__(self, retriever_func):
        self.retriever_func = retriever_func

    def run(self, decomposition: Dict, top_k: int = 3) -> List[Dict]:
        """Retrieve evidence for each sub-question."""
        evidence_bundle = []

        for item in decomposition["sub_questions"]:
            jurisdiction = item["jurisdiction"]
            sub_question = item["sub_question"]

            retrieved_chunks = self.retriever_func(sub_question, top_k=top_k)

            normalized_chunks = self._normalize_chunks(retrieved_chunks)

            evidence_bundle.append(
                {
                    "jurisdiction": jurisdiction,
                    "sub_question": sub_question,
                    "retrieved_chunks": normalized_chunks
                }
            )

        return evidence_bundle

    def _normalize_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Ensure all retrieved chunks follow one consistent format."""
        normalized = []

        for chunk in chunks:
            text = chunk.get("text", "").strip()
            metadata = chunk.get("metadata", {})
            score = chunk.get("score", None)

            normalized.append(
                {
                    "text": text,
                    "law_name": metadata.get("law_name"),
                    "article_number": metadata.get("article_number"),
                    "section_number": metadata.get("section_number"),
                    "chunk_id": metadata.get("chunk_id"),
                    "score": score
                }
            )

        return normalized