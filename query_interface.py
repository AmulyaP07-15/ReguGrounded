# query_interface.py

from rlm_engine import RLMEngine
from reasoning_orchestrator import ReasoningOrchestrator
from answer_synthesizer import AnswerSynthesizer


class QueryInterface:
    """Accepts a user query and runs the reasoning pipeline."""

    def __init__(self, retriever_func):
        self.rlm_engine = RLMEngine()
        self.orchestrator = ReasoningOrchestrator(retriever_func=retriever_func)
        self.synthesizer = AnswerSynthesizer()

    def run(self, user_query: str, top_k: int = 3) -> dict:
        """Run the full pipeline from query to structured answer."""
        decomposition = self.rlm_engine.decompose(user_query)
        evidence_bundle = self.orchestrator.run(decomposition, top_k=top_k)
        final_output = self.synthesizer.synthesize(
            original_query=user_query,
            decomposition=decomposition,
            evidence_bundle=evidence_bundle
        )
        return final_output


def main():
    """Simple CLI entry point for testing."""
    # replace this import with the actual retrieval function later
    from retriever import retrieve

    interface = QueryInterface(retriever_func=retrieve)

    user_query = input("Enter compliance question: ").strip()
    if not user_query:
        print("No query provided.")
        return

    result = interface.run(user_query=user_query, top_k=3)

    print("\nFINAL ANSWER\n")
    print(result["final_answer"])

    print("\nCITED REGULATIONS\n")
    for citation in result["cited_regulations"]:
        print(f"- {citation}")

    print("\nSUPPORTING EVIDENCE\n")
    for snippet in result["supporting_evidence"]:
        print(f"- {snippet}")


if __name__ == "__main__":
    main()