# rlm_engine.py

from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


class RLMEngine:
    """
    decomposes a compliance query into jurisdiction-specific sub-questions
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.known_jurisdictions = [
            "EU AI Act",
            "NYC Local Law 144",
            "Colorado AI Act",
            "NIST AI Risk Management Framework",
            "NIST AI RMF"
        ]

        # llm client
        self.client = OpenAI()
        self.model = model

    def decompose(self, user_query: str) -> Dict:
        """
        return structured decomposition of a query
        """

        jurisdictions = self._identify_jurisdictions(user_query)

        # if jurisdictions detected, use rule-based decomposition
        if jurisdictions:
            sub_questions = self._build_sub_questions(user_query, jurisdictions)

        # otherwise ask llm to identify jurisdictions
        else:
            llm_result = self._llm_decompose(user_query)
            jurisdictions = llm_result["jurisdictions"]
            sub_questions = llm_result["sub_questions"]

        return {
            "original_query": user_query,
            "jurisdictions": jurisdictions,
            "sub_questions": sub_questions
        }

    def _identify_jurisdictions(self, user_query: str) -> List[str]:
        """
        detect known regulations mentioned in the query
        """

        found = []
        query_lower = user_query.lower()

        for jurisdiction in self.known_jurisdictions:
            if jurisdiction.lower() in query_lower:
                found.append(jurisdiction)

        # remove duplicate naming
        if "NIST AI RMF" in found and "NIST AI Risk Management Framework" in found:
            found.remove("NIST AI RMF")

        return found

    def _build_sub_questions(self, user_query: str, jurisdictions: List[str]) -> List[Dict]:
        """
        create one sub-question per jurisdiction
        """

        if not jurisdictions:
            return [
                {
                    "jurisdiction": None,
                    "sub_question": user_query
                }
            ]

        sub_questions = []

        for jurisdiction in jurisdictions:
            sub_questions.append(
                {
                    "jurisdiction": jurisdiction,
                    "sub_question": f"{user_query} Focus only on {jurisdiction}."
                }
            )

        return sub_questions

    def _llm_decompose(self, user_query: str) -> Dict:
        """
        use llm to detect jurisdictions and produce sub-questions
        """

        prompt = f"""
Break the following regulatory compliance question into jurisdiction-specific sub-questions.

Known regulations include:
- EU AI Act
- NYC Local Law 144
- Colorado AI Act
- NIST AI Risk Management Framework

Return JSON with:
jurisdictions: list of regulations referenced
sub_questions: list of objects with fields
    jurisdiction
    sub_question

User Query:
{user_query}
"""

        response = self.client.responses.create(
            model=self.model,
            input=prompt
        )

        text_output = response.output_text.strip()

        # fallback simple parsing if needed
        try:
            import json
            parsed = json.loads(text_output)
            return parsed
        except Exception:
            return {
                "jurisdictions": [],
                "sub_questions": [
                    {
                        "jurisdiction": None,
                        "sub_question": user_query
                    }
                ]
            }