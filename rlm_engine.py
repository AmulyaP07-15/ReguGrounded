# rlm_engine.py

import json
import os
from typing import List, Dict

from openai import OpenAI
from pathlib import Path

from dotenv import load_dotenv

from utils.logger import setup_logger, log_function_call
from utils.cache import ComponentCache
from utils.rate_limiter import rate_limiter

load_dotenv(Path(__file__).parent / ".env")

logger = setup_logger("rlm_engine")
_component_cache = ComponentCache()

# Keyword → jurisdiction hints used when explicit names are absent and LLM fails
_KEYWORD_HINTS: Dict[str, List[str]] = {
    "bias audit":        ["NYC Local Law 144", "Colorado AI Act"],
    "hiring":            ["NYC Local Law 144", "Colorado AI Act"],
    "employment":        ["NYC Local Law 144"],
    "automated employ":  ["NYC Local Law 144"],
    "notice":            ["NYC Local Law 144"],
    "high-risk":         ["EU AI Act", "Colorado AI Act"],
    "high risk":         ["EU AI Act", "Colorado AI Act"],
    "conformity":        ["EU AI Act"],
    "prohibited":        ["EU AI Act"],
    "transparency":      ["EU AI Act"],
    "post-market":       ["EU AI Act"],
    "general purpose":   ["EU AI Act"],
    "general-purpose":   ["EU AI Act"],
    "impact assessment": ["Colorado AI Act"],
    "consumer":          ["Colorado AI Act"],
    "developer":         ["Colorado AI Act"],
    "govern":            ["NIST AI Risk Management Framework"],
    "trustworthy":       ["NIST AI Risk Management Framework"],
    "risk management":   ["EU AI Act", "NIST AI Risk Management Framework"],
    "risk framework":    ["NIST AI Risk Management Framework"],
    "map function":      ["NIST AI Risk Management Framework"],
    "measure":           ["NIST AI Risk Management Framework"],
}


class RLMEngine:
    """Decomposes a compliance query into jurisdiction-specific sub-questions."""

    def __init__(self, model: str = "gemini-2.0-flash"):
        self.known_jurisdictions = [
            "EU AI Act",
            "NYC Local Law 144",
            "Colorado AI Act",
            "NIST AI Risk Management Framework",
            "NIST AI RMF",
        ]
        self.client = OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        self.model = model

    @log_function_call(logger)
    def decompose(self, user_query: str) -> Dict:
        """Return structured decomposition of a query."""
        # Check component cache first
        cached = _component_cache.get_decomposition(user_query)
        if cached is not None:
            logger.info(f"Decomposition cache hit: {user_query[:60]!r}")
            return cached

        result = self._decompose_uncached(user_query)
        _component_cache.set_decomposition(user_query, result)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decompose_uncached(self, user_query: str) -> Dict:
        # 1. Try explicit mention detection (fast, no LLM)
        jurisdictions = self._identify_jurisdictions(user_query)

        if jurisdictions:
            logger.info(f"Explicit jurisdiction match: {jurisdictions}")
            sub_questions = self._build_sub_questions(user_query, jurisdictions)
        else:
            # 2. Try LLM decomposition
            try:
                llm_result    = self._llm_decompose(user_query)
                jurisdictions = llm_result.get("jurisdictions", [])
                sub_questions = llm_result.get("sub_questions", [])
                logger.info(f"LLM decomposition → jurisdictions: {jurisdictions}")
            except Exception as e:
                logger.warning(f"LLM decomposition failed ({e}); falling back to keyword hints")
                jurisdictions = []
                sub_questions = []

            # 3. If LLM produced nothing useful, fall back to keyword hints
            if not sub_questions:
                jurisdictions = self._identify_jurisdictions_from_keywords(user_query)
                sub_questions = self._build_sub_questions(user_query, jurisdictions)
                logger.info(f"Keyword-hint fallback → jurisdictions: {jurisdictions}")

        return {
            "original_query": user_query,
            "jurisdictions":  jurisdictions,
            "sub_questions":  sub_questions,
        }

    def _identify_jurisdictions(self, user_query: str) -> List[str]:
        """Detect known regulation names explicitly mentioned in the query."""
        found = []
        query_lower = user_query.lower()

        for jurisdiction in self.known_jurisdictions:
            if jurisdiction.lower() in query_lower:
                found.append(jurisdiction)

        # Deduplicate NIST aliases
        if "NIST AI RMF" in found and "NIST AI Risk Management Framework" in found:
            found.remove("NIST AI RMF")

        return found

    def _identify_jurisdictions_from_keywords(self, user_query: str) -> List[str]:
        """
        Infer likely jurisdictions from topic keywords when no regulation name
        is mentioned explicitly and the LLM is unavailable.
        """
        query_lower = user_query.lower()
        matched: List[str] = []
        seen = set()

        for keyword, jurisdictions in _KEYWORD_HINTS.items():
            if keyword in query_lower:
                for j in jurisdictions:
                    if j not in seen:
                        seen.add(j)
                        matched.append(j)

        # Deduplicate NIST aliases
        if "NIST AI RMF" in matched and "NIST AI Risk Management Framework" in matched:
            matched.remove("NIST AI RMF")

        logger.info(f"Keyword hint jurisdictions for '{user_query[:60]}': {matched}")
        return matched

    def _build_sub_questions(self, user_query: str, jurisdictions: List[str]) -> List[Dict]:
        """Create one sub-question per jurisdiction."""
        if not jurisdictions:
            return [{"jurisdiction": None, "sub_question": user_query}]

        return [
            {
                "jurisdiction": j,
                "sub_question": f"{user_query} Focus only on {j}.",
            }
            for j in jurisdictions
        ]

    def _llm_decompose(self, user_query: str) -> Dict:
        """Use LLM to detect jurisdictions and produce sub-questions."""
        prompt = f"""Break the following regulatory compliance question into jurisdiction-specific sub-questions.

Known regulations:
- EU AI Act
- NYC Local Law 144
- Colorado AI Act
- NIST AI Risk Management Framework

Return JSON with exactly these keys:
  "jurisdictions": list of regulation names that are relevant
  "sub_questions": list of objects, each with "jurisdiction" (string) and "sub_question" (string)

User Query: {user_query}"""

        rate_limiter.acquire(estimated_tokens=600)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        text_output = response.choices[0].message.content.strip()
        return json.loads(text_output)
