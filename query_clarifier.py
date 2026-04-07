# query_clarifier.py
#
# Sits between the raw user query and the reasoning pipeline.
# Assesses whether the query is ambiguous, asks up to 2 targeted clarifying
# questions, then uses the answers to produce a richer, more specific query.
#
# Only used in interactive (CLI) mode — QueryInterface.run() is unchanged
# so evals and programmatic callers are unaffected.

import json
import logging
import re
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

from utils.rate_limiter import rate_limiter

load_dotenv()

logger = logging.getLogger("query_clarifier")

# ---------------------------------------------------------------------------
# Rule-based fast-path signals
# ---------------------------------------------------------------------------

# Any of these strings appearing in the query means jurisdiction is already clear
_REGULATION_KEYWORDS = [
    "eu ai act",
    "nyc local law 144", "nyc ll144", "nyc ll 144", "local law 144",
    "colorado ai act",
    "nist ai", "nist rmf", "nist framework", "nist ai risk",
]

# These words indicate a comparison/contrast query — user knows what they want
_COMPARISON_KEYWORDS = ["compare", "comparison", "differ", "difference between",
                        " vs ", " vs.", "versus", "contrast"]

# §-style or Article N references are already specific enough
_SECTION_PATTERN  = re.compile(r'§\s*[\d][\d\w-]+', re.IGNORECASE)
_ARTICLE_PATTERN  = re.compile(r'\barticle\s+\d+\b', re.IGNORECASE)

# NIST function codes are already specific (GOVERN 1.1, MAP 2.1, etc.)
_NIST_FN_PATTERN  = re.compile(
    r'\b(govern|map|measure|manage)\s+\d+\.\d+\b', re.IGNORECASE
)

# ---------------------------------------------------------------------------
# LLM prompts — only reached when the fast path doesn't fire
# ---------------------------------------------------------------------------

_ASSESS_PROMPT = """\
You are a regulatory compliance assistant deciding whether to ask the user a \
clarifying question before searching.

The system covers:
  - EU AI Act
  - NYC Local Law 144 (automated employment decision tools, bias audits)
  - Colorado AI Act (high-risk AI, developer/deployer obligations)
  - NIST AI Risk Management Framework (GOVERN, MAP, MEASURE, MANAGE)

Ask a clarifying question ONLY when the answer would be fundamentally different \
depending on something the user hasn't specified. Specifically:
  • JURISDICTION — when no regulation is named and the topic spans multiple laws
  • ROLE — when employer vs developer vs consumer obligations differ significantly
  • USE CASE — when the industry/application (hiring, healthcare, etc.) changes scope

Do NOT ask when:
  • Any specific regulation is already named (even partially)
  • The query already names a specific section, article, or provision
  • The question is a comparison across jurisdictions
  • The question is detailed enough that jurisdiction can be inferred
  • Asking would be pedantic — the system can search broadly and still give a good answer

--- EXAMPLES OF CORRECT DECISIONS ---

Query: "What are the requirements?"
→ {{"needs_clarification": true, "questions": ["Which regulation are you asking about? (EU AI Act, NYC LL144, Colorado AI Act, or NIST?)", "What is your role? (employer, AI developer, deployer, or general research?)"], "reasoning": "No regulation or role specified."}}

Query: "What notice do I need to give?"
→ {{"needs_clarification": true, "questions": ["Which law are you asking about?", "Are you an employer giving notice to candidates, or a developer notifying users?"], "reasoning": "Notice requirements differ by law and role."}}

Query: "What do I need to do about AI bias?"
→ {{"needs_clarification": true, "questions": ["Which regulation applies to your situation?"], "reasoning": "Bias obligations vary significantly across jurisdictions."}}

Query: "What are bias audit requirements?"
→ {{"needs_clarification": true, "questions": ["Which jurisdiction? NYC Local Law 144, Colorado AI Act, or both?"], "reasoning": "Bias audit rules differ between NYC and Colorado."}}

Query: "What are the high-risk AI requirements?"
→ {{"needs_clarification": true, "questions": ["Which regulation? EU AI Act, Colorado AI Act, or both?"], "reasoning": "High-risk classification differs between EU and Colorado."}}

Query: "What does the Colorado AI Act require from developers?"
→ {{"needs_clarification": false, "questions": [], "reasoning": "Regulation and role (developer) are both specified."}}

Query: "What transparency requirements apply under the EU AI Act?"
→ {{"needs_clarification": false, "questions": [], "reasoning": "Regulation is named and topic is specific."}}

Query: "What are the post-market monitoring obligations under EU AI Act?"
→ {{"needs_clarification": false, "questions": [], "reasoning": "Specific topic within a named regulation."}}

Query: "How do bias audit requirements differ between NYC and Colorado?"
→ {{"needs_clarification": false, "questions": [], "reasoning": "Comparison query with both jurisdictions named."}}

--- END EXAMPLES ---

Generate AT MOST 2 questions. Prefer 1 if that's enough. \
Questions should be short and offer concrete choices where possible.

Return ONLY valid JSON:
{{
  "needs_clarification": true or false,
  "questions": [],
  "reasoning": "one sentence"
}}

User question: {query}"""

_REFINE_PROMPT = """\
A user asked a regulatory compliance question and answered some clarifying questions. \
Rewrite the original question as a single precise query that incorporates the answers \
so a regulatory search engine retrieves the most relevant results.

Rules:
- Keep the core question intact; only add specificity from the answers
- Include the jurisdiction(s) if the user specified them
- Include the role/perspective if relevant
- Output ONLY the refined question — no preamble, no explanation, no quotes

Original question: {original_query}

Clarifications:
{qa_text}

Refined question:"""


class QueryClarifier:
    """
    Assesses query ambiguity, asks targeted questions, and returns a refined query.

    A two-stage filter:
      1. Rule-based fast path — handles obvious cases with zero LLM calls
      2. LLM assessment — only for genuinely ambiguous queries

    Usage (interactive CLI):
        clarifier = QueryClarifier()
        refined, was_clarified = clarifier.run_interactive(user_query)
        result = pipeline.run(refined)
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model  = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess(self, query: str) -> dict:
        """
        Decide whether the query needs clarification and generate questions.

        Stage 1: rule-based fast path (no LLM, < 1 ms).
        Stage 2: LLM assessment (only for genuinely ambiguous queries).

        Returns:
            {
                "needs_clarification": bool,
                "questions":           [str],   # 0–2 questions
                "reasoning":           str,
                "fast_path":           bool,    # True if rule-based fast path fired
            }
        """
        # ── Stage 1: rule-based fast path ────────────────────────────
        reason = self._fast_path_reason(query)
        if reason:
            logger.info(f"Fast path: no clarification — {reason}")
            return {
                "needs_clarification": False,
                "questions":           [],
                "reasoning":           reason,
                "fast_path":           True,
            }

        # ── Stage 2: LLM assessment ───────────────────────────────────
        prompt = _ASSESS_PROMPT.format(query=query)
        try:
            rate_limiter.acquire(estimated_tokens=400)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            result["questions"] = result.get("questions", [])[:2]   # cap at 2
            result["fast_path"] = False
            return result
        except Exception as e:
            logger.warning(f"LLM assessment failed ({e}); skipping clarification")
            return {
                "needs_clarification": False,
                "questions":           [],
                "reasoning":           "LLM unavailable",
                "fast_path":           False,
            }

    def refine(self, original_query: str, qa_pairs: dict) -> str:
        """
        Rewrite the original query incorporating clarification answers.
        Falls back to original_query on LLM failure.
        """
        if not qa_pairs:
            return original_query

        qa_text  = "\n".join(f"  Q: {q}\n  A: {a}" for q, a in qa_pairs.items())
        prompt   = _REFINE_PROMPT.format(original_query=original_query, qa_text=qa_text)

        try:
            rate_limiter.acquire(estimated_tokens=300)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            refined = response.choices[0].message.content.strip().strip('"')
            logger.info(f"Refined query: {refined}")
            return refined
        except Exception as e:
            logger.warning(f"Query refinement failed ({e}); using original query")
            return original_query

    def run_interactive(self, original_query: str, input_fn=input) -> tuple:
        """
        Run the full interactive clarification loop.

        Asks up to 2 questions, collects answers, returns refined query.
        Skips entirely when the query is already specific enough.

        Args:
            original_query: Raw query from the user.
            input_fn:        Callable used to read user input (injectable for testing).

        Returns:
            (refined_query, was_clarified)
        """
        assessment = self.assess(original_query)

        if not assessment["needs_clarification"] or not assessment["questions"]:
            return original_query, False

        print("\nA couple of quick questions to give you the most relevant answer:\n")

        qa_pairs: dict = {}
        for i, question in enumerate(assessment["questions"], 1):
            print(f"  {i}. {question}")
            try:
                answer = input_fn("     Your answer: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if answer:
                qa_pairs[question] = answer
            else:
                print("     (skipped)")

        if not qa_pairs:
            return original_query, False

        refined = self.refine(original_query, qa_pairs)
        return refined, True

    # ------------------------------------------------------------------
    # Internal: rule-based fast path
    # ------------------------------------------------------------------

    def _fast_path_reason(self, query: str) -> Optional[str]:
        """
        Return a reason string if the query is clearly specific enough,
        or None if LLM assessment is needed.

        Checks (in order, short-circuit on first match):
          1. Any known regulation is explicitly named
          2. Contains a § section reference or Article N
          3. Comparison/contrast query
          4. NIST function code (GOVERN 1.1, MAP 2.1, etc.)
        """
        q = query.lower()

        # 1. Named regulation
        for keyword in _REGULATION_KEYWORDS:
            if keyword in q:
                return f"Regulation explicitly named ('{keyword}')."

        # 2. Specific section or article reference
        if _SECTION_PATTERN.search(query):
            return "Query contains a specific § section reference."
        if _ARTICLE_PATTERN.search(query):
            return "Query references a specific Article number."

        # 3. Comparison query
        if any(kw in q for kw in _COMPARISON_KEYWORDS):
            return "Comparison/contrast query — jurisdictions implied."

        # 4. NIST function code
        if _NIST_FN_PATTERN.search(query):
            return "Query contains a specific NIST function reference."

        return None  # ambiguous — let LLM decide
