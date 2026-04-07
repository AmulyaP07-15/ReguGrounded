# src/query_expander.py
#
# Regulatory query expansion.
# Maps informal user terminology to official regulatory language so the
# embedding and BM25 searches surface the right chunks even when users
# don't know the exact statutory phrases.
#
# Design:
#   - Rule-based only — no LLM calls, no latency impact
#   - Expansion limited to 3 synonyms per matched term (configurable)
#   - Skips proper nouns, legal citations, and already-official terms

import re
import logging
from typing import List, Tuple

logger = logging.getLogger("query_expander")

# ---------------------------------------------------------------------------
# Synonym dictionary
# key   = canonical user term (lowercase)
# value = list of official / related regulatory phrases to add
# ---------------------------------------------------------------------------

REGULATORY_SYNONYMS: dict[str, List[str]] = {
    # ── Bias / auditing ────────────────────────────────────────────────
    "bias audit": [
        "bias audit", "algorithmic audit",
        "discrimination testing", "independent bias audit",
    ],
    "audit": [
        "audit", "bias audit", "impact assessment", "algorithmic assessment",
    ],
    "algorithmic audit": [
        "bias audit", "algorithmic audit", "discrimination testing",
    ],

    # ── Hiring / employment ────────────────────────────────────────────
    "ai hiring": [
        "automated employment decision tool", "AEDT",
        "AI hiring", "automated hiring system",
    ],
    "hiring tool": [
        "automated employment decision tool", "AEDT",
        "employment screening tool", "candidate selection",
    ],
    "hiring algorithm": [
        "automated employment decision tool", "algorithmic hiring",
        "AEDT", "automated candidate screening",
    ],
    "employment screening": [
        "automated employment decision tool", "AEDT",
        "employment decision", "candidate evaluation",
    ],
    "resume screening": [
        "automated employment decision tool", "AEDT", "resume screening",
        "candidate screening", "hiring algorithm",
    ],

    # ── High-risk AI ───────────────────────────────────────────────────
    "high-risk": [
        "high-risk AI system", "high-risk AI",
        "systems presenting significant risk",
    ],
    "high risk": [
        "high-risk AI system", "high-risk AI",
        "systems presenting significant risk",
    ],
    "risky ai": [
        "high-risk AI system", "AI system presenting risk",
    ],

    # ── Transparency ───────────────────────────────────────────────────
    "transparency": [
        "transparency obligation", "disclosure requirement",
        "information requirement", "transparency",
    ],
    "explainability": [
        "explainability", "transparency", "right to explanation",
        "human oversight", "disclosure requirement",
    ],
    "explain": [
        "explanation", "transparency", "disclosure", "right to explanation",
    ],
    "disclose": [
        "disclosure", "transparency obligation", "notification requirement",
    ],
    "notice": [
        "notice requirement", "notification", "disclosure",
        "candidate notice", "employee notice",
    ],

    # ── Risk management ────────────────────────────────────────────────
    "risk management": [
        "risk management system", "risk management framework",
        "AI risk management", "risk assessment",
    ],
    "risk assessment": [
        "risk assessment", "impact assessment",
        "algorithmic impact assessment", "risk management",
    ],
    "impact assessment": [
        "impact assessment", "algorithmic impact assessment",
        "risk assessment", "developer impact assessment",
    ],

    # ── Data / training ────────────────────────────────────────────────
    "training data": [
        "training data", "data governance", "training dataset",
        "data quality", "dataset requirements",
    ],
    "data quality": [
        "data quality", "training data requirements",
        "data governance", "dataset quality",
    ],

    # ── Human oversight ────────────────────────────────────────────────
    "human oversight": [
        "human oversight", "human review", "right to human review",
        "human-in-the-loop", "meaningful human oversight",
    ],
    "human review": [
        "human review", "right to human review",
        "human oversight", "meaningful human decision",
    ],

    # ── Conformity / certification ─────────────────────────────────────
    "certification": [
        "conformity assessment", "CE marking", "certification",
        "notified body", "third-party assessment",
    ],
    "conformity": [
        "conformity assessment", "CE marking",
        "declaration of conformity",
    ],

    # ── Consumer / user rights ─────────────────────────────────────────
    "consumer rights": [
        "consumer right", "individual right",
        "right to explanation", "right to contest", "opt-out right",
    ],
    "opt out": [
        "opt-out", "right to opt out", "consumer right",
        "refusal right",
    ],

    # ── Deployer / developer obligations ──────────────────────────────
    "deployer": [
        "deployer", "deployer obligation", "entity deploying",
        "AI system deployer",
    ],
    "developer": [
        "developer", "provider", "developer obligation",
        "AI developer", "system provider",
    ],
    "provider": [
        "provider", "developer", "high-risk AI provider",
        "provider obligation",
    ],

    # ── Post-market / monitoring ───────────────────────────────────────
    "monitoring": [
        "post-market monitoring", "monitoring obligation",
        "performance monitoring", "system monitoring",
    ],
    "post-market": [
        "post-market monitoring", "post-deployment monitoring",
        "market surveillance",
    ],

    # ── NIST functions ─────────────────────────────────────────────────
    "govern": [
        "GOVERN function", "AI governance", "governance practices",
        "organisational governance",
    ],
    "trustworthy ai": [
        "trustworthy AI", "reliable AI", "NIST AI RMF",
        "AI trustworthiness",
    ],
}

# Patterns that indicate the query already uses official terminology
# (no expansion needed or wanted)
_OFFICIAL_TERM_PATTERNS = [
    re.compile(r"\barticle\s+\d+\b", re.I),
    re.compile(r"§\s*[\d\w-]+"),
    re.compile(r"\b\d+-\d+\b"),          # section numbers like 20-871
    re.compile(r"\b(GOVERN|MAP|MEASURE|MANAGE)\s+\d+\.\d+\b", re.I),
    re.compile(r"\bAEDT\b"),
]

# Proper nouns / jurisdiction names — never expand these
_PROPER_NOUNS = {
    "nyc", "eu", "colorado", "nist", "new york", "european union",
    "local law 144", "ll 144", "ai act", "sb 24-205",
}


class QueryExpander:
    """
    Expands a user query with regulatory synonyms before retrieval.

    Usage:
        expander = QueryExpander()
        expanded, terms_added = expander.expand("AI hiring tool requirements in NYC")
        # expanded = "AI hiring tool requirements in NYC automated employment
        #             decision tool AEDT employment screening tool"
    """

    def __init__(self, expansion_limit: int = 3, skip_short_tokens: int = 3):
        self.expansion_limit     = expansion_limit
        self.skip_short_tokens   = skip_short_tokens
        # Pre-sort by length descending so multi-word keys match before single-word
        self._sorted_keys = sorted(REGULATORY_SYNONYMS.keys(), key=len, reverse=True)

    def expand(self, query: str) -> Tuple[str, List[str]]:
        """
        Expand query with regulatory synonyms.

        Returns:
            (expanded_query, list_of_terms_added)
            If no expansion is needed, returns (original_query, [])
        """
        # Don't expand if query already uses official citations
        if self._is_already_official(query):
            logger.debug(f"No expansion: query uses official terminology")
            return query, []

        q_lower = query.lower()
        additions: List[str] = []
        already_added: set   = set()

        for key in self._sorted_keys:
            # Skip if proper noun
            if key in _PROPER_NOUNS:
                continue
            # Skip very short keys (noise)
            if len(key.split()) == 1 and len(key) < self.skip_short_tokens:
                continue

            if key in q_lower:
                synonyms = REGULATORY_SYNONYMS[key]
                count = 0
                for syn in synonyms:
                    if count >= self.expansion_limit:
                        break
                    # Don't add if already in query or already added
                    if syn.lower() in q_lower or syn.lower() in already_added:
                        continue
                    additions.append(syn)
                    already_added.add(syn.lower())
                    count += 1

        if not additions:
            return query, []

        expanded = query + " " + " ".join(additions)
        logger.info(
            f"Query expanded: added {len(additions)} term(s): "
            + ", ".join(f'{a!r}' for a in additions[:3])
            + ("..." if len(additions) > 3 else "")
        )
        return expanded, additions

    def _is_already_official(self, query: str) -> bool:
        """Return True if the query already contains official legal citations."""
        for pattern in _OFFICIAL_TERM_PATTERNS:
            if pattern.search(query):
                return True
        return False
