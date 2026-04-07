# utils/guardrails.py
#
# Guardrail layer for ReguGrounded.
#
# Two independent guard objects:
#   InputGuard  — runs BEFORE any LLM call; blocks or warns on bad input
#   OutputGuard — runs AFTER synthesis; adds warnings to metadata
#
# Severity levels:
#   "block"  — GuardrailBlockedError is raised; query never reaches the LLM
#   "warn"   — processing continues; warning string appended to result metadata
#
# All checks use rule-based fast paths (regex / keyword lists).
# No additional LLM calls are made by the guardrail layer itself.

import re
from dataclasses import dataclass, field
from typing import List, Optional

from utils.logger import setup_logger

logger = setup_logger("guardrails")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GuardrailViolation:
    category: str    # "LENGTH" | "INJECTION" | "PII" | "SCOPE"
    severity: str    # "block" | "warn"
    message:  str


@dataclass
class GuardrailResult:
    passed:     bool
    violations: List[GuardrailViolation] = field(default_factory=list)
    warnings:   List[str]                = field(default_factory=list)

    def blocks(self) -> bool:
        return any(v.severity == "block" for v in self.violations)

    def warning_strings(self) -> List[str]:
        return [v.message for v in self.violations if v.severity == "warn"] + self.warnings


class GuardrailBlockedError(ValueError):
    """Raised when an input guardrail with severity='block' fires."""
    def __init__(self, violation: GuardrailViolation):
        self.violation = violation
        super().__init__(f"[{violation.category}] {violation.message}")


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# Minimum / maximum query length (characters)
_MIN_QUERY_LEN = 10
_MAX_QUERY_LEN = 2000

# Tokens that indicate the query touches one of the 4 covered regulations
_IN_SCOPE_KEYWORDS = [
    # Explicit regulation names
    "eu ai act", "ai act",
    "nyc local law 144", "local law 144", "ll 144", "ll144",
    "colorado ai act",
    "nist ai", "nist rmf", "nist ai rmf", "nist framework",
    # Topical terms strongly associated with these specific laws
    "bias audit", "automated employment", "aeds",
    "high-risk ai", "high risk ai",
    "conformity assessment", "post-market monitoring",
    "ai risk management", "govern function", "map function",
    "deployer", "developer obligation",
    "transparency requirement", "explainability",
    "algorithmic", "ai system",
]

# Topics that are clearly outside the system's coverage — warn but don't block
_OUT_OF_SCOPE_SIGNALS = [
    "gdpr", "ccpa", "hipaa", "ferpa", "glba", "coppa",
    "sox", "sarbanes", "dodd-frank",
    "tax law", "tax code", "irs",
    "criminal law", "penal code", "sentencing",
    "immigration", "visa",
    "drug approval", "fda approval", "medical device",
    "antitrust", "competition law",
    "intellectual property", "patent", "copyright",
    "real estate", "zoning",
]

# Classic prompt-injection / jailbreak patterns
_INJECTION_PATTERNS: List[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?", re.I),
    re.compile(r"forget\s+(all\s+)?(previous|prior|above)\s+instructions?", re.I),
    re.compile(r"you\s+are\s+now\b", re.I),
    re.compile(r"pretend\s+(to\s+be|you('re| are))\b", re.I),
    re.compile(r"\bjailbreak\b", re.I),
    re.compile(r"\bdan\s+(mode|prompt)\b", re.I),
    re.compile(r"(print|repeat|reveal|output|show|display)\s+(your\s+)?(system\s+)?(prompt|instructions?)", re.I),
    re.compile(r"<\s*system\s*>", re.I),           # XML-style system tag injection
    re.compile(r"\[INST\]|\[\/INST\]", re.I),       # Llama-style injection markers
    re.compile(r"###\s*instruction", re.I),
    re.compile(r"act\s+as\s+(a|an)\s+\w+\s+with\s+no\s+restrictions?", re.I),
]

# PII patterns — warn only (don't block; redaction is optional)
_PII_EMAIL   = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_PII_PHONE   = re.compile(
    r"(\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}"
)
_PII_SSN     = re.compile(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b")
_PII_CREDIT  = re.compile(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b")

# Answer quality thresholds
_HALLUCINATION_WARN_THRESHOLD = 0.10   # warn if >10% citations are hallucinated
_MIN_CHUNKS_WARN = 1                   # warn if fewer than this many chunks retrieved


# ---------------------------------------------------------------------------
# InputGuard
# ---------------------------------------------------------------------------

class InputGuard:
    """
    Validates user queries before they reach the LLM pipeline.

    Checks (in order):
      1. LENGTH  — query too short or too long   → block
      2. INJECTION — prompt injection patterns   → block
      3. PII     — personal data in the query    → warn
      4. SCOPE   — query likely outside coverage → warn
    """

    def run(self, query: str) -> GuardrailResult:
        """
        Run all input checks on `query`.

        Returns a GuardrailResult. If any violation has severity='block',
        the caller should raise GuardrailBlockedError rather than proceeding.
        """
        violations: List[GuardrailViolation] = []

        v = self._check_length(query)
        if v:
            violations.append(v)

        v = self._check_injection(query)
        if v:
            violations.append(v)

        pii_warnings = self._check_pii(query)
        violations.extend(pii_warnings)

        v = self._check_scope(query)
        if v:
            violations.append(v)

        # Log summary
        blocks  = [v for v in violations if v.severity == "block"]
        warns   = [v for v in violations if v.severity == "warn"]
        if blocks:
            logger.warning(
                f"Input blocked ({blocks[0].category}): {blocks[0].message[:80]}"
            )
        elif warns:
            logger.info(
                f"Input warnings for query {query[:60]!r}: "
                + "; ".join(w.category for w in warns)
            )

        passed = len(blocks) == 0
        return GuardrailResult(passed=passed, violations=violations)

    # ── Individual checks ──────────────────────────────────────────────

    def _check_length(self, query: str) -> Optional[GuardrailViolation]:
        if len(query) < _MIN_QUERY_LEN:
            return GuardrailViolation(
                category="LENGTH",
                severity="block",
                message=(
                    f"Query is too short ({len(query)} chars). "
                    f"Please provide a more complete question (minimum {_MIN_QUERY_LEN} characters)."
                ),
            )
        if len(query) > _MAX_QUERY_LEN:
            return GuardrailViolation(
                category="LENGTH",
                severity="block",
                message=(
                    f"Query is too long ({len(query)} chars). "
                    f"Please shorten your question to {_MAX_QUERY_LEN} characters or fewer."
                ),
            )
        return None

    def _check_injection(self, query: str) -> Optional[GuardrailViolation]:
        for pattern in _INJECTION_PATTERNS:
            m = pattern.search(query)
            if m:
                logger.warning(
                    f"Prompt injection attempt detected — matched: {m.group(0)!r}"
                )
                return GuardrailViolation(
                    category="INJECTION",
                    severity="block",
                    message=(
                        "Your query appears to contain instructions directed at the AI system. "
                        "Please ask a straightforward compliance question."
                    ),
                )
        return None

    def _check_pii(self, query: str) -> List[GuardrailViolation]:
        violations = []
        if _PII_EMAIL.search(query):
            violations.append(GuardrailViolation(
                category="PII",
                severity="warn",
                message="Query appears to contain an email address. Avoid including personal data in compliance queries.",
            ))
        if _PII_PHONE.search(query):
            violations.append(GuardrailViolation(
                category="PII",
                severity="warn",
                message="Query appears to contain a phone number. Avoid including personal data in compliance queries.",
            ))
        if _PII_SSN.search(query):
            violations.append(GuardrailViolation(
                category="PII",
                severity="warn",
                message="Query appears to contain a Social Security Number. Do not include personal identifiers.",
            ))
        if _PII_CREDIT.search(query):
            violations.append(GuardrailViolation(
                category="PII",
                severity="warn",
                message="Query appears to contain a credit card number. Do not include financial identifiers.",
            ))
        return violations

    def _check_scope(self, query: str) -> Optional[GuardrailViolation]:
        q_lower = query.lower()

        # If any in-scope keyword is present, clearly relevant → pass
        if any(kw in q_lower for kw in _IN_SCOPE_KEYWORDS):
            return None

        # If out-of-scope signals are present, warn
        matched = [sig for sig in _OUT_OF_SCOPE_SIGNALS if sig in q_lower]
        if matched:
            return GuardrailViolation(
                category="SCOPE",
                severity="warn",
                message=(
                    f"This query may reference topics outside the system's coverage "
                    f"({', '.join(matched[:3])}). "
                    "ReguGrounded covers: EU AI Act, NYC Local Law 144, "
                    "Colorado AI Act, and NIST AI RMF."
                ),
            )

        # Ambiguous — no clear signal either way; don't warn (too many false positives)
        return None


# ---------------------------------------------------------------------------
# OutputGuard
# ---------------------------------------------------------------------------

class OutputGuard:
    """
    Validates pipeline results after synthesis.

    All checks add warnings to metadata — the answer is never blocked
    at the output stage so the user always gets a response.

    Checks:
      1. HALLUCINATION  — citation accuracy below threshold
      2. NO_EVIDENCE    — no chunks retrieved
      3. SCOPE_REMINDER — no jurisdiction identified (generic answer)
    """

    def run(self, result: dict) -> GuardrailResult:
        """
        Inspect a pipeline result dict and return warnings.
        Does NOT modify the result — caller appends warnings to metadata.
        """
        warnings: List[str] = []

        w = self._check_hallucination(result)
        if w:
            warnings.append(w)

        w = self._check_evidence(result)
        if w:
            warnings.append(w)

        w = self._check_jurisdiction_coverage(result)
        if w:
            warnings.append(w)

        if warnings:
            logger.warning(
                f"Output guardrail warnings for {result.get('question','')[:60]!r}: "
                + " | ".join(warnings)
            )

        return GuardrailResult(passed=True, warnings=warnings)

    # ── Individual checks ──────────────────────────────────────────────

    def _check_hallucination(self, result: dict) -> Optional[str]:
        validation = result.get("validation", {})
        hr  = validation.get("hallucination_rate", 0.0)
        inv = validation.get("invalid_citations", 0)
        tot = validation.get("total_citations", 0)

        if hr > _HALLUCINATION_WARN_THRESHOLD and tot > 0:
            return (
                f"Low citation confidence: {inv}/{tot} citation(s) could not be "
                f"verified against retrieved evidence ({hr*100:.0f}% hallucination rate). "
                "Verify these citations against the source regulations before relying on this answer."
            )
        return None

    def _check_evidence(self, result: dict) -> Optional[str]:
        chunks = result.get("metadata", {}).get("total_chunks_retrieved", None)
        if chunks is not None and chunks < _MIN_CHUNKS_WARN:
            return (
                "No regulatory evidence was retrieved for this query. "
                "The answer may be incomplete or unsupported. "
                "Try rephrasing or specifying a regulation (EU AI Act, NYC Local Law 144, "
                "Colorado AI Act, or NIST AI RMF)."
            )
        return None

    def _check_jurisdiction_coverage(self, result: dict) -> Optional[str]:
        jurisdictions = result.get("metadata", {}).get("jurisdictions_searched", [])
        if not jurisdictions:
            return (
                "No specific regulation was identified for this query. "
                "Results cover all supported regulations but may be broad. "
                "For more precise answers, mention a specific regulation."
            )
        return None
