# citation_validator.py

import re
from typing import Dict, List, Optional, Tuple

from utils.logger import setup_logger

logger = setup_logger("citation_validator")

# ---------------------------------------------------------------------------
# Full-form citation patterns
# Match law name + article/section together — precise and unambiguous.
# ---------------------------------------------------------------------------
_FULL_PATTERNS: List[Tuple[str, re.Pattern]] = [
    # NYC Local Law 144 § 20-871  /  NYC LL 144 § 20-871
    ("NYC Local Law 144",
     re.compile(r"NYC\s+(?:Local\s+Law|LL)\s*144\s*§\s*([\d][\d\w-]*)", re.IGNORECASE)),
    # EU AI Act Article 9  /  EU AI Act Art. 9
    ("EU AI Act",
     re.compile(r"EU\s+AI\s+Act\s+(?:Article|Art\.)\s*(\d+)", re.IGNORECASE)),
    # Colorado AI Act § 6-1-1707
    ("Colorado AI Act",
     re.compile(r"Colorado\s+AI\s+Act\s*§\s*([\d][\d\w-]*)", re.IGNORECASE)),
    # NIST AI RMF GOVERN-1.1  /  NIST AI Framework GOVERN-1.1
    ("NIST AI Risk Management Framework",
     re.compile(r"NIST\s+AI\s+(?:RMF|Framework)\s+([A-Z]+-\d+\.\d+)", re.IGNORECASE)),
]

# ---------------------------------------------------------------------------
# Short-form citation patterns
# Catch "§ 20-871" or "Article 9" written without the full law name.
# These are validated against chunk metadata (any matching chunk).
# ---------------------------------------------------------------------------
_SHORT_PATTERNS: List[Tuple[str, re.Pattern]] = [
    # § 20-871  (section number standalone)
    ("SHORT_SECTION",
     re.compile(r"(?<![/\w])§\s*([\d][\d\w-]+)", re.IGNORECASE)),
    # Article 9  (article number standalone, not preceded by law name already captured)
    ("SHORT_ARTICLE",
     re.compile(r"\bArticle\s+(\d+)\b", re.IGNORECASE)),
]

# Normalise law names to canonical keys
_LAW_ALIASES: Dict[str, str] = {
    "nyc local law 144":                 "NYC Local Law 144",
    "eu ai act":                         "EU AI Act",
    "eu ai act (annex)":                 "EU AI Act",
    "colorado ai act":                   "Colorado AI Act",
    "nist ai risk management framework": "NIST AI Risk Management Framework",
    "nist ai rmf":                       "NIST AI Risk Management Framework",
}


def _normalize_law(name: str) -> str:
    return _LAW_ALIASES.get(name.lower().strip(), name.strip())


def _normalize_ref(ref: str) -> str:
    """
    Normalize a section/article reference for comparison.
    Converts letter-digit dashes to spaces: "GOVERN-1.1" → "GOVERN 1.1"
    Leaves numeric dashes intact:           "20-871"     → "20-871"
    """
    return re.sub(r'([A-Za-z])-(\d)', r'\1 \2', ref.strip())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_citations(answer: str) -> List[Dict]:
    """
    Extract all citations from an answer string.

    Full-form citations are extracted first ("NYC Local Law 144 § 20-871").
    Short-form citations ("§ 20-871", "Article 9") are extracted only for
    spans not already covered by a full-form match.

    Returns list of dicts: {"law", "ref", "raw", "form"}
    """
    found: List[Dict] = []
    covered_spans: List[Tuple[int, int]] = []

    # 1. Full-form (highest confidence)
    for law_name, pattern in _FULL_PATTERNS:
        for m in pattern.finditer(answer):
            found.append({
                "law":  law_name,
                "ref":  m.group(1).strip(),
                "raw":  m.group(0).strip(),
                "form": "full",
            })
            covered_spans.append((m.start(), m.end()))

    # 2. Short-form (only spans not covered above)
    for form_name, pattern in _SHORT_PATTERNS:
        for m in pattern.finditer(answer):
            if any(lo <= m.start() and m.end() <= hi for lo, hi in covered_spans):
                continue  # already captured by a full-form match
            found.append({
                "law":  None,
                "ref":  m.group(1).strip(),
                "raw":  m.group(0).strip(),
                "form": form_name,
            })

    return found


def validate_citations(answer: str, retrieved_chunks: List[Dict]) -> Dict:
    """
    Extract all citations from the answer and verify each against retrieved chunks.

    Full-form citations are matched by law name + article/section.
    Short-form citations (§ X, Article X) are matched against any chunk's
    section/article number, regardless of law — they're grounded as long as
    the number appears in the evidence.

    Returns:
        {
            "valid_citations":    [str],
            "invalid_citations":  [str],
            "total_citations":    int,
            "valid_count":        int,
            "invalid_count":      int,
            "accuracy":           float,
            "hallucination_rate": float,
            "details":            {"valid": [str], "invalid": [str]},
        }
    """
    extracted = extract_citations(answer)

    valid:   List[str] = []
    invalid: List[str] = []

    for cite in extracted:
        if cite["form"] == "full":
            matched = any(_chunk_matches_full(cite, chunk) for chunk in retrieved_chunks)
        else:
            matched = any(_chunk_matches_short(cite, chunk) for chunk in retrieved_chunks)

        label = cite["raw"]
        (valid if matched else invalid).append(label)

    total       = len(extracted)
    valid_count = len(valid)
    accuracy    = valid_count / total if total > 0 else 1.0

    logger.debug(
        f"Validated {total} citations: {valid_count} valid, {len(invalid)} invalid "
        f"(accuracy={accuracy:.2%})"
    )

    return {
        "valid_citations":    valid,
        "invalid_citations":  invalid,
        "total_citations":    total,
        "valid_count":        valid_count,
        "invalid_count":      len(invalid),
        "accuracy":           round(accuracy, 4),
        "hallucination_rate": round(1.0 - accuracy, 4),
        "details": {
            "valid":   valid,
            "invalid": invalid,
        },
    }


# ---------------------------------------------------------------------------
# Internal matching helpers
# ---------------------------------------------------------------------------

def _chunk_matches_full(citation: Dict, chunk: Dict) -> bool:
    """
    Match a full-form citation against a chunk.
    Both law name and article/section must align.
    """
    chunk_law = _normalize_law(chunk.get("law_name", ""))
    cite_law  = _normalize_law(citation["law"])

    if chunk_law != cite_law:
        return False

    cite_ref = _normalize_ref(citation["ref"].strip())

    article = chunk.get("article_number") or ""
    if article and cite_ref == _normalize_ref(article):
        return True

    section = chunk.get("section_number") or ""
    if section:
        section_clean = _normalize_ref(section.lstrip("§").strip())
        if cite_ref.lstrip("§").strip() == section_clean:
            return True

    # Fallback: law matches but no article/section stored → accept
    if not article and not section:
        return True

    return False


def _chunk_matches_short(citation: Dict, chunk: Dict) -> bool:
    """
    Match a short-form citation (§ X or Article X) against any chunk
    by ref number alone, without requiring law name to match.
    """
    cite_ref = _normalize_ref(citation["ref"].strip()).lstrip("§").strip()
    form     = citation.get("form", "")

    if form == "SHORT_SECTION":
        section = chunk.get("section_number") or ""
        if section:
            return cite_ref == _normalize_ref(section.lstrip("§").strip())

    elif form == "SHORT_ARTICLE":
        article = chunk.get("article_number") or ""
        if article:
            return cite_ref == _normalize_ref(article)

    return False
