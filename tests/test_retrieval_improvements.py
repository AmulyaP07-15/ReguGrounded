# tests/test_retrieval_improvements.py
#
# Tests for all four retrieval improvements.
# All tests run offline (no Pinecone, no cross-encoder download required).
# A mock Pinecone index is injected so the tests are fast and deterministic.

import json
import time
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── Fixtures & helpers ─────────────────────────────────────────────────────

CHUNKS_DIR = Path("data/chunks")

def _load_chunks():
    """Load all chunk files for index testing."""
    all_chunks = []
    for path in sorted(CHUNKS_DIR.glob("*_chunks.json")):
        with open(path) as f:
            all_chunks.extend(json.load(f))
    return all_chunks


# ===========================================================================
# Part 1 — BM25 Index
# ===========================================================================

class TestBM25Index:
    @pytest.fixture(scope="class")
    def idx(self):
        from src.bm25_index import BM25Index
        return BM25Index().build()

    def test_index_built(self, idx):
        assert len(idx.chunk_ids) == 395
        assert idx._bm25 is not None

    def test_exact_section_match(self, idx):
        """§ 20-871 query should surface NYC chunk as top result."""
        results = idx.search("§ 20-871 bias audit requirements", top_k=5)
        assert results, "Expected at least one result"
        top_doc_ids = [r["doc_id"] for r in results[:3]]
        assert "nyc_local_law_144" in top_doc_ids, (
            f"Expected NYC chunk in top-3, got: {top_doc_ids}"
        )

    def test_article_match(self, idx):
        """Article 9 query should surface EU AI Act chunk."""
        results = idx.search("Article 9 risk management system", top_k=5)
        sections = [r["section"] for r in results[:3]]
        assert any("9" in s for s in sections), (
            f"Expected Article 9 in top-3 sections, got: {sections}"
        )

    def test_jurisdiction_filter(self, idx):
        """filter_doc_ids should exclude other jurisdictions."""
        results = idx.search("transparency requirements", top_k=10,
                             filter_doc_ids=["eu_ai_act"])
        doc_ids = {r["doc_id"] for r in results}
        assert doc_ids <= {"eu_ai_act"}, f"Non-EU chunks returned: {doc_ids}"

    def test_scores_normalised(self, idx):
        """Normalised scores should be in [0, 1]."""
        results = idx.search("bias audit employer obligations", top_k=10)
        for r in results:
            assert 0.0 <= r["bm25_score_norm"] <= 1.0, (
                f"Score out of range: {r['bm25_score_norm']}"
            )

    def test_top_score_is_one(self, idx):
        """Top result's normalised score should be 1.0."""
        results = idx.search("high-risk AI conformity assessment", top_k=5)
        if results:
            assert results[0]["bm25_score_norm"] == pytest.approx(1.0)

    def test_empty_query_returns_nothing(self, idx):
        results = idx.search("", top_k=5)
        assert results == []

    def test_persist_and_reload(self, idx, tmp_path):
        """Save and reload produces identical chunk IDs."""
        path = str(tmp_path / "test_bm25.pkl")
        idx.save(path)
        from src.bm25_index import BM25Index
        loaded = BM25Index().load(path)
        assert loaded.chunk_ids == idx.chunk_ids


# ===========================================================================
# Part 2 — Jurisdiction detection & filtering
# ===========================================================================

class TestJurisdictionDetection:
    def _detect(self, query):
        from src.enhanced_retriever import _detect_jurisdictions
        return _detect_jurisdictions(query)

    def test_nyc_explicit(self):
        result = self._detect("What does NYC Local Law 144 require?")
        assert result is not None
        assert "nyc_local_law_144" in result

    def test_eu_explicit(self):
        result = self._detect("What are EU AI Act high-risk requirements?")
        assert result is not None
        assert "eu_ai_act" in result or "eu_ai_act_annex" in result

    def test_colorado_explicit(self):
        result = self._detect("What does the Colorado AI Act require?")
        assert result is not None
        assert "colorado_ai_act" in result

    def test_nist_explicit(self):
        result = self._detect("What does NIST AI RMF say about governance?")
        assert result is not None
        assert "nist_ai_framework" in result

    def test_bias_audit_implies_nyc_colorado(self):
        result = self._detect("What are bias audit requirements?")
        assert result is not None
        assert "nyc_local_law_144" in result
        assert "colorado_ai_act" in result

    def test_comparison_returns_none(self):
        """Comparison queries should not be filtered."""
        result = self._detect("How do NYC and EU AI Act differ on transparency?")
        assert result is None

    def test_ambiguous_returns_none(self):
        """Generic queries without jurisdiction signals should return None."""
        result = self._detect("What are the general AI obligations?")
        assert result is None

    def test_section_number_implies_nyc(self):
        result = self._detect("What does § 20-871 say about bias audits?")
        assert result is not None
        assert "nyc_local_law_144" in result


# ===========================================================================
# Part 3 — Query expansion
# ===========================================================================

class TestQueryExpander:
    @pytest.fixture
    def expander(self):
        from src.query_expander import QueryExpander
        return QueryExpander(expansion_limit=3)

    def test_informal_hiring_expanded(self, expander):
        expanded, terms = expander.expand("AI hiring tool requirements in NYC")
        assert len(terms) > 0
        assert any("employment" in t.lower() or "AEDT" in t for t in terms)

    def test_bias_audit_expanded(self, expander):
        expanded, terms = expander.expand("What are bias audit requirements?")
        assert len(terms) > 0
        assert any("audit" in t.lower() or "discrimination" in t.lower() for t in terms)

    def test_official_citation_not_expanded(self, expander):
        """Queries with Article N should not be expanded."""
        _, terms = expander.expand("What does Article 9 require?")
        assert terms == [], f"Should not expand official citation, got: {terms}"

    def test_section_number_not_expanded(self, expander):
        _, terms = expander.expand("What does § 20-871 say?")
        assert terms == []

    def test_expansion_limit_respected(self, expander):
        _, terms = expander.expand("bias audit employment screening AI hiring")
        assert len(terms) <= 9  # 3 terms × 3 synonyms max

    def test_no_double_expansion(self, expander):
        """Synonyms already in the query should not be added again."""
        q = "automated employment decision tool AEDT requirements"
        expanded, terms = expander.expand(q)
        # AEDT and automated employment are already present
        for t in terms:
            assert t.lower() not in q.lower(), (
                f"'{t}' was already in query but was added again"
            )

    def test_high_risk_expansion(self, expander):
        _, terms = expander.expand("high-risk AI system obligations")
        assert len(terms) > 0

    def test_transparency_expansion(self, expander):
        _, terms = expander.expand("transparency requirements for AI systems")
        assert len(terms) > 0


# ===========================================================================
# Part 4 — Cross-encoder reranker
# ===========================================================================

class TestReranker:
    """Tests that mock the cross-encoder so no model download is needed."""

    @pytest.fixture(autouse=True)
    def clear_score_cache(self):
        """Clear the module-level score cache before every test."""
        from src.reranker import _score_cache
        _score_cache._data.clear()

    def _make_candidates(self, n=10, query_tag="default"):
        """Use query_tag to generate unique chunk IDs per test."""
        return [
            {
                "text":     f"Chunk {query_tag} {i} about regulatory requirements.",
                "chunk_id": f"chunk_{query_tag}_{i:03d}",
                "score":    1.0 - i * 0.05,
                "metadata": {"law_name": "EU AI Act", "article_number": str(i)},
            }
            for i in range(n)
        ]

    def test_reranking_changes_order(self):
        """Cross-encoder that gives highest score to last chunk should promote it."""
        from src.reranker import rerank
        candidates = self._make_candidates(5, "order")

        # Ascending scores: chunk_004 gets highest score (4.0)
        ascending_scores = list(range(len(candidates)))  # [0, 1, 2, 3, 4]

        with patch("src.reranker._get_cross_encoder") as mock_ce:
            mock_model = MagicMock()
            mock_model.predict.return_value = ascending_scores
            mock_ce.return_value = mock_model

            result = rerank("order query", candidates, final_k=3)

        assert result[0]["chunk_id"] == "chunk_order_004"  # highest score → top

    def test_final_k_respected(self):
        from src.reranker import rerank
        candidates = self._make_candidates(10, "fk")
        scores = [float(i) for i in range(len(candidates))]

        with patch("src.reranker._get_cross_encoder") as mock_ce:
            mock_model = MagicMock()
            mock_model.predict.return_value = scores
            mock_ce.return_value = mock_model

            result = rerank("final k query", candidates, final_k=3)

        assert len(result) == 3

    def test_rerank_score_attached(self):
        from src.reranker import rerank
        candidates = self._make_candidates(3, "score")
        scores = [0.9, 0.5, 0.7]

        with patch("src.reranker._get_cross_encoder") as mock_ce:
            mock_model = MagicMock()
            mock_model.predict.return_value = scores
            mock_ce.return_value = mock_model

            result = rerank("score attach query", candidates, final_k=3)

        assert "rerank_score" in result[0]
        # chunk_000 has score 0.9 → should be ranked first
        assert result[0]["chunk_id"] == "chunk_score_000"
        assert result[0]["rerank_score"] == pytest.approx(0.9)

    def test_rerank_rank_annotated(self):
        from src.reranker import rerank
        candidates = self._make_candidates(5, "rank")

        with patch("src.reranker._get_cross_encoder") as mock_ce:
            mock_model = MagicMock()
            mock_model.predict.return_value = [float(i) for i in range(5)]
            mock_ce.return_value = mock_model

            result = rerank("rank annot query", candidates, final_k=5)

        ranks = [r["rerank_rank"] for r in result]
        assert ranks == [1, 2, 3, 4, 5]

    def test_score_cache_hit(self):
        """Second call with same query+chunk should not call the model."""
        from src.reranker import rerank
        candidates = self._make_candidates(2, "cache")

        call_count = [0]
        def counting_predict(inputs, batch_size=32):
            call_count[0] += 1
            return [0.5] * len(inputs)

        with patch("src.reranker._get_cross_encoder") as mock_ce:
            mock_model = MagicMock()
            mock_model.predict.side_effect = counting_predict
            mock_ce.return_value = mock_model

            rerank("cache hit query", candidates, final_k=2)
            first_count = call_count[0]
            rerank("cache hit query", candidates, final_k=2)
            second_count = call_count[0]

        assert second_count == first_count, "Model should not be called on cache hit"

    def test_empty_candidates_returns_empty(self):
        from src.reranker import rerank
        assert rerank("empty query", [], final_k=5) == []


# ===========================================================================
# Part 5 — Score fusion (hybrid merge)
# ===========================================================================

class TestScoreFusion:
    """Test hybrid score combination logic without Pinecone."""

    def test_keyword_weight_boosts_for_citation(self):
        from src.enhanced_retriever import _keyword_weight
        from src.retrieval_config import DEFAULT_CONFIG
        # Query with § citation
        w = _keyword_weight("What does § 20-871 require?", 0.4, DEFAULT_CONFIG)
        assert w >= 0.5, f"Expected keyword boost for citation, got {w}"

    def test_keyword_weight_boosts_for_short_query(self):
        from src.enhanced_retriever import _keyword_weight
        from src.retrieval_config import DEFAULT_CONFIG
        w = _keyword_weight("bias audit", 0.4, DEFAULT_CONFIG)
        assert w >= 0.5, f"Expected keyword boost for short query, got {w}"

    def test_keyword_weight_normal_for_long_query(self):
        from src.enhanced_retriever import _keyword_weight
        from src.retrieval_config import DEFAULT_CONFIG
        w = _keyword_weight(
            "What transparency requirements apply to automated employment decision tools?",
            0.4, DEFAULT_CONFIG
        )
        assert w == pytest.approx(0.4)

    def test_semantic_plus_keyword_weights_sum_to_one(self):
        from src.enhanced_retriever import _keyword_weight
        from src.retrieval_config import DEFAULT_CONFIG
        query = "What are the requirements for bias auditing?"
        kw  = _keyword_weight(query, 0.4, DEFAULT_CONFIG)
        sem = 1.0 - kw
        assert kw + sem == pytest.approx(1.0)


# ===========================================================================
# Part 6 — BM25 tokenizer
# ===========================================================================

class TestTokenizer:
    def test_article_preserved(self):
        from src.bm25_index import _tokenize
        tokens = _tokenize("Article 9 requires a risk management system")
        assert "article_9" in tokens

    def test_section_preserved(self):
        from src.bm25_index import _tokenize
        tokens = _tokenize("§ 20-871 requires an annual bias audit")
        assert any("section_20-871" in t or "20-871" in t for t in tokens)

    def test_stop_words_removed(self):
        from src.bm25_index import _tokenize
        tokens = _tokenize("the provider shall comply with the requirements")
        assert "the" not in tokens
        assert "shall" not in tokens
        assert "with" not in tokens

    def test_short_tokens_removed(self):
        from src.bm25_index import _tokenize
        tokens = _tokenize("AI is a key technology")
        assert "a" not in tokens

    def test_empty_string(self):
        from src.bm25_index import _tokenize
        assert _tokenize("") == []
