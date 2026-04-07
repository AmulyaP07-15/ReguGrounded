# ReguGrounded

## Overview

ReguGrounded is a regulatory compliance assistant that answers multi-jurisdiction AI governance questions using **grounded generation with citations**.

The system combines:

* **LLM reasoning**
* **hybrid vector + keyword retrieval**
* **structured evidence synthesis**

The goal is to prevent hallucinated regulatory answers by grounding all responses in real regulatory text.

---

# System Architecture

```
User Query
     ↓
Query Interface  (input guardrails, trace ID, rate limiting)
     ↓
Query Clarifier  (ambiguity detection)
     ↓
RLM Decomposition (LLM query decomposition into sub-questions)
     ↓
Enhanced Retrieval (per sub-question)
   ├── Query Expansion     (informal → statutory terminology)
   ├── Jurisdiction Filter (auto-detect relevant regulation)
   ├── Hybrid Search       (BM25 keyword + Pinecone semantic)
   └── Cross-Encoder Rerank (precise 2nd-pass scoring)
     ↓
Evidence Collection
     ↓
Answer Synthesis   (LLM grounded in retrieved text only)
     ↓
Citation Validator (post-hoc hallucination detection)
     ↓
Grounded Response + Citations
```

---

# Repository Structure

```
ReguGrounded/
│
├── query_interface.py          # Entry point: input/output guardrails, trace IDs, caching
├── rlm_engine.py               # LLM query decomposition into sub-questions
├── reasoning_orchestrator.py   # Retrieval orchestration across jurisdictions
├── answer_synthesizer.py       # Answer + citation formatting via LLM
├── citation_validator.py       # Post-hoc citation grounding check
├── query_clarifier.py          # Detects and flags ambiguous queries
│
├── src/                        # Data & retrieval layer
│   ├── pdf_ingestion.py        # Extract and clean text from PDFs
│   ├── chunker.py              # Split by article/section with metadata
│   ├── embedder.py             # Generate embeddings + upload to Pinecone
│   ├── process_all_pdfs.py     # Run full ingestion pipeline
│   ├── retriever.py            # Base Pinecone retriever
│   ├── enhanced_retriever.py   # Hybrid retrieval with all 4 improvements
│   ├── bm25_index.py           # BM25 keyword index over all chunks
│   ├── query_expander.py       # Rule-based regulatory synonym expansion
│   ├── reranker.py             # Cross-encoder reranking (2nd pass)
│   └── retrieval_config.py     # Centralised retrieval tuning parameters
│
├── utils/
│   ├── cache.py                # LRU in-memory + SQLite disk cache
│   ├── guardrails.py           # Input/output safety validation
│   ├── logger.py               # Structured logging with decorators
│   ├── metrics.py              # Per-request latency and usage tracking
│   ├── rate_limiter.py         # Token-aware OpenAI rate limiter
│   └── trace.py                # Request-scoped trace IDs via contextvars
│
├── eval/
│   ├── evaluate_retrieval.py   # Retrieval precision/recall metrics
│   ├── evaluate_citations.py   # Citation accuracy evaluation
│   ├── evaluate_answers.py     # Answer quality scoring
│   ├── evaluate_e2e.py         # End-to-end pipeline evaluation
│   ├── run_all_evals.py        # Single entry point for all evaluations
│   ├── ground_truth.json       # Hand-labelled ground truth for eval
│   └── baselines/              # Standard RAG and agentic RAG baselines
│
├── tests/
│   ├── test_retrieval_improvements.py  # BM25, hybrid, expansion, reranker (39 tests)
│   ├── test_integration.py             # Full pipeline integration tests
│   ├── test_cache.py                   # Cache layer tests
│   ├── test_guardrails.py              # Guardrail validation tests
│   └── test_rate_limiter.py            # Rate limiter tests
│
├── data/
│   ├── raw/                    # Source PDFs (not committed)
│   ├── processed/              # Cleaned text per document
│   └── chunks/                 # Structured JSON chunks with metadata
│
├── scripts/
│   └── manage_cache.py         # CLI tool to inspect and clear caches
│
├── requirements.txt
├── .env                        # API keys (not committed)
└── README.md
```

---

# Responsibilities

### Amulya — Data & Retrieval Layer (`src/`, `utils/`, `eval/`, `tests/`)

* PDF ingestion and text cleaning
* Structured chunking by article/section
* Embedding generation (sentence-transformers) and Pinecone indexing
* Enhanced hybrid retrieval pipeline (BM25 + semantic + reranking + expansion)
* Utils layer: caching, guardrails, logging, metrics, rate limiting, trace IDs
* Evaluation framework and test suite

Interface exposed to the reasoning layer:

```python
retrieve(query: str, top_k: int) -> list[dict]
```

Returns:

```python
{
    "text": "regulatory snippet",
    "metadata": {
        "law_name": "EU AI Act",
        "article_number": "10",
        "section_number": "2",
        "chunk_id": "eu_ai_10_2"
    },
    "score": 0.89
}
```

Regulatory documents covered:
- EU AI Act
- EU AI Act Annex
- Colorado AI Act
- NIST AI Risk Management Framework
- NYC Local Law 144

---

### Prisha — Reasoning Layer

* `rlm_engine.py` — LLM query decomposition
* `reasoning_orchestrator.py` — retrieval orchestration
* `answer_synthesizer.py` — answer synthesis and citation formatting
* `query_interface.py` — query intake and pipeline coordination

---

# Engineering Challenges & Solutions

### 1. Legal citations breaking BM25 tokenization

**Problem:** Standard tokenizers split `§ 20-871` into `20` and `871`, and `Article 9` into separate tokens. A keyword search for `§ 20-871` would then return irrelevant chunks containing any number with a hyphen.

**Solution:** Custom `_tokenize()` in `bm25_index.py` pre-processes regulatory text before indexing. It converts `article 9` → `article_9` and `§ 20-871` → `section_20-871` using regex substitution so citations survive as single atomic tokens. Stop words and single-character tokens are stripped to reduce noise.

---

### 2. Hybrid score fusion across incompatible scales

**Problem:** BM25 raw scores and Pinecone cosine similarity scores are on completely different scales and distributions. A naïve weighted average would be dominated by whichever scale happened to produce larger numbers.

**Solution:** BM25 scores are normalised to [0, 1] before fusion by dividing by the top result's score. The final hybrid score is `semantic_weight × cosine + keyword_weight × bm25_norm`. Weights are dynamically adjusted: queries with legal citations (e.g. `§ 20-871`) or very short queries (≤3 tokens) get a higher keyword weight (0.5 vs 0.4 default) since exact term matching matters more than semantic similarity in those cases.

---

### 3. Jurisdiction filtering breaking multi-jurisdiction queries

**Problem:** Automatically filtering retrieved chunks to a detected jurisdiction is useful for specific queries ("What does NYC require?") but breaks comparison queries ("How do NYC and the EU AI Act differ?"). An aggressive filter would silently return only half the evidence.

**Solution:** `_detect_jurisdictions()` returns `None` (no filter) when the query contains multiple jurisdiction signals or comparison language ("vs", "differ", "compare", "both"). Only unambiguous single-jurisdiction queries trigger a Pinecone metadata filter. This was validated with dedicated tests covering both explicit and comparison query patterns.

---

### 4. Query expansion inflating noise for official citations

**Problem:** Expanding a query like "What does Article 9 require?" with synonyms adds irrelevant terms and dilutes the precise article match. Expanding "AEDT requirements" is useful; expanding `§ 20-871` is harmful.

**Solution:** `QueryExpander` checks for legal citation patterns (`Article N`, `§ X-Y`) and official statutory terms before expanding. If the query already contains precise regulatory language, expansion is skipped entirely. Synonyms already present in the query are also excluded from the expansion to avoid duplication.

---

### 5. Cross-encoder reranking latency at query time

**Problem:** The cross-encoder model (~80 MB) takes ~2 seconds to load and scores each (query, chunk) pair individually, making it too slow to run on every query from scratch.

**Solution:** Two-layer mitigation: (1) Lazy singleton loading — the model loads once on first use and is reused across requests. (2) TTL score cache keyed on `sha256(query)[:16] :: chunk_id` — if the same query is seen again within 1 hour, cached scores are returned without any model inference. In the test suite, all cross-encoder calls are patched out so tests run offline with no model download required.

---

### 6. Cache invalidation across process restarts

**Problem:** An in-memory LRU cache is fast but lost on every restart, forcing expensive LLM and Pinecone calls to repeat for previously seen queries.

**Solution:** `QueryCache` uses a two-layer architecture: LRU in-memory (fast, process-scoped) backed by SQLite on disk (survives restarts). On a memory miss, the disk is checked and the result is promoted back to memory on a hit. Sub-pipeline components (decomposition, retrieval results) use memory-only `ComponentCache` since they are cheaper to recompute than full query results.

---

### 7. Thread safety for concurrent requests

**Problem:** The SQLite store and LRU cache are accessed by multiple threads in a live serving context. Unprotected concurrent writes could corrupt cache state.

**Solution:** `_LRUCache` wraps all reads and writes in a `threading.Lock()`. `_SQLiteStore` uses `threading.local()` to give each thread its own SQLite connection, avoiding cross-thread connection sharing which SQLite does not support safely.

---

# Environment Setup

### 1. Clone the repository

```
git clone https://github.com/AmulyaP07-15/ReguGrounded.git
cd ReguGrounded
```

### 2. Create and activate the environment

```
python3 -m venv ReguGrounded_env
source ReguGrounded_env/bin/activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

# API Key Setup

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_key_here
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_INDEX_NAME=regugrounded
```

Do **not** commit `.env` to GitHub.

---

# Running the System

### Ingest and index documents (run once)

```
python src/process_all_pdfs.py
```

### Run the CLI interface

```
python query_interface.py
```

Example query:

```
What bias audit requirements apply to AI hiring systems under NYC Local Law 144 and the EU AI Act?
```

### Run the test suite

```
python -m pytest tests/
```

### Run evaluations

```
python eval/run_all_evals.py
```

---

# Development Progress

- [x] PDF ingestion and text cleaning
- [x] Structured chunking (395 chunks across 5 documents)
- [x] Embedding generation and Pinecone indexing
- [x] Base `retrieve(query)` interface
- [x] BM25 keyword index with legal citation-aware tokenizer
- [x] Hybrid BM25 + semantic search with dynamic score fusion
- [x] Jurisdiction detection and auto-filtering
- [x] Regulatory query expansion (rule-based, zero latency)
- [x] Cross-encoder reranking with TTL score cache
- [x] Utils layer: cache, guardrails, rate limiter, metrics, trace IDs
- [x] Citation validator (post-hoc hallucination detection)
- [x] 154-test suite (all passing, fully offline)
- [x] Evaluation framework with ground truth and baselines
- [ ] End-to-end grounded responses with citations (integration in progress)

---

# Notes

* The LLM **never answers directly from its own knowledge**
* All answers must be grounded in retrieved regulatory text
* Every response must include **citations and supporting evidence**
* Citation accuracy is measured per-response via `citation_validator.py`

---

# Future Phases

Phase 3 will include:

* Live end-to-end grounded response pipeline
* Hallucination rate benchmarking against baseline RAG
* Expanded ground truth and adversarial query evaluation
