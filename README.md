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

Two pipelines are implemented. Both share the same decomposition and synthesis layers but differ in how they retrieve evidence.

### RLM Pipeline (`query_interface.py`)
```
User Query
     ↓
Query Interface  (input guardrails, trace ID, rate limiting)
     ↓
RLM Decomposition  (LLM → jurisdiction-specific sub-questions)
     ↓
Parallel Retrieval  (one fixed retrieve call per sub-question)
   └── EnhancedRetriever: hybrid BM25 + semantic, rerank, expand
     ↓
Answer Synthesis   (LLM grounded in retrieved text only)
     ↓
Citation Validator (post-hoc hallucination detection)
     ↓
Grounded Response + Citations
```

### Agentic RAG Pipeline (`agentic_rag_pipeline.py`)
```
User Query
     ↓
RLM Decomposition  (same as above)
     ↓
Agent Loop  ← runs independently per jurisdiction, in parallel
   ├── ToolSelector  — picks semantic / keyword / hybrid per query
   ├── EnhancedRetriever  — fetches chunks with chosen search mode
   ├── EvidenceReflector  — LLM judges: "is this evidence sufficient?"
   └── if insufficient: reformulate query + switch tool → loop (max 3×)
     ↓
Answer Synthesis   (from all accumulated evidence across iterations)
     ↓
Citation Validator
     ↓
Grounded Response + Citations + Agent Trace
```

**Key differences:**

| | RLM | Agentic RAG |
|---|---|---|
| Retrieval rounds | 1 (fixed) | 1–3 (agent decides) |
| Tool selection | fixed hybrid | semantic / keyword / hybrid per query |
| Evidence check | none | LLM reflection after each round |
| Query reformulation | no | yes, if evidence is weak |
| Output | answer + citations | answer + citations + `agent_trace` |

---

# Repository Structure

```
ReguGrounded/
│
├── test_agentic_rag.py         # Agentic RAG demo + evaluation  ← START HERE
├── agentic_rag_pipeline.py     # Agentic RAG pipeline (tool selection + reflection loop)
├── query_interface.py          # RLM pipeline entry point + programmatic API
├── rlm_engine.py               # LLM query decomposition into sub-questions
├── reasoning_orchestrator.py   # Parallel evidence retrieval + deduplication
├── answer_synthesizer.py       # Grounded answer synthesis + citation formatting
├── citation_validator.py       # Post-hoc hallucination detection
├── query_clarifier.py          # Interactive clarification (CLI only)
│
├── src/                        # Data & retrieval layer (Amulya)
│   ├── process_all_pdfs.py     # One-time ingestion pipeline
│   ├── pdf_ingestion.py        # Extract and clean text from PDFs
│   ├── chunker.py              # Split by article/section with metadata
│   ├── embedder.py             # Generate embeddings + upload to Pinecone
│   ├── retriever.py            # Base Pinecone retrieval
│   ├── enhanced_retriever.py   # Hybrid BM25 + semantic, reranking, expansion
│   ├── bm25_index.py           # Legal citation-aware BM25 index
│   ├── query_expander.py       # Rule-based regulatory synonym expansion
│   ├── reranker.py             # Cross-encoder reranking (lazy-loaded)
│   └── retrieval_config.py     # Feature toggles for retrieval enhancements
│
├── utils/                      # Infrastructure
│   ├── cache.py                # QueryCache (LRU + SQLite) + ComponentCache
│   ├── guardrails.py           # InputGuard + OutputGuard
│   ├── rate_limiter.py         # OpenAI token rate limiting
│   ├── logger.py               # Structured logging + decorators
│   ├── metrics.py              # MetricsTracker (latency, errors, cache hits)
│   └── trace.py                # Request trace ID generation
│
├── eval/                       # Evaluation suite
│   ├── run_all_evals.py        # Master evaluation script
│   ├── evaluate_retrieval.py   # Precision@5, Recall@5, MRR
│   ├── evaluate_citations.py   # Citation precision, recall, hallucination rate
│   ├── evaluate_answers.py     # LLM-judge answer quality (0–5 scale)
│   ├── evaluate_e2e.py         # Success rate, latency percentiles
│   ├── benchmark_pipelines.py      # 3-way comparison: standard_rag vs RLM vs agentic_rag
│   ├── benchmark_hallucination.py  # Hallucination-focused comparison (2-way)
│   ├── ground_truth.json       # 18 annotated test questions
│   ├── baselines/
│   │   ├── standard_rag.py     # Baseline: single retrieve → synthesize
│   │   └── agentic_rag.py      # Baseline: wraps agentic_rag_pipeline.py
│   └── results/                # Timestamped JSON eval reports
│
├── tests/                      # 154-test offline suite
│   ├── test_cache.py
│   ├── test_guardrails.py
│   ├── test_integration.py
│   ├── test_rate_limiter.py
│   └── test_retrieval_improvements.py
│
├── data/
│   ├── raw/                    # Source PDFs (not committed)
│   ├── processed/              # Cleaned text per document
│   └── chunks/                 # Structured JSON chunks with metadata
│
├── logs/                       # Timestamped log files
├── requirements.txt
├── .env                        # API keys (not committed)
└── README.md
```

---

# Responsibilities

### Amulya — Data & Retrieval Layer (`src/`)

* PDF ingestion and text cleaning
* Structured chunking by article/section
* Embedding generation (sentence-transformers)
* Pinecone indexing
* Vector retrieval

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

* `query_interface.py` — query intake and CLI
* `rlm_engine.py` — LLM query decomposition
* `reasoning_orchestrator.py` — retrieval orchestration
* `answer_synthesizer.py` — answer synthesis and citation formatting

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

### 1. Ingest and index documents (run once)

```
python src/process_all_pdfs.py
```

Reads PDFs from `data/raw/`, chunks them by article/section, generates embeddings, and uploads 395 chunks to the Pinecone index. Only needed once per environment.

---

### 2. Agentic RAG demo

`test_agentic_rag.py` runs the `AgenticRAGPipeline` and shows the full agent loop at every stage.

**Quick mode (3 questions, compact output):**
```
python test_agentic_rag.py
```

**Full mode (all 18 ground-truth questions):**
```
python test_agentic_rag.py --mode full
```

**Verbose mode — shows every stage including the agent loop:**
```
python test_agentic_rag.py --verbose
```

**Single custom question:**
```
python test_agentic_rag.py --question "What are the bias audit requirements under NYC Local Law 144?"
```

**What verbose mode prints per query:**

```
[Stage 1] Query Decomposition
  Jurisdictions detected: NYC Local Law 144
  Sub-questions (1): [NYC Local Law 144] What are the bias audit requirements...

[Stage 2] Agent Loop  ← NEW vs RLM pipeline
  [NYC Local Law 144]  2 iteration(s)  →  8 chunks  avg_score=0.847

    iter 1
      tool:       semantic  (conceptual question — semantic search)
      query:      What are the bias audit requirements under NYC Local Law 144
      new chunks: 5
      reflect:    sufficient=False | Evidence quality low, missing penalty provisions

    iter 2
      tool:       hybrid  (retry: switching from semantic to hybrid)
      query:      NYC Local Law 144 bias audit requirements obligations compliance
      new chunks: 3
      reflect:    sufficient=True | Evidence covers audit timing, scope, and exemptions

[Stage 3] Grounded Answer Synthesis
  Answer (842 chars): ...

[Stage 4] Citation Validation
  [PASS] 3/3 citations matched to retrieved chunks
  Hallucination rate: 0.0%  (target: ≤10%)
```

**Compact mode adds `iters=` and `tools=` columns:**
```
[01] OK    3241ms  chunks=8  iters=2.0  tools=semantic,hybrid  cit=3/3  | What are the bias...
```

**Aggregate metrics:**

| Metric | Target |
|---|---|
| Success rate | ≥ 95% |
| Citation accuracy | ≥ 90% |
| Hallucination rate | ≤ 10% |
| Latency p90 | ≤ 10,000 ms |
| Avg iterations/juris | shows agent loop depth |

---

### 3. Interactive CLI

```
python query_interface.py
```

Prompts for a compliance question, runs optional clarification (up to 2 follow-up questions), then runs the full pipeline and displays the answer, citations, and validation stats.

---

### 4. Unit and integration tests

```
python -m pytest tests/ -v
```

All 154 tests run fully offline with mocked retrievers and LLM responses.

---

### 5. Pipeline comparison benchmark

Compare all three pipelines side-by-side on the same 18 ground-truth questions:

```
python eval/benchmark_pipelines.py
```

**Quick mode (first 5 questions):**
```
python eval/benchmark_pipelines.py --mode quick
```

**Verbose — per-question breakdown:**
```
python eval/benchmark_pipelines.py --verbose --save
```

**Run a subset of systems:**
```
python eval/benchmark_pipelines.py --systems rlm agentic_rag
```

**Example output:**

```
  Pipeline Comparison Benchmark
  ──────────────────────────────────────────────────────────────────────
  Metric                              standard_rag               rlm       agentic_rag
  ──────────────────────────────────────────────────────────────────────
  Quality
    Hallucination rate (all)              18.3%             8.1%             4.2%
    Citation accuracy                     81.7%            91.9%            95.8%
    Success rate                         100.0%           100.0%           100.0%

  Retrieval
    Avg chunks / query                      5.0              8.3             11.4
    Avg iterations / juris                  N/A              N/A             1.8

  Latency
    p50                                  1820 ms          2340 ms          4120 ms
    p90                                  2940 ms          3810 ms          7200 ms

  Improvement over standard_rag
    rlm          ↓ 10.2%  (better)
    agentic_rag  ↓ 14.1%  (better)
```

---

### 6. Full evaluation suite

```
python eval/run_all_evals.py --mode full --verbose
```

Runs four evaluation suites in sequence:

| Suite | Script | What it measures |
|---|---|---|
| Retrieval quality | `evaluate_retrieval.py` | Precision@5, Recall@5, F1, MRR |
| Citation accuracy | `evaluate_citations.py` | Precision, Recall, Hallucination Rate |
| Answer correctness | `evaluate_answers.py` | LLM-judge scores (0–5) for relevance, accuracy, completeness |
| End-to-end performance | `evaluate_e2e.py` | Success rate, latency p50/p90/p99 |

Saves a timestamped JSON report to `eval/results/`.

Individual suites:
```
python eval/evaluate_retrieval.py --top-k 5 --verbose
python eval/evaluate_citations.py --mode full
python eval/evaluate_answers.py --mode quick
python eval/benchmark_hallucination.py --mode full --verbose --save
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
- [x] End-to-end grounded responses with citations (`test_agentic_rag.py`)
- [x] Agentic RAG pipeline with dynamic tool selection + LLM reflection loop (`agentic_rag_pipeline.py`)
- [x] 3-way pipeline benchmark: standard_rag vs RLM vs agentic_rag (`eval/benchmark_pipelines.py`)

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
