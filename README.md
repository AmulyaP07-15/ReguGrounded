# ReguGrounded

A multi-stage RAG pipeline for regulatory compliance Q&A, grounded in retrieved evidence with built-in citation validation and a corrective retry loop.

The project benchmarks four systems - Standard RAG, Agentic RAG (ARAG), Cache-Augmented Generation baseline (CAG-cache), and ReguGrounded (CAG) - across 53 curated questions spanning EU AI Act, NYC Local Law 144, Colorado AI Act, NIST AI RMF, and cross-jurisdictional scenarios.

---

## Architecture

### Four Systems Compared

| System | Query Decomposition | Retrieval | Validation | Corrective Retry |
|---|---|---|---|---|
| Standard RAG | No | Pinecone top-5 | Post-hoc only | No |
| ARAG | Yes (RLM) | Pinecone multi-retrieval | Post-hoc only | No |
| CAG-cache | No | Full corpus in context | Post-hoc only | No |
| CAG (ReguGrounded) | Yes (RLM) | Full corpus in context | Built-in | Yes |

### ReguGrounded Pipeline

```
User Query
     ↓
Query Interface      - input guardrails, trace ID, rate limiting
     ↓
Query Clarifier      - optional ambiguity detection (interactive mode)
     ↓
RLM Engine           - LLM decomposition into jurisdiction-specific sub-questions
     ↓
Corpus Loader        - full regulatory corpus loaded from data/chunks/ (one-time,
                       ~130k tokens across 395 chunks; cached in-process)
     ↓
Answer Synthesizer   - LLM synthesis grounded in full corpus
                       (Anthropic: corpus in cached system message for KV reuse)
     ↓
Citation Validator   - hallucination detection: checks every cited source/article
     ↓                 against corpus chunk metadata
  [if hallucination_rate > 10%]
     ↓
Corrective Retry     - re-synthesize with forbidden citation list injected
     ↓
Grounded Response + Citations
```

---

## Repository Structure

```
regugrounded/
│
├── __init__.py                     # Package entry point
├── query_interface.py              # CLI + programmatic API (main entry point)
├── rlm_engine.py                   # LLM query decomposition
├── reasoning_orchestrator.py       # Parallel retrieval orchestration (used by ARAG baseline)
├── answer_synthesizer.py           # Grounded answer synthesis
├── citation_validator.py           # Hallucination detection
├── query_clarifier.py              # Interactive clarification
├── test_agentic_rag.py             # End-to-end demo + trace script for the agentic pipeline
│
├── src/                            # Data & retrieval layer
│   ├── process_all_pdfs.py         # One-time ingestion pipeline
│   ├── pdf_ingestion.py            # PDF text extraction and cleaning
│   ├── chunker.py                  # Split by article/section with metadata
│   ├── embedder.py                 # Embeddings + Pinecone upload
│   ├── retriever.py                # Base Pinecone retrieval
│   ├── enhanced_retriever.py       # Hybrid BM25 + semantic + reranking
│   ├── bm25_index.py               # Legal citation-aware BM25 index
│   ├── query_expander.py           # Regulatory synonym expansion
│   ├── reranker.py                 # Cross-encoder reranking (lazy-loaded)
│   ├── retrieval_config.py         # Feature toggles
│   └── fix_colorado_sections.py    # One-time patch: map bill sections to CRS section numbers
│
├── utils/                          # Infrastructure
│   ├── llm_client.py               # Unified LLM client (Anthropic / Groq / Gemini)
│   ├── cache.py                    # QueryCache (LRU + SQLite) + ComponentCache
│   ├── guardrails.py               # InputGuard + OutputGuard
│   ├── rate_limiter.py             # Token rate limiting
│   ├── logger.py                   # Structured logging + decorators
│   ├── metrics.py                  # Latency and accuracy tracking
│   └── trace.py                    # Request trace ID generation
│
├── scripts/
│   └── manage_cache.py             # CLI for inspecting and clearing query/component caches
│
├── cache/                          # Runtime cache files (not committed)
│   ├── bm25_index.pkl              # Serialised BM25 index
│   └── query_cache.db              # SQLite query result cache
│
├── eval/                           # Evaluation suite
│   ├── compare_models.py           # Three-way benchmark (Standard RAG / ARAG / CAG)
│   ├── run_all_evals.py            # Master script: runs all four eval suites in sequence
│   ├── evaluate_citations.py       # Citation precision / recall / hallucination rate
│   ├── evaluate_answers.py         # LLM-judge answer correctness (0–5 scale)
│   ├── evaluate_e2e.py             # End-to-end latency and success rate
│   ├── evaluate_retrieval.py       # Retrieval Recall@k / Precision@k / F1
│   ├── evaluate_retrieval_per_model.py  # Per-model retrieval breakdown
│   ├── benchmark_hallucination.py  # Hallucination stress-test across exhaustive questions
│   ├── ground_truth.json           # 53 annotated questions (IDs 1–53)
│   ├── questions/                  # Per-jurisdiction question files
│   │   ├── eu_ai_act.json          # 16 questions (Q1–16)
│   │   ├── nyc_local_law_144.json  # 10 questions (Q17–26)
│   │   ├── colorado_ai_act.json    # 10 questions (Q27–36)
│   │   ├── nist_ai_rmf.json        # 8 questions  (Q37–44)
│   │   └── cross_jurisdictional.json # 9 questions (Q45–53)
│   ├── baselines/
│   │   ├── standard_rag.py         # Baseline: single retrieve → LLM, no decomposition
│   │   ├── agentic_rag.py          # Baseline: RLM decomposition + Pinecone retrieval, no validation
│   │   └── cache_augmented_rag.py  # Baseline: full corpus in context, no decomposition, no validation
│   └── results/                    # Timestamped JSON benchmark reports
│
├── tests/                          # 154-test offline suite (all passing)
│   ├── test_cache.py
│   ├── test_guardrails.py
│   ├── test_integration.py
│   ├── test_rate_limiter.py
│   └── test_retrieval_improvements.py
│
├── data/
│   ├── raw/                        # Source PDFs (not committed)
│   ├── processed/                  # Cleaned text per document
│   └── chunks/                     # Structured JSON chunks with metadata
│
├── logs/                           # Timestamped log files
├── requirements.txt
├── .env                            # API keys (not committed)
└── README.md
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/AmulyaP07-15/ReguGrounded.git
cd ReguGrounded
python3 -m venv regugrounded_env
source regugrounded_env/bin/activate
pip install -r requirements.txt
```

### 2. Configure API keys

Create a `.env` file in the project root. The LLM client uses the first available provider in priority order: **Anthropic → Groq → Gemini**.

```
# LLM provider - set ONE of the following blocks:

# Option A: Anthropic (Claude)
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_MODEL=claude-haiku-4-5-20251001   # optional, this is the default

# Option B: Groq (free tier, 30 RPM / 15,000 TPM)
GROQ_API_KEY=your_key_here
GROQ_API_KEY_2=your_second_key   # optional - rotated on 429
GROQ_API_KEY_3=your_third_key    # optional

# Option C: Gemini (rotates across up to 3 keys on quota errors)
GEMINI_API_KEY=your_key_here
GEMINI_API_KEY_2=your_second_key  # optional
GEMINI_API_KEY_3=your_third_key   # optional

# Retrieval (required)
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=regugrounded
```

Do **not** commit `.env` to GitHub.

### 3. Ingest documents (run once)

```bash
python src/process_all_pdfs.py
```

Reads PDFs from `data/raw/`, chunks by article/section, generates embeddings, and uploads to Pinecone. Covers 5 regulatory documents: EU AI Act, EU AI Act Annex, NYC Local Law 144, Colorado AI Act, NIST AI RMF.

---

## Usage

### Programmatic API

```python
from query_interface import process_query

result = process_query("What are the bias audit requirements under NYC Local Law 144?")
print(result["answer"])

# Result keys:
# answer            - grounded natural-language response
# citations         - list of {source, article, excerpt, jurisdiction}
# validation        - {valid_citations, invalid_citations, hallucination_rate}
# metadata          - latency_ms, total_chunks_retrieved, corrective_retry_used, ...
```

### Interactive CLI

```bash
python query_interface.py
```

### Three-way benchmark

`compare_models.py` compares Standard RAG, ARAG, and CAG (ReguGrounded) side by side. A standalone `eval/baselines/cache_augmented_rag.py` script is also available to run the CAG-cache baseline independently.

```bash
# Quick mode - 10 curated questions (includes exhaustive multi-citation questions)
python eval/compare_models.py --mode quick

# Full mode - all 53 ground-truth questions
python eval/compare_models.py --mode full

# Verbose: print per-question traces for each model
python eval/compare_models.py --mode quick --verbose
```

### Full evaluation suite

```bash
# Run all four eval stages (retrieval, citations, answers, end-to-end)
python eval/run_all_evals.py

# Quick mode (5 questions) or skip a stage
python eval/run_all_evals.py --mode quick
python eval/run_all_evals.py --mode full --skip retrieval
```

### Run tests

```bash
python -m pytest tests/ -v
```

All 154 tests run fully offline with mocked retrievers and LLM responses.

---

## Evaluation Metrics

The three-way benchmark (`eval/compare_models.py`) measures Standard RAG, ARAG, and CAG (ReguGrounded) across:

| Metric | Description |
|---|---|
| `citation_accuracy` | % of cited sources/articles matched to retrieved chunks |
| `hallucination_rate` | % of citations NOT found in retrieved evidence |
| `avg_citations` | Average structured citations per answer |
| `latency_ms` | End-to-end wall-clock time per query |
| `corrective_action_taken` | CAG (ReguGrounded) only - how often the corrective retry loop fired |
| `fallback_rate` | % of queries where LLM was unavailable (excluded from accuracy stats) |

### Why the systems differ

**Standard RAG** - single retrieve (top-5 chunks), one LLM call. No decomposition means multi-jurisdiction questions get half the evidence. No validation means hallucinated citations go uncorrected.

**ARAG** - decomposes the query into jurisdiction-specific sub-questions, retrieves per sub-question from Pinecone using hybrid BM25 + semantic search and cross-encoder reranking. More coverage than Standard RAG, but still no corrective loop: if the LLM invents a citation, it stays in the answer.

**CAG-cache** - no retrieval at all. All 395 regulatory chunks (~130k tokens) are loaded into the LLM's context window once at startup. When using Anthropic, the corpus is placed in a cached system message (`cache_control=ephemeral`) so the KV state is reused across queries — only the question is re-encoded per call. Because the full corpus is always in context, exhaustive multi-citation questions cannot fail due to missed retrieval. No decomposition or validation.

**CAG (ReguGrounded)** - builds on CAG-cache: adds RLM query decomposition into jurisdiction-specific sub-questions, plus a citation validator that checks every cited article against the corpus chunk metadata. If `hallucination_rate > 10%`, the answer is re-synthesized with a corrective constraint that explicitly forbids the invalid citations. Exhaustive questions (requiring 6–13 citations) force hallucination in both retrieval-based baselines; only CAG recovers.

---

## Key Engineering Challenges

| # | Challenge | Solution |
|---|---|---|
| 1 | **Hybrid score fusion** - BM25 and cosine similarity are on incompatible scales | Normalise BM25 to [0,1]; dynamically increase keyword weight for queries containing legal citations or ≤3 tokens |
| 2 | **Jurisdiction filtering breaks multi-jurisdiction queries** | Return `None` (no filter) when the query contains multiple jurisdiction signals or comparison language ("vs", "differ", "compare") |
| 3 | **Query expansion inflates noise for official citations** | Skip expansion entirely when the query already contains precise regulatory language (`Article N`, `§ X-Y`) |
| 4 | **Cross-encoder reranking latency** | Lazy singleton load + TTL score cache keyed on query hash + chunk ID; repeated queries skip model inference |
| 5 | **LLM provider rate limits** | Unified `utils/llm_client.py` rotates across Anthropic → Groq → Gemini; Groq throttled at 2.4s/call; up to 3 keys per provider rotated on 429 |
| 6 | **Short-form citation matching producing false negatives** | Added exhaustive multi-citation questions (Q16: 13 citations, Q52: 9, Q53: 6) that exceed retrieval coverage, forcing hallucination even with lenient matching |

---

## Statement of Contributions

**Prisha Srivastava** designed and implemented the reasoning layer, including the RLM query decomposition engine (`rlm_engine.py`), reasoning orchestrator (`reasoning_orchestrator.py`), answer synthesizer (`answer_synthesizer.py`), and query interface (`query_interface.py`). Prisha also designed and implemented the evaluation framework, including the citation, answer quality, and end-to-end evaluation scripts, and the multi-pipeline comparison across Standard RAG, Agentic RAG, and ReguGrounded.

<<<<<<< HEAD
**Amulya Penikalapati** designed and implemented the data and retrieval layer, including PDF ingestion, structured chunking, embedding generation, and Pinecone indexing. Amulya also built the hybrid retrieval pipeline combining BM25 keyword search with semantic search, jurisdiction filtering, query expansion, and cross-encoder reranking, as well as the full utility layer covering caching, guardrails, logging, metrics, and rate limiting. Amulya also implemented the Cached Augmented Generation (CAG) architecture, including the Reasoning Language Model for query decomposition, the citation validation system, and the corrective retry loop that actively detects and fixes hallucinated citations.
=======
**Amulya Penikalapati** designed and implemented the data and retrieval layer, including PDF ingestion, structured chunking, embedding generation, and Pinecone indexing. Amulya also built the hybrid retrieval pipeline combining BM25 keyword search with semantic search, jurisdiction filtering, query expansion, and cross-encoder reranking, as well as the full utility layer covering caching, guardrails, logging, metrics, and rate limiting. Amulya also implemented the Cache-Augmented Generation (CAG) architecture, including the Reasoning Language Model for query decomposition, the full corpus loading and Anthropic prompt caching integration, the citation validation system, and the corrective retry loop that actively detects and fixes hallucinated citations.
>>>>>>> ea845bd (Update README: fix architecture descriptions, repo structure, and CAG naming)

---

## Ground Truth

53 curated questions with expected jurisdiction coverage and citation counts:

| Jurisdiction | IDs | Count |
|---|---|---|
| EU AI Act | Q1–Q16 | 16 |
| NYC Local Law 144 | Q17–Q26 | 10 |
| Colorado AI Act | Q27–Q36 | 10 |
| NIST AI RMF | Q37–Q44 | 8 |
| Cross-jurisdictional | Q45–Q53 | 9 |

Includes 3 exhaustive multi-citation questions (Q16: 13 expected citations, Q52: 9, Q53: 6) designed to force hallucination in systems that cannot retrieve all relevant articles.
