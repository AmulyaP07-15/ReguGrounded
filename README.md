# ReguGrounded

A multi-stage RAG pipeline for regulatory compliance Q&A, grounded in retrieved evidence with built-in citation validation and a corrective retry loop.

The project benchmarks three systems — Standard RAG, Agentic RAG (ARAG), and Corrective Agentic Generation (CAG) — across 53 curated questions spanning EU AI Act, NYC Local Law 144, Colorado AI Act, NIST AI RMF, and cross-jurisdictional scenarios.

---

## Architecture

### Three Systems Compared

| System | Query Decomposition | Validation | Corrective Retry |
|---|---|---|---|
| Standard RAG | No | Post-hoc only | No |
| ARAG | Yes (RLM) | Post-hoc only | No |
| CAG (ReguGrounded) | Yes (RLM) | Built-in | Yes |

### CAG Pipeline

```
User Query
     ↓
Query Interface      — input guardrails, trace ID, rate limiting
     ↓
Query Clarifier      — optional ambiguity detection (interactive mode)
     ↓
RLM Engine           — LLM decomposition into jurisdiction-specific sub-questions
     ↓
Reasoning Orchestrator — parallel retrieval across all sub-questions
   ├── Query Expansion       (informal → statutory terminology)
   ├── Jurisdiction Filter   (auto-detect relevant regulation)
   ├── Hybrid Search         (BM25 keyword + Pinecone semantic)
   └── Cross-Encoder Rerank  (precise 2nd-pass scoring)
     ↓
Answer Synthesizer   — LLM synthesis grounded in retrieved text only
     ↓
Citation Validator   — hallucination detection: checks every cited source/article
     ↓                 against retrieved chunk metadata
  [if hallucination_rate > 10%]
     ↓
Corrective Retry     — re-synthesize with forbidden citation list injected
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
├── reasoning_orchestrator.py       # Parallel retrieval orchestration
├── answer_synthesizer.py           # Grounded answer synthesis
├── citation_validator.py           # Hallucination detection
├── query_clarifier.py              # Interactive clarification
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
│   └── retrieval_config.py         # Feature toggles
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
├── eval/                           # Evaluation suite
│   ├── compare_models.py           # Three-way benchmark (Standard RAG / ARAG / CAG)
│   ├── evaluate_answers.py         # LLM-judge answer correctness (0–5 scale)
│   ├── ground_truth.json           # 53 annotated questions (IDs 1–53)
│   ├── questions/                  # Per-jurisdiction question files
│   │   ├── eu_ai_act.json          # 16 questions (Q1–16)
│   │   ├── nyc_local_law_144.json  # 10 questions (Q17–26)
│   │   ├── colorado_ai_act.json    # 10 questions (Q27–36)
│   │   ├── nist_ai_rmf.json        # 8 questions  (Q37–44)
│   │   └── cross_jurisdictional.json # 9 questions (Q45–53)
│   ├── baselines/
│   │   ├── standard_rag.py         # Baseline: single retrieve → LLM, no decomposition
│   │   └── agentic_rag.py          # Baseline: RLM decomposition, no corrective loop
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
# LLM provider — set ONE of the following blocks:

# Option A: Anthropic (Claude)
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_MODEL=claude-haiku-4-5-20251001   # optional, this is the default

# Option B: Groq (free tier, 30 RPM / 15,000 TPM)
GROQ_API_KEY=your_key_here
GROQ_API_KEY_2=your_second_key   # optional — rotated on 429
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
# answer            — grounded natural-language response
# citations         — list of {source, article, excerpt, jurisdiction}
# validation        — {valid_citations, invalid_citations, hallucination_rate}
# metadata          — latency_ms, total_chunks_retrieved, corrective_retry_used, ...
```

### Interactive CLI

```bash
python query_interface.py
```

### Three-way benchmark

```bash
# Quick mode — 10 curated questions (includes exhaustive multi-citation questions)
python eval/compare_models.py --mode quick

# Full mode — all 53 ground-truth questions
python eval/compare_models.py --mode full

# Save results to eval/results/
python eval/compare_models.py --mode quick --save
```

### Run tests

```bash
python -m pytest tests/ -v
```

All 154 tests run fully offline with mocked retrievers and LLM responses.

---

## Evaluation Metrics

The benchmark (`eval/compare_models.py`) measures three systems across:

| Metric | Description |
|---|---|
| `citation_accuracy` | % of cited sources/articles matched to retrieved chunks |
| `hallucination_rate` | % of citations NOT found in retrieved evidence |
| `avg_citations` | Average structured citations per answer |
| `latency_ms` | End-to-end wall-clock time per query |
| `corrective_retries` | CAG only — how often the retry loop fired |
| `fallback_rate` | % of queries where LLM was unavailable (excluded from accuracy stats) |

### Why the three systems differ

**Standard RAG** — single retrieve (top-5 chunks), one LLM call. No decomposition means multi-jurisdiction questions get half the evidence. No validation means hallucinated citations go uncorrected.

**ARAG** — decomposes the query into jurisdiction-specific sub-questions, retrieves per sub-question. More coverage, but still no corrective loop: if the LLM invents a citation, it stays in the answer.

**CAG** — same decomposition and retrieval as ARAG, plus a citation validator that checks every cited article against the retrieved chunk metadata. If `hallucination_rate > 10%`, the answer is re-synthesized with a CORRECTIVE CONSTRAINT that explicitly forbids the invalid citations. Exhaustive questions (requiring 6–13 citations) force hallucination in both baselines; only CAG recovers.

---

## Key Engineering Challenges

### 1. Hybrid score fusion across incompatible scales

BM25 raw scores and Pinecone cosine similarity scores are on completely different scales. A naïve weighted average is dominated by whichever scale happens to produce larger numbers.

**Solution:** BM25 scores are normalised to [0, 1] by dividing by the top result's score. Final hybrid score: `semantic_weight × cosine + keyword_weight × bm25_norm`. Weights are dynamically adjusted — queries with legal citations (`§ 20-871`) or very short queries (≤3 tokens) get a higher keyword weight (0.5 vs 0.4 default).

### 2. Jurisdiction filtering breaking multi-jurisdiction queries

Automatically filtering retrieved chunks to a detected jurisdiction is useful for specific queries ("What does NYC require?") but breaks comparison queries ("How do NYC and the EU AI Act differ?").

**Solution:** `_detect_jurisdictions()` returns `None` (no filter) when the query contains multiple jurisdiction signals or comparison language ("vs", "differ", "compare", "both"). Only unambiguous single-jurisdiction queries trigger a Pinecone metadata filter.

### 3. Query expansion inflating noise for official citations

Expanding a query like "What does Article 9 require?" with synonyms dilutes the precise article match. Expanding "AEDT requirements" is useful; expanding `§ 20-871` is harmful.

**Solution:** `QueryExpander` checks for legal citation patterns (`Article N`, `§ X-Y`) before expanding. If the query already contains precise regulatory language, expansion is skipped entirely.

### 4. Cross-encoder reranking latency

The cross-encoder model (~80 MB) takes ~2 seconds to load and scores each (query, chunk) pair individually.

**Solution:** Lazy singleton loading (model loads once, reused across requests) plus a TTL score cache keyed on `sha256(query)[:16] :: chunk_id`. Cached scores are returned without model inference for repeated queries within 1 hour.

### 5. LLM provider rate limits

Gemini free tier has a strict daily quota. Groq free tier allows 30 RPM / 15,000 TPM — the TPM limit is the real bottleneck on complex regulatory questions.

**Solution:** `utils/llm_client.py` provides a unified client that automatically selects and rotates across providers. Groq is throttled at 2.4s per call (25 RPM) to stay within the 30 RPM free tier. Up to 3 keys per provider are rotated on 429 errors.

### 6. Short-form citation matching in the validator

The citation validator uses regex to match cited sources against retrieved chunks. Short-form references ("Article 9", "§ 20-871") without a law name can match too broadly when many chunks are in context, producing false negatives (missed hallucinations) for ARAG and Standard RAG.

**Solution:** Validator designed to be strict on the CAG path (used for corrective retry decisions) while remaining comparable across systems for benchmark reporting. Exhaustive multi-citation questions (Q16, Q52, Q53) were added specifically to expose hallucination even with lenient short-form matching.

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
