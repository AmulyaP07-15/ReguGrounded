# ReguGrounded

A multi-stage RAG pipeline for regulatory compliance Q&A, grounded in retrieved evidence with built-in citation validation and a corrective retry loop.

The project benchmarks three systems - Standard RAG, Agentic RAG (ARAG), and ReguGrounded (Cache-Augmented Generation with corrective retry) - across 53 curated questions spanning EU AI Act, NYC Local Law 144, Colorado AI Act, NIST AI RMF, and cross-jurisdictional scenarios.

---

## Architecture

### Three Systems Compared

| System | Query Decomposition | Retrieval | Validation | Corrective Retry |
|---|---|---|---|---|
| Standard RAG | No | Pinecone top-5 | Post-hoc only | No |
| ARAG | Yes (RLM) | Pinecone multi-retrieval | Post-hoc only | No |
| ReguGrounded (CAG) | Yes (RLM) | Full corpus in context | Built-in | Yes |

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
│
├── src/                            # Data & retrieval layer
│   ├── process_all_pdfs.py         # One-time ingestion pipeline
│   ├── pdf_ingestion.py            # PDF text extraction and cleaning
│   ├── chunker.py                  # Split by article/section with metadata
│   ├── embedder.py                 # Embeddings + Pinecone upload
│   └── retriever.py                # Pinecone retrieval (used by ARAG baseline)
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
├── cache/
│   └── query_cache.db              # SQLite query result cache (not committed)
│
├── eval/                           # Evaluation suite
│   ├── compare_models.py           # Three-way benchmark (Standard RAG / ARAG / CAG)
│   ├── run_all_evals.py            # Master script: runs all four eval suites in sequence
│   ├── evaluate_citations.py       # Citation precision / recall / hallucination rate
│   ├── evaluate_answers.py         # LLM-judge answer correctness (0–5 scale)
│   ├── evaluate_e2e.py             # End-to-end latency and success rate
│   ├── evaluate_retrieval.py       # Retrieval Recall@k / Precision@k / F1
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
│   │   └── agentic_rag.py          # Baseline: RLM decomposition + Pinecone retrieval, no validation
│   └── results/                    # Timestamped JSON benchmark reports
│
├── tests/                          # Offline test suite (all passing)
│   ├── test_cache.py
│   ├── test_guardrails.py
│   ├── test_integration.py
│   └── test_rate_limiter.py
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

### Sample query and output

```python
result = process_query("What bias audit requirements apply to AI hiring tools in New York City?")
```

```
Question: What bias audit requirements apply to AI hiring tools in New York City?

Sub-questions:
  1. What bias audit requirements apply to AI hiring tools in New York City? Focus only on NYC Local Law 144.

Answer:
NYC Local Law 144 § 20-871 requires employers and employment agencies that use an
automated employment decision tool (AEDT) to conduct a bias audit no more than one
year prior to the tool's use. The bias audit must be conducted by an independent
auditor and must calculate selection rates and score distributions across sex, race,
and ethnicity categories. NYC Local Law 144 § 20-872 requires employers to notify
candidates who are residents of New York City at least ten business days before using
an AEDT, and to make the bias audit summary publicly available on their website.

Citations (2):
  - NYC Local Law 144 § 20-871
  - NYC Local Law 144 § 20-872

Validation: 100.0% citations valid (2/2)

Metadata:
  Jurisdictions: NYC Local Law 144
  Chunks retrieved: 395
  Latency: 3241 ms
```

### Interactive CLI

```bash
python query_interface.py
```

### Three-way benchmark

`compare_models.py` compares Standard RAG, ARAG, and CAG (ReguGrounded) side by side.

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

### Why the three systems differ

**Standard RAG** - single retrieve (top-5 chunks), one LLM call. No decomposition means multi-jurisdiction questions get half the evidence. No validation means hallucinated citations go uncorrected.

**ARAG** - decomposes the query into jurisdiction-specific sub-questions and retrieves per sub-question from Pinecone. More coverage than Standard RAG, but still no corrective loop: if the LLM invents a citation, it stays in the answer.

**ReguGrounded (CAG)** - loads the full regulatory corpus (~130k tokens, 395 chunks) into the LLM's context window once at startup instead of retrieving. When using Anthropic, the corpus is placed in a cached system message (`cache_control=ephemeral`) so the KV state is reused across queries. Adds RLM query decomposition and a citation validator that checks every cited article against the corpus chunk metadata. If `hallucination_rate > 10%`, the answer is re-synthesized with a corrective constraint that explicitly forbids the invalid citations. Exhaustive questions (requiring 6–13 citations) force hallucination in both baselines; only ReguGrounded recovers.

---

## Key Engineering Challenges

| # | Challenge | Solution |
|---|---|---|
| 1 | **Jurisdiction filtering breaks multi-jurisdiction queries** | Return `None` (no filter) when the query contains multiple jurisdiction signals or comparison language ("vs", "differ", "compare") |
| 2 | **LLM provider rate limits** | Unified `utils/llm_client.py` rotates across Anthropic → Groq → Gemini; Groq throttled at 2.4s/call; up to 3 keys per provider rotated on 429 |
| 3 | **Short-form citation matching producing false negatives** | Added exhaustive multi-citation questions (Q16: 13 citations, Q52: 9, Q53: 6) that exceed retrieval coverage, forcing hallucination even with lenient matching |
| 4 | **Colorado chunk metadata wrong** | Chunks were stored with bill section labels ("SECTION 1") instead of CRS section numbers ("§ 6-1-1701") — citation recall was 0% until metadata was re-mapped from chunk text |
| 5 | **Large context KV cost** | Full corpus (~130k tokens) placed in Anthropic cached system message (`cache_control=ephemeral`) so KV state is reused across queries; only the question is re-encoded per call |

---

## Statement of Contributions

**Prisha Srivastava** designed and implemented the reasoning layer, including the RLM query decomposition engine (`rlm_engine.py`), reasoning orchestrator (`reasoning_orchestrator.py`), answer synthesizer (`answer_synthesizer.py`), and query interface (`query_interface.py`). Prisha also designed and implemented the evaluation framework, including the citation, answer quality, and end-to-end evaluation scripts, and the multi-pipeline comparison across Standard RAG, Agentic RAG, and ReguGrounded.

**Amulya Penikalapati** designed and implemented the data and retrieval layer, including PDF ingestion, structured chunking, embedding generation, and Pinecone indexing, as well as the full utility layer covering caching, guardrails, logging, metrics, and rate limiting. Amulya also implemented the Cache-Augmented Generation (CAG) architecture, including the Reasoning Language Model for query decomposition, the full corpus loading and Anthropic prompt caching integration, the citation validation system, and the corrective retry loop that actively detects and fixes hallucinated citations.

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
