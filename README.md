# ReguGrounded

## Overview

ReguGrounded is a regulatory compliance assistant that answers multi-jurisdiction AI governance questions using **grounded generation with citations**.

The system combines:

* **LLM reasoning**
* **vector retrieval**
* **structured evidence synthesis**

The goal is to prevent hallucinated regulatory answers by grounding all responses in real regulatory text.

---

# System Architecture

```
User Query
     ↓
Query Interface
     ↓
RLM Decomposition (LLM reasoning)
     ↓
Sub-Questions
     ↓
Vector Retrieval (Pinecone)
     ↓
Evidence Collection
     ↓
Answer Synthesis
     ↓
Grounded Response + Citations
```

---

# Repository Structure

```
ReguGrounded/
│
├── query_interface.py          # CLI entry point
├── rlm_engine.py               # LLM query decomposition
├── reasoning_orchestrator.py   # Retrieval orchestration
├── answer_synthesizer.py       # Answer + citation formatting
│
├── src/                        # Data & retrieval layer
│   ├── pdf_ingestion.py        # Extract and clean text from PDFs
│   ├── chunker.py              # Split by article/section with metadata
│   ├── embedder.py             # Generate embeddings + upload to Pinecone
│   ├── process_all_pdfs.py     # Run full ingestion pipeline
│   └── retriever.py            # Query Pinecone and return ranked chunks
│
├── data/
│   ├── raw/                    # Source PDFs (not committed)
│   ├── processed/              # Cleaned text per document
│   └── chunks/                 # Structured JSON chunks with metadata
│
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

# Environment Setup

### 1. Clone the repository

```
git clone https://github.com/AmulyaP07-15/ReguGrounded.git
cd ReguGrounded
```

### 2. Create and activate the environment

```
python3 -m venv RegGrounded_env
source RegGrounded_env/bin/activate
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

---

# Development Progress

- [x] PDF ingestion and text cleaning
- [x] Structured chunking (395 chunks across 5 documents)
- [x] Embedding generation and Pinecone indexing
- [x] `retrieve(query)` interface
- [ ] Connect retriever to reasoning pipeline
- [ ] End-to-end grounded responses with citations

---

# Notes

* The LLM **never answers directly from its own knowledge**
* All answers must be grounded in retrieved regulatory text
* Every response must include **citations and supporting evidence**

---

# Future Phases

Phase 2 will include:

* Agentic RAG (A-RAG)
* Evaluation metrics
* Hallucination detection
* Benchmark comparisons