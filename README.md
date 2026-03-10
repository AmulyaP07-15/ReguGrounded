# ReguGrounded — Phase 1 Setup Guide

## Overview

ReguGrounded is a regulatory compliance assistant that answers multi-jurisdiction AI governance questions using **grounded generation with citations**.

The system combines:

* **LLM reasoning**
* **vector retrieval**
* **structured evidence synthesis**

The goal is to prevent hallucinated regulatory answers by grounding all responses in real regulatory text.

Phase 1 focuses on building the **core pipeline** before adding evaluation and agentic retrieval.

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
├── query_interface.py
├── rlm_engine.py
├── reasoning_orchestrator.py
├── answer_synthesizer.py
├── retriever.py
│
├── requirements.txt
├── .env
└── README.md
```

---

# Responsibilities

### Amulya — Data & Retrieval Layer

Responsible for:

* PDF ingestion
* structured chunking
* embedding generation
* Pinecone indexing
* vector retrieval

Expected modules:

```
pdf_ingestion.py
chunking.py
embedding_pipeline.py
retriever.py
```

The only function the reasoning layer requires is:

```python
retrieve(query: str, top_k: int) -> list[dict]
```

Expected return format:

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

---

### Prisha — Reasoning Layer

Modules:

```
query_interface.py
rlm_engine.py
reasoning_orchestrator.py
answer_synthesizer.py
```

Responsibilities:

* query intake
* LLM query decomposition
* retrieval orchestration
* answer synthesis
* citation formatting

---

# Environment Setup

### 1. Clone the repository

```
git clone <repo>
cd ReguGrounded
```

---

### 2. Create the environment

```
python3 -m venv RegGrounded_env
```

---

### 3. Activate the environment

macOS / Linux

```
source RegGrounded_env/bin/activate
```

You should see:

```
(RegGrounded_env)
```

---

### 4. Install dependencies

```
pip install -r requirements.txt
```

---

# API Key Setup

Create a `.env` file in the project root.

```
OPENAI_API_KEY=your_key_here
```

This key is used for:

* query decomposition
* answer synthesis

Do **not commit `.env` to GitHub**.

---

# Running the System

Run the CLI interface:

```
python query_interface.py
```

Example query:

```
What bias audit requirements apply to AI hiring systems under NYC Local Law 144 and the EU AI Act?
```

Expected pipeline flow:

```
Query
↓
Decomposition
↓
Retrieval
↓
Evidence synthesis
↓
Grounded answer
```

---

# Testing Without Pinecone

Until the retrieval layer is finished, the system can run using a **mock retriever**.

`retriever.py` currently returns sample regulatory snippets so the reasoning pipeline can be tested independently.

Once the Pinecone retriever is implemented, simply replace the mock function.

---

# Development Sequence

1. Complete PDF ingestion and chunking
2. Generate embeddings and index Pinecone
3. Implement `retrieve(query)`
4. Connect retriever to reasoning pipeline
5. Produce grounded responses with citations

---

# Notes

* The LLM **never answers directly from its own knowledge**
* All answers must be grounded in retrieved regulatory text
* Every response must include **citations and supporting evidence**

---

# Future Phases

Phase 2 will include:

* agentic RAG (A-RAG)
* evaluation metrics
* hallucination detection
* benchmark comparisons
