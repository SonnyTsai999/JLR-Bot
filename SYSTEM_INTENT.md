# JLR Technology Intelligence Assistant — Development Intent & System Constraints

## 1. Project Purpose

This project implements a **Retrieval-Augmented Generation (RAG)-based Technology Intelligence Assistant** for J.L. Richards (JLR).

The system transforms a curated academic journal corpus into structured, citation-grounded strategic insights for emerging AEC design technologies.

- **This is not** a general chatbot.
- **This is not** a document distribution system.
- **It is** a decision-support intelligence tool.

## 2. Core Objectives

The system must:

- Accept natural language queries.
- Retrieve relevant evidence from an embedded academic corpus.
- Generate structured analytical insights.
- Provide APA citation and DOI references.
- Avoid redistribution of copyrighted material.
- Operate as a strategic synthesis engine.

## 3. Non-Goals (Hard Constraints)

The system must **NOT**:

- Expose full PDF content.
- Allow reconstruction of entire papers.
- Act as a general-purpose LLM assistant.
- Provide unsupported or hallucinated citations.
- Retrieve external internet data.
- Replace expert human judgment.

## 4. High-Level Architecture

### Pipeline Overview

```
PDF → Text Extraction → Cleaning → Chunking → Metadata Enrichment
      ↓
Embedding Model → FAISS Vector Index
      ↓
Query Embedding → Metadata Filter → Top-K Retrieval
      ↓
LLM Synthesis (Structured Output Template)
      ↓
User Interface
```

## 5. Corpus Processing Requirements

### 5.1 Input

- Peer-reviewed journal PDFs
- Legally accessed via institutional subscription

### 5.2 Processing Rules

- Remove reference sections
- Remove appendices
- Remove tables where possible
- Normalize whitespace
- Preserve semantic paragraph integrity

### 5.3 Chunking

- Chunk size: 500–800 tokens
- Overlap: 100–150 tokens
- Do not break mid-sentence

## 6. Metadata Schema

Each chunk must include structured metadata:

```json
{
  "source": "filename.pdf",
  "title": "...",
  "year": 2022,
  "technology_L1": "AI",
  "technology_L2": "Digital Twin",
  "lifecycle_stage": "Emerging | Growth | Mature | Decline",
  "apa_citation": "...",
  "doi": "..."
}
```

Metadata filtering must be supported at retrieval stage.

## 7. Embedding & Storage

- Use OpenAI embedding model (`text-embedding-3-large` or latest equivalent).
- Store embeddings in FAISS.
- In deployment version:
  - Do not store full raw PDFs.
  - Store only chunked text and metadata.
  - No internet retrieval.

## 8. Retrieval Requirements

- Top-K semantic similarity search.
- Metadata-aware filtering (technology category, lifecycle stage).
- Prevent excessive chunk return.
- Avoid retrieval redundancy.

## 9. LLM Output Requirements

All responses must follow structured template:

### Output Format

1. **Executive Summary**
2. **Key Findings**
3. **Adoption Barriers**
4. **Technology Maturity Signal**
5. **Strategic Implications for JLR**
6. **APA References**

### Rules

- Use only retrieved evidence.
- Do not invent citations.
- If insufficient evidence → explicitly state.
- No raw long excerpts (>500 words).
- No verbatim full section reproduction.

## 10. Ethical Safeguards

The system must:

- Provide citation and DOI.
- Limit excerpt length.
- Avoid full-text reproduction.
- Function as a transformative analytical tool.

## 11. Intended Use Cases

- Compare emerging technologies
- Extract adoption barriers
- Assess maturity signals
- Support strategic technology prioritization
- Inform internal JLR discussions

## 12. Future Integration (Optional)

The architecture should allow future integration with:

- TLC curve data
- MCDM/VIKOR ranking scores
- Practitioner sentiment analysis
- Barrier intensity scoring

Design system to be **modular and extensible**.

## 13. Code Structure Expectations

```
/data
    /raw_pdfs
    /processed_chunks
/index
/backend
    ingest.py
    embed.py
    retrieve.py
    synthesize.py
/frontend
/config
SYSTEM_INTENT.md
README.md
```

All modules should be cleanly separated: **Ingestion** | **Embedding** | **Retrieval** | **Generation** | **Interface**.

## 14. System Identity

When prompting the LLM, it must assume the role of:

> **"Technology Strategy Analyst for JLR specializing in emerging AEC design technologies."**

This prevents generic responses.

## 15. Quality Principles

- Evidence-grounded
- Metadata-aware
- Structured output
- Strategic orientation
- Copyright-safe
- Deterministic where possible

---

*End of Development Intent*
