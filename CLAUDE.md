# Project Context

## What
A multimodal Retrieval Augmented Generation (RAG) system built to query the IFC Annual Report 2024 (financials). Handles three content types: text, tables, and images/charts. Starts as a naive text-only RAG (Phase 1) and progressively adds table extraction, image captioning, advanced retrieval, re-ranking, and evaluation. 

## Why
Financial reports mix narrative text, dense tables, and visual charts - a naive text-only RAG misses most of the useful signal. This project is designed to build and compare retrieval strategies across modalities, and to properly evaluate accuracy on complex, multi-part financial questions rather than relying on a single "looks right" pass.

## How
- LLM: Gemini 3.5 Flash, via Google GenAI SDK, Vertex AI auth (no API keys). Use streaming, function calling, and structured/JSON output where relevant.
- Vector DBs: FAISS and Qdrant (local), populate both, compare performance and use cases.
- PDF parsing: Docling / PyMuPDF / Gemini multimodal, for text, tables (Camelot/Tabula or LLM-based), and images (Gemini-generated captions).
- Framework: LangChain for RAG orchestration.
- UI: Streamlit or Gradio.
- Observability: Langfuse for tracing queries and responses.
- Evaluation: RAGAS.
- Deployment: Docker.

### Phased approach
1. Data parsing: extract text, tables, and images separately, with metadata and cross-referencing where feasible.
2. Phase 1 - Naive text RAG: chunk text, embed, index in FAISS + Qdrant, retrieve, generate via Gemini, basic Streamlit/Gradio UI, Langfuse tracing.
3. Later phases (not yet detailed): incorporate tables and images into retrieval, add re-ranking, and formal RAGAS-based evaluation.

## Critical rules
- Since we add new components progressively, we will progressively update the CLAUDE.md file. New required components are added per each new project phase. 
- Commit and push at reasonable intervals