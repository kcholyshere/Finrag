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
- UI: Streamlit.
- Observability: Langfuse for tracing queries and responses.
- Evaluation: RAGAS.
- Deployment: Docker.

### Phased approach
1. Data parsing: extract text, tables, and images separately, with metadata and cross-referencing where feasible.
2. Phase 1 - Naive text RAG: chunk text, embed, index in FAISS + Qdrant, retrieve, generate via Gemini, basic Streamlit UI, Langfuse tracing.
3. Phase 2 - Evaluation (`src/evaluation/`): a 200-row eval set (`data/processed/eval_dataset.csv` - 34 curated rows plus synthetically generated/critique-filtered rows, built by `python -m src.evaluation.synthetic_qa`) evaluated in two layers via `python -m src.evaluation.run_eval`:
   - Per-step diagnostics (`src/evaluation/diagnostics.py`): parsing/chunking coverage (fuzzy match), retrieval Hit Rate@k/MRR plus RAGAS `context_precision`/`context_recall`, and RAGAS `faithfulness`/`answer_relevancy` for generation.
   - End-to-end outcome: RAGAS `answer_correctness`, RAGAS's own LLM-graded metrics serving as the LLM-as-judge experiment.
   - RAGAS runs against Vertex AI Gemini via `src/evaluation/ragas_compat.py` (see ADR-0006 in `agent_docs/decisions.md` for the RAGAS/langchain-community compatibility shim this needs).
   - Each run is saved as a settings-tagged JSON under `data/processed/eval_runs/` so later phases can be compared against this baseline.
4. Later phases (not yet detailed): incorporate tables and images into retrieval, add re-ranking.

## Critical rules
- Since we add new components progressively, we will progressively update the CLAUDE.md file. New required components are added per each new project phase. 
- Commit and push at reasonable intervals
- After every run of the eval pipeline (`python -m src.evaluation.run_eval`), add the result to `reports/eval_results_comparison.md` and update its stage-by-stage diff/notes. That doc tracks an additive ladder (baseline -> +hybrid -> +reranking -> +tables -> +images, one component added per stage) - don't let it drift out of sync with `data/processed/eval_runs/`.
- We're currently in a speed push to get through several phases quickly. Balance speed with understanding: keep messages brief, succinct, and directly relevant to the query - don't diverge into tangents or exhaustive option surveys unless asked. Move fast, but don't skip verifying that things actually work.
- Curate `agent_docs/achievements.md` proactively as we go, using the `resumify` skill, whenever a piece of completed work is genuinely high-impact (a measurable before/after, a non-obvious debugging insight, a real architectural trade-off). Don't wait to be asked, and don't log routine commits or boilerplate.