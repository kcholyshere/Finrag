# Finrag

Multimodal Retrieval Augmented Generation (RAG) over the IFC Annual Report 2024
(Financials). The report mixes narrative text, dense tables, and charts - a
text-only RAG misses most of the useful signal, so this project builds and
compares retrieval strategies across all three content types, and evaluates
answer quality with a two-layer harness rather than a single "looks right"
pass.

Two retrieval pipelines are built and compared side by side:

- a tuned chunk pipeline (Phases 1-5): Docling parsing, table/image
  enrichment, hybrid search, cross-encoder reranking
- a ColPali-style page-image pipeline (Phase 6): ColQwen2 patch embeddings
  and late-interaction (MaxSim) retrieval over whole rendered pages, with no
  parsing, chunking, or enrichment step at all

Both are wired into the same Streamlit UI and the same evaluation harness, so
they can be judged on the same 200-question set.

See `ARCHITECTURE.md` for the module-level design and data flow, and
`reports/` for the full evaluation write-ups.

## Capabilities

- Docling-based parsing of text, tables, and images from the source PDF, with
  page/section metadata carried through to citations
- Table chunks headed by a cached Gemini summary, and image/chart chunks
  captioned (with figures) by Gemini multimodal, so both content types are
  retrievable by natural-language queries, not just bare markdown or pixels
- Hybrid retrieval (BM25 + dense embeddings via `EnsembleRetriever`) with
  cross-encoder reranking as the default pipeline
- A separate ColPali-like pipeline: PyMuPDF page renders, ColQwen2 patch
  embeddings, Qdrant native multivector MAX_SIM retrieval, whole page images
  fed directly to Gemini
- Gemini function calling for arithmetic (deltas, percentage changes, ratios)
  via an AST-sandboxed calculator tool, so the model never computes by hand
- Two-layer RAGAS evaluation: per-step diagnostics (parsing/chunking
  coverage, retrieval Hit Rate@k/MRR, context precision/recall, faithfulness,
  answer relevancy) plus an end-to-end outcome metric (answer correctness),
  against a 200-row dataset (34 curated pairs plus critique-filtered
  synthetic pairs)
- Every evaluation run saved as a settings-tagged JSON so pipeline stages can
  be compared against each other, forming an additive ladder
- Langfuse tracing of every query (retrieval span + generation span nested
  under one trace)
- FAISS and Qdrant populated in parallel from the same chunks for a genuine
  backend comparison (see `reports/faiss_vs_qdrant.md`)

## Results snapshot

FAISS backend, same 200-row eval set, k=4. Each column adds one component on
top of the previous one (additive ladder).

| Metric | Baseline (dense) | +Hybrid | +Reranking | +Tables | +Table fixes | +Images |
|---|---|---|---|---|---|---|
| Hit Rate@4 | 0.860 | 0.945 | 0.940 | 0.945 | 0.930 | 0.935 |
| MRR@4 | 0.7125 | 0.7858 | 0.8600 | 0.8153 | 0.8604 | 0.8629 |
| faithfulness | 0.6571 | 0.6592 | 0.6794 | 0.6800 | 0.6917 | 0.7184 |
| answer_correctness | 0.5963 | 0.6383 | 0.6377 | 0.6501 | 0.6945 | 0.7073 |

Full metrics (context_precision, context_recall, answer_relevancy) and
stage-by-stage commentary are in `reports/eval_results_comparison.md` - that
is the source of truth this table is drawn from, kept in sync after every
eval run.

Phase 6 ColPali comparison (not a ladder column - it swaps the whole
retrieval and context representation rather than adding to the chunk
pipeline; Qdrant only, by necessity of multivector support):

| Metric | +Images (chunks) | ColPali (pages) | Delta |
|---|---|---|---|
| Hit Rate@4 | 0.935 | 0.925 | -0.010 |
| MRR@4 | 0.8629 | 0.8021 | -0.0608 |
| answer_relevancy | 0.8784 | 0.8751 | -0.0033 |
| answer_correctness | 0.7073 | 0.6543 | -0.0530 |

context_precision, context_recall, and faithfulness are not applicable to the
ColPali row: the generator answers from page pixels, so scoring RAGAS
context metrics against the placeholder `page_content` string would measure
nothing real.

Headline reading: a single local VLM with no parsing, no table summaries, no
captions, no BM25, and no cross-encoder lands within 0.01 Hit Rate of a
pipeline that took four phases of tuning. Retrieval is near-parity; the real
gap is in generation - reading precise figures off a page render is harder
than reading them from extracted markdown/captions. Curated text remains the
better generation substrate; pages are the better retrieval substrate for
visually structured content (tables in particular: 0.667 -> 0.778 Hit Rate).
Full breakdown, per-content-type numbers, and operational trade-offs are in
`reports/eval_results_comparison.md`.

## Setup

Requires:

- Python 3.13
- A GCP project with Vertex AI enabled, and Application Default Credentials
  set up locally (`gcloud auth application-default login`) - there are no API
  keys anywhere in this stack
- Docker, for running Qdrant (and optionally the app) via `docker-compose.yml`

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env: set GOOGLE_CLOUD_PROJECT, optionally Langfuse keys
```

`.env` variables (see `.env.example`):

- `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` (defaults to `global` -
  `gemini-3.5-flash` 404s on regional Vertex endpoints in this project)
- `QDRANT_HOST`, `QDRANT_PORT` (default to the compose service)
- `LANGFUSE_BASE_URL`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` (optional -
  tracing is skipped if unset)

## Running

Start Qdrant (needed by ingestion and by the Qdrant backend/ColPali pipeline):

```bash
docker compose up -d qdrant
```

Run ingestion once (parses the PDF, enriches tables/images via Gemini, builds
both the FAISS and Qdrant indices - several minutes, most of it one-time
Gemini/Docling cost that is cached for later reruns):

```bash
python -m src.dataset
```

Build the Phase 6 ColPali page-image index (separate pipeline, local GPU/MPS
model, ~10 minutes one-time):

```bash
python -m src.colpali_dataset
```

Build the evaluation dataset, then run the evaluation harness:

```bash
python -m src.evaluation.synthetic_qa
python -m src.evaluation.run_eval --backend faiss --k 4 --retrieval-mode reranked
```

`--retrieval-mode` accepts `dense`, `hybrid`, `reranked`, or `colpali` (forces
`backend=qdrant`). `--n` runs a smaller sample for a quick smoke test.

Run the app locally:

```bash
streamlit run src/ui/app.py
```

Or via Docker (Streamlit on `localhost:8503`, mounting the host's Vertex AI
credentials, the built FAISS index, the processed chunks, and rendered page
images read-only, with Qdrant as a sidecar service):

```bash
docker compose up
```

## Project layout

```
src/
├── config.py                  <- GCP project/location, model IDs, paths, chunk size
├── dataset.py                 <- ingestion entrypoint: parse -> enrich -> chunk -> FAISS + Qdrant
├── colpali_dataset.py         <- Phase 6 entrypoint: render pages -> ColQwen2 embed -> Qdrant multivector
├── plots.py                   <- FAISS vs Qdrant latency benchmark
│
├── ingestion/
│   ├── parse.py               <- Docling parsing: text/table/image records + page-to-section map
│   ├── chunk.py               <- text/table/image chunking into langchain Documents
│   ├── enrich.py              <- cached Gemini table summaries + image captions/classification
│   └── page_images.py         <- PyMuPDF page-to-PNG rendering for the ColPali pipeline
│
├── embedding/
│   ├── embedder.py            <- GeminiEmbeddings (gemini-embedding-001)
│   └── colpali_embedder.py    <- ColQwen2 patch embeddings (local, MPS/CPU)
│
├── retrieval/
│   ├── faiss_store.py         <- FAISS HNSW index build/load
│   ├── qdrant_store.py        <- Qdrant collection build/load, incl. ColPali multivector collection
│   ├── reranker.py            <- cross-encoder reranking
│   └── retriever.py           <- dense/hybrid/reranked/colpali retrieval entrypoints
│
├── generation/
│   ├── answer.py              <- prompt assembly, streaming generation, tool-call loop
│   └── calculator.py          <- AST-sandboxed arithmetic tool
│
├── evaluation/
│   ├── synthetic_qa.py        <- synthetic QA generation + critique filtering
│   ├── diagnostics.py         <- coverage/rank/RAGAS metric computation
│   ├── ragas_compat.py        <- RAGAS-to-Vertex AI compatibility shim
│   └── run_eval.py            <- evaluation entrypoint, settings-tagged JSON output
│
├── services/
│   ├── genai_client.py        <- shared Vertex AI GenAI client
│   └── langfuse_client.py     <- shared Langfuse client
│
└── ui/
    └── app.py                 <- Streamlit app
```

Most artefacts under `data/interim`, `data/processed`, and `models/faiss` are
generated by the entrypoints above and not checked in (docling/embedding
caches, chunk JSONL, eval run JSONs, the FAISS index) - see `.gitignore` for
the exact list. `data/processed/eval_dataset.csv` is the exception: it is
checked in so the evaluation harness has a fixed question set to run against.

## Further reading

- `ARCHITECTURE.md` - module map, data flow diagrams, and deployment
- `agent_docs/decisions.md` - architecture decision records (ADRs) for the
  non-obvious choices (parsing library, embedding model, table/image
  retrievability fixes, ColPali design)
- `reports/eval_results_comparison.md` - the full eval ladder and Phase 6
  comparison
- `reports/faiss_vs_qdrant.md` - the FAISS vs Qdrant latency comparison
