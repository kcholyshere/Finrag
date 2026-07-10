# Architecture

Phase 1 of Finrag: a naive, text-only Retrieval Augmented Generation (RAG) system that answers
questions about the IFC Annual Report 2024 (Financials). It has two halves that run at different
times:

- **Ingestion** (offline, run once via `python -m src.dataset`): parses the source PDF, chunks it,
  embeds the chunks, and populates two interchangeable vector stores (FAISS and Qdrant).
- **Serving** (online, the Streamlit app): takes a user's question, retrieves relevant chunks from
  whichever backend is selected, and streams a grounded answer from Gemini, with every query traced
  in Langfuse.

Later phases (tables, images, re-ranking, RAGAS evaluation) build on top of this without changing
the shape below - see `agent_docs/decisions.md` for the reasoning behind each choice.

## System overview

```mermaid
flowchart TB
    PDF["references/ifc-annual-report-2024-financials.pdf"]

    subgraph Ingestion["Ingestion pipeline (offline) - python -m src.dataset"]
        Parse["ingestion/parse.py<br/>Docling"]
        Chunk["ingestion/chunk.py<br/>RecursiveCharacterTextSplitter"]
        Embed["embedding/embedder.py<br/>gemini-embedding-001"]
    end

    subgraph Stores["Vector stores"]
        FAISS[("FAISS index<br/>models/faiss/")]
        Qdrant[("Qdrant collection<br/>containerised service")]
    end

    subgraph Serving["Serving (online) - Streamlit app"]
        UI["ui/app.py"]
        Retriever["retrieval/retriever.py"]
        Generation["generation/answer.py"]
    end

    subgraph External["External services"]
        Vertex["Vertex AI<br/>gemini-3.5-flash + gemini-embedding-001"]
        Langfuse["Langfuse Cloud<br/>tracing"]
    end

    PDF --> Parse --> Chunk --> Embed
    Embed --> FAISS
    Embed --> Qdrant

    UI --> Generation --> Retriever
    Retriever --> FAISS
    Retriever --> Qdrant
    Retriever -.->|"embed query"| Vertex
    Generation -.->|"generate"| Vertex

    Retriever -.->|"trace"| Langfuse
    Generation -.->|"trace"| Langfuse
```

## Ingestion: PDF to two vector stores

`src/dataset.py` is the single entrypoint. It is idempotent for parsing (Docling's output is
cached as JSON so re-running doesn't repeat OCR) but rebuilds both vector stores from scratch each
time.

```mermaid
flowchart LR
    A["references/*.pdf"] --> B["parse_pdf()<br/>Docling DocumentConverter"]
    B --> C["data/interim/*.docling.json<br/>(cached, reused across phases)"]
    B --> D["extract_text_records()<br/>+ group_into_sections()"]
    D --> E["chunk_sections()<br/>chunk_size=1000, overlap=150"]
    E --> F["data/processed/chunks.jsonl"]
    E --> G["GeminiEmbeddings<br/>gemini-embedding-001, 3072-dim"]
    G --> H["faiss_store.build_index()"]
    G --> I["qdrant_store.build_index()"]
    H --> J[("models/faiss/*.faiss + *.pkl")]
    I --> K[("Qdrant collection:<br/>ifc_annual_report_2024")]
```

Each chunk carries `section` (nearest heading) and `start_page`/`end_page` metadata, captured at
parse time - this is what lets the UI show "page 5, SECTION I. EXECUTIVE SUMMARY" next to a
retrieved snippet.

## Serving: answering one question

`generation/answer.py:answer_query()` is the single entry point the UI calls. It nests retrieval
and generation under one Langfuse trace (`rag_query`) so a single user question shows up as one
trace with two spans, rather than two disconnected traces.

```mermaid
sequenceDiagram
    participant User
    participant UI as ui/app.py
    participant Gen as generation.answer_query()
    participant Ret as retrieval.retrieve()
    participant Store as FAISS / Qdrant
    participant Vertex as Vertex AI
    participant LF as Langfuse

    User->>UI: types a question, picks backend + k
    UI->>Gen: answer_query(query, backend, k)
    Gen->>Ret: retrieve(query, backend, k)
    Ret->>Vertex: embed_content(query, task_type=RETRIEVAL_QUERY)
    Vertex-->>Ret: query embedding
    Ret->>Store: similarity_search(embedding, k)
    Store-->>Ret: top-k chunks + metadata
    Ret--)LF: retriever span
    Ret-->>Gen: context_docs
    Gen->>Vertex: generate_content_stream(prompt + context)
    Vertex-->>Gen: streamed answer tokens
    Gen--)LF: generation span (nested under rag_query trace)
    Gen-->>UI: (context_docs, token stream)
    UI-->>User: streamed answer + expandable source snippets
```

## Module map

Import direction is one-way: `ui` depends on `generation`, which depends on `retrieval`, which
depends on `embedding`. Nothing imports back up the chain.

```mermaid
flowchart TB
    config["config.py<br/>GCP project/location, model IDs, paths, chunk size"]

    subgraph services["services/"]
        genai["genai_client.py<br/>shared Vertex AI Client"]
        langfuse_svc["langfuse_client.py<br/>shared Langfuse client"]
    end

    subgraph ingestion["ingestion/"]
        parse["parse.py"]
        chunk["chunk.py"]
    end

    subgraph embedding["embedding/"]
        embedder["embedder.py<br/>GeminiEmbeddings"]
    end

    subgraph retrieval["retrieval/"]
        faiss_store["faiss_store.py"]
        qdrant_store["qdrant_store.py"]
        retriever["retriever.py"]
    end

    subgraph generation["generation/"]
        answer["answer.py"]
    end

    subgraph ui["ui/"]
        app["app.py"]
    end

    dataset["dataset.py<br/>ingestion entrypoint"]

    parse --> config
    chunk --> config
    embedder --> config
    embedder --> genai
    faiss_store --> config
    faiss_store --> embedder
    qdrant_store --> config
    qdrant_store --> embedder
    retriever --> faiss_store
    retriever --> qdrant_store
    retriever --> langfuse_svc
    answer --> config
    answer --> genai
    answer --> retriever
    answer --> langfuse_svc
    app --> answer
    dataset --> parse
    dataset --> chunk
    dataset --> faiss_store
    dataset --> qdrant_store
```

## Deployment

`docker-compose.yml` runs two services. The app container reuses the host's already-built FAISS
index and Vertex AI credentials rather than re-running ingestion inside the container.

```mermaid
flowchart TB
    Browser["Browser<br/>localhost:8503"]

    subgraph Host["Host machine"]
        ADC["~/.config/gcloud/<br/>application_default_credentials.json"]
        FaissDir["models/faiss/"]

        subgraph Compose["docker-compose.yml"]
            App["app container<br/>Streamlit :8501 → host :8503"]
            QdrantC["qdrant container<br/>:6333"]
        end

        QVol[("qdrant_storage<br/>docker volume")]
    end

    VertexAI["Vertex AI<br/>(external)"]
    LangfuseCloud["Langfuse Cloud<br/>(external)"]

    Browser --> App
    ADC -.->|"mounted read-only as /gcp/adc.json"| App
    FaissDir -.->|"mounted read-only"| App
    App --> QdrantC
    QdrantC --> QVol
    App -.->|"auth via ADC"| VertexAI
    App -.->|"traces"| LangfuseCloud
```

## Evaluation: measuring Phase 1 quality

Phase 2 adds an offline evaluation harness that sits beside ingestion/serving rather than in
their request path - it runs the *existing* pipeline over a fixed question set and scores it, in
two layers: per-step diagnostics (is each stage doing its job?) and end-to-end outcome (is the
final answer any good?). See `agent_docs/decisions.md` (ADR-0006) for why the RAGAS/Vertex shim
below is necessary.

```mermaid
flowchart TB
    Curated["references/RAG_evaluation_dataset.csv<br/>33 hand-curated Q&A pairs"]
    Synth["synthetic_qa.py<br/>generate + critique-filter<br/>(groundedness/relevance/standalone)"]
    EvalSet["data/processed/eval_dataset.csv<br/>~200 rows, Source=curated|synthetic"]

    Curated --> EvalSet
    Chunks2["data/processed/chunks.jsonl"] --> Synth --> EvalSet

    subgraph RunEval["run_eval.py - python -m src.evaluation.run_eval --backend faiss|qdrant"]
        Retrieve2["retrieval.retrieve()<br/>per question"]
        Generate2["generation.generate_answer()<br/>per question"]
        Coverage["diagnostics.parse_chunk_coverage()<br/>rapidfuzz vs live chunks"]
        RankMetrics["diagnostics.hit_rate_and_mrr()<br/>page-overlap relevance label"]
        RagasMetrics["diagnostics.run_ragas_metrics()<br/>single evaluate() call"]
    end

    subgraph RagasCompat["ragas_compat.py"]
        Shim["langchain_community.chat_models.vertexai<br/>stub module (pre-import)"]
        LLMWrap["ChatVertexAI → LangchainLLMWrapper"]
        EmbedWrap["GeminiEmbeddings → LangchainEmbeddingsWrapper"]
    end

    EvalSet --> Retrieve2 --> Generate2
    Generate2 --> Coverage
    Generate2 --> RankMetrics
    Generate2 --> RagasMetrics
    RagasMetrics --> RagasCompat
    RagasCompat -.->|"grades via"| Vertex2["Vertex AI<br/>gemini-3.5-flash"]

    Coverage --> Result["settings-tagged JSON<br/>data/processed/eval_runs/*.json"]
    RankMetrics --> Result
    RagasMetrics --> Result
```

**What each stage checks:**

| Layer | Metric | Question it answers |
|---|---|---|
| Parsing/chunking | `parse_chunk_coverage` (rapidfuzz `partial_ratio`, threshold 80) | Does a curated, text-only ground-truth snippet actually survive inside some real chunk? |
| Retrieval (rank) | Hit Rate@k / MRR@k (ground-truth page as relevance label) | Is the right-page chunk in the top-k, and how high is it ranked? |
| Retrieval (RAGAS) | `context_precision` / `context_recall` | Are the retrieved contexts sufficient and precise for the reference answer? |
| Generation (RAGAS) | `faithfulness` / `answer_relevancy` | Is the answer grounded in retrieved context, and does it address the question? |
| Outcome (RAGAS) | `answer_correctness` | Does the final answer match the reference answer end to end (the LLM-as-judge experiment)? |

**Non-obvious constraints:**

- `ragas_compat.py` must be imported *before* anything else touches `ragas` - ragas 0.4.3
  unconditionally imports a `langchain_community.chat_models.vertexai.ChatVertexAI` symbol that no
  longer exists in current `langchain-community`; the shim registers a stub module so that dead
  import resolves, then wires RAGAS to the real `langchain-google-vertexai` `ChatVertexAI` and to
  this project's own `GeminiEmbeddings` instead of RAGAS's OpenAI default.
- `run_ragas_metrics` runs every RAGAS metric in one `evaluate()` call rather than one call per
  metric: RAGAS tears down its internal asyncio event loop after `evaluate()` returns, and the
  cached `ChatVertexAI`'s grpc.aio channel is bound to that loop - a second call in the same
  process silently returns NaN for every row.
- `parse_chunk_coverage` is scoped to `Source == "curated"` and text-only rows: synthetic rows were
  generated *from* the same chunks (trivially 100%), and table/image rows can't match since Phase 1
  only chunks text.
- Each run is saved as a settings-tagged JSON (backend, k, chunk size/overlap, model IDs,
  timestamp) so later phases (tables, images, re-ranking) can be compared against this baseline
  without re-running earlier ones.

## Key facts worth remembering

- **Auth**: everything goes through Vertex AI Application Default Credentials - no API keys
  anywhere. `GCP_LOCATION` defaults to `"global"` because `gemini-3.5-flash` 404s on regional
  Vertex AI endpoints in this project (`gemini-embedding-001` works on both).
- **Both vector stores are kept in sync**: `dataset.py` populates FAISS and Qdrant identically from
  the same chunks, so the UI's backend picker is a genuine A/B, not a stub. See
  `reports/faiss_vs_qdrant.md` for the measured comparison.
- **Docling's output is cached** (`data/interim/*.docling.json`) specifically so later phases
  (tables, images) can reuse the same parse without re-running OCR.
