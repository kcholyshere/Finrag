# FAISS vs Qdrant: Phase 1 comparison

Both vector stores are populated with the same 673 chunks (`gemini-embedding-001`, 3072 dims) from the IFC Annual Report 2024. FAISS runs in-process against a local index file (`models/faiss/`); Qdrant runs as a containerised service (`docker-compose.yml`, `qdrant/qdrant` image) accessed over HTTP.

Benchmark: `python -m src.plots` - 6 representative queries, 5 repetitions each, `k=4`.

## Latency

| Backend | Mean (s) | Median (s) | p95 (s) |
|---|---|---|---|
| FAISS   | 0.397 | 0.400 | 0.556 |
| Qdrant  | 0.421 | 0.425 | 0.561 |

FAISS is marginally faster (~6% on mean), but the gap is small relative to the total. Both numbers are dominated by the `embed_query` round trip to the Vertex AI embedding API (~0.35-0.4s per call in this environment) rather than by the vector search itself - at 673 vectors, both an in-memory FAISS index scan and a local Qdrant HTTP call are near-instant compared to that network call. This benchmark therefore measures "index backend overhead on top of an unavoidable embedding call," not raw ANN search speed.

## Result quality

Top-4 retrieved chunks (by `section` + `start_page`) were identical between the two backends for all 6 test queries (Jaccard overlap = 1.00). This is notable because FAISS defaults to Euclidean (L2) distance with unnormalised vectors, while Qdrant defaults to cosine distance - a mismatch that can reorder results in general. It didn't here, most likely because `gemini-embedding-001` vectors are close enough to unit norm that L2 and cosine rankings coincide at this corpus size. This should be re-checked if the corpus grows substantially or the embedding model changes.

## Use-case analysis

- **FAISS**: simplest option for a fixed, read-mostly corpus that fits in memory and is rebuilt wholesale on each ingestion run (our `dataset.py` flow). No service to run or operate; the index is just two files on disk. Downsides: no built-in persistence/versioning story beyond "save/load a folder," no metadata filtering at query time beyond what LangChain bolts on, and no concurrent-write support - it doesn't fit a scenario with multiple ingestion jobs or live updates.
- **Qdrant**: adds an operational component (the container in `docker-compose.yml`) but gets us a real database: persisted storage independent of the app process, payload-based filtering (e.g. restrict retrieval to a page range or section), and a path to horizontal scaling and multi-client access if this app ever needs concurrent ingestion or serving from multiple processes.

For this project's current scale (one 147-page PDF, single-process Streamlit app, index rebuilt from scratch on each ingestion run) FAISS is sufficient on its own. Qdrant earns its keep once later phases add incremental updates (tables/images in separate collections, re-ranking that needs payload filters) or once the app needs to serve multiple concurrent users against a shared, independently-updatable index - which is the direction this project is headed, so keeping both wired up now (rather than picking one) is the right call.
