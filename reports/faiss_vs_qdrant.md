# FAISS vs Qdrant: Phase 1 comparison

Both vector stores are populated with the same 673 chunks (`gemini-embedding-001`, 3072 dims) from the IFC Annual Report 2024. FAISS runs in-process against a local index file (`models/faiss/`); Qdrant runs as a containerised service (`docker-compose.yml`, `qdrant/qdrant` image) accessed over HTTP.

Benchmark: `python -m src.plots` - 6 representative queries, 5 repetitions each, `k=4`. Run 5 times (5-fold) on separate invocations to average out noise from the embedding API call.

## Latency: FAISS `IndexFlatL2` (exact, brute-force) vs Qdrant HNSW

Per-run results:

| Run | FAISS mean | FAISS median | FAISS p95 | Qdrant mean | Qdrant median | Qdrant p95 |
|---|---|---|---|---|---|---|
| 1 | 0.397 | 0.400 | 0.556 | 0.421 | 0.425 | 0.561 |
| 2 | 0.407 | 0.306 | 0.706 | 0.346 | 0.295 | 0.604 |
| 3 | 0.344 | 0.299 | 0.621 | 0.284 | 0.283 | 0.316 |
| 4 | 0.350 | 0.298 | 0.651 | 0.285 | 0.276 | 0.338 |
| 5 | 0.344 | 0.285 | 0.641 | 0.279 | 0.278 | 0.312 |

Averaged across the 5 runs:

| Backend | Mean (s) | Median (s) | p95 (s) |
|---|---|---|---|
| FAISS   | 0.368 | 0.318 | 0.635 |
| Qdrant  | 0.323 | 0.311 | 0.426 |

Qdrant is faster on average across runs (~12% on mean, and notably tighter on p95 - 0.43s vs 0.64s), reversing the single-run result from the first pass. Run 1 was the outlier where FAISS came out ahead; every subsequent run favoured Qdrant, including on tail latency, which suggests FAISS is more prone to occasional slow calls in-process (e.g. GC or cold-cache effects) than Qdrant's separate service. That said, both numbers are still dominated by the `embed_query` round trip to the Vertex AI embedding API (~0.3-0.4s per call in this environment) rather than by the vector search itself - at 673 vectors, both an in-memory FAISS index scan and a local Qdrant HTTP call are near-instant compared to that network call. This benchmark therefore measures "index backend overhead on top of an unavoidable embedding call," not raw ANN search speed, and the gap between backends is small relative to that shared cost either way.

One confound in the above: `FAISS.from_documents` (LangChain's default) builds an `IndexFlatL2` - an exact, brute-force O(n) linear scan - while Qdrant's default index is HNSW, an approximate graph-based index with O(log n) query complexity. So the comparison above wasn't purely "in-process vs HTTP," it was also "linear scan vs log-time graph search." To isolate the container/HTTP variable, `src/retrieval/faiss_store.py` was switched to build a `faiss.IndexHNSWFlat` (`M=32`, `efConstruction=200`, `efSearch=128`) instead, giving both backends the same algorithmic complexity class.

## Latency: FAISS `IndexHNSWFlat` (log-time) vs Qdrant HNSW

Per-run results:

| Run | FAISS mean | FAISS median | FAISS p95 | Qdrant mean | Qdrant median | Qdrant p95 |
|---|---|---|---|---|---|---|
| 1 | 0.394 | 0.327 | 0.667 | 0.317 | 0.281 | 0.391 |
| 2 | 0.424 | 0.356 | 0.625 | 0.290 | 0.271 | 0.318 |
| 3 | 0.334 | 0.306 | 0.542 | 0.274 | 0.273 | 0.309 |
| 4 | 0.339 | 0.280 | 0.664 | 0.392 | 0.309 | 0.658 |
| 5 | 0.344 | 0.288 | 0.555 | 0.271 | 0.271 | 0.311 |

Averaged across the 5 runs:

| Backend | Mean (s) | Median (s) | p95 (s) |
|---|---|---|---|
| FAISS (HNSW) | 0.367 | 0.311 | 0.611 |
| Qdrant        | 0.309 | 0.281 | 0.397 |

Giving FAISS the same log-time HNSW algorithm barely changes anything (mean 0.367s vs 0.368s for the flat index) - Qdrant is still faster by roughly the same margin as before. This confirms the earlier read: at 673 vectors, index search itself (whether O(n) or O(log n)) is negligible next to the ~0.3-0.4s embedding API round trip, so the algorithmic complexity class isn't what's driving the latency gap between backends. The gap is more likely explained by Qdrant's HTTP server handling the search off the main Python process/GIL versus FAISS's in-process call competing with the rest of the benchmark script.

## Result quality

Top-4 retrieved chunks (by `section` + `start_page`) were identical between the two backends for all 6 test queries (Jaccard overlap = 1.00). This is notable because FAISS defaults to Euclidean (L2) distance with unnormalised vectors, while Qdrant defaults to cosine distance - a mismatch that can reorder results in general. It didn't here, most likely because `gemini-embedding-001` vectors are close enough to unit norm that L2 and cosine rankings coincide at this corpus size. This should be re-checked if the corpus grows substantially or the embedding model changes.

## Use-case analysis

- **FAISS**: simplest option for a fixed, read-mostly corpus that fits in memory and is rebuilt wholesale on each ingestion run (our `dataset.py` flow). No service to run or operate; the index is just two files on disk. Downsides: no built-in persistence/versioning story beyond "save/load a folder," no metadata filtering at query time beyond what LangChain bolts on, and no concurrent-write support - it doesn't fit a scenario with multiple ingestion jobs or live updates.
- **Qdrant**: adds an operational component (the container in `docker-compose.yml`) but gets us a real database: persisted storage independent of the app process, payload-based filtering (e.g. restrict retrieval to a page range or section), and a path to horizontal scaling and multi-client access if this app ever needs concurrent ingestion or serving from multiple processes.

For this project's current scale (one 147-page PDF, single-process Streamlit app, index rebuilt from scratch on each ingestion run) FAISS is sufficient on its own. Qdrant earns its keep once later phases add incremental updates (tables/images in separate collections, re-ranking that needs payload filters) or once the app needs to serve multiple concurrent users against a shared, independently-updatable index - which is the direction this project is headed, so keeping both wired up now (rather than picking one) is the right call.
