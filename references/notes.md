# Notes

## Why two vector databases (FAISS + Qdrant)?

- **FAISS**: runs in-process with no network hop, but it isn't containerised - it's a local index file, so it doesn't travel well across devices/deployments and has no concurrent-write or incremental-update story (the whole index is rebuilt from scratch each ingestion run). Averaged over 5 benchmark runs it's actually ~12% slower on mean latency than Qdrant (see below), so the in-process argument is about avoiding an extra container, not about raw speed.
- **Qdrant**: adds an extra container to operate (`docker-compose.yml`), but that buys a real database - persisted, addressable independently of the app process, with payload filtering and support for updating embedding indexes mid-run rather than requiring a full rebuild.
- Net trade-off: keeping Qdrant means an extra container and a small, inconsistent latency delta either way, in exchange for the flexibility to meet a range of client use cases/needs (multi-device access, incremental updates, concurrent serving) as the project scales beyond a single-process prototype.

See `reports/faiss_vs_qdrant.md` for the full benchmark and analysis.

## FAISS index type: flat vs HNSW

The first benchmarks above used LangChain's default `IndexFlatL2` - exact, brute-force O(n) search - which isn't a fair comparison against Qdrant's default HNSW (O(log n)). Since the project requirement only says "use FAISS" (no index type specified), switched `src/retrieval/faiss_store.py` to build `faiss.IndexHNSWFlat` instead, so both backends use the same log-time graph-search algorithm. Re-ran the 5-fold benchmark: FAISS's numbers barely moved (mean 0.367s vs 0.368s before), and Qdrant is still faster by about the same margin - confirming the latency gap is dominated by the embedding API call, not by search algorithm complexity, at this corpus size (673 vectors).
