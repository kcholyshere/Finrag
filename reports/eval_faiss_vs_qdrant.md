# Phase 2 eval: FAISS vs Qdrant

Same 200-row eval set, `k=4`, both backends indexing identical chunks/embeddings.

- FAISS run: `data/processed/eval_runs/2026-07-06T16-10-15.276688+00-00_faiss_k4.json`
- Qdrant run: `data/processed/eval_runs/2026-07-07T09-28-33.814935+00-00_qdrant_k4.json`

| Metric | FAISS | Qdrant | Diff |
|---|---|---|---|
| Hit Rate@4 | 0.860 | 0.860 | 0.000 |
| MRR@4 | 0.7125 | 0.7125 | 0.000 |
| context_precision | 0.7332 | 0.7278 | -0.0054 |
| context_recall | 0.8075 | 0.8075 | 0.000 |
| faithfulness | 0.6520 | 0.6464 | -0.0056 |
| answer_relevancy | 0.7611 | 0.7652 | +0.0041 |
| answer_correctness | 0.5925 | 0.5901 | -0.0024 |

## Reading

Rank metrics (Hit Rate/MRR) are exactly identical - expected, since both backends now use the same HNSW algorithm class over the same vectors (see `reports/faiss_vs_qdrant.md`), so they retrieve the same top-k chunks. The RAGAS-scored metrics differ by under 1%, which is within LLM-judge noise rather than a real quality gap.

**Conclusion:** backend choice has no meaningful effect on retrieval or answer quality at this corpus size. The two stores remain a genuine A/B on operational grounds only (see `reports/faiss_vs_qdrant.md` for the latency/use-case comparison), not a quality trade-off.

## Weakest area (both backends)

`table` questions: Hit Rate@4 = 0.556, MRR@4 = 0.278 - well below the `text` questions (0.875 / 0.747). Phase 1 only chunks text, so table content is retrieved incidentally at best. This is the clearest signal for where Phase 3 (dedicated table extraction) should focus.
