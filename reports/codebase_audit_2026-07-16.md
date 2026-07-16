# Codebase audit - 2026-07-16

How this was run: five parallel auditors, one per pipeline slice (ingestion, indexing/embedding, retrieval/generation, evaluation, UI/infra), each reading its slice in full plus the interfaces it touches. Findings were then cross-checked at the seams (chunk schema -> index -> retriever metadata -> UI/eval expectations), and every P0 plus the two highest-impact P1s were independently re-verified against the code before publishing. The audit covered the working tree including the then-uncommitted `colpali_embedder.py`/`run_eval.py`/`diagnostics.py` changes; those landed unchanged as commit `198644d` (ColPali eval wiring + Phase 6 comparison results) while the audit ran, so all findings apply to HEAD.

Totals: 4 P0, 14 P1, 20 P2. Actionable checklist lives in `agent_docs/TODOS.md` under "Codebase audit (2026-07-16)".

## P0 - wrong results or broken as shipped

### A1. Section attribution is wrong for almost all table and image chunks
`src/ingestion/parse.py:32-43` (`_build_page_to_section`), consumed at `parse.py:58` and `parse.py:87`.

The page-to-section map is built by overwriting the entry each time any text item on a page is visited, so each page ends up labelled with whichever section header appears last on it in reading order. Verified against the parsed document: 131 of 147 pages carry more than one distinct section, and only 1 of 154 tables (13 of 36 pictures) has a real Docling caption to override the fallback. Net effect: ~153 tables and 23 images lead with a misattributed `Table: {section}` / `Figure: {section}` heading - the exact text those chunks are retrieved by, and the label shown in the UI expander. `extract_text_records` already does this correctly by tracking the current section in document order; tables/pictures should be attributed the same way (e.g. via `iterate_items()`).

Fix chain warning (cross-slice): fixing this changes chunk text, and the enrichment caches key only on table markdown / image bytes, not on the section context injected into the prompt (A7). Fixing A1 without also fixing/invalidating those caches silently serves stale summaries. Full sequence: fix cache keys -> fix attribution -> re-enrich (~190 Gemini calls) -> re-chunk -> rebuild both indices -> ladder stages strictly need re-running. This is the most expensive fix in the audit; see sequencing at the end.

### A2. Hit Rate/MRR miscounts hits for sub-chunks of multi-page sections
`src/evaluation/diagnostics.py:78`, root cause in `src/ingestion/chunk.py:37-51`.

Two compounding problems. First, `chunk_sections` passes one metadata dict per section to the splitter, so every sub-chunk of a split section inherits the whole section's page span (verified: 13 sections split into multi-chunk groups all sharing identical wide spans, e.g. 12 "INVESTMENT PRODUCTS" sub-chunks all tagged pages 10-12). Second, the hit test checks only `{start_page, end_page} & page_numbers` - the two endpoints, not the inclusive range - so a ground-truth page 11 question can never match a pages 10-12 chunk (false negative), while a chunk whose text is really from page 12 can spuriously hit a page 10 question (false positive). Affects ~9% of text chunks, and the noise is asymmetric versus the ColPali stage, whose page metadata is exact - the ladder comparison is not currently apples-to-apples.

Cheap correct fix available now: match the full inclusive range (`start_page <= p <= end_page`) in `hit_rate_and_mrr` - no re-chunking or re-indexing needed. The deeper fix (true per-sub-chunk pages via character-offset mapping) can follow A1's rebuild.

Note: the Phase 6 ColPali comparison (commit `198644d`) landed mid-audit with this bug in place. Its headline near-parity result (0.925 vs 0.935 Hit Rate@4) carries A2's noise on the text-pipeline side only, so the true gap could be in either direction; caveat the comparison until the affected stages are re-scored with range matching. Per-sample retrievals are not persisted in the run JSONs, so re-scoring means re-running the stages, not just recomputing.

### A3. Docker deployment cannot answer a single query
`docker-compose.yml:17-19`, `Dockerfile:8`.

The compose file mounts only `./models/faiss`; it never mounts `data/processed/chunks.jsonl` (needed by BM25 inside the default reranked pipeline) or `data/interim/page_images/` (needed by ColPali source rendering and Gemini image input). The Dockerfile copies only `src/`, and `.dockerignore` excludes `data/` and `models/` from the build context, so the data reaches the container by no path at all. `docker compose up` + any question = `FileNotFoundError`. Confirmed via git history: the Docker files predate hybrid retrieval, tables, images, and ColPali, and were never revisited. Fix: add read-only mounts for `data/processed` and `data/interim/page_images`, then verify inside the container, not just locally. If Friday's demo runs locally, this can slip past the demo - decide explicitly.

### A4. UI error handling only covers Gemini API errors
`src/ui/app.py:80-98` (single `except genai_errors.APIError`), source-rendering loop `app.py:100-133` unguarded.

A missing FAISS index (fresh clone - `models/` is gitignored), Qdrant not running, a missing ColPali collection, or a missing page PNG all raise exceptions the guard does not catch, rendering a raw traceback in the client-facing UI. Compounds with A3 and with the unguarded ColPali paths (A6). Fix: a broad guard around retrieval/generation with a friendly message, plus a per-source try/except in the rendering loop degrading to `st.error`.

## P1 - likely to bite soon; demo or eval-run risk

Eval-run cluster (fix before the next eval run):

- A5. One failed sample kills the whole eval run. `run_eval.py:87` - `[f.result() for f in futures]` propagates the first exception; nothing is written, no resume. The comment above `SAMPLE_ATTEMPTS` says the opposite. Not hypothetical: the `198644d` commit message records that the first full ColPali run crashed (concurrent VLM cold-loads) and had to be redone end to end. Catch per-future failures, record and skip, and persist samples incrementally.
- A8. RAGAS means silently drop NaN rows. `diagnostics.py:153` - pandas `mean()` with default `skipna` averages over an unrecorded effective N, so two ladder stages can differ partly by coverage while both claim n=200. Store per-metric counts alongside means.
- A9. Hybrid stage feeds RAGAS an unbounded context. `retriever.py:88-92` - `EnsembleRetriever` returns the deduplicated union of both retrievers (up to 2 x k docs, never truncated to k), which `run_eval.py:60-61` passes unsliced to both RAGAS and generation, while Hit Rate slices to k. The committed +Hybrid ladder column was scored on roughly double the context of its neighbours. Slice to k in the harness; annotate the existing +Hybrid column with the caveat.
- A10. Interrupted ColPali cache writes leave permanently corrupt `.npy` files. `colpali_embedder.py:70-80` (same pattern in `page_images.py:26-28`) - direct `np.save` to the final path; a truncated file passes the existence check forever and `np.load` then crashes every subsequent run at that page. Write to temp + `os.replace`.

Demo cluster (fix before Friday):

- A6. ColPali query path unguarded. `retriever.py:109-114` (Qdrant `query_points`) and `answer.py:87` (`read_bytes` on the page PNG) - both raise plain exceptions the UI guard misses. This is the flagship Phase 6 feature.
- A11. Malformed tool call crashes the answer stream. `answer.py:146` - `dict(fc.args)` runs outside `_run_calculate`'s try/except and `FunctionCall.args` is `Optional` in the SDK; a glitchy call (same class as the observed `turn_to_user` leak) yields `dict(None)` -> `TypeError` -> raw traceback mid-answer. Fix: `dict(fc.args or {})`.
- A12. The `$`-escaping fix missed table caption preambles. `app.py:128` - `st.caption(preamble)` renders the Gemini table summary, precisely the text most likely to contain two dollar amounts on one line, through the same LaTeX-mangling path fixed at lines 92/132. One-line fix.
- A13. Multi-GB lazy model downloads at first query. `reranker.py:11` (cross-encoder) and `colpali_embedder.py:38` (~5GB ColQwen2) download from Hugging Face on first use, hidden behind the retrieval spinner. On demo day a cold cache reads as a hung app. Pre-warm both before the demo; for Docker, bake into the image or persist `HF_HOME`.

Robustness/cost cluster:

- A7. Enrichment cache keys ignore prompt context. `enrich.py:39-40,148` - keys hash only table markdown / image bytes, but the prompts also inject `caption or section`; changed attribution (including the A1 fix) serves stale cached output. Fold context into the key or version the cache.
- A14. Image captioning has no bad-response fallback. `enrich.py:126-139` - unlike `_summarise`, a `None` `response.parsed` raises `AttributeError` and kills the whole captioning batch.
- A15. Non-atomic enrichment cache writes. `enrich.py:49-51,158-159` - full-file rewrite per item; an interrupt can corrupt the JSON and lose the entire cache. Temp + `os.replace`.
- A16. Text chunks embedded twice per index rebuild. `dataset.py:55-59` - FAISS build embeds all ~852 chunks, then `QdrantVectorStore.from_documents` re-embeds the same texts. Double Vertex cost/latency and a drift path between stores. Compute once, feed both.
- A17. Interrupted Qdrant builds leave a partially populated live collection. `qdrant_store.py:14-22,44-76` - delete/recreate then batched upserts with no count verification or staging alias; a mid-build crash leaves a silently incomplete collection that retrieval happily queries.
- A18. requirements.txt pins nothing and omits `langchain-classic`. `retriever.py:5` imports it directly but it resolves only transitively; this exact import path already broke once on a langchain bump. Pin the langchain family, torch, and colpali-engine before any fresh install near the demo.

## P2 - polish and maintainability

- FAISS uses L2 while Qdrant uses cosine; equivalent today only because vectors are unit-norm (verified empirically), with nothing enforcing it (`faiss_store.py:23`). Assert or switch to inner product.
- Non-atomic two-file FAISS save can desync `.faiss`/`.pkl` on interrupt (`faiss_store.py:36`).
- `load_chunks()` re-parses the full JSONL from disk on every query, up to twice, in `retrieve_reranked`'s structural-candidate helpers (`retriever.py:152,176`); `lru_cache` it like its siblings.
- `MAX_TOOL_TURNS` exhaustion ends the answer silently mid-thought (`answer.py:119-151`); yield a fallback line.
- `_stream_turn` retries only `ServerError`, not pre-first-chunk timeouts (`answer.py:100-111`).
- FAISS filtered search scores the whole index (`fetch_k=ntotal`, `retriever.py:69-71`); fine at this scale, will not scale.
- Parse/page-image caches are existence-only, no content/version check (`parse.py:26-29`, `page_images.py:26`); already required one manual cache-bust (ADR log).
- Latent `None` end_page overwrite in section grouping (`chunk.py:29`); does not trigger on current data.
- Stale comment: "16 logos/signatures" is actually 11 (`enrich.py:97-98`).
- Enrichment is fully sequential with no per-item retry/backoff; a full cache invalidation forces a slow, fragile serial re-run.
- Eval CSV row 35 has an unquoted comma truncating `Context_Content_Type` to `combination (text` - a bogus n=1 bucket in every per-content-type breakdown (`references/RAG_evaluation_dataset.csv:35`).
- ColPali is not an additive ladder rung (different retrieval unit, different embedder, three RAGAS metrics None by design; `parse_chunk_coverage` reports on a corpus the pipeline never touches) - caveat it explicitly in `reports/eval_results_comparison.md` when the run lands.
- Dead config: `EMBEDDING_DIMENSIONS` (`config.py:17`) and the three Langfuse constants (`config.py:64-66`, SDK reads env directly, and `LANGFUSE_HOST` is a deprecated alias anyway).
- `langchain_community` (FAISS store imports) is sunset upstream; migrate alongside the earlier `langchain_classic` move.
- ColPali Qdrant point IDs are bare page numbers - a second document would silently overwrite; fine single-doc, needs a composite ID if scope grows.
- No Qdrant pre-flight check; a down container surfaces as a raw httpx error rather than "run docker compose up -d qdrant".
- ARCHITECTURE.md still describes only the Phase 1 naive text pipeline - no hybrid, reranking, tables, images, ColPali, or tool calling. Update before Friday if the mentor may open it.
- README.md is the unedited cookiecutter template: lists directories that do not exist, omits every real module, and misdescribes the data gitignore policy. Repo polish is a stated pre-Friday task.
- Qdrant is exposed on 6333 with no auth - fine locally, flag if the compose file ever runs on a shared host.
- Coarse page ranges on multi-page chunks make citation spans misleadingly wide in the UI (subsumed by A2's deeper fix).

## Seam checks and verified-sound notes

- Page numbering is consistent across all five slices: raw Docling 1-indexed everywhere in metadata, filenames, Qdrant payloads, and the eval CSV; `config.display_page()` converts only at display/prompt time. No off-by-one seams found.
- The `chunks.jsonl` schema is consistent between producer (`chunk.py`) and all consumers (indexing, BM25, structural candidates, eval).
- The uncommitted `colpali_embedder.py` change (model-load lock) is a correct, well-targeted fix for the concurrent cold-start problem; no new issues.
- `run_eval.py`'s settings tagging correctly reflects the forced `backend="qdrant"` for ColPali runs - filename and JSON both right even if invoked with `--backend faiss`.
- `calculator.calculate` is safe by construction (AST allow-list, no eval/names/calls).

## Suggested sequencing

The ColPali comparison run landed mid-audit (`198644d`), so the eval fixes are no longer gating that run - they now matter for the integrity of the published ladder and for any re-runs.

1. Before Friday's demo: A4, A6, A11, A12, A13 warm-up, A18 pins; decide Docker (A3) vs local demo; caveat the Phase 6 comparison in `reports/eval_results_comparison.md` per A2/A9; README/ARCHITECTURE if the repo will be shown.
2. Before the next eval run (post-demo re-scoring included): A2's cheap fix, A5, A8, A9's harness slice, A10; then consider re-running the ladder stages to publish corrected Hit Rate/MRR.
3. After the demo: the A1 fix chain (A7 first), A16, A17, A14/A15, remaining P2s.
