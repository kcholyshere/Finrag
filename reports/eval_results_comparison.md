# Eval results comparison

FAISS backend only, same 200-row eval set, `k=4`. Each column adds exactly one component on top of the previous one (additive ladder), not an isolated A/B.

> Maintenance rule: every time the eval benchmark is run, add the result to the table below and update the diff/notes for that stage - don't let this doc drift out of sync with `data/processed/eval_runs/`.

| Metric | Baseline (dense) | +Hybrid (BM25) | +Reranking (cross-encoder) | +Tables (Phase 5.1) |
|---|---|---|---|---|
| Hit Rate@4 | 0.860 | 0.945 | 0.940 | 0.945 |
| MRR@4 | 0.7125 | 0.7858 | 0.8600 | 0.8153 |
| context_precision | 0.7195 | 0.7637 | 0.8811 | 0.7590 |
| context_recall | 0.8015 | 0.9141 | 0.8821 | 0.9219 |
| faithfulness | 0.6571 | 0.6592 | 0.6794 | 0.6800 |
| answer_relevancy | 0.7746 | 0.8272 | 0.8395 | 0.8301 |
| answer_correctness | 0.5963 | 0.6383 | 0.6377 | 0.6501 |

Source files:
- Baseline: `data/processed/eval_runs/2026-07-10T07-04-26.367728+00-00_faiss_dense_k4.json`
- +Hybrid: `data/processed/eval_runs/2026-07-10T07-23-27.572883+00-00_faiss_hybrid_k4.json`
- +Reranking: `data/processed/eval_runs/2026-07-10T12-54-46.134674+00-00_faiss_reranked_k4.json` - reverted `parse.py`/`chunk.py`/`dataset.py` to pre-table commit `028db64~1` to keep the index text-only for this stage, then restored current code and rebuilt the table-inclusive index afterward.
- +Tables: `data/processed/eval_runs/2026-07-10T10-35-40.149738+00-00_faiss_reranked_k4.json` (hybrid + reranking + table-inclusive index, all together)

Caveat: the +Reranking run uses the current (table-aware) generation prompt wording in `answer.py`, not the pre-table prompt - negligible effect since no table chunks exist yet at that stage to trigger the table-specific instructions.

## Reading Hit Rate@4 and MRR@4

- Hit Rate@4: of the top 4 retrieved chunks, did any of them contain the ground-truth answer? Yes/no per question, averaged over 200 questions. 0.945 means the right chunk showed up somewhere in the top 4 for 94.5% of questions.
- MRR@4 (Mean Reciprocal Rank): same top-4 window, but scored by how high the right chunk ranked, not just whether it appeared. 1st place = 1.0, 2nd = 0.5, 3rd = 0.33, 4th = 0.25, not present = 0. Averaged over 200 questions.
- Hit Rate answers "did we find it at all", MRR answers "how far down did the model have to dig". A big MRR jump with a flat/small Hit Rate move (as in +Reranking below) means the right chunk was usually already in the top 4, reranking just moved it higher within that window.

## What improved, stage by stage

- Baseline -> +Hybrid: everything moved up, nothing regressed. context_recall jumped most (+0.113). answer_correctness +0.042. faithfulness barely moved (+0.002) - better material, not more honesty.
- +Hybrid -> +Reranking: MRR +0.074 (biggest single-stage MRR jump) and context_precision +0.117, but Hit Rate@4 dipped slightly (-0.005) and answer_correctness is flat (+0.0 vs -0.0006, noise). Reranking's job is ordering, not recall - it can't find chunks hybrid didn't retrieve into the candidate pool, it can only push the right one higher once it's there. That's exactly what the numbers show: big precision/ranking gains, no real recall or end-to-end movement.
- +Reranking -> +Tables: Hit Rate@4 +0.005, context_recall +0.040, answer_correctness +0.012. Modest, as expected given only 9 of 200 questions are table-typed - see below for why the table-specific numbers matter more than the blended ones here.

## Still weak

`table` questions (n=9 of 200):

| Stage | Hit Rate@4 | MRR@4 |
|---|---|---|
| Baseline (dense) | 0.556 | 0.278 |
| +Hybrid | 0.667 | 0.426 |
| +Reranking (no tables) | 0.556 | 0.426 |
| +Tables | 0.667 | 0.509 |

Table Hit Rate never got past 0.667 even after tables were actually indexed - no better than the +Hybrid stage which had no table chunks at all. MRR did climb steadily. Reranking alone regressed Hit Rate for tables specifically (0.667 -> 0.556) even though it helped every other category - consistent with reranking only reordering a candidate pool that dense+BM25 assembled, and that pool apparently wasn't reliably including the right table chunk pre-Phase-5.1 anyway. Worth digging into before Phase 5.2 (images): possibly dense/BM25 aren't matching table markdown content well, or n=9 is too small to trust as a real signal either way.

## Faithfulness experiments

Baseline faithfulness (0.6800, +Tables row above) barely moved across the whole retrieval ladder, so three isolated generation-side changes were tested in parallel, each on its own git worktree/branch off current main, one change each, run against the same `reranked` pipeline:

| Metric | Baseline (+Tables) | +Grounding instructions | +Lower temperature (0.1) | +Post-hoc verification |
|---|---|---|---|---|
| faithfulness | 0.6800 | 0.6364 | 0.6714 | 0.6729 |
| answer_correctness | 0.6501 | 0.6319 | 0.6452 | 0.6501 |
| answer_relevancy | 0.8301 | 0.8424 | 0.8506 | 0.8512 |
| MRR@4 | 0.8153 | 0.8621 | 0.8621 | 0.8621 |
| context_precision | 0.7590 | 0.8784 | 0.8765 | 0.8815 |
| context_recall | 0.9219 | 0.8950 | 0.8950 | 0.8990 |
| Hit Rate@4 | 0.945 | 0.940 | 0.940 | 0.940 |

Branches: `experiment/grounding-prompt` (`efd4cf2`), `experiment/low-temperature` (`e782370`), `experiment/post-hoc-verify` (`69f5201`) - none merged to main yet, pending a decision below.

Retrieval-side caveat: MRR@4/context_precision/context_recall/Hit Rate are identical across all three experiments (as expected - none of them touch retrieval code) but differ from the +Tables row above, because the FAISS index was rebuilt in between (the additive-ladder work reverted and rebuilt it to isolate the +Reranking stage). FAISS's HNSW graph build isn't seeded deterministically, so rebuilding from identical embeddings can still shift a few borderline rankings. This means the three experiment columns are cleanly comparable to *each other*, but their diff against the +Tables column carries a small amount of index-rebuild noise on top of the real generation-side effect.

None of the three improved faithfulness over baseline - all three came in lower:

- Grounding instructions performed worst (-0.044 faithfulness, -0.018 correctness) - explicit "don't infer, cite claim-by-claim" instructions didn't make the model more grounded, if anything the opposite.
- Lower temperature (-0.009 faithfulness, -0.005 correctness) - close to a wash, small regression.
- Post-hoc verification (-0.007 faithfulness, correctness flat at 0.6501) - best of the three, essentially a wash on correctness, still a small faithfulness dip. Costs ~2x generation latency for a result indistinguishable from doing nothing, given the index-rebuild noise band above.

Take-away: none of these three are worth merging as-is. The faithfulness ceiling here likely isn't a prompt-wording or decoding-temperature problem - post-hoc verification (the most direct lever, a dedicated grounding-check pass) still didn't move it, which suggests either RAGAS's faithfulness judge is noisy at this sample size, or the ungrounded claims are concentrated in a subset of questions (e.g. table/multi-hop) that these generic fixes don't target. Worth checking per-category faithfulness (not just the blended score) before trying another variant.
