# Eval results comparison

FAISS backend only, same 200-row eval set, `k=4`. Each column adds exactly one component on top of the previous one (additive ladder), not an isolated A/B.

> Maintenance rule: every time the eval benchmark is run, add the result to the table below and update the diff/notes for that stage - don't let this doc drift out of sync with `data/processed/eval_runs/`.

| Metric | Baseline (dense) | +Hybrid (BM25) | +Reranking (cross-encoder) | +Tables (Phase 5.1) | +Table retrievability fixes |
|---|---|---|---|---|---|
| Hit Rate@4 | 0.860 | 0.945 | 0.940 | 0.945 | 0.930 |
| MRR@4 | 0.7125 | 0.7858 | 0.8600 | 0.8153 | 0.8604 |
| context_precision | 0.7195 | 0.7637 | 0.8811 | 0.7590 | 0.8668 |
| context_recall | 0.8015 | 0.9141 | 0.8821 | 0.9219 | 0.9100 |
| faithfulness | 0.6571 | 0.6592 | 0.6794 | 0.6800 | 0.6917 |
| answer_relevancy | 0.7746 | 0.8272 | 0.8395 | 0.8301 | 0.8598 |
| answer_correctness | 0.5963 | 0.6383 | 0.6377 | 0.6501 | 0.6945 |

Source files:
- Baseline: `data/processed/eval_runs/2026-07-10T07-04-26.367728+00-00_faiss_dense_k4.json`
- +Hybrid: `data/processed/eval_runs/2026-07-10T07-23-27.572883+00-00_faiss_hybrid_k4.json`
- +Reranking: `data/processed/eval_runs/2026-07-10T12-54-46.134674+00-00_faiss_reranked_k4.json` - reverted `parse.py`/`chunk.py`/`dataset.py` to pre-table commit `028db64~1` to keep the index text-only for this stage, then restored current code and rebuilt the table-inclusive index afterward.
- +Tables: `data/processed/eval_runs/2026-07-10T10-35-40.149738+00-00_faiss_reranked_k4.json` (hybrid + reranking + table-inclusive index, all together)
- +Table retrievability fixes: `data/processed/eval_runs/2026-07-13T17-17-40.291555+00-00_faiss_reranked_k4.json`

Unlike earlier stages, the "+Table retrievability fixes" column bundles several changes shipped together on 2026-07-13 (see ADR-0007 and commits `036e7a8`..`48c66fa`): table chunks enriched with inline heading + cached Gemini summary, cross-encoder scoring tables by their heading/summary preamble, BM25 tokenisation fixed (lowercased word tokens instead of bare `str.split()`), "table N" queries injecting section-matched chunks into the reranker pool, calculator function calling, and generation temperature 0.2 (was default 1.0).

Caveat: the +Reranking run uses the current (table-aware) generation prompt wording in `answer.py`, not the pre-table prompt - negligible effect since no table chunks exist yet at that stage to trigger the table-specific instructions.

## Reading metrics

- Hit Rate@4: did any of the top 4 retrieved chunks contain the ground-truth answer (yes/no per question, averaged)?
- MRR@4: how high the right chunk ranked within the top 4 (1st = 1.0, 2nd = 0.5, 3rd = 0.33, 4th = 0.25, absent = 0, averaged).
- context_precision: how much of the retrieved context is actually relevant to the question (RAGAS, LLM-graded).
- context_recall: how much of the ground-truth answer is covered by the retrieved context (RAGAS, LLM-graded).
- faithfulness: how well the generated answer's claims are supported by the retrieved context, i.e. absence of hallucination (RAGAS, LLM-graded).
- answer_relevancy: how directly the generated answer addresses the question asked (RAGAS, LLM-graded).
- answer_correctness: how well the generated answer matches the ground-truth answer, the end-to-end outcome metric (RAGAS, LLM-graded).

## What improved, stage by stage

- Baseline -> +Hybrid: everything moved up, nothing regressed. context_recall jumped most (+0.113). answer_correctness +0.042. faithfulness barely moved (+0.002) - better material, not more honesty.
- +Hybrid -> +Reranking: MRR +0.074 (biggest single-stage MRR jump) and context_precision +0.117, but Hit Rate@4 dipped slightly (-0.005) and answer_correctness is flat (+0.0 vs -0.0006, noise). Reranking's job is ordering, not recall - it can't find chunks hybrid didn't retrieve into the candidate pool, it can only push the right one higher once it's there. That's exactly what the numbers show: big precision/ranking gains, no real recall or end-to-end movement.
- +Reranking -> +Tables: Hit Rate@4 +0.005, context_recall +0.040, answer_correctness +0.012. Modest, as expected given only 9 of 200 questions are table-typed - see below for why the table-specific numbers matter more than the blended ones here.
- +Tables -> +Table retrievability fixes: answer_correctness +0.044 - the largest single-stage end-to-end gain on the whole ladder, beating even Baseline -> +Hybrid (+0.042). MRR@4 +0.045 (0.8604, best of any stage), context_precision +0.108, faithfulness +0.012 (also a ladder best), answer_relevancy +0.030. Hit Rate@4 dipped -0.015 and context_recall -0.012; some of that sits within the documented index-rebuild noise band (the FAISS HNSW build is not seeded deterministically), and the rest is consistent with the reranking trade-off seen before: sharper ordering (precision/MRR up) at the cost of a few borderline candidates dropping out of the top 4. The end-to-end outcome moving +0.044 while Hit Rate dipped says the chunks that ranked higher were the ones that actually mattered for answers.

## Table questions (n=9 of 200)

| Stage | Hit Rate@4 | MRR@4 |
|---|---|---|
| Baseline (dense) | 0.556 | 0.278 |
| +Hybrid | 0.667 | 0.426 |
| +Reranking (no tables) | 0.556 | 0.426 |
| +Tables | 0.667 | 0.509 |
| +Table retrievability fixes | 0.667 | 0.667 |

Status after the 2026-07-13 retrievability fixes (root cause and fix documented in ADR-0007): every retrieved table hit now lands at rank 1 (MRR = Hit Rate). Of the 3 remaining misses, 2 are metric artefacts (the answer is retrieved verbatim from narrative text elsewhere and answered correctly, the page-based label just doesn't credit it) and 1 is genuinely hard ("total value of assets" is lexically closer to the report's many "fair value of assets" chunks than to the balance sheet).

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

Take-away: none of these three are worth merging as-is.

### Round 2 (re-run on top of +Table retrievability fixes, 2026-07-13)

The same experiments were re-run as fresh branches off current main (`experiment/grounding-prompt-r2`, `experiment/post-hoc-verify-r2`, `experiment/low-temperature-r2`), each in its own worktree with a byte-copy of the current FAISS index - so unlike round 1, retrieval metrics are exactly identical to the baseline column (Hit Rate@4 0.930, MRR@4 0.8604, no index-rebuild noise). Note the baseline itself changed generation-side since round 1: temperature is now 0.2 (was the 1.0 default) and the prompt gained table/calculator instructions.

| Metric | Baseline (+Table retr. fixes) | +Grounding instructions (r2) | +Post-hoc verification (r2) |
|---|---|---|---|
| faithfulness | 0.6917 | 0.7015 | 0.7003 |
| answer_correctness | 0.6945 | 0.6907 | 0.6946 |
| answer_relevancy | 0.8598 | 0.8604 | 0.8604 |
| context_precision | 0.8668 | 0.8719 | 0.8694 |
| context_recall | 0.9100 | 0.9091 | 0.9091 |

Source files: `data/processed/eval_runs/2026-07-13T18-49-44.847663+00-00_faiss_reranked_k4.json` (grounding), `data/processed/eval_runs/2026-07-13T18-28-28.282793+00-00_faiss_reranked_k4.json` (post-hoc).

The low-temperature re-run was deliberately cancelled mid-flight: with the baseline already at temperature 0.2, it would have measured 0.1 vs 0.2 - a far smaller delta than round 1's 0.1 vs 1.0, with no realistic chance of a decision-changing result.

Round-2 take-away: the conclusion stands - neither variant is worth merging (both within roughly +-0.01 of baseline on faithfulness and correctness). The genuinely useful datum is that grounding instructions flipped from round 1's worst performer (-0.044 faithfulness) to a small positive (+0.010) with nothing but baseline changes in between, which puts round 1's scariest number squarely inside the noise band of RAGAS faithfulness at n=200. Single-run deltas of this size should not drive merge decisions on their own. The faithfulness ceiling here likely isn't a prompt-wording or decoding-temperature problem - post-hoc verification (the most direct lever, a dedicated grounding-check pass) still didn't move it, which suggests either RAGAS's faithfulness judge is noisy at this sample size, or the ungrounded claims are concentrated in a subset of questions (e.g. table/multi-hop) that these generic fixes don't target. Worth checking per-category faithfulness (not just the blended score) before trying another variant.
