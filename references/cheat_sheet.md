# Demo cheat sheet

## Eval metrics

- Hit Rate@4: did any of the top 4 retrieved chunks contain the ground-truth answer (yes/no per question, averaged)?
- MRR@4: how high the right chunk ranked within the top 4 (1st = 1.0, 2nd = 0.5, 3rd = 0.33, 4th = 0.25, absent = 0, averaged).
- context_precision: how much of the retrieved context is actually relevant to the question (RAGAS, LLM-graded).
- context_recall: how much of the ground-truth answer is covered by the retrieved context (RAGAS, LLM-graded).
- faithfulness: how well the generated answer's claims are supported by the retrieved context, i.e. absence of hallucination (RAGAS, LLM-graded).
- answer_relevancy: how directly the generated answer addresses the question asked (RAGAS, LLM-graded).
- answer_correctness: how well the generated answer matches the ground-truth answer, the end-to-end outcome metric (RAGAS, LLM-graded).

Cross-encoder model used: cross-encoder/ms-marco-MiniLM-L-6-v2 for re-ranking.

## Business questions and use cases (2 min)

- Problem: manual report analysis is slow and expert-bound - this speeds it up.
- Friction cut: from reading 147 pages to asking questions with cited answers.
- Business question: cost optimisation - fewer analyst-hours per filing, report insight opened up to less analysis-heavy roles.
- Research question: can multimodal retrieval + generation answer financial questions reliably enough to trust?
- Use cases: analyst copilot, due diligence, IR Q&A, internal audit.
- Non-negotiable in finance: every answer cites its source.

## What was built

- Two pipelines, one product: tuned chunk RAG (parse -> table/chart enrichment -> hybrid -> rerank -> Gemini + calculator tool) vs ColPali page-image late interaction.
- Same UI, same eval harness, Docker deployment.
- Sidebar pipeline radio = the pivot moment of the demo.

## Live demo - queries below

## How we know it works

- Not "looks right": 200-question benchmark, two layers - per-step diagnostics + end-to-end correctness (LLM-as-judge).
- Additive ladder: answer_correctness 0.596 -> 0.720, one component at a time.
- ColPali headline: one untuned VLM within 0.015 Hit Rate@4 of four phases of tuning - and wins tables outright (0.667 -> 0.778).

## Strengths and limitations

Strengths:

- Per-modality wins: chart captions -> image hit rate 1.0; ColPali -> tables without any parsing.
- Attribution down to the literal page pixels.
- Honest "not in the context" instead of hallucination.
- Every claim carries a measured number; every serving path verified end to end (caught a silent model-load failure in Docker exactly this way).

Limitations (three):

- Reading figures off page renders: retrieval at parity, generation gap (answer_correctness 0.654 vs 0.720); chart-derived values approximate by nature.
- Multi-part questions needing two modalities in one top-4 window: 0.5 hit rate.
- Single-document scope; per-modality eval subsets small (n=6-9), so those numbers are coarse.

## Roadmap close

- Pixel-level bounding-box attribution, multi-document corpus, per-content-type routing between the two pipelines.

## Questions

1. "What was IFC's net income in FY24, and what drove the increase from FY23?" - flagship: $1,485M, drivers, page citations.
2. "What were IFC's key financial ratios in FY24?" - table at rank 1, dataframe render + summary caption, corrected section heading (last night's fix, live).
3. "By what percentage did IFC's net income change from FY23 to FY24?" - calculator tool call, no freehand arithmetic.
4. "How did IFC's non-performing loans evolve from FY20 to FY24?" - chart caption answering for the pixels (all ten values verified).
5. Pivot to Page images, ask exactly: What were IFC's total "Charges on borrowings" for the fiscal year ended June 30, 2023? - page 9 at rank 1, full page render + MaxSim score. Exact wording matters: a paraphrase misses (late interaction is phrasing-sensitive - worth a sentence if asked).
6. If time: "What was IFC's headcount in Brazil in FY24?" - the honesty fallback, grounded refusal.

## How would you deploy it?

Ship the existing Docker image to Cloud Run for the Streamlit app, with Vertex AI auth via workload identity federation (no keys to rotate). Qdrant needs persistent storage Cloud Run doesn't offer, so it runs on a small managed instance (Qdrant Cloud or a persistent-disk VM) instead; the FAISS index and chunk/caption caches are pulled from a GCS bucket at container startup rather than baked into the image.

## How would you improve the architecture?

Replace the manual chunk-vs-ColPali pipeline toggle with a query router that picks or fuses both pipelines per question. The eval already shows why: multi-part questions needing two modalities in one top-4 window score only 0.5 hit rate, exactly the case a routing/orchestration layer over both retrievers would fix.
