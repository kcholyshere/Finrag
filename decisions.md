# Decision log

## ADR-0001 - PDF text/table/image extraction library: Docling

- Date: 2026-07-03
- Status: accepted

### Context
Phase 1 needs text extraction from `ifc-annual-report-2024-financials.pdf`. Later phases need tables and images from the same PDF. The practice requirements list Docling, PyMuPDF, or Gemini multimodal as acceptable options.

### Options considered
- PyPDF2 - simplest API - rejected because it has the weakest layout fidelity on multi-column financial documents and no structure detection.
- pdfminer.six - mature plain-text extraction - rejected because it has no structure detection, so later phases would need a second tool for tables/images anyway.
- Docling (chosen) - structure-aware (headings, reading order, table/figure detection built in), one tool reusable across all phases.

### Decision
Use Docling for all PDF parsing: text in Phase 1, tables and images in later phases.

### Consequences
Slightly more setup cost up front. Avoids re-parsing the PDF with a different tool per phase, and gives structure metadata (headings, page numbers, reading order) for free, which later phases and citation snippets need.

### Transferable principle
When a document has to be parsed incrementally across project phases, pick the one tool that covers the hardest future phase first, rather than the simplest tool for today's phase - re-parsing the same source with a second tool later is wasted work and a metadata-consistency risk.

## ADR-0002 - Embedding model: Vertex AI `gemini-embedding-001`

- Date: 2026-07-03
- Status: accepted

### Context
The requirements mandate Vertex AI auth (no API keys) for the LLM. Embeddings needed a matching choice, and no embedding model was prescribed.

### Options considered
- Local sentence-transformers - free, fully offline - rejected because it introduces a second auth/infra path alongside Vertex, breaking the "everything through one GCP project, no API keys" pattern.
- Vertex AI `gemini-embedding-001` (chosen) - current GA text embedding model, same auth/project as Gemini generation.

### Decision
Use `gemini-embedding-001` via Vertex AI for chunk embeddings.

### Consequences
One GCP project and auth mechanism for the whole pipeline. Embedding calls incur Vertex AI cost/quota instead of running locally for free.

### Transferable principle
When a stack mandates a specific auth story for one component (here: Vertex AI, no API keys, for the LLM), extend that same auth story to adjacent components (embeddings) by default - mixing auth paths multiplies operational surface area for no clear benefit.

## ADR-0003 - UI framework: Streamlit

- Date: 2026-07-03
- Status: accepted

### Context
The requirements allow either Streamlit or Gradio for the query UI.

### Options considered
- Gradio - faster to stand up a minimal chat-style Q&A box - rejected because it is a less natural fit once the UI needs side-by-side panels for multimodal results and FAISS vs Qdrant comparisons.
- Streamlit (chosen) - richer layout control, better suited to later multimodal result panels and vector-store comparison views.

### Decision
Use Streamlit for the query interface.

### Consequences
Slightly more boilerplate for a basic query box now, in exchange for an easier path to richer layouts (tabs, side-by-side panels) in later phases.

### Transferable principle
Pick UI tooling based on the shape of the final feature set, not the shape of the first milestone, when the roadmap already specifies later features that need more layout flexibility.

## ADR-0004 - `src/` layout: restructured for RAG

- Date: 2026-07-03
- Status: accepted

### Context
The repo started from a cookiecutter data-science template (`config.py`, `dataset.py`, `features.py`, `modeling/`, `services/`) aimed at ML training pipelines, which doesn't map to a RAG application. All existing files were empty stubs, so no code was at risk.

### Options considered
- Keep the scaffold, shoehorn RAG code into the existing stub files - rejected because module names (e.g. `features.py`) would stop matching what the code does, confusing future readers.
- Restructure `src/` into RAG-shaped modules (chosen) - e.g. ingestion, embedding, retrieval, generation, ui - and drop modules that don't map to this project (`features.py`, `modeling/`).

### Decision
Restructure `src/` for RAG; remove ML-training modules that don't apply.

### Consequences
Module names match what the code does; no unused ML-training scaffolding left to confuse future readers. One-time reorganisation cost, absorbed now while all files are still empty stubs.

### Transferable principle
When inheriting a generic template whose module names encode assumptions that don't hold for the actual project, rename/restructure early - before real code accumulates in the mismatched files - rather than paying the confusion cost indefinitely to save a one-time rename.

## ADR-0005 - Docker included in Phase 1, Qdrant runs as a real containerised service

- Date: 2026-07-03
- Status: accepted

### Context
Docker is listed in the overall project tech stack, but the Phase 1 task list (ingestion, embedding/indexing, retrieval, generation & UI, observability) doesn't call it out explicitly. It was unclear whether to defer Dockerisation to a later phase, in line with the project's "progressively add components per phase" rule.

### Options considered
- Defer to a later phase - rejected by the user; they chose to include it now.
- Include now (chosen) - Dockerise the Streamlit app and run Qdrant as a proper service via `docker-compose.yml` (official `qdrant/qdrant` image) rather than Qdrant's embedded/local-path mode.

### Decision
Include Docker in Phase 1. Qdrant runs as a containerised service (not embedded mode), with a `docker-compose.yml` defining both the `app` (Streamlit) and `qdrant` services.

### Consequences
The FAISS-vs-Qdrant comparison reflects a realistic deployment (network calls to a real service) rather than an in-process shortcut. Slightly more setup work now (Dockerfile, compose file, service wiring) instead of deferring it, but it lands while the codebase is still small.

### Transferable principle
When a requirement mentions infrastructure (e.g. "use Docker") without specifying which phase, resolve the ambiguity by asking rather than assuming a default of "defer" or "include" - the two choices have different downstream shapes (e.g. embedded vs client-server mode for a database), and picking wrong costs a rework later.

## ADR-0006 - Phase 2 evaluation: two-layer metrics (per-step diagnostics + end-to-end outcome), synthetic dataset expansion, RAGAS-only judging

- Date: 2026-07-06
- Status: accepted

### Context
The mentor asked two distinct questions: (1) how to check the quality of each data-processing step (parsing, chunking, embedding/retrieval), and (2) how to assess the quality of the whole solution. Phase 2 requirements (`agent_docs/phase2-requirements.md`) call for RAGAS-based evaluation and experimenting with LLM-as-judge, using `references/RAG_evaluation_dataset.csv` (33 curated Q&A pairs) and `references/rag_evaluation.md` (a Hugging Face cookbook on a different stack: HF datasets, Mixtral, Zephyr, GPT-4) as a pattern reference.

### Options considered
- **Metric structure**: a single end-to-end score vs a two-layer split - rejected the single-score approach because it can't answer "which step is broken," which is exactly the mentor's first question.
- **Dataset size**: use the 33 curated pairs as-is vs also build the cookbook's synthetic generation + critique-agent filtering pipeline (groundedness/relevance/standalone scoring) to expand to ~100-200 pairs - chose to also build synthetic generation, since 33 pairs give noisy metric averages.
- **LLM-as-judge**: RAGAS's built-in LLM-graded metrics (faithfulness, answer_relevancy, answer_correctness) only vs also building a separate custom rubric-based judge on top - chose RAGAS-only to avoid duplicating what RAGAS already does internally.
- **Pre-embedding coverage**: skip parsing/chunking checks (rely solely on downstream retrieval metrics) vs add a lightweight fuzzy-match coverage check (does the parsed/chunked corpus contain each ground-truth context snippet, scoped to `Context_Content_Type == text` since table ground truth is often a paraphrased fragment) - chose to add it, since a parsing/chunking bug and an embedding-quality problem would otherwise look identical in the retrieval numbers.

### Decision
Structure Phase 2 evaluation in two layers, mapped directly onto the mentor's two questions:
- **Per-step diagnostics** (answers "quality of each step"): a text-coverage check for parsing/chunking; Hit Rate@k / MRR / Recall@k plus RAGAS `context_precision` / `context_recall` for retrieval (the measurable proxy for embedding quality, since embeddings themselves aren't directly inspectable); RAGAS `faithfulness` / `answer_relevancy` for generation.
- **End-to-end outcome** (answers "quality of the whole solution"): RAGAS `answer_correctness` against ground-truth answers, using RAGAS's own LLM-graded metrics as the "LLM-as-judge" experiment the requirements ask for.
- Expand the 33-pair curated dataset with a synthetic generation + critique-filtering pipeline (cookbook-style), adapted to this project's stack (Gemini via Vertex AI, not Mixtral/GPT-4).
- Persist each evaluation run as a JSON file tagged with its settings (chunk size, embedding model, backend), following the cookbook's `test_settings` pattern, so future phases can be compared against the Phase 1 baseline without a dashboard.

### Consequences
RAGAS defaults to OpenAI internally; wiring it to Gemini/`gemini-embedding-001` via Vertex AI (no API keys) is untested on this stack and is the main implementation risk - to be spiked with a single metric before building the full harness. Synthetic dataset generation adds real work (question-generation prompts, critique agents, manual spot-checks) beyond just running RAGAS over the existing 33 pairs.

### Transferable principle
When a stakeholder asks two evaluation questions at different granularities ("each step" vs "the whole thing"), don't collapse them into one score - pick a metrics framework whose own taxonomy already splits along that line (here, RAGAS's retrieval-specific vs generation-specific vs outcome metrics), and use that split as the answer's structure rather than inventing a parallel one.

## ADR-0007 - Table retrievability: enrich table chunks with inline heading + LLM summary

- Date: 2026-07-13
- Status: accepted

### Context
After Phase 5.1 indexed tables, table-typed eval questions still capped at Hit Rate@4 = 0.667 (6/9) - no better than before tables existed in the index. Reproducing the 3 misses end to end showed the ground-truth table never even entered the reranker's candidate pool: dense rank 16-20+ and BM25 rank 12+/absent. Root cause was an asymmetry in what gets embedded: text chunks carry their section header inline in `page_content`, but table chunks were bare Docling markdown - the caption/section lived only in metadata, which is never embedded. A wall of pipe-delimited numbers matches question-shaped queries poorly in both dense embedding (signal dilution) and BM25 (length normalisation), and for one miss the section header itself carried no signal ("AS OF THE YEAR ENDED JUNE 30").

### Options considered
- **Inline heading + one-time Gemini summary per table** (chosen): prepend `Table: <caption/section>` plus a 2-4 sentence LLM description (subject, line items, periods - no figures) to each table chunk's embedded text. Strongest semantic signal; fixes the miss whose section header is useless. Costs 154 one-time Gemini calls at ingestion, cached in `data/interim/table_summaries.json` keyed by markdown hash.
- **Inline heading only**: cheap and deterministic, but weak where the caption/section carries no signal (the "Charges on borrowings" miss).
- **Raise reranker `candidate_k` from 10 to ~30**: one-line change, but the worst miss wasn't in the top 30 of either retriever, so it fixes at most 1 of 3 misses and adds reranker latency.

### Decision
Add `src/ingestion/enrich.py` (`summarise_tables`, hash-keyed JSON cache) and build table `page_content` as heading + summary + blank line + raw markdown. The raw table stays in the chunk for generation; the Streamlit UI separates preamble from table by filtering pipe-prefixed lines, showing the preamble as a caption above the rendered dataframe.

### Consequences
Ingestion now needs Vertex AI access for table summaries (cache makes re-runs free until table extraction changes). Table chunks are ~300-500 chars longer, slightly increasing embedding cost and prompt size. Retrieval and generation code paths are untouched - the fix is purely in what gets indexed. Table Hit Rate@4 on the 9-question eval subset: 0.667 -> measured after rebuild (see reports/eval_results_comparison.md).

### Transferable principle
When one content type retrieves worse than another through the same pipeline, diff what actually gets embedded for each type before touching the retriever - an asymmetry in chunk construction (metadata-only context vs inline context) masquerades as a retrieval-algorithm problem but is fixed at ingestion.

## ADR-0008 - Image retrievability: figure-bearing Gemini captions with classification-based filtering

- Date: 2026-07-14
- Status: accepted

### Context
Phase 5.2 required making the report's charts and diagrams retrievable. The Phase 1 Docling parse stored only picture geometry (page/bbox) - `generate_picture_images` defaults to False, so no pixel data existed to caption. Of the 36 pictures in the report, only ~25 carry information (16 charts, 9 diagrams); the rest are logos and signatures. Unlike tables, an image has no raw textual body to travel with the chunk: whatever text the captioning step produces is the only representation retrieval and generation ever see.

### Options considered
Image pixels:
- **Docling re-parse with `generate_picture_images=True`, `images_scale=2`** (chosen): one-time re-parse cost (~10 min) and a larger docling.json, but a single parsing source of truth and the planned `get_image()` API. Verified zero drift: identical text/table extraction, all 154 table-summary cache hits survived.
- **PyMuPDF bbox rendering from the existing docling.json**: no re-parse, but adds a second parser dependency plus BOTTOMLEFT-to-TOPLEFT coordinate conversion - two extraction paths to maintain.

Decorative-image filtering:
- **Gemini structured-output classification** (chosen): the captioning call also returns `kind` (chart/diagram/logo/signature/decorative); only charts and diagrams are indexed. Robust to layout edge cases; costs nothing extra since every image is captioned once anyway.
- **Size heuristic**: no LLM dependency, but brittle - the page-1 logo is mid-sized while some real charts are small.
- **Index everything**: signature/logo chunks dilute retrieval for zero benefit.

Caption content:
- **Include the chart's actual figures** (chosen): the description substitutes for the image at generation time, so it must carry the data, not just describe the topic.
- **Topic-only summary (mirroring ADR-0007's table summaries)**: wrong here - table summaries omit figures precisely because the raw markdown travels with the chunk; images have no such body.

### Decision
`parse.py::extract_image_records` (re-parse with picture images), `enrich.py::caption_images` (Gemini multimodal, pydantic `ImageCaption` schema, cache keyed by sha1 of PNG bytes in `data/interim/image_captions.json`), `chunk.py::chunk_images` (one chunk per informative image, `content_type: "image"`, decoratives dropped - mirroring how NOISE_LABELS filtering lives in chunking, not parsing). `answer.py`'s prompt tells the model image context is a chart description whose values may be approximate.

### Consequences
Ingestion now needs Vertex multimodal access (cache makes re-runs free). docling.json grows with embedded images. 25 image chunks joined the index (852 total). Eval (+Images ladder stage): image-question Hit Rate@4 0.833 -> 1.000, blended answer_correctness 0.6945 -> 0.7073 and faithfulness 0.6917 -> 0.7184 (both ladder bests), zero regressions; the NPL chart answer was verified value-for-value against the rendered image. Known limits: figures duplicated in the report (e.g. Figure 14) index twice, and combination image+text questions stay at 0.5 hit rate.

### Transferable principle
Whether derived text should summarise or transcribe depends on whether the raw content travels with the chunk: table summaries omit figures because the markdown follows them; image captions must include figures because the pixels never reach the generator. Getting this backwards either duplicates data or silently discards it.

## ADR-0009 - Phase 6 multimodal retrieval: ColQwen2 patch embeddings, Qdrant-only late interaction

- Date: 2026-07-15
- Status: accepted

### Context
Phase 6 (`agent_docs/phase6-requirements.md`) asks for a ColPali-like approach: a VLM producing contextualised embeddings from visual document patches, stored and retrieved via a late-interaction/MaxSim (ColBERT-style) mechanism, run as a separate pipeline compared against the Phase 5.2 (+images) baseline rather than replacing it. This is a narrower decision than earlier ADRs - only two options were weighed per fork, not a broad market survey, since the requirements doc already names the model family (ColPali/PaliGemma-like) and the project is in a Phase 6 speed push.

### Options considered
Embedding model:
- Vertex AI multimodal embeddings (reusing the existing Gemini/Vertex auth path, no new heavy dependency) - rejected because the API returns one dense vector per image, not per-patch vectors, so there is no real late interaction to implement; would water down the phase's core requirement.
- `colpali-engine` / ColQwen2 (chosen) - genuine multi-vector per-patch embeddings, enabling actual MaxSim late interaction. Runs locally on the M3 Pro via MPS; feasible as a one-time batch job even at the report's actual size (147 pages, not the ~20 originally assumed - corrected mid-planning). New ~6GB model dependency, the first heavy local Torch/transformers component in the stack (distinct from the small cross-encoder reranker already in use).

Vector DB / late-interaction storage:
- Qdrant + FAISS parity, mirroring the dual-backend comparison pattern from earlier phases - rejected because FAISS has no native late-interaction/MaxSim support; matching it would mean hand-rolling brute-force MaxSim scoring with no engine support behind it.
- Qdrant only (chosen) - native `MultiVectorConfig` + `MAX_SIM` comparator is a direct fit for storing and scoring per-patch vectors.

### Decision
Use ColQwen2 via `colpali-engine`, run locally, for patch embeddings. Store and retrieve them in a new Qdrant collection using `MultiVectorConfig`/`MAX_SIM`. Drop FAISS parity for this phase only. Page images themselves are rendered via a decoupled PyMuPDF script (`src/ingestion/page_images.py`) rather than reusing the shared Docling parse/cache, to avoid re-running Docling's full layout/OCR pass over 147 pages and bloating the `docling.json` cache every other pipeline script already loads.

### Consequences
Indexing 147 pages (~150k patch vectors) is a real one-time compute + disk-caching job, not a quick pass - needs batching and a resumable cache, unlike the smaller Phase 5.2 image-captioning cache. The stack gains its first local heavy Torch/transformers dependency. FAISS stops being an apples-to-apples comparison target for this phase specifically (Phase 3-5.2 dual-backend parity is unaffected). Query-time retrieval needs a VLM forward pass per query instead of a cheap embedding lookup - fine for a demo, not representative of production query latency.

### Transferable principle
When a requirements doc already names the model family and the project is time-boxed, it's fine to resolve implementation forks with a small number of concrete options rather than an exhaustive survey - but still write down what was rejected and why, since "we didn't deeply compare" is itself useful context for whoever revisits the decision later.
