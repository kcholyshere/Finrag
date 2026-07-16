"""Phase 2 evaluation entrypoint: runs the RAG pipeline over the eval dataset and
computes both per-step diagnostics and the end-to-end outcome metric, then saves a
settings-tagged JSON so later phases can be compared against this baseline.

Run with `python -m src.evaluation.run_eval [--backend faiss|qdrant] [--k 4] [--n N]
[--retrieval-mode dense|hybrid|reranked]`.
Requires the FAISS/Qdrant indexes to already be built (`python -m src.dataset`), and
`data/processed/eval_dataset.csv` to exist (`python -m src.evaluation.synthetic_qa`).
"""

import os

# faiss and one of ragas' dependencies both bundle their own OpenMP runtime; without
# this, faiss aborts the process with "OMP: Error #15" as soon as both are imported
# in the same process (harmless in this single-process, CPU-only pipeline).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Literal

import pandas as pd

from src import config
from src.evaluation import diagnostics
from src.evaluation.diagnostics import EvalSample
from src.generation.answer import generate_answer
from src.ingestion.chunk import load_chunks
from src.retrieval.retriever import (
    Backend,
    retrieve,
    retrieve_colpali,
    retrieve_hybrid,
    retrieve_reranked,
)

MAX_WORKERS = 6

# A 200-question run makes ~1,000 API calls over half an hour - the odd network
# blip is expected, and one lost sample shouldn't cost the whole run.
SAMPLE_ATTEMPTS = 3

RetrievalMode = Literal["dense", "hybrid", "reranked", "colpali"]
RETRIEVE_FNS = {
    "dense": retrieve,
    "hybrid": retrieve_hybrid,
    "reranked": retrieve_reranked,
    # Page-image late interaction (Phase 6); Qdrant-only, so backend is ignored.
    "colpali": lambda query, backend, k: retrieve_colpali(query, k=k),
}


def _build_sample(row: pd.Series, backend: Backend, k: int, retrieval_mode: RetrievalMode) -> EvalSample:
    question = row["Question"]
    for attempt in range(1, SAMPLE_ATTEMPTS + 1):
        try:
            retrieved_docs = RETRIEVE_FNS[retrieval_mode](question, backend=backend, k=k)
            generated_answer = generate_answer(question, retrieved_docs)
            break
        except Exception as exc:
            if attempt == SAMPLE_ATTEMPTS:
                raise
            print(f"Retrying sample after {type(exc).__name__} (attempt {attempt}): {question[:60]}")
            time.sleep(5 * attempt)
    return EvalSample(
        question=question,
        reference_answer=row["Ground_Truth_Answer"],
        reference_context=row["Ground_Truth_Context"],
        page_numbers=diagnostics.parse_page_numbers(row["Page_Number"]),
        content_type=row["Context_Content_Type"],
        retrieved_docs=retrieved_docs,
        generated_answer=generated_answer,
    )


def run_pipeline(
    eval_df: pd.DataFrame, backend: Backend, k: int, retrieval_mode: RetrievalMode
) -> list[EvalSample]:
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(_build_sample, row, backend, k, retrieval_mode)
            for _, row in eval_df.iterrows()
        ]
        return [f.result() for f in futures]


def run(
    backend: Backend = "faiss",
    k: int = 4,
    n: int | None = None,
    retrieval_mode: RetrievalMode = "dense",
) -> dict:
    if retrieval_mode == "colpali":
        backend = "qdrant"  # the page-image collection only exists in Qdrant

    eval_df = pd.read_csv(config.EVAL_DATASET_PATH)
    if n is not None:
        eval_df = eval_df.sample(n=n, random_state=0)

    print(
        f"Running retrieval + generation for {len(eval_df)} questions "
        f"(backend={backend}, k={k}, retrieval_mode={retrieval_mode})..."
    )
    samples = run_pipeline(eval_df, backend, k, retrieval_mode)

    print("Checking parsing/chunking coverage...")
    chunks = load_chunks()
    coverage = diagnostics.parse_chunk_coverage(eval_df, chunks)

    print("Computing retrieval rank metrics (Hit Rate@k / MRR)...")
    rank_metrics = diagnostics.hit_rate_and_mrr(samples, k)

    # ColPali answers from page images; the context-grounded RAGAS metrics would
    # score placeholder strings the model never saw, so they are reported as
    # None (page-based hit rate/MRR covers retrieval for that pipeline).
    context_based = retrieval_mode != "colpali"
    print("Computing RAGAS metrics "
          + ("(context_precision, context_recall, faithfulness, answer_relevancy, "
             "answer_correctness)..." if context_based
             else "(answer_relevancy, answer_correctness; context metrics N/A)..."))
    ragas_metrics = diagnostics.run_ragas_metrics(samples, context_based=context_based)

    result = {
        "settings": {
            "backend": backend,
            "k": k,
            "retrieval_mode": retrieval_mode,
            "n_samples": len(samples),
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "gemini_model": config.GEMINI_MODEL,
            "embedding_model": (
                config.COLPALI_MODEL if retrieval_mode == "colpali" else config.EMBEDDING_MODEL
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "diagnostics": {
            "parse_chunk_coverage": coverage,
            "retrieval_rank_metrics": rank_metrics,
            "retrieval_ragas_metrics": {
                "context_precision": ragas_metrics["context_precision"],
                "context_recall": ragas_metrics["context_recall"],
            },
            "generation_ragas_metrics": {
                "faithfulness": ragas_metrics["faithfulness"],
                "answer_relevancy": ragas_metrics["answer_relevancy"],
            },
        },
        "outcome": {
            "answer_correctness": ragas_metrics["answer_correctness"],
        },
    }

    config.EVAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp_slug = result["settings"]["timestamp"].replace(":", "-")
    output_path = config.EVAL_RUNS_DIR / f"{timestamp_slug}_{backend}_{retrieval_mode}_k{k}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Saved run to {output_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["faiss", "qdrant"], default="faiss")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--n", type=int, default=None, help="Sample size for a quick smoke test")
    parser.add_argument(
        "--retrieval-mode", choices=["dense", "hybrid", "reranked", "colpali"], default="dense"
    )
    args = parser.parse_args()

    run(backend=args.backend, k=args.k, n=args.n, retrieval_mode=args.retrieval_mode)
