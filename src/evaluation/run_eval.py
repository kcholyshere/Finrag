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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
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
            # retrieve_hybrid returns the RRF union of both retrievers (up to 2 x k,
            # not truncated) - slice here so every mode feeds rank metrics, RAGAS,
            # and generation the same top-k (audit finding A9).
            retrieved_docs = retrieved_docs[:k]
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
    eval_df: pd.DataFrame,
    backend: Backend,
    k: int,
    retrieval_mode: RetrievalMode,
    partial_path: Path | None = None,
) -> tuple[list[EvalSample], list[dict]]:
    """Runs retrieval + generation for every row concurrently.

    A sample that exhausts all SAMPLE_ATTEMPTS retries is recorded and dropped,
    not propagated - one flaky question must not sink a 30-minute, ~1,000-call
    run that otherwise produced 199 good results (audit finding A5). Successful
    samples are appended to `partial_path` (a JSONL sidecar) as they complete -
    written only from this thread via `as_completed`, so no lock is needed - so a
    hard crash mid-run still leaves the completed samples on disk to inspect.
    """
    samples = []
    failures = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_question = {
            executor.submit(_build_sample, row, backend, k, retrieval_mode): row["Question"]
            for _, row in eval_df.iterrows()
        }
        for future in as_completed(future_to_question):
            question = future_to_question[future]
            try:
                sample = future.result()
            except Exception as exc:
                print(f"Sample failed after {SAMPLE_ATTEMPTS} attempts ({type(exc).__name__}): {question[:60]}")
                failures.append({"question": question, "error": f"{type(exc).__name__}: {exc}"})
                continue
            samples.append(sample)
            if partial_path is not None:
                with open(partial_path, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "question": sample.question,
                                "generated_answer": sample.generated_answer,
                                "retrieved_pages": [
                                    [doc.metadata.get("start_page"), doc.metadata.get("end_page")]
                                    for doc in sample.retrieved_docs
                                ],
                            }
                        )
                        + "\n"
                    )
    return samples, failures


def _ragas_mean(ragas_metrics: dict, name: str) -> float | None:
    entry = ragas_metrics[name]
    return entry["mean"] if entry is not None else None


def _ragas_count(ragas_metrics: dict, name: str) -> int | None:
    entry = ragas_metrics[name]
    return entry["count"] if entry is not None else None


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
    # Timestamp fixed up front so the partial sidecar and the final output file
    # share one run slug - the sidecar is `<slug>.partial.jsonl` (audit finding A5).
    config.EVAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    run_slug = f"{timestamp.replace(':', '-')}_{backend}_{retrieval_mode}_k{k}"
    partial_path = config.EVAL_RUNS_DIR / f"{run_slug}.partial.jsonl"

    samples, failures = run_pipeline(eval_df, backend, k, retrieval_mode, partial_path)
    if failures:
        print(f"{len(failures)} of {len(eval_df)} samples failed and were dropped from scoring.")

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
            "n_failed": len(failures),
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "gemini_model": config.GEMINI_MODEL,
            "embedding_model": (
                config.COLPALI_MODEL if retrieval_mode == "colpali" else config.EMBEDDING_MODEL
            ),
            "timestamp": timestamp,
        },
        # Failed samples never enter a metric denominator (n_samples above counts
        # only what was actually scored) - kept alongside settings so a run with
        # dropped samples is visibly non-clean rather than silently short (A5).
        "failed_samples": failures,
        "diagnostics": {
            "parse_chunk_coverage": coverage,
            "retrieval_rank_metrics": rank_metrics,
            "retrieval_ragas_metrics": {
                "context_precision": _ragas_mean(ragas_metrics, "context_precision"),
                "context_recall": _ragas_mean(ragas_metrics, "context_recall"),
                "context_precision_n": _ragas_count(ragas_metrics, "context_precision"),
                "context_recall_n": _ragas_count(ragas_metrics, "context_recall"),
            },
            "generation_ragas_metrics": {
                "faithfulness": _ragas_mean(ragas_metrics, "faithfulness"),
                "answer_relevancy": _ragas_mean(ragas_metrics, "answer_relevancy"),
                "faithfulness_n": _ragas_count(ragas_metrics, "faithfulness"),
                "answer_relevancy_n": _ragas_count(ragas_metrics, "answer_relevancy"),
            },
        },
        "outcome": {
            "answer_correctness": _ragas_mean(ragas_metrics, "answer_correctness"),
            "answer_correctness_n": _ragas_count(ragas_metrics, "answer_correctness"),
        },
    }

    output_path = config.EVAL_RUNS_DIR / f"{run_slug}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Saved run to {output_path}")

    # Samples are now durably in output_path; the sidecar was only insurance
    # against a mid-run crash, so drop it once the run has actually completed.
    partial_path.unlink(missing_ok=True)

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
