"""Phase 2 evaluation entrypoint: runs the RAG pipeline over the eval dataset and
computes both per-step diagnostics and the end-to-end outcome metric, then saves a
settings-tagged JSON so later phases can be compared against this baseline.

Run with `python -m src.evaluation.run_eval [--backend faiss|qdrant] [--k 4] [--n N]`.
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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd

from src import config
from src.evaluation import diagnostics
from src.evaluation.diagnostics import EvalSample
from src.generation.answer import generate_answer
from src.ingestion.chunk import load_chunks
from src.retrieval.retriever import Backend, retrieve

MAX_WORKERS = 6


def _build_sample(row: pd.Series, backend: Backend, k: int) -> EvalSample:
    question = row["Question"]
    retrieved_docs = retrieve(question, backend=backend, k=k)
    generated_answer = generate_answer(question, retrieved_docs)
    return EvalSample(
        question=question,
        reference_answer=row["Ground_Truth_Answer"],
        reference_context=row["Ground_Truth_Context"],
        page_numbers=diagnostics.parse_page_numbers(row["Page_Number"]),
        content_type=row["Context_Content_Type"],
        retrieved_docs=retrieved_docs,
        generated_answer=generated_answer,
    )


def run_pipeline(eval_df: pd.DataFrame, backend: Backend, k: int) -> list[EvalSample]:
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(_build_sample, row, backend, k) for _, row in eval_df.iterrows()]
        return [f.result() for f in futures]


def run(backend: Backend = "faiss", k: int = 4, n: int | None = None) -> dict:
    eval_df = pd.read_csv(config.EVAL_DATASET_PATH)
    if n is not None:
        eval_df = eval_df.sample(n=n, random_state=0)

    print(f"Running retrieval + generation for {len(eval_df)} questions (backend={backend}, k={k})...")
    samples = run_pipeline(eval_df, backend, k)

    print("Checking parsing/chunking coverage...")
    chunks = load_chunks()
    coverage = diagnostics.parse_chunk_coverage(eval_df, chunks)

    print("Computing retrieval rank metrics (Hit Rate@k / MRR)...")
    rank_metrics = diagnostics.hit_rate_and_mrr(samples, k)

    print("Computing RAGAS metrics (context_precision, context_recall, faithfulness, "
          "answer_relevancy, answer_correctness)...")
    ragas_metrics = diagnostics.run_ragas_metrics(samples)

    result = {
        "settings": {
            "backend": backend,
            "k": k,
            "n_samples": len(samples),
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "gemini_model": config.GEMINI_MODEL,
            "embedding_model": config.EMBEDDING_MODEL,
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
    output_path = config.EVAL_RUNS_DIR / f"{timestamp_slug}_{backend}_k{k}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Saved run to {output_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["faiss", "qdrant"], default="faiss")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--n", type=int, default=None, help="Sample size for a quick smoke test")
    args = parser.parse_args()

    run(backend=args.backend, k=args.k, n=args.n)
