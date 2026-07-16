"""Per-step diagnostics for the RAG pipeline: parsing/chunking coverage, retrieval
rank metrics, and RAGAS metrics scoped to the retrieval and generation steps.

These answer "how do we check the quality of each step" as distinct from
`outcome.py`, which answers "how good is the whole solution end to end".
"""

from dataclasses import dataclass, field

import pandas as pd
from langchain_core.documents import Document
from rapidfuzz import fuzz

from src.evaluation import ragas_compat

FUZZY_MATCH_THRESHOLD = 80


@dataclass
class EvalSample:
    question: str
    reference_answer: str
    reference_context: str
    page_numbers: set[int]
    content_type: str
    retrieved_docs: list[Document] = field(default_factory=list)
    generated_answer: str = ""


def parse_page_numbers(raw: str) -> set[int]:
    return {int(part.strip()) for part in str(raw).split(";") if part.strip().isdigit()}


def parse_chunk_coverage(eval_df: pd.DataFrame, chunks: list[Document]) -> dict:
    """Does the parsed/chunked corpus actually contain each ground-truth context snippet?

    Scoped to curated, text-only rows: synthetic rows were generated *from* these same
    chunks, so they'd trivially score 100% and tell us nothing; table/image rows can't
    match since Phase 1 only chunks text.
    """
    candidates = eval_df[
        (eval_df["Source"] == "curated") & eval_df["Context_Content_Type"].str.startswith("text")
    ]

    misses = []
    scores = []
    for _, row in candidates.iterrows():
        context = str(row["Ground_Truth_Context"])
        best = max((fuzz.partial_ratio(context, c.page_content) for c in chunks), default=0)
        scores.append(best)
        if best < FUZZY_MATCH_THRESHOLD:
            misses.append({"question": row["Question"], "best_match_score": best})

    coverage = sum(1 for s in scores if s >= FUZZY_MATCH_THRESHOLD) / len(scores) if scores else None
    return {
        "rows_checked": len(scores),
        "coverage_rate": coverage,
        "mean_fuzzy_score": sum(scores) / len(scores) if scores else None,
        "misses": misses,
    }


def hit_rate_and_mrr(samples: list[EvalSample], k: int) -> dict:
    """Hit Rate@k / MRR@k using the ground-truth page number as the relevance label -
    the measurable proxy for embedding/retrieval quality, since embeddings themselves
    aren't directly inspectable.
    """

    def _score(subset: list[EvalSample]) -> dict:
        if not subset:
            return {"n": 0, "hit_rate": None, "mrr": None}
        hits = 0
        reciprocal_ranks = []
        for sample in subset:
            ranks = [
                rank
                for rank, doc in enumerate(sample.retrieved_docs[:k], start=1)
                if {doc.metadata.get("start_page"), doc.metadata.get("end_page")} & sample.page_numbers
            ]
            if ranks:
                hits += 1
                reciprocal_ranks.append(1 / ranks[0])
            else:
                reciprocal_ranks.append(0.0)
        return {"n": len(subset), "hit_rate": hits / len(subset), "mrr": sum(reciprocal_ranks) / len(subset)}

    by_content_type = {
        content_type: _score([s for s in samples if s.content_type == content_type])
        for content_type in sorted({s.content_type for s in samples})
    }
    return {"k": k, "overall": _score(samples), "by_content_type": by_content_type}


def to_ragas_dataset(samples: list[EvalSample]):
    from ragas import EvaluationDataset, SingleTurnSample

    return EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=s.question,
                response=s.generated_answer,
                retrieved_contexts=[doc.page_content for doc in s.retrieved_docs],
                reference=s.reference_answer,
            )
            for s in samples
        ]
    )


def run_ragas_metrics(samples: list[EvalSample], context_based: bool = True) -> dict:
    """Runs the RAGAS metrics (retrieval, generation, and outcome) in a single
    `evaluate()` call.

    Must be one call: ragas tears down its internal asyncio event loop after each
    `evaluate()` returns, and the cached ChatVertexAI's grpc.aio channel is bound to
    the loop it was first used on - a second `evaluate()` call in the same process
    silently returns NaN for every row once that loop is closed. (A single call
    with a metric subset is fine - the trap is a second call.)

    - context_precision / context_recall: are the actually-retrieved contexts
      sufficient and precise for supporting the reference answer? (retrieval step)
    - faithfulness / answer_relevancy: is the generated answer grounded in the
      retrieved contexts, and does it address the question asked? (generation step)
    - answer_correctness: does the final answer match the reference answer?
      (end-to-end outcome, and RAGAS's own LLM-graded scoring is the "LLM-as-judge"
      experiment the Phase 2 requirements ask for)

    context_based=False (ColPali pipeline) skips the three context-grounded
    metrics and reports them as None: the model answers from page images, and
    the docs' page_content is only a path placeholder - scoring precision/
    recall/faithfulness against it would measure a string the model never saw.
    Page-based hit rate/MRR covers the retrieval step for that pipeline instead.
    """
    from ragas import evaluate
    from ragas.metrics import AnswerCorrectness, AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    metrics = {"answer_relevancy": AnswerRelevancy(), "answer_correctness": AnswerCorrectness()}
    if context_based:
        metrics = {
            "context_precision": ContextPrecision(),
            "context_recall": ContextRecall(),
            "faithfulness": Faithfulness(),
            **metrics,
        }

    result = evaluate(
        dataset=to_ragas_dataset(samples),
        metrics=list(metrics.values()),
        llm=ragas_compat.get_ragas_llm(),
        embeddings=ragas_compat.get_ragas_embeddings(),
        show_progress=False,
    )
    scores = result.to_pandas()[list(metrics)].mean().to_dict()
    all_columns = ["context_precision", "context_recall", "faithfulness", "answer_relevancy", "answer_correctness"]
    return {column: scores.get(column) for column in all_columns}
