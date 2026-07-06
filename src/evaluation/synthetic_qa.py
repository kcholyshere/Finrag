"""Synthetic QA generation for the Phase 2 evaluation dataset.

Expands the 33 hand-curated rows in `references/RAG_evaluation_dataset.csv` up to
`config.EVAL_DATASET_TARGET_SIZE` total rows by generating factoid questions from
the Phase 1 text chunks and filtering them with an LLM critique pass (adapted from
the groundedness/relevance/standalone pattern in `references/rag_evaluation.md`,
ported from Mixtral/GPT-4 to Gemini via Vertex AI). Run with
`python -m src.evaluation.synthetic_qa`.
"""

import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from google.genai import types
from langchain_core.documents import Document

from src import config
from src.ingestion.chunk import load_chunks
from src.services.genai_client import get_client

MIN_CHUNK_CHARS = 200
CRITIQUE_THRESHOLD = 4
MAX_WORKERS = 6

QA_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "question": {"type": "STRING"},
        "answer": {"type": "STRING"},
    },
    "required": ["question", "answer"],
}

CRITIQUE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "groundedness": {"type": "INTEGER"},
        "relevance": {"type": "INTEGER"},
        "standalone": {"type": "INTEGER"},
    },
    "required": ["groundedness", "relevance", "standalone"],
}

QA_GENERATION_PROMPT = """You are writing evaluation questions for a RAG system over IFC's 2024 Annual Report (Financials).

Write one factoid question and its answer, using ONLY the context below. The question must be:
- answerable with a specific, concise fact from the context (a number, name, date, or short statement)
- phrased the way a financial analyst would type it into a search bar
- self-contained: it must NOT refer to "the context", "the passage", "the document", "this table", or similar

Context:
{context}

Respond with the question and its answer."""

CRITIQUE_PROMPT = """Rate the following question/answer pair, generated from a context snippet of IFC's 2024 Annual Report, on three 1-5 scales.

Context:
{context}

Question: {question}
Answer: {answer}

Score each on a 1 (worst) to 5 (best) scale:
- groundedness: can this question be answered unambiguously and correctly from the context alone?
- relevance: how useful would this question be to someone analysing IFC's financial performance?
- standalone: is the question understandable on its own, with no reference to "the context/document/table/passage"?

Respond with the three integer scores."""


def _generate_qa(client, context: str) -> dict | None:
    resp = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=QA_GENERATION_PROMPT.format(context=context),
        config=types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=QA_SCHEMA, temperature=0.7
        ),
    )
    return resp.parsed


def _critique_qa(client, context: str, question: str, answer: str) -> dict | None:
    resp = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=CRITIQUE_PROMPT.format(context=context, question=question, answer=answer),
        config=types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=CRITIQUE_SCHEMA, temperature=0.0
        ),
    )
    return resp.parsed


def _generate_and_critique(chunk: Document) -> dict | None:
    client = get_client()
    context = chunk.page_content

    qa = _generate_qa(client, context)
    if not qa or not qa.get("question") or not qa.get("answer"):
        return None

    critique = _critique_qa(client, context, qa["question"], qa["answer"])
    if not critique:
        return None

    scores = (critique.get("groundedness", 0), critique.get("relevance", 0), critique.get("standalone", 0))
    if min(scores) < CRITIQUE_THRESHOLD:
        return None

    start_page = chunk.metadata.get("start_page")
    end_page = chunk.metadata.get("end_page", start_page)
    page_number = str(start_page) if start_page == end_page else f"{start_page}; {end_page}"

    return {
        "Question": qa["question"],
        "Ground_Truth_Context": context,
        "Ground_Truth_Answer": qa["answer"],
        "Page_Number": page_number,
        "Context_Content_Type": "text",
        "Source": "synthetic",
        "Groundedness_Score": scores[0],
        "Relevance_Score": scores[1],
        "Standalone_Score": scores[2],
    }


def generate_synthetic_rows(target_count: int) -> pd.DataFrame:
    chunks = [c for c in load_chunks() if len(c.page_content) >= MIN_CHUNK_CHARS]
    random.Random(0).shuffle(chunks)

    accepted: list[dict] = []
    pool = iter(chunks)
    exhausted = False

    while len(accepted) < target_count and not exhausted:
        batch = []
        for _ in range(min(40, (target_count - len(accepted)) * 2)):
            chunk = next(pool, None)
            if chunk is None:
                exhausted = True
                break
            batch.append(chunk)

        if not batch:
            break

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(_generate_and_critique, chunk) for chunk in batch]
            for future in as_completed(futures):
                try:
                    row = future.result()
                except Exception as exc:
                    print(f"  skipped a candidate after error: {exc}")
                    continue
                if row is not None:
                    accepted.append(row)

        print(f"  {len(accepted)}/{target_count} synthetic rows accepted so far...")

    return pd.DataFrame(accepted[:target_count])


def build_eval_dataset() -> pd.DataFrame:
    curated = pd.read_csv(config.CURATED_EVAL_PATH)
    curated["Source"] = "curated"
    for col in ("Groundedness_Score", "Relevance_Score", "Standalone_Score"):
        curated[col] = pd.NA

    target_synthetic = max(0, config.EVAL_DATASET_TARGET_SIZE - len(curated))
    print(f"Curated rows: {len(curated)}. Generating up to {target_synthetic} synthetic rows...")
    synthetic = generate_synthetic_rows(target_synthetic)
    print(f"Synthetic rows generated: {len(synthetic)}")

    combined = pd.concat([curated, synthetic], ignore_index=True)
    return combined


def run() -> None:
    combined = build_eval_dataset()
    config.EVAL_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(config.EVAL_DATASET_PATH, index=False)
    print(f"Wrote {len(combined)} rows to {config.EVAL_DATASET_PATH}")


if __name__ == "__main__":
    run()
