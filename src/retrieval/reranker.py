from functools import lru_cache

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=None)
def _load_model() -> CrossEncoder:
    return CrossEncoder(CROSS_ENCODER_MODEL)


def _scoring_text(doc: Document) -> str:
    """Cross-encoders are trained on natural-language passages, so raw pipe-table
    markdown scores poorly however relevant the table is. Table chunks carry a
    heading + summary preamble (see ingestion/enrich.py) - score on that instead,
    keeping the full content untouched for generation.
    """
    if doc.metadata.get("content_type") != "table":
        return doc.page_content
    preamble = "\n".join(
        line for line in doc.page_content.splitlines() if not line.strip().startswith("|")
    ).strip()
    return preamble or doc.page_content


def rerank(query: str, docs: list[Document], top_n: int) -> list[Document]:
    if not docs:
        return docs

    pairs = [(query, _scoring_text(doc)) for doc in docs]
    scores = _load_model().predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
    return [doc for doc, _ in ranked[:top_n]]
