from functools import lru_cache

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=None)
def _load_model() -> CrossEncoder:
    return CrossEncoder(CROSS_ENCODER_MODEL)


def rerank(query: str, docs: list[Document], top_n: int) -> list[Document]:
    if not docs:
        return docs

    pairs = [(query, doc.page_content) for doc in docs]
    scores = _load_model().predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
    return [doc for doc, _ in ranked[:top_n]]
