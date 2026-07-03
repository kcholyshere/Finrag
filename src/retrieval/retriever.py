from functools import lru_cache
from typing import Literal

from langchain_core.documents import Document
from langfuse import observe

from src.retrieval import faiss_store, qdrant_store

Backend = Literal["faiss", "qdrant"]


@lru_cache(maxsize=None)
def _load_index(backend: Backend):
    if backend == "faiss":
        return faiss_store.load_index()
    if backend == "qdrant":
        return qdrant_store.load_index()
    raise ValueError(f"Unknown backend: {backend}")


@observe(as_type="retriever")
def retrieve(query: str, backend: Backend = "faiss", k: int = 4) -> list[Document]:
    index = _load_index(backend)
    return index.similarity_search(query, k=k)
