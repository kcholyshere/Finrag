from functools import lru_cache
from typing import Literal, Optional

from langchain_core.documents import Document
from langfuse import observe
from qdrant_client import models as qdrant_models

from src.retrieval import faiss_store, qdrant_store

Backend = Literal["faiss", "qdrant"]

# Exact-match metadata filter, e.g. {"content_type": "text", "start_page": 12}.
MetadataFilter = dict[str, str | int]


@lru_cache(maxsize=None)
def _load_index(backend: Backend):
    if backend == "faiss":
        return faiss_store.load_index()
    if backend == "qdrant":
        return qdrant_store.load_index()
    raise ValueError(f"Unknown backend: {backend}")


def _to_qdrant_filter(metadata_filter: MetadataFilter) -> qdrant_models.Filter:
    return qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key=f"metadata.{key}", match=qdrant_models.MatchValue(value=value)
            )
            for key, value in metadata_filter.items()
        ]
    )


@observe(as_type="retriever")
def retrieve(
    query: str,
    backend: Backend = "faiss",
    k: int = 4,
    metadata_filter: Optional[MetadataFilter] = None,
) -> list[Document]:
    index = _load_index(backend)
    if metadata_filter is None:
        return index.similarity_search(query, k=k)

    if backend == "faiss":
        return index.similarity_search(query, k=k, filter=metadata_filter)
    return index.similarity_search(query, k=k, filter=_to_qdrant_filter(metadata_filter))
