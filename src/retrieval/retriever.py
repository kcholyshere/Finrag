from functools import lru_cache
from typing import Literal, Optional

from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langfuse import observe
from qdrant_client import models as qdrant_models

from src.ingestion.chunk import load_chunks
from src.retrieval import faiss_store, qdrant_store, reranker

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


@lru_cache(maxsize=None)
def _load_bm25_retriever() -> BM25Retriever:
    return BM25Retriever.from_documents(load_chunks())


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
        # FAISS's filter is a post-filter over the top fetch_k nearest neighbours
        # (default 20), not a true pre-filter - fetch_k must cover the whole index
        # or a filtered match outside that window is silently dropped.
        return index.similarity_search(
            query, k=k, filter=metadata_filter, fetch_k=index.index.ntotal
        )
    return index.similarity_search(query, k=k, filter=_to_qdrant_filter(metadata_filter))


@observe(as_type="retriever")
def retrieve_hybrid(
    query: str,
    backend: Backend = "faiss",
    k: int = 4,
    dense_weight: float = 0.5,
) -> list[Document]:
    """Combine dense similarity search with BM25 lexical search via reciprocal rank fusion."""
    dense_retriever = _load_index(backend).as_retriever(search_kwargs={"k": k})

    bm25_retriever = _load_bm25_retriever()
    bm25_retriever.k = k

    ensemble = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[dense_weight, 1 - dense_weight],
    )
    return ensemble.invoke(query)


@observe(as_type="retriever")
def retrieve_reranked(
    query: str,
    backend: Backend = "faiss",
    k: int = 4,
    candidate_k: int = 10,
    dense_weight: float = 0.5,
) -> list[Document]:
    """Cast a wider hybrid net, then use a cross-encoder to re-rank down to top-k."""
    candidates = retrieve_hybrid(query, backend=backend, k=candidate_k, dense_weight=dense_weight)
    return reranker.rerank(query, candidates, top_n=k)
