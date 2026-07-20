import re
from functools import lru_cache
from typing import Literal, Optional

from langchain_classic.retrievers.ensemble import EnsembleRetriever

# langchain_community is sunset upstream, but langchain_classic.retrievers.BM25Retriever
# is itself just a deprecated lazy re-export of this same class (confirmed against the
# installed package) - moving to it would add a deprecation warning without removing the
# dependency. Pinned deliberately until BM25Retriever gets a real non-community home.
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langfuse import observe
from qdrant_client import models as qdrant_models
from qdrant_client.http.exceptions import ApiException

from src import config
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


def _bm25_tokenise(text: str) -> list[str]:
    """Lowercase word tokens. BM25Retriever's default is a bare str.split(), so
    "Table:" never matched a query's "table" and "2:" never matched "2" - case
    and adjacent punctuation silently broke lexical matching across the corpus.
    """
    return re.findall(r"\w+", text.lower())


@lru_cache(maxsize=None)
def _load_bm25_retriever() -> BM25Retriever:
    return BM25Retriever.from_documents(load_chunks(), preprocess_func=_bm25_tokenise)


def _bm25_retriever_for(metadata_filter: Optional[MetadataFilter]) -> BM25Retriever:
    """BM25Retriever has no built-in metadata filter, so a filtered request builds
    a fresh index over the matching chunk subset rather than reusing the cached
    full-corpus one - cheap at this corpus size, and keeps the common unfiltered
    path on the cached retriever.
    """
    if metadata_filter is None:
        return _load_bm25_retriever()
    chunks = [
        chunk
        for chunk in load_chunks()
        if all(chunk.metadata.get(key) == value for key, value in metadata_filter.items())
    ]
    return BM25Retriever.from_documents(chunks, preprocess_func=_bm25_tokenise)


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
    metadata_filter: Optional[MetadataFilter] = None,
) -> list[Document]:
    """Combine dense similarity search with BM25 lexical search via reciprocal rank fusion."""
    index = _load_index(backend)
    search_kwargs: dict = {"k": k}
    if metadata_filter is not None:
        if backend == "faiss":
            # Same fetch_k caveat as retrieve(): FAISS filters post-search, so
            # fetch_k must cover the whole index or a match outside the default
            # top-20 window is silently dropped.
            search_kwargs["filter"] = metadata_filter
            search_kwargs["fetch_k"] = index.index.ntotal
        else:
            search_kwargs["filter"] = _to_qdrant_filter(metadata_filter)
    dense_retriever = index.as_retriever(search_kwargs=search_kwargs)

    bm25_retriever = _bm25_retriever_for(metadata_filter)
    bm25_retriever.k = k

    ensemble = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[dense_weight, 1 - dense_weight],
    )
    return ensemble.invoke(query)


@observe(as_type="retriever")
def retrieve_colpali(query: str, k: int = 4) -> list[Document]:
    """Rank whole pages by ColQwen2 late interaction (Phase 6 pipeline).

    Qdrant scores the query's token matrix against each page's patch matrix
    with its native MAX_SIM comparator server-side - no reranker or hybrid
    layering on top, late interaction is the ranking mechanism. Documents
    carry the page image path in metadata; the pixels, not page_content, are
    what generation consumes downstream.
    """
    # Heavy torch/VLM stack - imported here so the text pipeline (Streamlit
    # app, eval harness) can import this module without loading it.
    from src.embedding.colpali_embedder import embed_query

    try:
        result = qdrant_store.get_client().query_points(
            collection_name=config.QDRANT_COLPALI_COLLECTION,
            query=embed_query(query).tolist(),
            limit=k,
            with_payload=True,
        )
    except ApiException as exc:
        # Covers both a down Qdrant (connection failure wrapped as
        # ResponseHandlingException) and a missing collection (non-2xx response
        # as UnexpectedResponse) - either way the raw httpx/qdrant error is
        # meaningless to a client-facing user, so surface an actionable message.
        raise RuntimeError(
            "Could not reach the ColPali page collection in Qdrant. Check that "
            "Qdrant is running (`docker compose up -d qdrant`) and that the "
            "collection has been built (`python -m src.colpali_dataset`)."
        ) from exc
    return [
        Document(
            page_content=(
                f"Page image: {point.payload['image_path']} "
                f"(printed page {config.display_page(point.payload['page_no'])})"
            ),
            metadata={
                "content_type": "page_image",
                "start_page": point.payload["page_no"],
                "end_page": point.payload["page_no"],
                "image_path": point.payload["image_path"],
                "source_pdf": point.payload["source_pdf"],
                "score": point.score,
            },
        )
        for point in result.points
    ]


def _matches_filter(chunk: Document, metadata_filter: Optional[MetadataFilter]) -> bool:
    if metadata_filter is None:
        return True
    return all(chunk.metadata.get(key) == value for key, value in metadata_filter.items())


_TABLE_REFERENCE_PATTERN = re.compile(r"\btable\s+(\d+)\b", re.IGNORECASE)


def _structural_table_candidates(
    query: str, metadata_filter: Optional[MetadataFilter] = None
) -> list[Document]:
    """Chunks whose section names a table number the query references directly.

    "Fetch table 2 data" is a lookup by document structure, not by content -
    bag-of-words and dense similarity rank the actual Table 2 chunk 50th-80th
    because "table" and single digits are near-stopwords in this corpus. A
    direct section-title match puts the chunk in front of the cross-encoder,
    which then ranks it correctly.
    """
    numbers = _TABLE_REFERENCE_PATTERN.findall(query)
    if not numbers:
        return []
    prefixes = tuple(f"table {n}:" for n in numbers)
    return [
        chunk
        for chunk in load_chunks()
        if str(chunk.metadata.get("section", "")).lower().startswith(prefixes)
        and _matches_filter(chunk, metadata_filter)
    ]


_PAGE_REFERENCE_PATTERN = re.compile(r"\bpage\s+(\d+)\b", re.IGNORECASE)


def _structural_page_candidates(
    query: str, metadata_filter: Optional[MetadataFilter] = None
) -> list[Document]:
    """Chunks covering a printed page number the query references directly.

    "Get the data from page 58" is a lookup by document structure, and page
    numbers appear nowhere in the embedded text - similarity search cannot
    target a page. The report's printed page numbers also run one behind
    Docling's raw page index kept in chunk metadata (the cover page is
    unnumbered - see config.PDF_PAGE_NUMBER_OFFSET), so the printed reference
    is translated to raw numbering before matching against start/end_page.
    """
    numbers = _PAGE_REFERENCE_PATTERN.findall(query)
    if not numbers:
        return []
    raw_pages = {int(n) + config.PDF_PAGE_NUMBER_OFFSET for n in numbers}
    return [
        chunk
        for chunk in load_chunks()
        if chunk.metadata.get("start_page") is not None
        and any(
            chunk.metadata["start_page"] <= page <= chunk.metadata["end_page"]
            for page in raw_pages
        )
        and _matches_filter(chunk, metadata_filter)
    ]


@observe(as_type="retriever")
def retrieve_reranked(
    query: str,
    backend: Backend = "faiss",
    k: int = 4,
    candidate_k: int = 10,
    dense_weight: float = 0.5,
    metadata_filter: Optional[MetadataFilter] = None,
) -> list[Document]:
    """Cast a wider hybrid net, then use a cross-encoder to re-rank down to top-k."""
    # Page references are handled before anything else: unlike table candidates
    # (whose "Table N:" heading gives the cross-encoder a lexical anchor), page
    # chunks contain no page tokens, so thrown into the global pool they lose to
    # unrelated matches. Reranking them among themselves keeps the answer on the
    # requested page, ordered by relevance to the rest of the query; the normal
    # pipeline tops up only when the page yields fewer than k chunks.
    page_results = reranker.rerank(
        query, _structural_page_candidates(query, metadata_filter), top_n=k
    )
    if len(page_results) >= k:
        return page_results

    candidates = retrieve_hybrid(
        query,
        backend=backend,
        k=candidate_k,
        dense_weight=dense_weight,
        metadata_filter=metadata_filter,
    )
    seen = {candidate.page_content for candidate in candidates}
    candidates += [
        extra
        for extra in _structural_table_candidates(query, metadata_filter)
        if extra.page_content not in seen
    ]
    seen_pages = {doc.page_content for doc in page_results}
    pool = [c for c in candidates if c.page_content not in seen_pages]
    return page_results + reranker.rerank(query, pool, top_n=k - len(page_results))
