import faiss
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src import config
from src.embedding.embedder import GeminiEmbeddings

INDEX_NAME = "ifc_annual_report_2024"

# HNSW gives FAISS the same O(log n) graph-search profile as Qdrant's default
# index, instead of the flat/brute-force O(n) scan `FAISS.from_documents` builds.
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 128


def build_index(chunks: list[Document], vectors: list[list[float]]) -> FAISS:
    """Build the HNSW index from chunks and their precomputed embedding vectors.

    Vectors are computed once by the caller (dataset.py) and shared with the
    Qdrant build (A16) - embedding all ~852 chunks is the single most
    expensive/costly step in a rebuild, so doing it twice (once per store) was
    pure waste and a drift path between the two indexes.
    """
    if len(vectors) != len(chunks):
        raise ValueError(f"Got {len(vectors)} vectors for {len(chunks)} chunks")
    embeddings = GeminiEmbeddings()  # kept on the store for query-time embed_query

    index = faiss.IndexHNSWFlat(len(vectors[0]), HNSW_M)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH
    index.add(np.array(vectors, dtype="float32"))

    docstore = InMemoryDocstore({str(i): chunk for i, chunk in enumerate(chunks)})
    index_to_docstore_id = {i: str(i) for i in range(len(chunks))}

    return FAISS(embeddings, index, docstore, index_to_docstore_id)


def save_index(index: FAISS) -> None:
    config.FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index.save_local(str(config.FAISS_INDEX_DIR), index_name=INDEX_NAME)


def load_index() -> FAISS:
    return FAISS.load_local(
        str(config.FAISS_INDEX_DIR),
        GeminiEmbeddings(),
        index_name=INDEX_NAME,
        allow_dangerous_deserialization=True,
    )
