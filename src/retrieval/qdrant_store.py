import uuid

import numpy as np
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from src import config
from src.embedding.embedder import GeminiEmbeddings

# Must match langchain_qdrant.QdrantVectorStore's own defaults exactly (checked
# against the installed package source): the unnamed default vector, and these
# two payload keys. retriever.py still queries this collection through
# QdrantVectorStore, so build_index has to lay points out exactly as that class
# would, even though it no longer calls into it to do so.
VECTOR_NAME = ""
CONTENT_PAYLOAD_KEY = "page_content"
METADATA_PAYLOAD_KEY = "metadata"

BATCH_SIZE = 64


def get_client() -> QdrantClient:
    return QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)


def build_index(chunks: list[Document], vectors: list[list[float]]) -> None:
    """Recreate the text collection from chunks and their precomputed vectors.

    Building via the raw client instead of QdrantVectorStore.from_documents
    avoids re-embedding every chunk a second time (A16) - dataset.py computes
    the embedding matrix once and passes it to both this and the FAISS build.
    Batched upserts mirror build_colpali_index below; a point count check
    afterwards (A17-lite) turns a silently incomplete collection - e.g. from a
    build interrupted partway through - into a loud failure instead of a
    collection that retrieval happily queries anyway.
    """
    if len(vectors) != len(chunks):
        raise ValueError(f"Got {len(vectors)} vectors for {len(chunks)} chunks")

    client = get_client()
    if client.collection_exists(config.QDRANT_COLLECTION):
        client.delete_collection(config.QDRANT_COLLECTION)
    client.create_collection(
        collection_name=config.QDRANT_COLLECTION,
        vectors_config={
            VECTOR_NAME: models.VectorParams(
                size=len(vectors[0]), distance=models.Distance.COSINE
            )
        },
    )

    points = [
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector={VECTOR_NAME: vector},
            payload={
                CONTENT_PAYLOAD_KEY: chunk.page_content,
                METADATA_PAYLOAD_KEY: chunk.metadata,
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    for start in range(0, len(points), BATCH_SIZE):
        client.upsert(
            collection_name=config.QDRANT_COLLECTION, points=points[start : start + BATCH_SIZE]
        )

    count = client.count(config.QDRANT_COLLECTION, exact=True).count
    if count != len(chunks):
        raise RuntimeError(
            f"Qdrant collection {config.QDRANT_COLLECTION!r} has {count} points "
            f"after build, expected {len(chunks)} - build was likely interrupted"
        )


def load_index() -> QdrantVectorStore:
    return QdrantVectorStore(
        client=get_client(),
        collection_name=config.QDRANT_COLLECTION,
        embedding=GeminiEmbeddings(),
    )


def build_colpali_index(pages: list[tuple[int, np.ndarray, str]]) -> None:
    """Recreate the Phase 6 multivector collection from (page_no, patch matrix,
    image path) triples.

    One point per page; each point's "vector" is the whole n_patches x 128
    matrix, scored against query token matrices with Qdrant's native MAX_SIM
    comparator - no LangChain wrapper exists for multivectors, hence the raw
    client here unlike the text collection above. image_path is stored relative
    to the project root so generation/UI can load the PNG regardless of where
    the process runs from.
    """
    client = get_client()
    client.delete_collection(config.QDRANT_COLPALI_COLLECTION)
    client.create_collection(
        collection_name=config.QDRANT_COLPALI_COLLECTION,
        vectors_config=models.VectorParams(
            size=config.COLPALI_EMBEDDING_DIMENSIONS,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
        ),
    )
    points = [
        models.PointStruct(
            id=page_no,
            vector=matrix.astype(np.float32).tolist(),
            payload={
                "page_no": page_no,
                "image_path": image_path,
                "source_pdf": config.PDF_PATH.name,
            },
        )
        for page_no, matrix, image_path in pages
    ]
    # Upsert in small batches: one page is ~1.4MB as JSON (747 patches x 128
    # floats), and Qdrant's REST payload limit is 32MB - all 147 pages in one
    # call is a 200MB request that gets rejected outright.
    batch_size = 8
    for start in range(0, len(points), batch_size):
        client.upsert(
            collection_name=config.QDRANT_COLPALI_COLLECTION,
            points=points[start : start + batch_size],
        )

    # A17-lite: catch a build interrupted partway through before it becomes a
    # silently incomplete collection that retrieval happily queries anyway.
    count = client.count(config.QDRANT_COLPALI_COLLECTION, exact=True).count
    if count != len(points):
        raise RuntimeError(
            f"Qdrant collection {config.QDRANT_COLPALI_COLLECTION!r} has {count} points "
            f"after build, expected {len(points)} - build was likely interrupted"
        )
