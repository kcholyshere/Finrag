import numpy as np
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from src import config
from src.embedding.embedder import GeminiEmbeddings


def get_client() -> QdrantClient:
    return QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)


def build_index(chunks: list[Document]) -> QdrantVectorStore:
    return QdrantVectorStore.from_documents(
        chunks,
        GeminiEmbeddings(),
        host=config.QDRANT_HOST,
        port=config.QDRANT_PORT,
        collection_name=config.QDRANT_COLLECTION,
        force_recreate=True,
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
