from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

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
