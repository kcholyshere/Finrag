from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src import config
from src.embedding.embedder import GeminiEmbeddings

INDEX_NAME = "ifc_annual_report_2024"


def build_index(chunks: list[Document]) -> FAISS:
    return FAISS.from_documents(chunks, GeminiEmbeddings())


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
