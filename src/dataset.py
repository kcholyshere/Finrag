"""Ingestion pipeline entrypoint: parse the PDF, chunk it, embed the chunks,
and persist both the FAISS and Qdrant indexes. Run with `python -m src.dataset`.
Requires the Qdrant service to be up (`docker compose up -d qdrant`).
"""

import os

# Building the FAISS HNSW index segfaults on macOS ("OMP: Error #15") because faiss
# and numpy/torch each bundle their own OpenMP runtime; single-threading it avoids
# the crash (harmless in this single-process, CPU-only pipeline).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from src.embedding.embedder import GeminiEmbeddings
from src.ingestion.chunk import (
    chunk_images,
    chunk_sections,
    chunk_tables,
    group_into_sections,
    save_chunks,
)
from src.ingestion.enrich import caption_images, summarise_tables
from src.ingestion.parse import (
    extract_image_records,
    extract_table_records,
    extract_text_records,
    load_or_parse_pdf,
)
from src.retrieval import faiss_store, qdrant_store


def run() -> None:
    document = load_or_parse_pdf()
    print(
        f"Parsed {document.num_pages()} pages, {len(document.texts)} text items, "
        f"{len(document.tables)} tables, {len(document.pictures)} pictures"
    )

    records = extract_text_records(document)
    sections = group_into_sections(records)
    text_chunks = chunk_sections(sections)

    table_records = summarise_tables(extract_table_records(document))
    table_chunks = chunk_tables(table_records)

    image_records = caption_images(extract_image_records(document))
    image_chunks = chunk_images(image_records)

    chunks = text_chunks + table_chunks + image_chunks
    save_chunks(chunks)
    print(
        f"{len(sections)} sections -> {len(text_chunks)} text chunks, "
        f"{len(table_chunks)} table chunks, {len(image_chunks)} image chunks"
    )

    print(f"Embedding {len(chunks)} chunks...")
    # Computed once and shared between both stores (A16) - embedding is by far
    # the most expensive step here, and QdrantVectorStore.from_documents used
    # to redo it from scratch after FAISS already had.
    vectors = GeminiEmbeddings().embed_documents([chunk.page_content for chunk in chunks])

    faiss_index = faiss_store.build_index(chunks, vectors)
    faiss_store.save_index(faiss_index)
    print("FAISS index built and saved")

    qdrant_store.build_index(chunks, vectors)
    print("Qdrant index built and saved")


if __name__ == "__main__":
    run()
