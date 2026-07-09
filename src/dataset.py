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

from src.ingestion.chunk import chunk_sections, group_into_sections, save_chunks
from src.ingestion.parse import extract_text_records, load_or_parse_pdf
from src.retrieval import faiss_store, qdrant_store


def run() -> None:
    document = load_or_parse_pdf()
    print(f"Parsed {document.num_pages()} pages, {len(document.texts)} text items")

    records = extract_text_records(document)
    sections = group_into_sections(records)
    chunks = chunk_sections(sections)
    save_chunks(chunks)
    print(f"{len(sections)} sections -> {len(chunks)} chunks")

    faiss_index = faiss_store.build_index(chunks)
    faiss_store.save_index(faiss_index)
    print("FAISS index built and saved")

    qdrant_store.build_index(chunks)
    print("Qdrant index built and saved")


if __name__ == "__main__":
    run()
