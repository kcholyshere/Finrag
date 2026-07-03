"""Ingestion pipeline entrypoint: parse the PDF, chunk it, embed the chunks,
and persist the FAISS index. Run with `python -m src.dataset`.
"""

from src.ingestion.chunk import chunk_sections, group_into_sections, save_chunks
from src.ingestion.parse import extract_text_records, load_or_parse_pdf
from src.retrieval.faiss_store import build_index, save_index


def run() -> None:
    document = load_or_parse_pdf()
    print(f"Parsed {document.num_pages()} pages, {len(document.texts)} text items")

    records = extract_text_records(document)
    sections = group_into_sections(records)
    chunks = chunk_sections(sections)
    save_chunks(chunks)
    print(f"{len(sections)} sections -> {len(chunks)} chunks")

    index = build_index(chunks)
    save_index(index)
    print("FAISS index built and saved")


if __name__ == "__main__":
    run()
