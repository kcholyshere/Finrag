import json
from functools import lru_cache

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config
from src.ingestion.enrich import INFORMATIVE_KINDS

NOISE_LABELS = {"page_footer", "page_header"}

CHUNKS_PATH = config.PROCESSED_DIR / "chunks.jsonl"


def group_into_sections(records: list[dict]) -> list[dict]:
    """Merge consecutive records sharing a section into one text block per section."""
    sections: list[dict] = []
    current = None

    for record in records:
        if record["label"] in NOISE_LABELS:
            continue

        section = record["section"]
        if current is None or current["section"] != section:
            current = {"section": section, "start_page": record["page"], "text": record["text"]}
            sections.append(current)
        else:
            current["text"] += "\n" + record["text"]
            # A record with no page (e.g. an item Docling couldn't place) must
            # never overwrite an already-known end_page with None.
            if record["page"] is not None:
                current["end_page"] = record["page"]

    for section in sections:
        section.setdefault("end_page", section["start_page"])

    return sections


def chunk_sections(sections: list[dict]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    texts = [s["text"] for s in sections]
    metadatas = [
        {
            "section": s["section"],
            "start_page": s["start_page"],
            "end_page": s["end_page"],
            "content_type": "text",
        }
        for s in sections
    ]
    return splitter.create_documents(texts, metadatas=metadatas)


def _table_page_content(record: dict) -> str:
    """Prefix the raw table markdown with question-shaped text for retrieval.

    Bare markdown tables match queries poorly in both dense and BM25 search, so
    the caption/section header and LLM summary (see enrich.py) go into the
    embedded text itself - mirroring how text chunks carry their section header
    inline. The raw table stays below for generation; the UI splits the two
    apart again by filtering on pipe-prefixed lines.
    """
    heading = record.get("caption") or record.get("section")
    parts = [f"Table: {heading}" if heading else "Table"]
    if record.get("summary"):
        parts.append(record["summary"])
    return "\n".join(parts) + "\n\n" + record["text"]


def chunk_tables(table_records: list[dict]) -> list[Document]:
    """One chunk per table - a table is already a coherent unit, no recursive splitting."""
    return [
        Document(
            page_content=_table_page_content(record),
            metadata={
                "section": record["section"],
                "start_page": record["page"],
                "end_page": record["page"],
                "content_type": "table",
            },
        )
        for record in table_records
    ]


def _image_page_content(record: dict) -> str:
    """The Gemini description is the whole retrievable/generatable text for an
    image - the pixels never reach the answer model - so it goes in verbatim,
    headed by the report's own caption where one exists."""
    heading = record.get("caption") or record.get("section")
    header = f"Figure: {heading}" if heading else "Figure"
    return f"{header}\n{record['description']}"


def chunk_images(image_records: list[dict]) -> list[Document]:
    """One chunk per informative image; logos, signatures, and decorations are
    captioned upstream but dropped here (mirrors NOISE_LABELS for text)."""
    return [
        Document(
            page_content=_image_page_content(record),
            metadata={
                "section": record["section"],
                "start_page": record["page"],
                "end_page": record["page"],
                "content_type": "image",
            },
        )
        for record in image_records
        if record["kind"] in INFORMATIVE_KINDS
    ]


def save_chunks(chunks: list[Document]) -> None:
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_PATH, "w") as f:
        for i, chunk in enumerate(chunks):
            record = {"id": i, "text": chunk.page_content, **chunk.metadata}
            f.write(json.dumps(record) + "\n")


@lru_cache(maxsize=1)
def load_chunks() -> list[Document]:
    # Cached because retrieve_reranked's structural-candidate helpers
    # (retriever.py) call this up to twice per query, each re-parsing the
    # full JSONL from disk. Safe within one process: nothing writes
    # chunks.jsonl and then calls load_chunks() expecting fresh data in the
    # same run (dataset.py holds chunks in memory across a rebuild instead of
    # re-reading them back).
    chunks = []
    with open(CHUNKS_PATH) as f:
        for line in f:
            record = json.loads(line)
            text = record.pop("text")
            record.pop("id")
            chunks.append(Document(page_content=text, metadata=record))
    return chunks


if __name__ == "__main__":
    from src.ingestion.enrich import caption_images, summarise_tables
    from src.ingestion.parse import (
        extract_image_records,
        extract_table_records,
        extract_text_records,
        load_or_parse_pdf,
    )

    doc = load_or_parse_pdf()
    records = extract_text_records(doc)
    sections = group_into_sections(records)
    text_chunks = chunk_sections(sections)

    table_records = summarise_tables(extract_table_records(doc))
    table_chunks = chunk_tables(table_records)

    image_records = caption_images(extract_image_records(doc))
    image_chunks = chunk_images(image_records)

    chunks = text_chunks + table_chunks + image_chunks
    save_chunks(chunks)
    print(
        f"{len(sections)} sections -> {len(text_chunks)} text chunks, "
        f"{len(table_chunks)} table chunks, {len(image_chunks)} image chunks, "
        f"saved to {CHUNKS_PATH}"
    )
