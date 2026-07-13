import json

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config

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


def save_chunks(chunks: list[Document]) -> None:
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_PATH, "w") as f:
        for i, chunk in enumerate(chunks):
            record = {"id": i, "text": chunk.page_content, **chunk.metadata}
            f.write(json.dumps(record) + "\n")


def load_chunks() -> list[Document]:
    chunks = []
    with open(CHUNKS_PATH) as f:
        for line in f:
            record = json.loads(line)
            text = record.pop("text")
            record.pop("id")
            chunks.append(Document(page_content=text, metadata=record))
    return chunks


if __name__ == "__main__":
    from src.ingestion.enrich import summarise_tables
    from src.ingestion.parse import extract_table_records, extract_text_records, load_or_parse_pdf

    doc = load_or_parse_pdf()
    records = extract_text_records(doc)
    sections = group_into_sections(records)
    text_chunks = chunk_sections(sections)

    table_records = summarise_tables(extract_table_records(doc))
    table_chunks = chunk_tables(table_records)

    chunks = text_chunks + table_chunks
    save_chunks(chunks)
    print(
        f"{len(sections)} sections -> {len(text_chunks)} text chunks, "
        f"{len(table_chunks)} table chunks, saved to {CHUNKS_PATH}"
    )
