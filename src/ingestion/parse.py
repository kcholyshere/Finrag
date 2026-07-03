from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import DoclingDocument

from src import config

DOCLING_JSON_PATH = config.INTERIM_DIR / "ifc-annual-report-2024-financials.docling.json"


def parse_pdf() -> DoclingDocument:
    converter = DocumentConverter()
    result = converter.convert(config.PDF_PATH)
    document = result.document

    DOCLING_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    document.save_as_json(DOCLING_JSON_PATH)
    return document


def load_or_parse_pdf() -> DoclingDocument:
    if DOCLING_JSON_PATH.exists():
        return DoclingDocument.load_from_json(DOCLING_JSON_PATH)
    return parse_pdf()


def extract_text_records(document: DoclingDocument) -> list[dict]:
    """Flatten the parsed document into text records with page/section metadata.

    Kept unfiltered (headers, footers included) - chunking decides what to drop,
    since "collect all possible metadata" is this module's job, not chunk.py's.
    """
    records = []
    current_section = None

    for item in document.texts:
        if item.label == "section_header":
            current_section = item.text

        page_no = item.prov[0].page_no if item.prov else None
        records.append(
            {
                "text": item.text,
                "label": str(item.label),
                "page": page_no,
                "section": current_section,
            }
        )

    return records


if __name__ == "__main__":
    doc = parse_pdf()
    print(f"Parsed {doc.num_pages()} pages, {len(doc.texts)} text items, saved to {DOCLING_JSON_PATH}")
