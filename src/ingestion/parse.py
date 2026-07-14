from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import DoclingDocument

from src import config

DOCLING_JSON_PATH = config.INTERIM_DIR / "ifc-annual-report-2024-financials.docling.json"


def parse_pdf() -> DoclingDocument:
    # Picture images are off by default and Phase 5.2 needs the pixels for
    # captioning; scale 2 keeps small chart labels legible for the vision model.
    pipeline_options = PdfPipelineOptions(generate_picture_images=True, images_scale=2.0)
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    result = converter.convert(config.PDF_PATH)
    document = result.document

    DOCLING_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    document.save_as_json(DOCLING_JSON_PATH)
    return document


def load_or_parse_pdf() -> DoclingDocument:
    if DOCLING_JSON_PATH.exists():
        return DoclingDocument.load_from_json(DOCLING_JSON_PATH)
    return parse_pdf()


def _build_page_to_section(document: DoclingDocument) -> dict[int, str]:
    """Map each page number to whichever section header was active on that page."""
    page_to_section: dict[int, str] = {}
    current_section = None

    for item in document.texts:
        if item.label == "section_header":
            current_section = item.text
        if item.prov:
            page_to_section[item.prov[0].page_no] = current_section

    return page_to_section


def extract_table_records(document: DoclingDocument) -> list[dict]:
    """Flatten the parsed document's tables into markdown records with page/section metadata."""
    page_to_section = _build_page_to_section(document)
    records = []

    for table in document.tables:
        page_no = table.prov[0].page_no if table.prov else None
        records.append(
            {
                "text": table.export_to_markdown(document),
                "caption": table.caption_text(document) or None,
                "page": page_no,
                "section": page_to_section.get(page_no),
            }
        )

    return records


def extract_image_records(document: DoclingDocument) -> list[dict]:
    """Flatten the parsed document's pictures into records with page/section metadata.

    "image" holds a PIL image (from the parse run with generate_picture_images=True);
    pictures parsed without pixel data are skipped with a warning rather than
    crashing captioning downstream.
    """
    page_to_section = _build_page_to_section(document)
    records = []

    for picture in document.pictures:
        image = picture.get_image(document)
        if image is None:
            print(f"Skipping picture {picture.self_ref}: no image data in parse")
            continue

        page_no = picture.prov[0].page_no if picture.prov else None
        records.append(
            {
                "image": image,
                "caption": picture.caption_text(document) or None,
                "page": page_no,
                "section": page_to_section.get(page_no),
            }
        )

    return records


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
    print(
        f"Parsed {doc.num_pages()} pages, {len(doc.texts)} text items, "
        f"{len(doc.tables)} tables, {len(doc.pictures)} pictures, "
        f"saved to {DOCLING_JSON_PATH}"
    )
