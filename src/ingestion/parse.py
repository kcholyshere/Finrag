import hashlib
import json

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import DoclingDocument, PictureItem, TableItem

from src import config

DOCLING_JSON_PATH = config.INTERIM_DIR / "ifc-annual-report-2024-financials.docling.json"
DOCLING_META_PATH = DOCLING_JSON_PATH.with_suffix(".meta.json")

# Options that change parse output; parse_pdf() builds PdfPipelineOptions from
# this same dict so the cache-validity check can never drift from the options
# actually used.
PARSE_OPTIONS = {"generate_picture_images": True, "images_scale": 2.0}


def _cache_meta() -> dict:
    return {"pdf_sha1": hashlib.sha1(config.PDF_PATH.read_bytes()).hexdigest(), **PARSE_OPTIONS}


def _cache_status(meta_path, current: dict) -> str:
    """Returns 'valid', 'missing', or 'stale' against a sidecar meta file -
    existence-only caches (the previous behaviour here) already forced one
    manual cache-bust when images_scale changed (see ADR log), so a
    version+hash check replaces "hope nothing changed" with an explicit
    comparison."""
    if not meta_path.exists():
        return "missing"
    return "valid" if json.loads(meta_path.read_text()) == current else "stale"


def _write_meta(meta_path, current: dict) -> None:
    meta_path.write_text(json.dumps(current, indent=2))


def parse_pdf() -> DoclingDocument:
    # Picture images are off by default and Phase 5.2 needs the pixels for
    # captioning; scale 2 keeps small chart labels legible for the vision model.
    pipeline_options = PdfPipelineOptions(**PARSE_OPTIONS)
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    result = converter.convert(config.PDF_PATH)
    document = result.document

    DOCLING_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    document.save_as_json(DOCLING_JSON_PATH)
    _write_meta(DOCLING_META_PATH, _cache_meta())
    return document


def load_or_parse_pdf() -> DoclingDocument:
    if not DOCLING_JSON_PATH.exists():
        return parse_pdf()

    current = _cache_meta()
    status = _cache_status(DOCLING_META_PATH, current)
    if status == "stale":
        print(
            f"Parse cache at {DOCLING_JSON_PATH} no longer matches the source PDF "
            "or parse options - re-parsing"
        )
        return parse_pdf()
    if status == "missing":
        print(f"No cache metadata found for {DOCLING_JSON_PATH.name}, grandfathering existing cache")
        _write_meta(DOCLING_META_PATH, current)

    return DoclingDocument.load_from_json(DOCLING_JSON_PATH)


def _build_item_to_section(document: DoclingDocument) -> dict[str, str | None]:
    """Map each table/picture's self_ref to the section header governing it.

    Walks the document once via iterate_items(), which interleaves text,
    tables, and pictures in true reading order, tracking the most recent
    section header seen. This replaces the old page-keyed map, which
    overwrote its entry with whichever header appeared last on a page - wrong
    whenever a page holds more than one section (131 of 147 pages here), e.g.
    two tables on the same page ended up sharing one label instead of each
    getting the header that actually precedes it.
    """
    item_to_section: dict[str, str | None] = {}
    current_section = None

    for item, _level in document.iterate_items():
        if str(getattr(item, "label", "")) == "section_header":
            current_section = item.text
        elif isinstance(item, (TableItem, PictureItem)):
            item_to_section[item.self_ref] = current_section

    return item_to_section


def extract_table_records(document: DoclingDocument) -> list[dict]:
    """Flatten the parsed document's tables into markdown records with page/section metadata."""
    item_to_section = _build_item_to_section(document)
    records = []

    for table in document.tables:
        page_no = table.prov[0].page_no if table.prov else None
        records.append(
            {
                "text": table.export_to_markdown(document),
                "caption": table.caption_text(document) or None,
                "page": page_no,
                "section": item_to_section.get(table.self_ref),
            }
        )

    return records


def extract_image_records(document: DoclingDocument) -> list[dict]:
    """Flatten the parsed document's pictures into records with page/section metadata.

    "image" holds a PIL image (from the parse run with generate_picture_images=True);
    pictures parsed without pixel data are skipped with a warning rather than
    crashing captioning downstream.
    """
    item_to_section = _build_item_to_section(document)
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
                "section": item_to_section.get(picture.self_ref),
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
