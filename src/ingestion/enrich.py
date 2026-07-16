"""LLM enrichment for table and image records - generates natural-language text
per item so retrieval can match them against question-shaped queries.

Why: raw markdown tables embed poorly (a wall of numbers dilutes the dense
vector) and score badly in BM25 (length normalisation penalises long tables),
so questions like "What was Net Income for FY24?" never surfaced the right
table. A summary naming the table's subject, line items, and periods gives
both retrievers something question-shaped to match, without duplicating the
figures themselves (those stay in the table body for generation).

Summaries are cached in data/interim/table_summaries.json keyed by a hash of
the table markdown plus the caption/section context injected into the prompt,
so re-running ingestion only calls Gemini for new or changed tables (and a
changed section attribution correctly counts as changed).
"""

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Literal

from google.genai import types
from pydantic import BaseModel

from src import config
from src.services.genai_client import get_client

SUMMARIES_PATH = config.INTERIM_DIR / "table_summaries.json"
CAPTIONS_PATH = config.INTERIM_DIR / "image_captions.json"


def _context(caption: str | None, section: str | None) -> str | None:
    """Shared caption/section fallback, used both to build the prompt and to
    derive the cache key, so the two can never drift apart (A7)."""
    return caption or section


SUMMARY_PROMPT = (
    "You are indexing tables from the IFC Annual Report 2024 (Financials) for a "
    "search engine. Write a 2-4 sentence plain-text description of the table "
    "below. State what the table reports, name the key line items, measures, or "
    "entities that appear in its rows, and the fiscal years or periods covered. "
    "Do not repeat any figures. Output only the description, no preamble.\n\n"
)


def _table_key(markdown: str, context: str | None) -> str:
    # Context (caption/section) is folded into the key because it is injected
    # into the prompt below: if attribution changes (e.g. the A1 fix) the old
    # key would otherwise keep serving a summary generated under the wrong
    # context instead of being treated as a miss.
    return hashlib.sha1(f"{markdown}\x00{context or ''}".encode()).hexdigest()


def _load_cache() -> dict[str, str]:
    if SUMMARIES_PATH.exists():
        return json.loads(SUMMARIES_PATH.read_text())
    return {}


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write-temp-then-replace so an interrupt mid-write never corrupts the
    cache (A15) - same pattern as page_images.py's page render cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp.json")
    tmp_path.write_text(json.dumps(data, indent=2))
    os.replace(tmp_path, path)


def _save_cache(cache: dict[str, str]) -> None:
    _atomic_write_json(SUMMARIES_PATH, cache)


def _summarise(markdown: str, caption: str | None, section: str | None) -> str:
    context = _context(caption, section)
    prompt = SUMMARY_PROMPT
    if context:
        prompt += f"Table caption/section: {context}\n\n"
    prompt += markdown

    # Degenerate tables (e.g. a lone header row) can make the model return no
    # text part at all; a missing summary just means that table is indexed
    # without enrichment, which is not worth failing the whole pipeline over.
    for _ in range(2):
        response = get_client().models.generate_content(model=config.GEMINI_MODEL, contents=prompt)
        if response.text:
            return response.text.strip()
    print(f"  No summary returned for table (caption/section: {context!r}), indexing without one")
    return ""


def summarise_tables(table_records: list[dict]) -> list[dict]:
    """Add a "summary" field to each table record, calling Gemini only on cache misses."""
    cache = _load_cache()
    keys = [
        _table_key(r["text"], _context(r.get("caption"), r.get("section"))) for r in table_records
    ]
    misses = [(r, k) for r, k in zip(table_records, keys) if k not in cache]
    if misses:
        print(f"Summarising {len(misses)} of {len(table_records)} tables via Gemini...")

    for i, (record, key) in enumerate(misses, start=1):
        cache[key] = _summarise(record["text"], record.get("caption"), record.get("section"))
        _save_cache(cache)
        if i % 10 == 0 or i == len(misses):
            print(f"  {i}/{len(misses)}")

    for record, key in zip(table_records, keys):
        record["summary"] = cache[key]
    return table_records


# --- Image captioning (Phase 5.2) ---------------------------------------------
#
# Unlike table summaries, image descriptions must include the key figures and
# trends: a table chunk carries its raw markdown below the summary, but for an
# image the description is the only text the retriever and generator ever see.
# Gemini also classifies each picture so logos/signatures (16 of the report's
# 36 pictures) are captioned once but never indexed.

CAPTION_PROMPT = (
    "You are indexing figures from the IFC Annual Report 2024 (Financials) for "
    "a search engine. Classify the image, then describe it.\n"
    "- kind: 'chart' for bar/line/pie/area charts and graphs, 'diagram' for "
    "flowcharts, org charts, or schematic figures, 'logo', 'signature', or "
    "'decorative' for anything with no data content.\n"
    "- description: for a chart or diagram, 3-6 sentences stating what it shows, "
    "the axes or categories, the series and periods covered, and the key values "
    "and trends visible (include the actual numbers - this text substitutes for "
    "the image). For logos, signatures, or decorative images, one short sentence."
)

INFORMATIVE_KINDS = {"chart", "diagram"}

# Returned for an image Gemini never manages to classify (A14): "decorative"
# with an empty description drops it from indexing via INFORMATIVE_KINDS
# rather than crashing the whole captioning batch.
FALLBACK_CAPTION = {"kind": "decorative", "description": ""}


class ImageCaption(BaseModel):
    kind: Literal["chart", "diagram", "logo", "signature", "decorative"]
    description: str


def _image_png_bytes(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _image_key(png_bytes: bytes, context: str | None) -> str:
    # See _table_key: context is folded in so changed attribution invalidates
    # the cached caption instead of silently serving a stale one (A7).
    return hashlib.sha1(png_bytes + b"\x00" + (context or "").encode()).hexdigest()


def _caption(png_bytes: bytes, caption: str | None, section: str | None) -> dict:
    context = _context(caption, section)
    prompt = CAPTION_PROMPT
    if context:
        prompt += f"\n\nFigure caption/section: {context}"

    # Mirrors _summarise's retry/fallback: a None response.parsed (bad or
    # blocked generation) used to raise AttributeError and kill the whole
    # captioning batch instead of just this one image.
    for _ in range(2):
        response = get_client().models.generate_content(
            model=config.GEMINI_MODEL,
            contents=[prompt, types.Part.from_bytes(data=png_bytes, mime_type="image/png")],
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_schema=ImageCaption
            ),
        )
        if response.parsed is not None:
            return response.parsed.model_dump()
    print(f"  No caption returned for image (caption/section: {context!r}), marking decorative")
    return dict(FALLBACK_CAPTION)


def caption_images(image_records: list[dict]) -> list[dict]:
    """Add "kind" and "description" fields to each image record, calling Gemini
    only on cache misses (keyed by a hash of the PNG bytes plus context)."""
    cache = json.loads(CAPTIONS_PATH.read_text()) if CAPTIONS_PATH.exists() else {}

    for record in image_records:
        context = _context(record.get("caption"), record.get("section"))
        record["_key"] = _image_key(_image_png_bytes(record["image"]), context)

    misses = [r for r in image_records if r["_key"] not in cache]
    if misses:
        print(f"Captioning {len(misses)} of {len(image_records)} images via Gemini...")

    for i, record in enumerate(misses, start=1):
        cache[record["_key"]] = _caption(
            _image_png_bytes(record["image"]), record.get("caption"), record.get("section")
        )
        _atomic_write_json(CAPTIONS_PATH, cache)
        if i % 10 == 0 or i == len(misses):
            print(f"  {i}/{len(misses)}")

    for record in image_records:
        record.update(cache[record.pop("_key")])
    return image_records
