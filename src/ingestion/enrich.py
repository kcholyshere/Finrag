"""LLM enrichment for table records - generates a short natural-language summary
per table so retrieval can match tables against question-shaped queries.

Why: raw markdown tables embed poorly (a wall of numbers dilutes the dense
vector) and score badly in BM25 (length normalisation penalises long tables),
so questions like "What was Net Income for FY24?" never surfaced the right
table. A summary naming the table's subject, line items, and periods gives
both retrievers something question-shaped to match, without duplicating the
figures themselves (those stay in the table body for generation).

Summaries are cached in data/interim/table_summaries.json keyed by a hash of
the table markdown, so re-running ingestion only calls Gemini for new or
changed tables.
"""

import hashlib
import json

from src import config
from src.services.genai_client import get_client

SUMMARIES_PATH = config.INTERIM_DIR / "table_summaries.json"

SUMMARY_PROMPT = (
    "You are indexing tables from the IFC Annual Report 2024 (Financials) for a "
    "search engine. Write a 2-4 sentence plain-text description of the table "
    "below. State what the table reports, name the key line items, measures, or "
    "entities that appear in its rows, and the fiscal years or periods covered. "
    "Do not repeat any figures. Output only the description, no preamble.\n\n"
)


def _table_key(markdown: str) -> str:
    return hashlib.sha1(markdown.encode()).hexdigest()


def _load_cache() -> dict[str, str]:
    if SUMMARIES_PATH.exists():
        return json.loads(SUMMARIES_PATH.read_text())
    return {}


def _save_cache(cache: dict[str, str]) -> None:
    SUMMARIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARIES_PATH.write_text(json.dumps(cache, indent=2))


def _summarise(markdown: str, caption: str | None, section: str | None) -> str:
    context = caption or section
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
    misses = [r for r in table_records if _table_key(r["text"]) not in cache]
    if misses:
        print(f"Summarising {len(misses)} of {len(table_records)} tables via Gemini...")

    for i, record in enumerate(misses, start=1):
        cache[_table_key(record["text"])] = _summarise(
            record["text"], record.get("caption"), record.get("section")
        )
        _save_cache(cache)
        if i % 10 == 0 or i == len(misses):
            print(f"  {i}/{len(misses)}")

    for record in table_records:
        record["summary"] = cache[_table_key(record["text"])]
    return table_records
