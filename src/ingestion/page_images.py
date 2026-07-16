import hashlib
import json
import os
from pathlib import Path

import fitz

from src import config

PAGE_IMAGES_DIR = config.INTERIM_DIR / "page_images"
PAGE_IMAGES_META_PATH = PAGE_IMAGES_DIR / "_meta.json"
PAGE_IMAGE_DPI = 150


def _cache_meta() -> dict:
    return {"pdf_sha1": hashlib.sha1(config.PDF_PATH.read_bytes()).hexdigest(), "dpi": PAGE_IMAGE_DPI}


def _cache_status(meta_path, current: dict) -> str:
    """Returns 'valid', 'missing', or 'stale' - see parse.py's identical check;
    the per-file existence test alone can't tell a genuinely-cached page apart
    from one rendered at a different DPI or from a since-replaced PDF."""
    if not meta_path.exists():
        return "missing"
    return "valid" if json.loads(meta_path.read_text()) == current else "stale"


def render_page_images() -> list[Path]:
    """Render each PDF page to a cached PNG for the Phase 6 ColPali pipeline.

    Filenames use Docling's raw 1-indexed page numbering (fitz page i -> page_no
    i + 1, see config.PDF_PAGE_NUMBER_OFFSET) so page images line up with the
    page/section metadata already used elsewhere. Idempotent - skips pages
    already rendered, since a full 147-page render is a real one-time cost.
    """
    PAGE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    current = _cache_meta()
    status = _cache_status(PAGE_IMAGES_META_PATH, current)
    if status == "stale":
        print(
            f"Page image cache in {PAGE_IMAGES_DIR} no longer matches the source PDF "
            "or DPI setting - re-rendering all pages"
        )
        for stale_png in PAGE_IMAGES_DIR.glob("page_*.png"):
            stale_png.unlink()
    elif status == "missing":
        print(f"No cache metadata found in {PAGE_IMAGES_DIR}, grandfathering existing page images")
    PAGE_IMAGES_META_PATH.write_text(json.dumps(current, indent=2))

    paths = []

    with fitz.open(config.PDF_PATH) as pdf:
        for i, page in enumerate(pdf):
            page_no = i + 1
            path = PAGE_IMAGES_DIR / f"page_{page_no:04d}.png"
            if not path.exists():
                pixmap = page.get_pixmap(dpi=PAGE_IMAGE_DPI)
                # Write-temp-then-replace: a direct save left half-written on
                # interrupt would pass this exists() check forever (A10).
                tmp_path = path.with_suffix(".tmp.png")
                pixmap.save(tmp_path)
                os.replace(tmp_path, path)
            paths.append(path)

    return paths


if __name__ == "__main__":
    rendered = render_page_images()
    print(f"Rendered {len(rendered)} page images to {PAGE_IMAGES_DIR}")
