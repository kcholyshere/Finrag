import os
from pathlib import Path

import fitz

from src import config

PAGE_IMAGES_DIR = config.INTERIM_DIR / "page_images"
PAGE_IMAGE_DPI = 150


def render_page_images() -> list[Path]:
    """Render each PDF page to a cached PNG for the Phase 6 ColPali pipeline.

    Filenames use Docling's raw 1-indexed page numbering (fitz page i -> page_no
    i + 1, see config.PDF_PAGE_NUMBER_OFFSET) so page images line up with the
    page/section metadata already used elsewhere. Idempotent - skips pages
    already rendered, since a full 147-page render is a real one-time cost.
    """
    PAGE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
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
