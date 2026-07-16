"""Phase 6 indexing entrypoint: render page images, embed them with ColQwen2,
and build the Qdrant multivector collection. Run with `python -m src.colpali_dataset`.
Requires the Qdrant service to be up (`docker compose up -d qdrant`).

Separate from src.dataset so the heavy torch/VLM stack is only loaded when
actually indexing pages - the text pipeline and app never import it.
"""

import time

from src import config
from src.embedding.colpali_embedder import embed_page_images
from src.ingestion.page_images import render_page_images
from src.retrieval import qdrant_store


def run() -> None:
    started = time.perf_counter()
    paths = render_page_images()
    rendered = time.perf_counter()
    print(f"{len(paths)} page images ready ({rendered - started:.1f}s)")

    pages = [
        (page_no, matrix, str(path.relative_to(config.PROJECT_ROOT)))
        for (page_no, matrix), path in zip(embed_page_images(paths), paths)
    ]
    embedded = time.perf_counter()
    print(f"{len(pages)} pages embedded ({embedded - rendered:.1f}s)")

    qdrant_store.build_colpali_index(pages)
    print(
        f"Qdrant collection '{config.QDRANT_COLPALI_COLLECTION}' built "
        f"({time.perf_counter() - embedded:.1f}s, total {time.perf_counter() - started:.1f}s)"
    )


if __name__ == "__main__":
    run()
