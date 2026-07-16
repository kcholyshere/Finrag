"""ColQwen2 patch embeddings for the Phase 6 ColPali-like pipeline.

Plays the role embedder.py plays for Gemini text embeddings, but produces one
matrix per page (n_patches x 128) instead of one vector per chunk, enabling
late-interaction (MaxSim) retrieval. Runs locally - the model never sees the
network after the initial weights download.
"""

from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src import config

BATCH_SIZE = 2  # 2B-param VLM on 18GB unified memory - keep forward passes small


def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


@lru_cache(maxsize=None)
def _load_model_and_processor():
    from colpali_engine.models import ColQwen2, ColQwen2Processor

    model = ColQwen2.from_pretrained(
        config.COLPALI_MODEL, torch_dtype=torch.bfloat16, device_map=_device()
    ).eval()
    processor = ColQwen2Processor.from_pretrained(config.COLPALI_MODEL)
    return model, processor


def _cache_path(page_no: int) -> Path:
    return config.COLPALI_EMBEDDINGS_DIR / f"page_{page_no:04d}.npy"


def _embed_image_batch(paths: list[Path]) -> list[np.ndarray]:
    model, processor = _load_model_and_processor()
    images = [Image.open(path) for path in paths]
    batch = processor.process_images(images).to(model.device)
    with torch.no_grad():
        embeddings = model(**batch)
    # fp16 halves the cache size; the ~3-decimal precision loss is irrelevant
    # to MaxSim ranking.
    return [e.to(torch.float16).cpu().numpy() for e in embeddings]


def embed_page_images(paths: list[Path]) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (page_no, patch matrix) per page image, cached on disk per page.

    Page numbers are parsed from the page_NNNN.png filenames produced by
    ingestion/page_images.py. Cached pages skip the forward pass entirely, so
    an interrupted 147-page run resumes where it left off - same pattern as
    enrich.py's caches, but one .npy file per page instead of one JSON, since
    each entry is a ~200KB float matrix rather than a text snippet.
    """
    config.COLPALI_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    uncached = [p for p in paths if not _cache_path(int(p.stem.split("_")[1])).exists()]
    for start in range(0, len(uncached), BATCH_SIZE):
        batch_paths = uncached[start : start + BATCH_SIZE]
        for path, embedding in zip(batch_paths, _embed_image_batch(batch_paths)):
            np.save(_cache_path(int(path.stem.split("_")[1])), embedding)

    for path in paths:
        page_no = int(path.stem.split("_")[1])
        yield page_no, np.load(_cache_path(page_no))


def embed_query(text: str) -> np.ndarray:
    """Embed a query into its token-level matrix (n_tokens x 128) for MaxSim."""
    model, processor = _load_model_and_processor()
    batch = processor.process_queries([text]).to(model.device)
    with torch.no_grad():
        embeddings = model(**batch)
    return embeddings[0].to(torch.float32).cpu().numpy()
