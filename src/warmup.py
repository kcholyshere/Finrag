"""Pre-download and load the lazy Hugging Face models before the demo.

The cross-encoder reranker and ColQwen2 (~5GB) both download from Hugging Face
on first use and are otherwise hidden behind the retrieval spinner - a cold
cache on demo day reads as a hung app. Run this once after `docker compose up`
(with HF_HOME pointed at a persistent volume) so both models are downloaded
and loaded into memory-mapped weights on disk before the first real query.

    python -m src.warmup
"""

import time

from langchain_core.documents import Document


def _warm_cross_encoder() -> float:
    from src.retrieval.reranker import rerank

    start = time.perf_counter()
    rerank("warm-up query", [Document(page_content="warm-up passage")], top_n=1)
    return time.perf_counter() - start


def _warm_colpali() -> float:
    from src.embedding.colpali_embedder import embed_query

    start = time.perf_counter()
    embed_query("warm-up query")
    return time.perf_counter() - start


def main() -> None:
    from src.retrieval.reranker import CROSS_ENCODER_MODEL
    from src import config

    ce_seconds = _warm_cross_encoder()
    print(f"Cross-encoder ({CROSS_ENCODER_MODEL}) ready in {ce_seconds:.1f}s")

    colpali_seconds = _warm_colpali()
    print(f"ColQwen2 ({config.COLPALI_MODEL}) ready in {colpali_seconds:.1f}s")


if __name__ == "__main__":
    main()
