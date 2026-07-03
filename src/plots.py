"""FAISS vs Qdrant retrieval benchmark for the Phase 1 comparison writeup.
Run with `python -m src.plots` (requires `docker compose up -d qdrant` and a
built index in both backends - see `src/dataset.py`).
"""

import statistics
import time

from src.retrieval import faiss_store, qdrant_store

TEST_QUERIES = [
    "What is IFC's mission and how many member countries does it have?",
    "What was the Net Income for FY24 and FY23?",
    "How does IFC manage liquidity risk?",
    "What is IFC's approach to climate finance?",
    "Describe IFC's funding resources and borrowings.",
    "What are IFC's critical accounting policies?",
]

REPS = 5


def _time_backend(index, label: str) -> dict:
    latencies = []
    overlap_by_query: dict[str, set[str]] = {}

    for query in TEST_QUERIES:
        query_latencies = []
        for _ in range(REPS):
            start = time.perf_counter()
            docs = index.similarity_search(query, k=4)
            query_latencies.append(time.perf_counter() - start)
        latencies.extend(query_latencies)
        overlap_by_query[query] = {
            f"{d.metadata.get('section')}|{d.metadata.get('start_page')}" for d in docs
        }

    return {
        "label": label,
        "mean": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "p95": sorted(latencies)[int(len(latencies) * 0.95) - 1],
        "overlap_by_query": overlap_by_query,
    }


def run_benchmark() -> None:
    faiss_index = faiss_store.load_index()
    qdrant_index = qdrant_store.load_index()

    faiss_stats = _time_backend(faiss_index, "FAISS")
    qdrant_stats = _time_backend(qdrant_index, "Qdrant")

    print(f"{'backend':<10} {'mean (s)':<10} {'median (s)':<12} {'p95 (s)':<10}")
    for stats in (faiss_stats, qdrant_stats):
        print(f"{stats['label']:<10} {stats['mean']:<10.3f} {stats['median']:<12.3f} {stats['p95']:<10.3f}")

    print("\ntop-4 overlap (section|page) per query:")
    for query in TEST_QUERIES:
        faiss_set = faiss_stats["overlap_by_query"][query]
        qdrant_set = qdrant_stats["overlap_by_query"][query]
        jaccard = len(faiss_set & qdrant_set) / len(faiss_set | qdrant_set)
        print(f"  {jaccard:.2f}  {query}")


if __name__ == "__main__":
    run_benchmark()
