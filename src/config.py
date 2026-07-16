import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Vertex AI auth (no API keys - relies on Application Default Credentials)
GCP_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GCP_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "global")

# Models
GEMINI_MODEL = "gemini-3.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 3072

# Phase 6 ColPali-like pipeline. If ColQwen2 proves too slow on MPS, swap to
# "vidore/colSmol-500M" - same colpali-engine API, one-line change (cached
# embeddings under COLPALI_EMBEDDINGS_DIR must be regenerated).
COLPALI_MODEL = "vidore/colqwen2-v1.0"
COLPALI_EMBEDDING_DIMENSIONS = 128

# Source document
PDF_PATH = PROJECT_ROOT / "references" / "ifc-annual-report-2024-financials.pdf"

# Docling's raw page index runs 1 ahead of the report's own printed page number
# throughout (one unnumbered cover page precedes printed page 1) - verified
# against all 145 page-footer items, zero exceptions. Chunk metadata
# (start_page/end_page) keeps Docling's raw numbering unchanged, since
# retrieval filtering and the eval ground truth are already built against it;
# this offset is applied only where a page number is shown to a person.
PDF_PAGE_NUMBER_OFFSET = 1


def display_page(raw_page_no: int | None) -> int | None:
    """Convert Docling's raw page index to the report's own printed page number."""
    return raw_page_no - PDF_PAGE_NUMBER_OFFSET if raw_page_no is not None else None

# Ingestion pipeline data
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
COLPALI_EMBEDDINGS_DIR = INTERIM_DIR / "colpali_embeddings"

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Vector stores
FAISS_INDEX_DIR = PROJECT_ROOT / "models" / "faiss"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = "ifc_annual_report_2024"
QDRANT_COLPALI_COLLECTION = "ifc_annual_report_2024_colpali"

# Evaluation (Phase 2)
CURATED_EVAL_PATH = PROJECT_ROOT / "references" / "RAG_evaluation_dataset.csv"
EVAL_DATASET_PATH = PROCESSED_DIR / "eval_dataset.csv"
EVAL_RUNS_DIR = PROCESSED_DIR / "eval_runs"
EVAL_DATASET_TARGET_SIZE = 200

# Langfuse
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
