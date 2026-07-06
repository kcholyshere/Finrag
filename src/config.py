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

# Source document
PDF_PATH = PROJECT_ROOT / "references" / "ifc-annual-report-2024-financials.pdf"

# Ingestion pipeline data
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Vector stores
FAISS_INDEX_DIR = PROJECT_ROOT / "models" / "faiss"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = "ifc_annual_report_2024"

# Evaluation (Phase 2)
CURATED_EVAL_PATH = PROJECT_ROOT / "references" / "RAG_evaluation_dataset.csv"
EVAL_DATASET_PATH = PROCESSED_DIR / "eval_dataset.csv"
EVAL_RUNS_DIR = PROCESSED_DIR / "eval_runs"
EVAL_DATASET_TARGET_SIZE = 200

# Langfuse
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
