"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv()

def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_list(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


# --- API Keys ---
OPENAI_API_KEY = _env_str("OPENAI_API_KEY", "")
COHERE_API_KEY = _env_str("COHERE_API_KEY", "")
JINA_API_KEY = _env_str("JINA_API_KEY", "")

# --- Qdrant ---
QDRANT_HOST = _env_str("QDRANT_HOST", "localhost")
QDRANT_PORT = _env_int("QDRANT_PORT", 6333)
COLLECTION_NAME = _env_str("COLLECTION_NAME", "lab18_production")
NAIVE_COLLECTION = _env_str("NAIVE_COLLECTION", "lab18_naive")

# --- Models ---
EMBEDDING_MODEL = _env_str("EMBEDDING_MODEL", "jina-embeddings-v5-text-small")
SEMANTIC_EMBEDDING_MODEL = _env_str("SEMANTIC_EMBEDDING_MODEL", EMBEDDING_MODEL)
RERANK_MODEL = _env_str("RERANK_MODEL", "rerank-multilingual-v3.0")
FLASHRANK_MODEL = _env_str("FLASHRANK_MODEL", "ms-marco-MiniLM-L-12-v2")
FLASHRANK_MAX_LENGTH = _env_int("FLASHRANK_MAX_LENGTH", 512)
FLASHRANK_CACHE_DIR = _env_str("FLASHRANK_CACHE_DIR", "")
GENERATION_MODEL = _env_str("GENERATION_MODEL", "gpt-5.4-mini")
RAGAS_EMBEDDING_MODEL = _env_str("RAGAS_EMBEDDING_MODEL", "jina-embeddings-v5-text-small")
RAGAS_EMBEDDING_PROVIDER = _env_str("RAGAS_EMBEDDING_PROVIDER", "jina")
JINA_BASE_URL = _env_str("JINA_BASE_URL", "https://api.jina.ai/v1/embeddings")
JINA_USER_AGENT = _env_str("JINA_USER_AGENT", "lab18-production-rag/1.0")
JINA_EMBEDDING_TYPE = _env_str("JINA_EMBEDDING_TYPE", "float")
JINA_NORMALIZED = _env_int("JINA_NORMALIZED", 1) == 1
JINA_TRUNCATE = _env_int("JINA_TRUNCATE", 1) == 1
JINA_BATCH_SIZE = _env_int("JINA_BATCH_SIZE", 32)
JINA_TIMEOUT_SECONDS = _env_float("JINA_TIMEOUT_SECONDS", 30.0)
JINA_MAX_RETRIES = _env_int("JINA_MAX_RETRIES", 5)
JINA_BACKOFF_INITIAL_SECONDS = _env_float("JINA_BACKOFF_INITIAL_SECONDS", 1.0)
JINA_BACKOFF_MAX_SECONDS = _env_float("JINA_BACKOFF_MAX_SECONDS", 16.0)
JINA_TASK_RETRIEVAL_QUERY = _env_str("JINA_TASK_RETRIEVAL_QUERY", "retrieval.query")
JINA_TASK_RETRIEVAL_PASSAGE = _env_str("JINA_TASK_RETRIEVAL_PASSAGE", "retrieval.passage")
JINA_TASK_TEXT_MATCHING = _env_str("JINA_TASK_TEXT_MATCHING", "text-matching")

# --- Embedding ---
EMBEDDING_DIM = _env_int("EMBEDDING_DIM", 1024)
JINA_DIMENSIONS = _env_int("JINA_DIMENSIONS", EMBEDDING_DIM)

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = _env_int("HIERARCHICAL_PARENT_SIZE", 2048)
HIERARCHICAL_CHILD_SIZE = _env_int("HIERARCHICAL_CHILD_SIZE", 256)
SEMANTIC_THRESHOLD = _env_float("SEMANTIC_THRESHOLD", 0.85)

# --- Search ---
BM25_TOP_K = _env_int("BM25_TOP_K", 20)
DENSE_TOP_K = _env_int("DENSE_TOP_K", 20)
HYBRID_TOP_K = _env_int("HYBRID_TOP_K", 20)
RERANK_TOP_K = _env_int("RERANK_TOP_K", 3)
DEFAULT_CONTEXT_TOP_K = _env_int("DEFAULT_CONTEXT_TOP_K", 3)

# --- Evaluation ---
EVAL_METRICS = _env_list(
    "EVAL_METRICS",
    "faithfulness,answer_relevancy,context_precision,context_recall",
)
PASS_THRESHOLD = _env_float("PASS_THRESHOLD", 0.75)
FAILURE_BOTTOM_N = _env_int("FAILURE_BOTTOM_N", 10)
FAILURE_FAITHFULNESS_THRESHOLD = _env_float("FAILURE_FAITHFULNESS_THRESHOLD", 0.85)
FAILURE_CONTEXT_RECALL_THRESHOLD = _env_float("FAILURE_CONTEXT_RECALL_THRESHOLD", 0.75)
FAILURE_CONTEXT_PRECISION_THRESHOLD = _env_float("FAILURE_CONTEXT_PRECISION_THRESHOLD", 0.75)
FAILURE_ANSWER_RELEVANCY_THRESHOLD = _env_float("FAILURE_ANSWER_RELEVANCY_THRESHOLD", 0.80)

# --- Enrichment ---
DEFAULT_ENRICHMENT_METHODS = _env_list("DEFAULT_ENRICHMENT_METHODS", "contextual,hyqa,metadata")
ENRICHMENT_DEFAULT_QUESTIONS = _env_int("ENRICHMENT_DEFAULT_QUESTIONS", 3)
ENRICHMENT_SUMMARY_MAX_CHARS = _env_int("ENRICHMENT_SUMMARY_MAX_CHARS", 240)
ENRICHMENT_SUMMARY_SHORT_MAX_CHARS = _env_int("ENRICHMENT_SUMMARY_SHORT_MAX_CHARS", 180)

# --- Reports ---
REPORT_DIR = _env_str("REPORT_DIR", "reports")
RAGAS_REPORT_PATH = _env_str("RAGAS_REPORT_PATH", os.path.join(REPORT_DIR, "ragas_report.json"))
NAIVE_REPORT_PATH = _env_str("NAIVE_REPORT_PATH", os.path.join(REPORT_DIR, "naive_baseline_report.json"))

# --- Paths ---
SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
TESTS_DIR = os.path.join(os.path.dirname(__file__), "tests")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
