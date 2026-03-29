"""LightRAG configuration for Project 2."""

from pathlib import Path

# Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_DIR / "data" / "corpus"
LIGHTRAG_WORK_DIR = PROJECT_DIR / "output" / "lightrag"

# LLM settings
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
MAX_TOKENS = 4000
TEMPERATURE = 0.0

# Chunking
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# LightRAG-specific settings
LIGHTRAG_CONFIG = {
    "working_dir": str(LIGHTRAG_WORK_DIR),
    "chunk_token_size": CHUNK_SIZE,
    "chunk_overlap_token_size": CHUNK_OVERLAP,
    "tiktoken_model_name": "gpt-4o-mini",
}
