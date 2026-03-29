"""Central configuration and environment variable loading."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the repo root
_repo_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_repo_root / ".env")


# ── LLM Provider Keys ────────────────────────────────────────────────

def get_openai_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY not set. Copy .env.example to .env and fill in your key.")
    return key


def get_anthropic_api_key() -> str:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY not set. Copy .env.example to .env and fill in your key.")
    return key


def get_tavily_api_key() -> str:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        raise ValueError("TAVILY_API_KEY not set. Copy .env.example to .env and fill in your key.")
    return key


# ── Graph Database Configs ────────────────────────────────────────────

def get_neo4j_config() -> dict:
    """Neo4j connection config (Projects 3, 4, 6, 8)."""
    return {
        "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "user": os.getenv("NEO4J_USER", "neo4j"),
        "password": os.getenv("NEO4J_PASSWORD", "password"),
    }


def get_arangodb_config() -> dict:
    """ArangoDB connection config (Project 4)."""
    return {
        "url": os.getenv("ARANGO_URL", "http://localhost:8529"),
        "user": os.getenv("ARANGO_USER", "root"),
        "password": os.getenv("ARANGO_PASSWORD", ""),
        "database": os.getenv("ARANGO_DB", "knowledgegraph"),
    }


def get_memgraph_config() -> dict:
    """Memgraph connection config (Project 4)."""
    return {
        "host": os.getenv("MEMGRAPH_HOST", "localhost"),
        "port": int(os.getenv("MEMGRAPH_PORT", "7688")),
    }


def get_falkordb_config() -> dict:
    """FalkorDB connection config (Project 4)."""
    return {
        "host": os.getenv("FALKORDB_HOST", "localhost"),
        "port": int(os.getenv("FALKORDB_PORT", "6379")),
    }


# ── Convenience: repo paths ──────────────────────────────────────────

REPO_ROOT = _repo_root
PROJECTS_DIR = _repo_root / "projects"
KNOWLEDGE_BASE_DIR = _repo_root / "knowledge-base"
