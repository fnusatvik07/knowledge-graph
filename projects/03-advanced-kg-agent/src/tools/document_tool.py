"""LangChain-compatible tool for document ingestion.

Wraps the shared document_loader utilities to provide a tool interface
for loading and chunking documents.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from langchain_core.tools import BaseTool

from shared.document_loader import load_text_files, chunk_text


class DocumentLoaderTool(BaseTool):
    """Tool for loading documents from a directory."""

    name: str = "document_loader"
    description: str = (
        "Load text documents from a directory. Returns a list of documents "
        "with title, content, and source. Input should be an absolute path "
        "to a directory containing .txt or .md files."
    )

    def _run(self, directory: str) -> str:
        """Load documents from the specified directory."""
        try:
            path = Path(directory)
            if not path.exists():
                return json.dumps({"error": f"Directory not found: {directory}"})
            if not path.is_dir():
                return json.dumps({"error": f"Not a directory: {directory}"})

            documents = load_text_files(path)
            # Return metadata only (content can be large)
            summary = [
                {
                    "title": doc["title"],
                    "source": doc["source"],
                    "content_length": len(doc["content"]),
                    "preview": doc["content"][:200] + "...",
                }
                for doc in documents
            ]
            return json.dumps({
                "count": len(documents),
                "documents": summary,
            }, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def _arun(self, directory: str) -> str:
        return self._run(directory)


class DocumentChunkerTool(BaseTool):
    """Tool for chunking text into overlapping segments."""

    name: str = "document_chunker"
    description: str = (
        "Split text into overlapping chunks for processing. "
        "Input should be a JSON string with 'text' (required), "
        "'chunk_size' (optional, default 1000), and 'overlap' "
        "(optional, default 200)."
    )

    def _run(self, input_str: str) -> str:
        """Chunk the provided text."""
        try:
            params = json.loads(input_str)
            text = params["text"]
            chunk_size = params.get("chunk_size", 1000)
            overlap = params.get("overlap", 200)
        except (json.JSONDecodeError, KeyError) as e:
            return json.dumps({"error": f"Invalid input: {e}. Expected JSON with 'text' key."})

        try:
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            return json.dumps({
                "num_chunks": len(chunks),
                "chunk_size": chunk_size,
                "overlap": overlap,
                "chunks": [
                    {"index": i, "length": len(c), "preview": c[:100] + "..."}
                    for i, c in enumerate(chunks)
                ],
                "full_chunks": chunks,
            }, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def _arun(self, input_str: str) -> str:
        return self._run(input_str)


class DocumentIngestTool(BaseTool):
    """Combined tool: load documents from a directory and chunk them."""

    name: str = "document_ingest"
    description: str = (
        "Load and chunk documents from a directory in one step. "
        "Input should be a JSON string with 'directory' (required), "
        "'chunk_size' (optional, default 1000), and 'overlap' (optional, default 200). "
        "Returns chunked documents ready for entity extraction."
    )

    def _run(self, input_str: str) -> str:
        """Load and chunk documents."""
        try:
            params = json.loads(input_str)
            directory = params["directory"]
            chunk_size = params.get("chunk_size", 1000)
            overlap = params.get("overlap", 200)
        except (json.JSONDecodeError, KeyError) as e:
            return json.dumps({"error": f"Invalid input: {e}. Expected JSON with 'directory' key."})

        try:
            path = Path(directory)
            documents = load_text_files(path)

            all_chunks: list[dict[str, Any]] = []
            for doc in documents:
                chunks = chunk_text(doc["content"], chunk_size=chunk_size, overlap=overlap)
                for i, chunk in enumerate(chunks):
                    all_chunks.append({
                        "source": doc["source"],
                        "title": doc["title"],
                        "chunk_index": i,
                        "text": chunk,
                        "length": len(chunk),
                    })

            return json.dumps({
                "documents_loaded": len(documents),
                "total_chunks": len(all_chunks),
                "chunks": all_chunks,
            }, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def _arun(self, input_str: str) -> str:
        return self._run(input_str)
