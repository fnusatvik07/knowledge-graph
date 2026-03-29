"""Load documents from various file formats."""

from pathlib import Path


def load_text_files(directory: str | Path) -> list[dict]:
    """Load all .txt and .md files from a directory.

    Returns a list of dicts with 'title', 'content', and 'source' keys.
    """
    directory = Path(directory)
    documents = []
    for ext in ("*.txt", "*.md"):
        for filepath in sorted(directory.glob(ext)):
            content = filepath.read_text(encoding="utf-8")
            documents.append({
                "title": filepath.stem.replace("_", " ").replace("-", " ").title(),
                "content": content,
                "source": str(filepath),
            })
    return documents


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks by character count."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks
