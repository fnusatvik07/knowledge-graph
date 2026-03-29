"""Fetch PubMed abstracts using BioPython's Entrez API.

Falls back to sample data if BioPython is not installed or if the API is unavailable.

Usage:
    python src/01_fetch_pubmed.py --query "cancer drug target" --max_results 50
    python src/01_fetch_pubmed.py --use_sample  # Use bundled sample data

References:
    - BioPython Entrez: https://biopython.org/wiki/Entrez
    - PubMed E-utilities: https://www.ncbi.nlm.nih.gov/books/NBK25499/
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

# ── Shared imports ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
SAMPLE_FILE = RAW_DIR / "sample_abstracts.json"
OUTPUT_FILE = RAW_DIR / "pubmed_abstracts.json"


def fetch_pubmed_abstracts(query: str, max_results: int = 50, email: str = "user@example.com") -> list[dict]:
    """Fetch abstracts from PubMed using BioPython's Entrez API.

    Args:
        query: PubMed search query (e.g., "BRCA1 breast cancer").
        max_results: Maximum number of abstracts to fetch.
        email: Email address for NCBI Entrez (required by their API policy).

    Returns:
        List of dicts with keys: pmid, title, abstract.
    """
    try:
        from Bio import Entrez
    except ImportError:
        print("BioPython not installed. Install with: pip install biopython")
        print("Falling back to sample data.")
        return None

    Entrez.email = email

    # Step 1: Search for PubMed IDs
    print(f"Searching PubMed for: '{query}' (max {max_results} results)...")
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
    search_results = Entrez.read(handle)
    handle.close()

    pmids = search_results["IdList"]
    if not pmids:
        print("No results found.")
        return []

    print(f"Found {len(pmids)} articles. Fetching abstracts...")

    # Step 2: Fetch article details
    handle = Entrez.efetch(db="pubmed", id=pmids, rettype="xml", retmode="xml")
    from Bio import Medline
    # Use XML parsing for richer data
    records = Entrez.read(handle)
    handle.close()

    abstracts = []
    articles = records.get("PubmedArticle", [])
    for article in articles:
        medline = article.get("MedlineCitation", {})
        article_data = medline.get("Article", {})

        pmid = str(medline.get("PMID", ""))
        title = str(article_data.get("ArticleTitle", ""))

        # Abstract may have multiple sections
        abstract_obj = article_data.get("Abstract", {})
        abstract_texts = abstract_obj.get("AbstractText", [])
        abstract = " ".join(str(t) for t in abstract_texts) if abstract_texts else ""

        if abstract:  # Only include articles with abstracts
            abstracts.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
            })

    print(f"Fetched {len(abstracts)} abstracts with text.")
    return abstracts


def load_sample_data() -> list[dict]:
    """Load bundled sample abstracts as fallback.

    Returns:
        List of sample abstract dicts.
    """
    print(f"Loading sample data from {SAMPLE_FILE}...")
    with open(SAMPLE_FILE) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} sample abstracts.")
    return data


def main():
    parser = argparse.ArgumentParser(description="Fetch PubMed abstracts for biomedical KG construction.")
    parser.add_argument("--query", type=str, default="cancer drug target gene",
                        help="PubMed search query")
    parser.add_argument("--max_results", type=int, default=50,
                        help="Maximum number of abstracts to fetch")
    parser.add_argument("--email", type=str, default="user@example.com",
                        help="Email for NCBI Entrez API")
    parser.add_argument("--use_sample", action="store_true",
                        help="Use bundled sample data instead of fetching from PubMed")
    args = parser.parse_args()

    if args.use_sample:
        abstracts = load_sample_data()
    else:
        abstracts = fetch_pubmed_abstracts(
            query=args.query,
            max_results=args.max_results,
            email=args.email,
        )
        # Fall back to sample data if fetch failed
        if abstracts is None:
            abstracts = load_sample_data()

    # Save results
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(abstracts, f, indent=2)

    print(f"\nSaved {len(abstracts)} abstracts to {OUTPUT_FILE}")

    # Show preview
    for i, a in enumerate(abstracts[:3]):
        print(f"\n--- Abstract {i+1} (PMID: {a['pmid']}) ---")
        print(f"Title: {a['title']}")
        print(f"Abstract: {a['abstract'][:150]}...")


if __name__ == "__main__":
    main()
