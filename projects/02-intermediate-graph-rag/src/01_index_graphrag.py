"""Step 1: Index the corpus using Microsoft GraphRAG.

Configures a GraphRAG workspace, copies corpus documents into the input
directory, and runs the indexing pipeline to extract entities, relationships,
and community summaries.
"""

import os
import shutil
import sys
from pathlib import Path

# Add projects dir to path for shared imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared.config import get_openai_api_key, REPO_ROOT

PROJECT_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_DIR / "data" / "corpus"
GRAPHRAG_DIR = PROJECT_DIR / "output" / "graphrag"
CONFIG_FILE = PROJECT_DIR / "config" / "graphrag_settings.yaml"


def check_graphrag_installed():
    """Verify that the graphrag package is available."""
    try:
        import graphrag  # noqa: F401
        return True
    except ImportError:
        print("=" * 60)
        print("ERROR: Microsoft GraphRAG is not installed.")
        print()
        print("Install it with:")
        print("  pip install graphrag")
        print()
        print("For more information, see:")
        print("  https://github.com/microsoft/graphrag")
        print("=" * 60)
        return False


def setup_workspace():
    """Prepare the GraphRAG workspace directory structure."""
    # Create workspace directories
    input_dir = GRAPHRAG_DIR / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    # Copy corpus documents to the GraphRAG input directory
    corpus_files = list(CORPUS_DIR.glob("*.txt"))
    if not corpus_files:
        print(f"No .txt files found in {CORPUS_DIR}")
        sys.exit(1)

    print(f"Copying {len(corpus_files)} documents to GraphRAG input directory...")
    for src_file in corpus_files:
        dst_file = input_dir / src_file.name
        shutil.copy2(src_file, dst_file)
        print(f"  {src_file.name}")

    # Copy settings file
    settings_dst = GRAPHRAG_DIR / "settings.yaml"
    if CONFIG_FILE.exists():
        shutil.copy2(CONFIG_FILE, settings_dst)
        print(f"\nSettings copied to {settings_dst}")

    return input_dir


def run_indexing():
    """Run the GraphRAG indexing pipeline."""
    try:
        from graphrag.cli.main import app as graphrag_app
        from graphrag.config import create_graphrag_config
    except ImportError:
        # Fallback: try running via CLI subprocess
        import subprocess
        print("\nRunning GraphRAG indexing via CLI...")
        env = os.environ.copy()
        env["GRAPHRAG_API_KEY"] = get_openai_api_key()

        result = subprocess.run(
            ["graphrag", "index", "--root", str(GRAPHRAG_DIR)],
            env=env,
            capture_output=False,
            text=True,
        )
        return result.returncode == 0

    # Use the Python API if available
    print("\nRunning GraphRAG indexing via Python API...")
    os.environ["GRAPHRAG_API_KEY"] = get_openai_api_key()

    try:
        import subprocess
        result = subprocess.run(
            ["graphrag", "index", "--root", str(GRAPHRAG_DIR)],
            capture_output=False,
            text=True,
            env={**os.environ, "GRAPHRAG_API_KEY": get_openai_api_key()},
        )
        if result.returncode == 0:
            print("\nIndexing completed successfully!")
            return True
        else:
            print(f"\nIndexing failed with return code {result.returncode}")
            return False
    except Exception as e:
        print(f"\nError running indexing: {e}")
        return False


def verify_output():
    """Check that indexing produced the expected output files."""
    output_dir = GRAPHRAG_DIR / "output"
    if not output_dir.exists():
        # GraphRAG may store artifacts directly in the root
        output_dir = GRAPHRAG_DIR

    parquet_files = list(GRAPHRAG_DIR.rglob("*.parquet"))
    if parquet_files:
        print(f"\nFound {len(parquet_files)} output parquet files:")
        for f in sorted(parquet_files)[:10]:
            print(f"  {f.relative_to(GRAPHRAG_DIR)}")
        if len(parquet_files) > 10:
            print(f"  ... and {len(parquet_files) - 10} more")
        return True

    graphml_files = list(GRAPHRAG_DIR.rglob("*.graphml"))
    if graphml_files:
        print(f"\nFound {len(graphml_files)} GraphML files:")
        for f in graphml_files:
            print(f"  {f.relative_to(GRAPHRAG_DIR)}")
        return True

    print("\nWarning: No output artifacts found. Indexing may not have completed.")
    return False


def main():
    print("=" * 60)
    print("GraphRAG Indexing Pipeline")
    print("=" * 60)

    # Check prerequisites
    if not check_graphrag_installed():
        sys.exit(1)

    # Ensure API key is available
    api_key = get_openai_api_key()
    print(f"API key loaded (ends with ...{api_key[-4:]})")

    # Setup workspace
    print(f"\nProject directory: {PROJECT_DIR}")
    print(f"Corpus directory:  {CORPUS_DIR}")
    print(f"Output directory:  {GRAPHRAG_DIR}")
    setup_workspace()

    # Run indexing
    success = run_indexing()

    if success:
        verify_output()
        print("\n" + "=" * 60)
        print("GraphRAG indexing complete!")
        print(f"Output stored in: {GRAPHRAG_DIR}")
        print("=" * 60)
    else:
        print("\nIndexing encountered issues. Check the logs above for details.")
        print("You may need to run manually:")
        print(f"  cd {GRAPHRAG_DIR}")
        print(f"  GRAPHRAG_API_KEY=<your-key> graphrag index --root .")


if __name__ == "__main__":
    main()
