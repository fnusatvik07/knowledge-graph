"""Run this BEFORE the workshop to verify everything works."""

import sys

print("=" * 50)
print("  Workshop Setup Verification")
print("=" * 50)

# 1. Python version
print(f"\n1. Python: {sys.version.split()[0]}", end=" ")
print("✓" if sys.version_info >= (3, 11) else "✗ Need 3.11+")

# 2. Required packages
packages = ["openai", "neo4j", "chromadb", "pyvis", "pandas", "dotenv"]
for pkg in packages:
    try:
        __import__(pkg)
        print(f"2. {pkg}: ✓")
    except ImportError:
        print(f"2. {pkg}: ✗  → pip install -r requirements.txt")

# 3. OpenAI API key
import os
from dotenv import load_dotenv
load_dotenv()
key = os.getenv("OPENAI_API_KEY", "")
print(f"\n3. OpenAI API Key: {'✓ Set' if key and not key.startswith('sk-proj-your') else '✗ Edit .env file'}")

# 4. Neo4j connection
try:
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "workshop2024"))
    with driver.session() as session:
        session.run("RETURN 1")
    driver.close()
    print("4. Neo4j: ✓ Connected")
except Exception as e:
    print(f"4. Neo4j: ✗ Run 'docker compose up -d' first")

print("\n" + "=" * 50)
print("  If all checks pass, you're ready!")
print("=" * 50)
