"""Diagnostic and setup helper for MCP servers used in this project.

Checks if required MCP servers and their dependencies are available,
prints setup instructions for any that are missing, and verifies connectivity.

MCP Specification: https://spec.modelcontextprotocol.io/
MCP Server Registry: https://github.com/modelcontextprotocol/servers
"""

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def check_command(cmd: str) -> bool:
    """Check if a command is available on PATH."""
    return shutil.which(cmd) is not None


def check_npm_package(package: str) -> bool:
    """Check if a global npm package is installed."""
    try:
        result = subprocess.run(
            ["npm", "list", "-g", package, "--depth=0"],
            capture_output=True, text=True, timeout=10,
        )
        return package in result.stdout
    except Exception:
        return False


def check_python_package(package: str) -> bool:
    """Check if a Python package is importable."""
    try:
        __import__(package)
        return True
    except ImportError:
        return False


def check_neo4j_connectivity() -> bool:
    """Check if Neo4j is reachable."""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            "bolt://localhost:7687", auth=("neo4j", "password123")
        )
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception:
        return False


def check_docker_container(name: str) -> bool:
    """Check if a Docker container is running."""
    try:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", name],
            capture_output=True, text=True, timeout=10,
        )
        return "true" in result.stdout.lower()
    except Exception:
        return False


def print_status(name: str, available: bool, fix: str = ""):
    """Print a status line with checkmark or X."""
    icon = "[OK]" if available else "[!!]"
    print(f"  {icon} {name}")
    if not available and fix:
        print(f"       Fix: {fix}")


def main():
    print("=" * 70)
    print("MCP SERVER SETUP DIAGNOSTIC")
    print("=" * 70)

    # ── Prerequisites ────────────────────────────────────────────────────
    print("\n1. Prerequisites")
    print("-" * 40)

    node_ok = check_command("node")
    print_status("Node.js", node_ok, "Install from https://nodejs.org/")

    npm_ok = check_command("npm")
    print_status("npm", npm_ok, "Comes with Node.js")

    npx_ok = check_command("npx")
    print_status("npx", npx_ok, "Comes with npm 5.2+")

    docker_ok = check_command("docker")
    print_status("Docker", docker_ok, "Install from https://docker.com/")

    python_ok = sys.version_info >= (3, 11)
    print_status(
        f"Python {sys.version_info.major}.{sys.version_info.minor}",
        python_ok,
        "Python 3.11+ required",
    )

    claude_ok = check_command("claude")
    print_status("Claude Code CLI", claude_ok, "Install: npm install -g @anthropic-ai/claude-code")

    # ── MCP Servers ──────────────────────────────────────────────────────
    print("\n2. MCP Servers")
    print("-" * 40)

    # Memory MCP
    memory_ok = check_npm_package("@modelcontextprotocol/server-memory")
    print_status(
        "MCP Memory Server (@modelcontextprotocol/server-memory)",
        memory_ok,
        "npm install -g @modelcontextprotocol/server-memory",
    )

    # Neo4j MCP — runs via npx, no global install needed
    print_status(
        "Neo4j MCP Server (@neo4j/mcp-neo4j)",
        True,  # npx fetches on demand
        "Runs via npx — no install needed if npx is available",
    )

    # Graphiti MCP
    graphiti_ok = check_python_package("graphiti_mcp")
    print_status(
        "Graphiti MCP Server (graphiti-mcp)",
        graphiti_ok,
        "pip install graphiti-mcp",
    )

    # ── Backend Services ─────────────────────────────────────────────────
    print("\n3. Backend Services")
    print("-" * 40)

    neo4j_container = check_docker_container("kg-neo4j")
    print_status(
        "Neo4j Docker container (kg-neo4j)",
        neo4j_container,
        "docker run -d --name kg-neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password123 neo4j:5-community",
    )

    neo4j_conn = check_neo4j_connectivity()
    print_status(
        "Neo4j connectivity (bolt://localhost:7687)",
        neo4j_conn,
        "Start Neo4j first, then retry",
    )

    # ── Python Dependencies ──────────────────────────────────────────────
    print("\n4. Python Dependencies")
    print("-" * 40)

    for pkg, pip_name in [
        ("neo4j", "neo4j"),
        ("httpx", "httpx"),
        ("mcp", "mcp"),
        ("graphiti_core", "graphiti-core"),
    ]:
        ok = check_python_package(pkg)
        print_status(pkg, ok, f"pip install {pip_name}")

    # ── Claude Code MCP Configuration ────────────────────────────────────
    print("\n5. Claude Code MCP Setup Commands")
    print("-" * 40)
    print("""
  To register MCP servers with Claude Code, run:

    # Memory MCP (JSONL-based KG)
    claude mcp add memory -- npx -y @modelcontextprotocol/server-memory

    # Neo4j MCP (graph database)
    claude mcp add neo4j -e NEO4J_URI=bolt://localhost:7687 \\
      -e NEO4J_USERNAME=neo4j -e NEO4J_PASSWORD=password123 \\
      -- npx -y @neo4j/mcp-neo4j

    # Graphiti MCP (temporal KG)
    claude mcp add graphiti -e NEO4J_URI=bolt://localhost:7687 \\
      -e NEO4J_USERNAME=neo4j -e NEO4J_PASSWORD=password123 \\
      -- python -m graphiti_mcp

  See src/05_claude_code_setup.md for detailed instructions.
""")

    # ── Summary ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("Setup diagnostic complete. Fix any [!!] items above before running demos.")
    print("=" * 70)


if __name__ == "__main__":
    main()
