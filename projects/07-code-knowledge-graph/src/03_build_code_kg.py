"""Build a code knowledge graph from parsed Python and JavaScript data using NetworkX.

Entity types: Module, Class, Function, Import
Relationships: IMPORTS, DEFINES, CALLS, INHERITS_FROM, CONTAINS

Metadata includes line numbers, file paths, and language.

Usage:
    python src/03_build_code_kg.py
    python src/03_build_code_kg.py --python output/parsed_python.json --js output/parsed_javascript.json

References:
    - NetworkX docs: https://networkx.org/documentation/stable/
    - LangChain graph integration: https://python.langchain.com/docs/integrations/graphs/
"""

import argparse
import json
import sys
from pathlib import Path

# ── Shared imports ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import networkx as nx

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"


def create_code_kg(parsed_files: list[dict]) -> nx.DiGraph:
    """Build a directed knowledge graph from parsed code structures.

    Args:
        parsed_files: List of parsed file dicts (from 01_parse_python / 02_parse_javascript).

    Returns:
        A NetworkX DiGraph representing the code knowledge graph.
    """
    G = nx.DiGraph()

    for file_data in parsed_files:
        module_name = file_data["module"]
        language = file_data.get("language", "unknown")
        file_path = file_data.get("file_path", "")
        line_count = file_data.get("line_count", 0)

        # ── Add Module node ──────────────────────────────────────────
        module_id = f"module:{module_name}"
        G.add_node(module_id,
                    entity_type="Module",
                    name=module_name,
                    language=language,
                    file_path=file_path,
                    line_count=line_count)

        # ── Add Import nodes and IMPORTS relationships ───────────────
        for imp in file_data.get("imports", []):
            imported_module = imp.get("module", "")
            if not imported_module:
                continue

            import_id = f"module:{imported_module}"
            if not G.has_node(import_id):
                G.add_node(import_id,
                           entity_type="Module",
                           name=imported_module,
                           language="external",
                           file_path="",
                           line_count=0)

            G.add_edge(module_id, import_id,
                       relationship="IMPORTS",
                       line=imp.get("line", 0))

            # Add specific imported names
            for name in imp.get("names", []):
                name_id = f"import:{imported_module}.{name}"
                if not G.has_node(name_id):
                    G.add_node(name_id,
                               entity_type="Import",
                               name=name,
                               source_module=imported_module)
                G.add_edge(module_id, name_id,
                           relationship="IMPORTS",
                           line=imp.get("line", 0))

        # ── Add Class nodes ──────────────────────────────────────────
        for cls in file_data.get("classes", []):
            class_name = cls.get("name", "")
            if not class_name:
                continue

            class_id = f"class:{module_name}.{class_name}"
            G.add_node(class_id,
                       entity_type="Class",
                       name=class_name,
                       module=module_name,
                       line_start=cls.get("line_start", 0),
                       line_end=cls.get("line_end", 0),
                       docstring=cls.get("docstring", ""))

            # Module CONTAINS Class
            G.add_edge(module_id, class_id,
                       relationship="CONTAINS")

            # Module DEFINES Class
            G.add_edge(module_id, class_id,
                       relationship="DEFINES")

            # Inheritance: INHERITS_FROM
            for base in cls.get("bases", []):
                if base in ("object", "Exception"):
                    continue  # Skip trivial bases
                # Try to find the base class node, or create a placeholder
                base_id = _find_or_create_class(G, base, parsed_files, module_name)
                G.add_edge(class_id, base_id,
                           relationship="INHERITS_FROM")

            # Add Method nodes (functions inside classes)
            for method in cls.get("methods", []):
                method_name = method.get("name", "")
                if not method_name:
                    continue

                method_id = f"function:{module_name}.{class_name}.{method_name}"
                G.add_node(method_id,
                           entity_type="Function",
                           name=method_name,
                           qualified_name=f"{class_name}.{method_name}",
                           module=module_name,
                           is_method=True,
                           params=method.get("params", []),
                           decorators=method.get("decorators", []),
                           line_start=method.get("line_start", 0),
                           line_end=method.get("line_end", 0),
                           docstring=method.get("docstring", ""))

                # Class CONTAINS Method
                G.add_edge(class_id, method_id,
                           relationship="CONTAINS")

                # Class DEFINES Method
                G.add_edge(class_id, method_id,
                           relationship="DEFINES")

                # Method CALLS other functions
                for call in method.get("calls", []):
                    _add_call_edge(G, method_id, call, module_name, parsed_files)

        # ── Add top-level Function nodes ─────────────────────────────
        for func in file_data.get("functions", []):
            func_name = func.get("name", "")
            if not func_name:
                continue

            func_id = f"function:{module_name}.{func_name}"
            G.add_node(func_id,
                       entity_type="Function",
                       name=func_name,
                       qualified_name=func_name,
                       module=module_name,
                       is_method=False,
                       params=func.get("params", []),
                       decorators=func.get("decorators", []),
                       is_arrow=func.get("is_arrow", False),
                       line_start=func.get("line_start", 0),
                       line_end=func.get("line_end", 0),
                       docstring=func.get("docstring", ""))

            # Module CONTAINS Function
            G.add_edge(module_id, func_id,
                       relationship="CONTAINS")

            # Module DEFINES Function
            G.add_edge(module_id, func_id,
                       relationship="DEFINES")

            # Function CALLS other functions
            for call in func.get("calls", []):
                _add_call_edge(G, func_id, call, module_name, parsed_files)

    return G


def _find_or_create_class(G: nx.DiGraph, class_name: str,
                           parsed_files: list[dict], current_module: str) -> str:
    """Find an existing class node or create a placeholder.

    Searches all parsed files for the class, prioritizing the current module.
    """
    # Check current module first
    candidate = f"class:{current_module}.{class_name}"
    if G.has_node(candidate):
        return candidate

    # Search other modules
    for file_data in parsed_files:
        mod = file_data["module"]
        for cls in file_data.get("classes", []):
            if cls.get("name") == class_name:
                found_id = f"class:{mod}.{class_name}"
                if G.has_node(found_id):
                    return found_id

    # Create placeholder
    placeholder_id = f"class:external.{class_name}"
    if not G.has_node(placeholder_id):
        G.add_node(placeholder_id,
                   entity_type="Class",
                   name=class_name,
                   module="external",
                   line_start=0,
                   line_end=0)
    return placeholder_id


def _add_call_edge(G: nx.DiGraph, caller_id: str, call_text: str,
                   current_module: str, parsed_files: list[dict]):
    """Add a CALLS edge, resolving the target function if possible.

    Handles simple names (func), qualified names (obj.method), and
    self.method calls.
    """
    # Handle self.method calls
    if call_text.startswith("self."):
        method_name = call_text.split(".", 1)[1]
        # Try to find in the same class
        caller_parts = caller_id.split(":")
        if len(caller_parts) >= 2:
            path_parts = caller_parts[1].split(".")
            if len(path_parts) >= 2:
                class_name = path_parts[1] if len(path_parts) >= 3 else path_parts[0]
                module = path_parts[0]
                target_id = f"function:{module}.{class_name}.{method_name}"
                if G.has_node(target_id):
                    G.add_edge(caller_id, target_id, relationship="CALLS")
                    return

    # Handle this.method for JS
    if call_text.startswith("this."):
        method_name = call_text.split(".", 1)[1]
        # Similar resolution as self.method
        call_text = method_name

    # Try to resolve simple function name in current module
    simple_candidates = [
        f"function:{current_module}.{call_text}",
    ]

    # Try all modules
    for file_data in parsed_files:
        mod = file_data["module"]
        simple_candidates.append(f"function:{mod}.{call_text}")
        # Also check class methods
        for cls in file_data.get("classes", []):
            simple_candidates.append(f"function:{mod}.{cls['name']}.{call_text}")

    for candidate in simple_candidates:
        if G.has_node(candidate):
            G.add_edge(caller_id, candidate, relationship="CALLS")
            return

    # If we cannot resolve, create a placeholder function node
    if "." in call_text:
        parts = call_text.rsplit(".", 1)
        func_name = parts[-1]
    else:
        func_name = call_text

    # Skip built-in/common calls
    skip_names = {"print", "len", "str", "int", "float", "list", "dict", "set",
                  "range", "enumerate", "zip", "map", "filter", "sorted", "super",
                  "isinstance", "type", "getattr", "setattr", "hasattr",
                  "console.log", "console.error", "JSON.parse", "JSON.stringify",
                  "Math.max", "Math.min", "Math.floor", "Math.log", "Math.abs",
                  "Object.keys", "Array.isArray"}
    if call_text in skip_names or func_name in skip_names:
        return

    placeholder_id = f"function:external.{func_name}"
    if not G.has_node(placeholder_id):
        G.add_node(placeholder_id,
                   entity_type="Function",
                   name=func_name,
                   module="external",
                   is_method=False)
    G.add_edge(caller_id, placeholder_id, relationship="CALLS")


def print_graph_summary(G: nx.DiGraph):
    """Print a summary of the code knowledge graph."""
    print(f"\nCode Knowledge Graph Summary")
    print(f"{'='*60}")
    print(f"Total nodes: {G.number_of_nodes()}")
    print(f"Total edges: {G.number_of_edges()}")

    # Nodes by type
    type_counts = {}
    for _, data in G.nodes(data=True):
        t = data.get("entity_type", "Unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    print("\nNodes by type:")
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")

    # Edges by relationship
    rel_counts = {}
    for _, _, data in G.edges(data=True):
        r = data.get("relationship", "UNKNOWN")
        rel_counts[r] = rel_counts.get(r, 0) + 1
    print("\nEdges by relationship:")
    for r, c in sorted(rel_counts.items()):
        print(f"  {r}: {c}")

    # Most connected nodes
    print("\nMost connected nodes (by degree):")
    degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    for node_id, degree in degrees[:10]:
        data = G.nodes[node_id]
        print(f"  {data.get('name', node_id)} ({data.get('entity_type', '?')}): {degree} connections")


def main():
    parser = argparse.ArgumentParser(description="Build code knowledge graph from parsed data.")
    parser.add_argument("--python", type=str, default=None, help="Path to parsed Python JSON")
    parser.add_argument("--js", type=str, default=None, help="Path to parsed JavaScript JSON")
    args = parser.parse_args()

    parsed_files = []

    # Load parsed Python data
    py_file = Path(args.python) if args.python else OUTPUT_DIR / "parsed_python.json"
    if py_file.exists():
        print(f"Loading parsed Python data from {py_file}...")
        with open(py_file) as f:
            parsed_files.extend(json.load(f))
    else:
        print(f"Warning: Python parse data not found at {py_file}")
        print("Run 01_parse_python.py first.")

    # Load parsed JavaScript data
    js_file = Path(args.js) if args.js else OUTPUT_DIR / "parsed_javascript.json"
    if js_file.exists():
        print(f"Loading parsed JavaScript data from {js_file}...")
        with open(js_file) as f:
            parsed_files.extend(json.load(f))
    else:
        print(f"Warning: JavaScript parse data not found at {js_file}")
        print("Run 02_parse_javascript.py first.")

    if not parsed_files:
        print("ERROR: No parsed data found. Run the parsers first.")
        sys.exit(1)

    print(f"\nBuilding code knowledge graph from {len(parsed_files)} files...")
    G = create_code_kg(parsed_files)

    print_graph_summary(G)

    # Save as GraphML for interoperability
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    graphml_file = OUTPUT_DIR / "code_kg.graphml"
    nx.write_graphml(G, str(graphml_file))
    print(f"\nSaved GraphML to {graphml_file}")

    # Save as JSON (node-link format) for programmatic use
    json_data = nx.node_link_data(G)
    json_file = OUTPUT_DIR / "code_kg.json"
    with open(json_file, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"Saved JSON to {json_file}")

    # Save adjacency list for quick reference
    adj_file = OUTPUT_DIR / "code_kg_adjacency.json"
    adjacency = {}
    for node_id in G.nodes():
        node_data = G.nodes[node_id]
        neighbors = []
        for _, target, edge_data in G.out_edges(node_id, data=True):
            target_data = G.nodes[target]
            neighbors.append({
                "target": target,
                "name": target_data.get("name", ""),
                "type": target_data.get("entity_type", ""),
                "relationship": edge_data.get("relationship", ""),
            })
        adjacency[node_id] = {
            "data": {k: v for k, v in node_data.items()},
            "edges": neighbors,
        }
    with open(adj_file, "w") as f:
        json.dump(adjacency, f, indent=2, default=str)
    print(f"Saved adjacency list to {adj_file}")


if __name__ == "__main__":
    main()
