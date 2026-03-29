"""Impact analysis: find the blast radius of changing a function.

Given a function name, traverses the code KG in reverse to find all functions
that directly or transitively depend on it (call it, or call something that
calls it). This is reverse dependency analysis.

Usage:
    python src/06_impact_analysis.py --function "validate_input"
    python src/06_impact_analysis.py --function "format_result" --max_depth 5
    python src/06_impact_analysis.py --function "add" --show_paths

References:
    - NetworkX traversal: https://networkx.org/documentation/stable/reference/algorithms/traversal.html
    - NetworkX shortest paths: https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html
    - LangChain graph queries: https://python.langchain.com/docs/integrations/graphs/
"""

import argparse
import json
import sys
from pathlib import Path
from collections import deque

# ── Shared imports ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import networkx as nx

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"


def find_function_node(G: nx.DiGraph, function_name: str) -> str | None:
    """Find a function node by name (fuzzy matching).

    Tries exact match first, then partial match on function name.

    Args:
        G: The code knowledge graph.
        function_name: Name of the function to find.

    Returns:
        Node ID if found, None otherwise.
    """
    # Exact match on node IDs
    for node_id, data in G.nodes(data=True):
        if data.get("entity_type") == "Function" and data.get("name") == function_name:
            return node_id

    # Partial match
    candidates = []
    for node_id, data in G.nodes(data=True):
        if data.get("entity_type") == "Function":
            name = data.get("name", "")
            if function_name.lower() in name.lower():
                candidates.append(node_id)

    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        print(f"Multiple matches found for '{function_name}':")
        for c in candidates:
            print(f"  - {c} ({G.nodes[c].get('name', '')})")
        print(f"Using first match: {candidates[0]}")
        return candidates[0]

    return None


def find_reverse_callers(G: nx.DiGraph, target_node: str, max_depth: int = 10) -> dict:
    """Find all functions that directly or transitively call the target function.

    Uses BFS on the reverse graph (following CALLS edges backwards).

    Args:
        G: The code knowledge graph.
        target_node: Node ID of the target function.
        max_depth: Maximum traversal depth.

    Returns:
        Dict mapping node_id to depth (distance from target).
    """
    callers = {}  # node_id -> depth
    visited = set()
    queue = deque([(target_node, 0)])

    while queue:
        current, depth = queue.popleft()

        if current in visited:
            continue
        visited.add(current)

        if depth > 0:  # Don't include the target itself
            callers[current] = depth

        if depth >= max_depth:
            continue

        # Find all nodes that have a CALLS edge pointing to current
        for predecessor in G.predecessors(current):
            edge_data = G.edges[predecessor, current]
            if edge_data.get("relationship") == "CALLS" and predecessor not in visited:
                queue.append((predecessor, depth + 1))

    return callers


def find_call_paths(G: nx.DiGraph, source_node: str, target_node: str) -> list[list[str]]:
    """Find all call paths from source to target.

    Args:
        G: The code knowledge graph.
        source_node: Starting function node.
        target_node: Ending function node.

    Returns:
        List of paths, where each path is a list of node IDs.
    """
    # Build a subgraph with only CALLS edges
    calls_edges = [
        (u, v) for u, v, d in G.edges(data=True)
        if d.get("relationship") == "CALLS"
    ]
    calls_graph = nx.DiGraph(calls_edges)

    try:
        paths = list(nx.all_simple_paths(calls_graph, source_node, target_node, cutoff=6))
        return paths
    except (nx.NetworkXError, nx.NodeNotFound):
        return []


def get_direct_callers(G: nx.DiGraph, target_node: str) -> list[dict]:
    """Get immediate callers of a function.

    Args:
        G: The code knowledge graph.
        target_node: Node ID of the target function.

    Returns:
        List of caller info dicts.
    """
    callers = []
    for predecessor in G.predecessors(target_node):
        edge_data = G.edges[predecessor, target_node]
        if edge_data.get("relationship") == "CALLS":
            node_data = G.nodes[predecessor]
            callers.append({
                "node_id": predecessor,
                "name": node_data.get("name", ""),
                "entity_type": node_data.get("entity_type", ""),
                "module": node_data.get("module", ""),
                "line_start": node_data.get("line_start", 0),
            })
    return callers


def get_direct_callees(G: nx.DiGraph, source_node: str) -> list[dict]:
    """Get functions directly called by a function.

    Args:
        G: The code knowledge graph.
        source_node: Node ID of the source function.

    Returns:
        List of callee info dicts.
    """
    callees = []
    for _, successor, edge_data in G.out_edges(source_node, data=True):
        if edge_data.get("relationship") == "CALLS":
            node_data = G.nodes[successor]
            callees.append({
                "node_id": successor,
                "name": node_data.get("name", ""),
                "entity_type": node_data.get("entity_type", ""),
                "module": node_data.get("module", ""),
            })
    return callees


def generate_impact_report(G: nx.DiGraph, function_name: str,
                           max_depth: int = 10, show_paths: bool = False) -> dict:
    """Generate a comprehensive impact analysis report.

    Args:
        G: The code knowledge graph.
        function_name: Name of the function to analyze.
        max_depth: Maximum reverse traversal depth.
        show_paths: Whether to compute and show full call paths.

    Returns:
        Impact analysis report dict.
    """
    # Find the target function
    target_node = find_function_node(G, function_name)
    if not target_node:
        return {"error": f"Function '{function_name}' not found in the knowledge graph."}

    target_data = G.nodes[target_node]
    report = {
        "target_function": {
            "node_id": target_node,
            "name": target_data.get("name", ""),
            "module": target_data.get("module", ""),
            "line_start": target_data.get("line_start", 0),
            "line_end": target_data.get("line_end", 0),
        },
        "direct_callers": get_direct_callers(G, target_node),
        "direct_callees": get_direct_callees(G, target_node),
        "blast_radius": {},
        "impact_by_depth": {},
        "affected_modules": set(),
    }

    # Find all transitive callers (reverse dependencies)
    callers = find_reverse_callers(G, target_node, max_depth=max_depth)

    # Organize by depth
    for node_id, depth in callers.items():
        node_data = G.nodes[node_id]
        caller_info = {
            "node_id": node_id,
            "name": node_data.get("name", ""),
            "entity_type": node_data.get("entity_type", ""),
            "module": node_data.get("module", ""),
            "line_start": node_data.get("line_start", 0),
        }
        report["blast_radius"][node_id] = caller_info

        depth_key = str(depth)
        if depth_key not in report["impact_by_depth"]:
            report["impact_by_depth"][depth_key] = []
        report["impact_by_depth"][depth_key].append(caller_info)

        module = node_data.get("module", "")
        if module and module != "external":
            report["affected_modules"].add(module)

    report["affected_modules"] = sorted(report["affected_modules"])
    report["total_affected_functions"] = len(callers)

    # Optionally compute full call paths
    if show_paths:
        report["call_paths"] = {}
        for caller_id in callers:
            paths = find_call_paths(G, caller_id, target_node)
            if paths:
                path_strs = []
                for path in paths[:3]:  # Limit to 3 paths per caller
                    names = [G.nodes[n].get("name", n) for n in path]
                    path_strs.append(" -> ".join(names))
                report["call_paths"][caller_id] = path_strs

    return report


def print_impact_report(report: dict):
    """Pretty-print the impact analysis report."""
    if "error" in report:
        print(f"\nERROR: {report['error']}")
        return

    target = report["target_function"]
    print(f"\n{'='*60}")
    print(f"  IMPACT ANALYSIS: {target['name']}")
    print(f"{'='*60}")
    print(f"  Module: {target['module']}")
    print(f"  Lines: {target['line_start']}-{target['line_end']}")

    # Direct callees (what it calls)
    print(f"\n  Functions called by {target['name']}:")
    if report["direct_callees"]:
        for callee in report["direct_callees"]:
            print(f"    -> {callee['name']} ({callee['module']})")
    else:
        print(f"    (none)")

    # Direct callers
    print(f"\n  Direct callers ({len(report['direct_callers'])}):")
    if report["direct_callers"]:
        for caller in report["direct_callers"]:
            print(f"    <- {caller['name']} ({caller['module']}, line {caller['line_start']})")
    else:
        print(f"    (none)")

    # Blast radius by depth
    print(f"\n  BLAST RADIUS: {report['total_affected_functions']} functions affected")
    for depth, callers in sorted(report["impact_by_depth"].items(), key=lambda x: int(x[0])):
        distance_label = "direct" if depth == "1" else f"{depth} hops away"
        print(f"\n    Depth {depth} ({distance_label}): {len(callers)} functions")
        for caller in callers:
            print(f"      - {caller['name']} ({caller['module']})")

    # Affected modules
    print(f"\n  Affected modules ({len(report['affected_modules'])}):")
    for module in report["affected_modules"]:
        print(f"    - {module}")

    # Call paths (if computed)
    if "call_paths" in report and report["call_paths"]:
        print(f"\n  Call paths to {target['name']}:")
        for caller_id, paths in list(report["call_paths"].items())[:5]:
            for path_str in paths:
                print(f"    {path_str}")

    print(f"\n{'='*60}")
    print(f"  Summary: Changing '{target['name']}' may affect "
          f"{report['total_affected_functions']} functions across "
          f"{len(report['affected_modules'])} modules.")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Impact analysis for code changes.")
    parser.add_argument("--function", type=str, required=True,
                        help="Name of the function to analyze")
    parser.add_argument("--max_depth", type=int, default=10,
                        help="Maximum reverse traversal depth")
    parser.add_argument("--show_paths", action="store_true",
                        help="Show full call paths to the target function")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to code_kg.json")
    args = parser.parse_args()

    # Load the graph
    input_file = Path(args.input) if args.input else OUTPUT_DIR / "code_kg.json"
    if not input_file.exists():
        print(f"ERROR: Graph file not found at {input_file}")
        print("Run 03_build_code_kg.py first.")
        sys.exit(1)

    print(f"Loading code knowledge graph from {input_file}...")
    with open(input_file) as f:
        json_data = json.load(f)
    G = nx.node_link_graph(json_data)
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Run impact analysis
    report = generate_impact_report(
        G,
        function_name=args.function,
        max_depth=args.max_depth,
        show_paths=args.show_paths,
    )

    print_impact_report(report)

    # Save report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"impact_analysis_{args.function}.json"

    # Convert sets to lists for JSON serialization
    serializable_report = json.loads(json.dumps(report, default=list))
    with open(output_file, "w") as f:
        json.dump(serializable_report, f, indent=2)
    print(f"\nReport saved to {output_file}")


if __name__ == "__main__":
    main()
