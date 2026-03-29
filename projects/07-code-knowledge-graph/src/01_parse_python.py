"""Parse Python files using tree-sitter to extract code structure.

Extracts: modules, imports, class definitions (name, bases, methods),
function definitions (name, params, decorators), and function calls.
Outputs structured JSON for knowledge graph construction.

Usage:
    python src/01_parse_python.py
    python src/01_parse_python.py --input data/sample_repos/calculator.py
    python src/01_parse_python.py --dir data/sample_repos/

References:
    - tree-sitter Python bindings: https://github.com/tree-sitter/py-tree-sitter
    - tree-sitter-python grammar: https://github.com/tree-sitter/tree-sitter-python
    - LangChain code analysis: https://python.langchain.com/docs/integrations/document_loaders/source_code/
"""

import argparse
import json
import sys
from pathlib import Path

# ── Shared imports ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "sample_repos"
OUTPUT_DIR = PROJECT_DIR / "output"

# ── Parser setup ─────────────────────────────────────────────────────────
PY_LANGUAGE = Language(tspython.language())


def create_parser() -> Parser:
    """Create a tree-sitter parser for Python."""
    parser = Parser(PY_LANGUAGE)
    return parser


def get_node_text(node, source_code: bytes) -> str:
    """Extract the text content of a tree-sitter node."""
    return source_code[node.start_byte:node.end_byte].decode("utf-8")


def extract_imports(tree, source_code: bytes) -> list[dict]:
    """Extract all import statements from the syntax tree.

    Handles both 'import x' and 'from x import y' forms.
    """
    imports = []

    def visit(node):
        if node.type == "import_statement":
            # import x, y, z
            for child in node.children:
                if child.type == "dotted_name":
                    imports.append({
                        "type": "import",
                        "module": get_node_text(child, source_code),
                        "names": [],
                        "line": node.start_point[0] + 1,
                    })
                elif child.type == "aliased_import":
                    name_node = child.children[0]
                    imports.append({
                        "type": "import",
                        "module": get_node_text(name_node, source_code),
                        "names": [],
                        "line": node.start_point[0] + 1,
                    })

        elif node.type == "import_from_statement":
            # from x import y, z
            module_name = ""
            names = []
            for child in node.children:
                if child.type == "dotted_name" or child.type == "relative_import":
                    module_name = get_node_text(child, source_code)
                elif child.type == "import_prefix":
                    module_name = get_node_text(child, source_code)
                elif child.type in ("dotted_name", "identifier") and module_name:
                    names.append(get_node_text(child, source_code))
                elif child.type == "aliased_import":
                    name_node = child.children[0]
                    names.append(get_node_text(name_node, source_code))

            # Re-parse imported names from the full text if needed
            if not names:
                full_text = get_node_text(node, source_code)
                if "import" in full_text:
                    import_part = full_text.split("import", 1)[1].strip()
                    names = [n.strip().split(" as ")[0].strip()
                             for n in import_part.split(",")]

            imports.append({
                "type": "from_import",
                "module": module_name,
                "names": names,
                "line": node.start_point[0] + 1,
            })

        for child in node.children:
            visit(child)

    visit(tree.root_node)
    return imports


def extract_classes(tree, source_code: bytes) -> list[dict]:
    """Extract class definitions including bases, methods, and decorators."""
    classes = []

    def visit(node, parent_class=None):
        if node.type == "class_definition":
            class_info = {
                "name": "",
                "bases": [],
                "methods": [],
                "decorators": [],
                "line_start": node.start_point[0] + 1,
                "line_end": node.end_point[0] + 1,
                "docstring": None,
            }

            for child in node.children:
                if child.type == "identifier":
                    class_info["name"] = get_node_text(child, source_code)
                elif child.type == "argument_list":
                    # Base classes
                    for arg in child.children:
                        if arg.type in ("identifier", "dotted_name"):
                            class_info["bases"].append(get_node_text(arg, source_code))
                elif child.type == "block":
                    # Extract methods and docstring
                    for block_child in child.children:
                        if block_child.type == "expression_statement":
                            # Check for docstring
                            for expr_child in block_child.children:
                                if expr_child.type == "string" and not class_info["docstring"]:
                                    class_info["docstring"] = get_node_text(expr_child, source_code).strip('"""').strip("'''").strip()
                        elif block_child.type == "decorated_definition":
                            method = extract_function_node(block_child, source_code, is_method=True)
                            if method:
                                class_info["methods"].append(method)
                        elif block_child.type == "function_definition":
                            method = extract_function_node(block_child, source_code, is_method=True)
                            if method:
                                class_info["methods"].append(method)

            # Check for class decorators
            prev = node.prev_named_sibling
            if prev and prev.type == "decorator":
                class_info["decorators"].append(get_node_text(prev, source_code).lstrip("@"))

            classes.append(class_info)

        for child in node.children:
            visit(child)

    visit(tree.root_node)
    return classes


def extract_function_node(node, source_code: bytes, is_method: bool = False) -> dict | None:
    """Extract function/method definition from a tree-sitter node."""
    func_info = {
        "name": "",
        "params": [],
        "decorators": [],
        "is_method": is_method,
        "line_start": node.start_point[0] + 1,
        "line_end": node.end_point[0] + 1,
        "docstring": None,
        "calls": [],
    }

    # Handle decorated definitions
    actual_func = node
    if node.type == "decorated_definition":
        for child in node.children:
            if child.type == "decorator":
                decorator_text = get_node_text(child, source_code).lstrip("@").strip()
                func_info["decorators"].append(decorator_text)
            elif child.type == "function_definition":
                actual_func = child
        func_info["line_start"] = node.start_point[0] + 1

    for child in actual_func.children:
        if child.type == "identifier":
            func_info["name"] = get_node_text(child, source_code)
        elif child.type == "parameters":
            for param in child.children:
                if param.type in ("identifier", "typed_parameter", "default_parameter",
                                  "typed_default_parameter", "list_splat_pattern",
                                  "dictionary_splat_pattern"):
                    param_text = get_node_text(param, source_code)
                    func_info["params"].append(param_text)
        elif child.type == "block":
            # Extract docstring and function calls
            for block_child in child.children:
                if block_child.type == "expression_statement":
                    for expr_child in block_child.children:
                        if expr_child.type == "string" and not func_info["docstring"]:
                            func_info["docstring"] = get_node_text(expr_child, source_code).strip('"""').strip("'''").strip()

            # Extract function calls within the body
            func_info["calls"] = extract_calls_in_node(child, source_code)

    return func_info if func_info["name"] else None


def extract_functions(tree, source_code: bytes) -> list[dict]:
    """Extract top-level function definitions (not methods inside classes)."""
    functions = []

    for child in tree.root_node.children:
        if child.type == "function_definition":
            func = extract_function_node(child, source_code)
            if func:
                functions.append(func)
        elif child.type == "decorated_definition":
            func = extract_function_node(child, source_code)
            if func:
                functions.append(func)

    return functions


def extract_calls_in_node(node, source_code: bytes) -> list[str]:
    """Recursively extract all function call names within a node."""
    calls = []

    def visit(n):
        if n.type == "call":
            func_node = n.children[0] if n.children else None
            if func_node:
                call_text = get_node_text(func_node, source_code)
                # Simplify attribute calls: obj.method -> method
                if "." in call_text:
                    parts = call_text.split(".")
                    calls.append(call_text)  # Full qualified name
                else:
                    calls.append(call_text)

        for child in n.children:
            visit(child)

    visit(node)
    return list(set(calls))  # Deduplicate


def parse_python_file(file_path: Path) -> dict:
    """Parse a Python file and extract all code entities.

    Args:
        file_path: Path to the Python file.

    Returns:
        Dict with module name, imports, classes, functions, and metadata.
    """
    source_code = file_path.read_bytes()
    parser = create_parser()
    tree = parser.parse(source_code)

    module_name = file_path.stem

    result = {
        "module": module_name,
        "file_path": str(file_path),
        "language": "python",
        "imports": extract_imports(tree, source_code),
        "classes": extract_classes(tree, source_code),
        "functions": extract_functions(tree, source_code),
        "line_count": source_code.count(b"\n") + 1,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Parse Python files with tree-sitter.")
    parser.add_argument("--input", type=str, default=None, help="Path to a single Python file")
    parser.add_argument("--dir", type=str, default=None, help="Directory of Python files to parse")
    args = parser.parse_args()

    files_to_parse = []

    if args.input:
        files_to_parse.append(Path(args.input))
    elif args.dir:
        dir_path = Path(args.dir)
        files_to_parse.extend(dir_path.glob("*.py"))
    else:
        # Default: parse sample repos
        files_to_parse.extend(DATA_DIR.glob("*.py"))

    all_results = []
    for file_path in sorted(files_to_parse):
        print(f"\nParsing: {file_path}")
        result = parse_python_file(file_path)
        all_results.append(result)

        # Print summary
        print(f"  Module: {result['module']}")
        print(f"  Imports: {len(result['imports'])}")
        print(f"  Classes: {len(result['classes'])}")
        for cls in result["classes"]:
            print(f"    - {cls['name']} (bases: {cls['bases']}, methods: {len(cls['methods'])})")
        print(f"  Functions: {len(result['functions'])}")
        for func in result["functions"]:
            print(f"    - {func['name']}({', '.join(func['params'][:3])}{'...' if len(func['params']) > 3 else ''})")
            if func["calls"]:
                print(f"      Calls: {', '.join(func['calls'][:5])}")

    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "parsed_python.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Parsed {len(all_results)} Python files")
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
