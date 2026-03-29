"""Parse JavaScript files using tree-sitter to extract code structure.

Extracts: imports, function/class definitions, exports, and function calls.
Outputs structured JSON for knowledge graph construction.

Usage:
    python src/02_parse_javascript.py
    python src/02_parse_javascript.py --input data/sample_repos/app.js

References:
    - tree-sitter JS grammar: https://github.com/tree-sitter/tree-sitter-javascript
    - tree-sitter Python bindings: https://github.com/tree-sitter/py-tree-sitter
    - LangChain code analysis: https://python.langchain.com/docs/integrations/document_loaders/source_code/
"""

import argparse
import json
import sys
from pathlib import Path

# ── Shared imports ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import tree_sitter_javascript as tsjavascript
from tree_sitter import Language, Parser

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "sample_repos"
OUTPUT_DIR = PROJECT_DIR / "output"

# ── Parser setup ─────────────────────────────────────────────────────────
JS_LANGUAGE = Language(tsjavascript.language())


def create_parser() -> Parser:
    """Create a tree-sitter parser for JavaScript."""
    parser = Parser(JS_LANGUAGE)
    return parser


def get_node_text(node, source_code: bytes) -> str:
    """Extract the text content of a tree-sitter node."""
    return source_code[node.start_byte:node.end_byte].decode("utf-8")


def extract_imports(tree, source_code: bytes) -> list[dict]:
    """Extract ES6 import statements.

    Handles:
      - import { a, b } from 'module'
      - import x from 'module'
      - const x = require('module')
    """
    imports = []

    def visit(node):
        if node.type == "import_statement":
            import_info = {
                "type": "es6_import",
                "module": "",
                "names": [],
                "default": None,
                "line": node.start_point[0] + 1,
            }

            for child in node.children:
                if child.type == "string":
                    import_info["module"] = get_node_text(child, source_code).strip("'\"")
                elif child.type == "import_clause":
                    for clause_child in child.children:
                        if clause_child.type == "identifier":
                            import_info["default"] = get_node_text(clause_child, source_code)
                        elif clause_child.type == "named_imports":
                            for spec in clause_child.children:
                                if spec.type == "import_specifier":
                                    name = ""
                                    for spec_child in spec.children:
                                        if spec_child.type == "identifier":
                                            name = get_node_text(spec_child, source_code)
                                            break
                                    if name:
                                        import_info["names"].append(name)

            imports.append(import_info)

        # Handle CommonJS require
        elif node.type == "lexical_declaration" or node.type == "variable_declaration":
            text = get_node_text(node, source_code)
            if "require(" in text:
                import_info = {
                    "type": "commonjs_require",
                    "module": "",
                    "names": [],
                    "default": None,
                    "line": node.start_point[0] + 1,
                }
                # Simple parsing for require
                if "'" in text or '"' in text:
                    start = text.find("require(") + 8
                    end = text.find(")", start)
                    import_info["module"] = text[start:end].strip("'\"")
                imports.append(import_info)

        for child in node.children:
            visit(child)

    visit(tree.root_node)
    return imports


def extract_classes(tree, source_code: bytes) -> list[dict]:
    """Extract JavaScript class definitions."""
    classes = []

    def visit(node):
        if node.type == "class_declaration":
            class_info = {
                "name": "",
                "extends": None,
                "methods": [],
                "line_start": node.start_point[0] + 1,
                "line_end": node.end_point[0] + 1,
            }

            for child in node.children:
                if child.type == "identifier":
                    class_info["name"] = get_node_text(child, source_code)
                elif child.type == "class_heritage":
                    for heritage_child in child.children:
                        if heritage_child.type == "identifier":
                            class_info["extends"] = get_node_text(heritage_child, source_code)
                elif child.type == "class_body":
                    for body_child in child.children:
                        if body_child.type == "method_definition":
                            method = extract_method_node(body_child, source_code)
                            if method:
                                class_info["methods"].append(method)

            classes.append(class_info)

        for child in node.children:
            visit(child)

    visit(tree.root_node)
    return classes


def extract_method_node(node, source_code: bytes) -> dict | None:
    """Extract a method definition from a class body."""
    method_info = {
        "name": "",
        "params": [],
        "is_static": False,
        "is_getter": False,
        "is_setter": False,
        "line_start": node.start_point[0] + 1,
        "line_end": node.end_point[0] + 1,
        "calls": [],
    }

    for child in node.children:
        if child.type == "property_identifier":
            method_info["name"] = get_node_text(child, source_code)
        elif child.type == "formal_parameters":
            for param in child.children:
                if param.type in ("identifier", "assignment_pattern", "rest_pattern"):
                    method_info["params"].append(get_node_text(param, source_code))
        elif child.type == "statement_block":
            method_info["calls"] = extract_calls_in_node(child, source_code)

    # Check for static/get/set modifiers
    text = get_node_text(node, source_code)
    if text.startswith("static "):
        method_info["is_static"] = True
    if text.startswith("get "):
        method_info["is_getter"] = True
    if text.startswith("set "):
        method_info["is_setter"] = True

    return method_info if method_info["name"] else None


def extract_functions(tree, source_code: bytes) -> list[dict]:
    """Extract top-level function declarations and expressions."""
    functions = []

    def visit_top_level(node):
        if node.type == "function_declaration":
            func_info = extract_function_decl(node, source_code)
            if func_info:
                functions.append(func_info)
        elif node.type in ("lexical_declaration", "variable_declaration"):
            # Handle: const myFunc = function() {} or const myFunc = () => {}
            for child in node.children:
                if child.type == "variable_declarator":
                    name_node = None
                    value_node = None
                    for vc in child.children:
                        if vc.type == "identifier":
                            name_node = vc
                        elif vc.type in ("arrow_function", "function"):
                            value_node = vc
                    if name_node and value_node:
                        func_info = {
                            "name": get_node_text(name_node, source_code),
                            "params": [],
                            "is_arrow": value_node.type == "arrow_function",
                            "line_start": node.start_point[0] + 1,
                            "line_end": node.end_point[0] + 1,
                            "calls": [],
                        }
                        for fc in value_node.children:
                            if fc.type == "formal_parameters":
                                for param in fc.children:
                                    if param.type in ("identifier", "assignment_pattern"):
                                        func_info["params"].append(get_node_text(param, source_code))
                            elif fc.type == "statement_block":
                                func_info["calls"] = extract_calls_in_node(fc, source_code)
                        functions.append(func_info)

    for child in tree.root_node.children:
        visit_top_level(child)

    return functions


def extract_function_decl(node, source_code: bytes) -> dict | None:
    """Extract a function declaration node."""
    func_info = {
        "name": "",
        "params": [],
        "is_arrow": False,
        "line_start": node.start_point[0] + 1,
        "line_end": node.end_point[0] + 1,
        "calls": [],
    }

    for child in node.children:
        if child.type == "identifier":
            func_info["name"] = get_node_text(child, source_code)
        elif child.type == "formal_parameters":
            for param in child.children:
                if param.type in ("identifier", "assignment_pattern", "rest_pattern"):
                    func_info["params"].append(get_node_text(param, source_code))
        elif child.type == "statement_block":
            func_info["calls"] = extract_calls_in_node(child, source_code)

    return func_info if func_info["name"] else None


def extract_calls_in_node(node, source_code: bytes) -> list[str]:
    """Recursively extract function call names within a node."""
    calls = []

    def visit(n):
        if n.type == "call_expression":
            func_node = n.children[0] if n.children else None
            if func_node:
                call_text = get_node_text(func_node, source_code)
                calls.append(call_text)

        for child in n.children:
            visit(child)

    visit(node)
    return list(set(calls))


def extract_exports(tree, source_code: bytes) -> list[dict]:
    """Extract export statements."""
    exports = []

    def visit(node):
        if node.type == "export_statement":
            export_info = {
                "type": "export",
                "names": [],
                "default": False,
                "line": node.start_point[0] + 1,
            }

            text = get_node_text(node, source_code)
            if "default" in text:
                export_info["default"] = True

            for child in node.children:
                if child.type == "export_clause":
                    for spec in child.children:
                        if spec.type == "export_specifier":
                            for sc in spec.children:
                                if sc.type == "identifier":
                                    export_info["names"].append(get_node_text(sc, source_code))
                                    break
                elif child.type == "identifier":
                    export_info["names"].append(get_node_text(child, source_code))

            exports.append(export_info)

        for child in node.children:
            visit(child)

    visit(tree.root_node)
    return exports


def parse_javascript_file(file_path: Path) -> dict:
    """Parse a JavaScript file and extract all code entities.

    Args:
        file_path: Path to the JavaScript file.

    Returns:
        Dict with module name, imports, classes, functions, exports, and metadata.
    """
    source_code = file_path.read_bytes()
    parser = create_parser()
    tree = parser.parse(source_code)

    module_name = file_path.stem

    result = {
        "module": module_name,
        "file_path": str(file_path),
        "language": "javascript",
        "imports": extract_imports(tree, source_code),
        "classes": extract_classes(tree, source_code),
        "functions": extract_functions(tree, source_code),
        "exports": extract_exports(tree, source_code),
        "line_count": source_code.count(b"\n") + 1,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Parse JavaScript files with tree-sitter.")
    parser.add_argument("--input", type=str, default=None, help="Path to a single JS file")
    parser.add_argument("--dir", type=str, default=None, help="Directory of JS files to parse")
    args = parser.parse_args()

    files_to_parse = []

    if args.input:
        files_to_parse.append(Path(args.input))
    elif args.dir:
        dir_path = Path(args.dir)
        files_to_parse.extend(dir_path.glob("*.js"))
    else:
        files_to_parse.extend(DATA_DIR.glob("*.js"))

    all_results = []
    for file_path in sorted(files_to_parse):
        print(f"\nParsing: {file_path}")
        result = parse_javascript_file(file_path)
        all_results.append(result)

        # Print summary
        print(f"  Module: {result['module']}")
        print(f"  Imports: {len(result['imports'])}")
        for imp in result["imports"]:
            print(f"    - {imp['module']} ({imp['type']}): {imp.get('names', [])}")
        print(f"  Classes: {len(result['classes'])}")
        for cls in result["classes"]:
            extends = f" extends {cls['extends']}" if cls.get("extends") else ""
            print(f"    - {cls['name']}{extends} ({len(cls['methods'])} methods)")
        print(f"  Functions: {len(result['functions'])}")
        for func in result["functions"]:
            arrow = " (arrow)" if func.get("is_arrow") else ""
            print(f"    - {func['name']}{arrow}({', '.join(func['params'][:3])})")
            if func["calls"]:
                print(f"      Calls: {', '.join(func['calls'][:5])}")
        print(f"  Exports: {len(result.get('exports', []))}")

    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "parsed_javascript.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Parsed {len(all_results)} JavaScript files")
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
