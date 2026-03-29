# Project 7: Code Knowledge Graph

Parse Python and JavaScript codebases with Tree-sitter, extract functions, classes, imports, and call relationships, then build a code knowledge graph. Enable semantic code search and impact analysis.

## Overview

This project demonstrates how to build a knowledge graph from source code:

1. **Parse** Python files with tree-sitter to extract AST-level entities
2. **Parse** JavaScript files with tree-sitter for multi-language support
3. **Build** a code KG using NetworkX (modules, classes, functions, imports, calls)
4. **Visualize** the graph with pyvis, color-coded by entity type
5. **Search** code semantically using LLM embeddings
6. **Analyze** impact/blast radius of changing any function

## Requirements

- Python 3.11+
- tree-sitter, tree-sitter-python, tree-sitter-javascript
- NetworkX, pyvis
- Shared LLM layer (LangChain-based) for semantic search

```bash
pip install tree-sitter tree-sitter-python tree-sitter-javascript networkx pyvis
```

## Entity Types

| Type | Description | Color (viz) |
|------|-------------|-------------|
| Module | Python/JS files | Blue |
| Class | Class definitions | Green |
| Function | Function/method definitions | Orange |
| Import | Imported modules/names | Purple |

## Relationship Types

| Relationship | Description |
|-------------|-------------|
| IMPORTS | Module imports another module/name |
| DEFINES | Module/class defines a function/class |
| CALLS | Function calls another function |
| INHERITS_FROM | Class extends another class |
| CONTAINS | Module contains classes/functions |

## Pipeline

```bash
# 1. Parse Python files
python src/01_parse_python.py

# 2. Parse JavaScript files
python src/02_parse_javascript.py

# 3. Build the code knowledge graph
python src/03_build_code_kg.py

# 4. Visualize the graph
python src/04_visualize_code_kg.py

# 5. Semantic code search
python src/05_semantic_code_search.py --query "error handling"

# 6. Impact analysis
python src/06_impact_analysis.py --function "calculate"
```

## Sample Data

- `data/sample_repos/calculator.py` -- Python file with classes, functions, imports, inheritance
- `data/sample_repos/utils.py` -- Helper module imported by calculator.py
- `data/sample_repos/app.js` -- JavaScript file with functions, imports, classes
