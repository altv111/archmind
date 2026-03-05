# ArchMind

ArchMind is a lightweight code intelligence pipeline that:

1. Loads repository source files
2. Parses source into Tree-sitter ASTs
3. Extracts symbols and dependencies
4. Resolves dependency targets to concrete symbols
5. Builds symbol and module graphs
6. Produces LLM-ready context views

## Project Structure

- `ingestion/`
  - `repo_loader.py`: repository file loading
  - `code_parser.py`: Tree-sitter parsing
  - `symbol_extractor.py`: AST -> symbols
  - `dependency_extractor.py`: AST -> dependencies (`calls`, `imports`, `inherits`)
  - `containment_extractor.py`: structural `contains` edges
- `graph/`
  - `symbol_resolver.py`: resolve textual deps to symbol IDs
  - `graph_builder.py`: build final code graph
  - `module_graph_builder.py`: aggregate module-level dependencies
  - `code_graph.py`: graph query surface
- `query/`
  - `query_engine.py`: query helpers on top of graph
- `context/`
  - `context_builder.py`: LLM-oriented context payloads
- `tester.py`
  - end-to-end demo pipeline

## Installation

```bash
python3 -m venv env
source env/bin/activate
pip install -e .
```

## Quickstart

Run pipeline against this repository:

```bash
python tester.py /home/alpha/Workspace/archmind
```

The output is printed by layers:

- Layer 1: source files
- Layer 2: AST parse
- Layer 3: symbols + dependencies
- Layer 4: symbol resolution
- Layer 5: final graph (+ module edges)

## Core Data Types

- `SourceFile` (repo loader)
- `ParsedFile` (AST + source bytes)
- `Symbol` (semantic code symbols)
- `Dependency` (relationship edges)
- `ResolvedDependency` (symbol-linked dependency)

## Context APIs (LLM-Facing)

`ContextBuilder` currently supports:

- `symbol_context(symbol)`
- `class_context(class_symbol)`
- `module_context(module)`
- `call_chain(symbol)`
- `impact_context(symbol)`

These return structured dictionaries with:

- `focus`
- `summary`
- `facts`
- `warnings`

## Notes

- Tree-sitter dependencies are pinned in `setup.py` for compatibility.
- Repository loading skips common build/tool directories and auto-detects Python virtualenv directories.
- Needs more testing for Java and C++ support.

