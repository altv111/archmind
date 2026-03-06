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


## SQLite Index Store (MVP)

A starter persistence layer is available in `storage/index_store.py`.

It stores:
- index runs (`index_runs`)
- file state + content hash (`files`)
- AST payloads (`ast_artifacts`)
- symbols (`symbols`)
- dependencies (`dependencies`)
- module edges (`module_edges`)

Example:

```python
from storage import IndexStore

store = IndexStore("archmind.db")
run = store.start_run(commit="abc123")

# ...for each file...
file_id, changed = store.upsert_file(
    run_id=run.run_id,
    repo="serviceA",
    path="src/main.py",
    language="python",
    content=source_text,
)

# optional: store AST payload
# store.store_ast_json(run_id=run.run_id, file_id=file_id, parser_language="python", ast_payload=ast_json)

# bulk replace for this run
# store.replace_symbols_for_run(run.run_id, symbols)
# store.replace_dependencies_for_run(run.run_id, dependencies)
# store.replace_module_edges_for_run(run.run_id, module_edges)

store.complete_run(run.run_id)
store.close()
```

## CLI

The package provides a CLI entrypoint: `archmind`.

Index repos and persist to SQLite:

```bash
archmind index --repo-list repo_list --store archmind.db
```

Update index:

```bash
archmind update --repo-list repo_list --store archmind.db
```

If `--store` is omitted, artifacts are written to local filesystem under `.archmind_runs/<timestamp>/` (or `--output-dir`).

`repo_list` format:
- one absolute or relative repo path per line
- lines starting with `#` are ignored

Reset stored index data (keep schema, wipe rows):

```bash
archmind reset_store --store archmind.db
```

Query from SQLite (no re-index):

```bash
archmind query --store archmind.db --mode symbol_lookup --symbol GraphBuilder
archmind query --store archmind.db --mode dependencies --symbol GraphBuilder
archmind query --store archmind.db --mode module_dependencies --module graph
```

Explain a symbol quickly:

```bash
archmind explain-symbol GraphBuilder
```

`explain-symbol` uses `archmind.db` by default; pass `--store <path>` to override.

Run deterministic impact analysis for a symbol:

```bash
archmind impact --symbol GraphBuilder --store archmind.db --depth 3
```

This returns impacted symbols by level, affected files/repos, direct callees, and an impact context payload.

Ask a natural-language architecture question:

```bash
archmind ask --question "What is the impact if GraphBuilder changes?" --store archmind.db
archmind ask --question "Explain GraphBuilder" --store archmind.db
archmind ask --question "Show architecture for module graph" --store archmind.db
```

Use Ollama for final answer generation:

```bash
archmind ask --question "Explain GraphBuilder" --store archmind.db --source ollama
archmind ask --question "Explain GraphBuilder" --store archmind.db --source ollama --llm-timeout 900
archmind ask --question "Explain GraphBuilder" --store archmind.db --source ollama --stream
```

`--source` options:
- `archmind` (default): heuristic planning + structured context only (no LLM answer).
- `ollama` (or alias `ollamal`): uses Ollama for intent detection and symbol/module extraction during planning, and generates final `llm_answer`.

`ask` progress and latency controls:
- progress updates (intent/focus symbol/module) are printed to stderr while the query runs.
- `--llm-timeout <seconds>` increases HTTP timeout for slower local models.
- `--stream` streams final Ollama answer tokens to stderr while generation is in progress.

Generate context from SQLite:

```bash
archmind generate_context --store archmind.db --context class_context --symbol GraphBuilder
archmind generate_context --store archmind.db --context call_chain --symbol GraphBuilder --depth 2 --direction both
```

Generate all context views for one symbol:

```bash
archmind generate_context --store archmind.db --context all --scope symbol --symbol GraphBuilder
```

Generate batch context for one module:

```bash
archmind generate_context --store archmind.db --context all --scope module --module graph --out-dir ./contexts_graph
```

Generate batch context across all symbols/modules in the run:

```bash
archmind generate_context --store archmind.db --context all --scope all --max-symbols 500 --out-dir ./contexts_all
```

Batch filters:
- `--kinds class,function,method` limits symbol kinds in module/all scope.
- `--max-symbols N` bounds batch size.
- `--out-dir <dir>` writes one JSON file per context item.
- `--repo-root <path>` is optional and only needed for richer source excerpts/signatures.

Use a specific index run:

```bash
archmind query --store archmind.db --run-id 42 --mode symbol_lookup --symbol GraphBuilder
```

## Example: Generating Context for LLM Assisted Impact Analysis

Use these commands to build focused context payloads for a symbol/module and then feed them to an LLM.

```bash
archmind generate_context --store archmind.db --context impact_context --scope symbol --symbol "<SYMBOL>" --depth 3 --out impact_<SYMBOL>.json
archmind generate_context --store archmind.db --context call_chain --scope symbol --symbol "<SYMBOL>" --depth 2 --direction both --out call_chain_<SYMBOL>.json
archmind generate_context --store archmind.db --context symbol_context --scope symbol --symbol "<SYMBOL>" --out symbol_<SYMBOL>.json
archmind generate_context --store archmind.db --context module_context --scope module --module "<MODULE>" --out module_<MODULE>.json
```
