# ArchMind

ArchMind is a lightweight code intelligence system for repository-level architecture questions.

It builds a static representation of one or more codebases, stores that representation in SQLite, and exposes deterministic and LLM-oriented workflows for:

- symbol explanation
- impact analysis
- caller/callee tracing
- module dependency inspection
- PR risk analysis
- structured context generation for downstream LLMs

At a high level, ArchMind:

1. Loads repository source files
2. Parses source into Tree-sitter ASTs
3. Extracts symbols, containment, and dependencies
4. Resolves dependency targets to concrete symbols where possible
5. Builds symbol and module graphs
6. Produces query results and LLM-ready context payloads

## What ArchMind Is Good For

Use ArchMind when you want fast, static answers to questions like:

- "What breaks if I change `GraphBuilder`?"
- "Which modules depend on `graph`?"
- "Which symbols were touched by this PR, and what is the likely blast radius?"
- "Give me a structured context payload for this class or module so an LLM can reason over it."

## Documentation

- [Approach](docs/approach.md): design goals, pipeline, tradeoffs, and current architectural direction
- This README: installation, primary workflows, and CLI-oriented usage

## Project Structure

- `ingestion/`
  - `repo_loader.py`: repository file loading
  - `code_parser.py`: Tree-sitter parsing
  - `symbol_extractor.py`: AST -> symbols
  - `dependency_extractor.py`: AST -> dependencies (`calls`, `imports`, `inherits`)
  - `containment_extractor.py`: structural `contains` edges
- `storage/`
  - `index_store.py`: SQLite persistence for runs, files, ASTs, symbols, and edges
  - `graph_loader.py`: load persisted runs back into graph structures
- `graph/`
  - `symbol_resolver.py`: resolve textual dependencies to symbol IDs
  - `graph_builder.py`: build final code graph
  - `module_graph_builder.py`: aggregate module-level dependencies
  - `code_graph.py`: graph query surface
- `query/`
  - `query_engine.py`: deterministic graph queries
  - `query_planner.py`: heuristic or LLM-assisted natural-language planning
- `context/`
  - `context_builder.py`: LLM-oriented context payloads
- `agentic/`
  - `tool_registry.py`: tool specs and registration
  - `tool_executor.py`: unified query/context tool execution with cost hints
  - `ask_agent.py`: iterative tool loop orchestration
- `llm/`
  - `ollama_client.py`: local-model integration
  - `gemini_client.py`: Gemini integration
- `archmind_cli.py`
  - CLI entrypoint
- `tester.py`
  - end-to-end demo pipeline

## Installation

```bash
python3 -m venv env
source env/bin/activate
pip install -e .
```

## Quickstart

Index this repository into SQLite:

```bash
archmind index --repo /home/alpha/Workspace/archmind --store archmind.db
```

Ask a deterministic question against the stored graph:

```bash
archmind ask --question "What is the impact if GraphBuilder changes?" --store archmind.db
```

Run focused impact analysis:

```bash
archmind impact --symbol GraphBuilder --store archmind.db --depth 3
```

Analyze PR risk from a local git diff:

```bash
archmind pr-risk --base main --head HEAD --store archmind.db --format summary
```

Generate structured context for LLM use:

```bash
archmind generate_context --store archmind.db --context all --scope symbol --symbol GraphBuilder
```

## Demo Path

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

`tester.py` is useful for pipeline inspection and debugging. For normal usage, prefer the `archmind` CLI commands above.

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

## Current Capabilities

- Static symbol and dependency extraction for Python, Java, and C++ through Tree-sitter-based parsing
- SQLite-backed index runs with reloadable graph state
- Deterministic query workflows for symbol lookup, dependencies, callers/callees, module dependencies, and impact
- Structured context generation for symbols, classes, modules, call chains, and impact analysis
- PR-risk analysis from local git diffs
- Optional natural-language interfaces via heuristic planning, Ollama, or Gemini

## Limitations

- ArchMind is primarily a static analysis system. Dynamic dispatch, reflection, runtime imports, metaprogramming, and framework-specific wiring may be partially resolved or missed.
- Resolution quality depends on extracted symbol quality and language-specific heuristics.
- Cross-language architectural reasoning is still limited by per-language extractor coverage.
- `update` currently performs a full refresh rather than a fully incremental index update.
- LLM-backed modes improve question understanding and answer presentation, but their reasoning quality is bounded by the underlying static context.

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

Typical workflow:

1. Index one or more repositories into `archmind.db`
2. Run deterministic queries such as `query`, `impact`, `stack-trace`, or `pr-risk`
3. Generate `generate_context` payloads for downstream LLM use
4. Optionally use `ask` or `ask-agent` for natural-language and tool-loop workflows

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

Generate static caller/callee stack trace for a symbol:

```bash
archmind stack-trace --symbol GraphBuilder --store archmind.db --depth 2
archmind stack-trace --symbol GraphBuilder --store archmind.db --depth 2 --format json --out stack_trace.json
```

This shows all possible static callers/callees (depth-bounded) and includes file/line, signature, docstring, and source excerpts.

Analyze PR risk from local git diff:

```bash
archmind pr-risk --base main --head HEAD --store archmind.db --depth 3
archmind pr-risk --base main --head HEAD --store archmind.db --depth 3 --format full --out pr_risk_full.json
```

This maps changed lines to touched symbols, runs per-symbol impact analysis, and returns:
- touched symbols
- per-symbol impact summaries and call chains
- aggregated affected symbols/files/repos
- risk level (`low` / `medium` / `high`)
- targeted symbol/module contexts for LLM-assisted review.

`--format summary` is the default compact output for terminal use.

Deterministic PR analysis (compact):

```bash
archmind pr-risk --base main --head HEAD --store archmind.db --format summary
```

Agentic PR analysis (LLM tool loop):

```bash
export GEMINI_API_KEY="your_key"
archmind ask-agent \
  --question "Analyze PR risk for base=main head=HEAD in repo_root=/home/alpha/Workspace/archmind. Focus on likely breakages, impacted symbols, and affected modules." \
  --store archmind.db \
  --source gemini \
  --model gemini-2.5-flash \
  --max-steps 6 \
  --budget-chars 24000 \
  --llm-timeout 900
```

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
archmind ask --question "Explain GraphBuilder" --store archmind.db --source gemini --model gemini-2.5-flash
```

`--source` options:
- `archmind` (default): heuristic planning + structured context only (no LLM answer).
- `ollama`: uses Ollama for intent detection and symbol/module extraction during planning, and generates final `llm_answer`.
- `gemini`: uses Gemini API for intent/symbol/module extraction and final `llm_answer` (set `GEMINI_API_KEY` or pass `--api-key`).

`ask` progress and latency controls:
- progress updates (intent/focus symbol/module) are printed to stderr while the query runs.
- `--llm-timeout <seconds>` increases HTTP timeout for slower local models.
- `--stream` streams final Ollama answer tokens to stderr while generation is in progress.

Agentic tool loop (LLM selects tools iteratively):

```bash
archmind ask-agent --question "What breaks if I change GraphBuilder?" --store archmind.db --source ollama
archmind ask-agent --question "What breaks if I change GraphBuilder?" --store archmind.db --source gemini --model gemini-2.5-flash
```

Useful controls:
- `--max-steps 6`
- `--budget-chars 24000`
- `--confidence-threshold 0.75`
- `--llm-timeout 900`
- `--mode auto|general|pr_review`

For PR analysis with agentic loop, ask explicitly with refs so the agent can call `pr_diff_context`:

```bash
archmind ask-agent --question "Analyze PR risk for base=main head=HEAD in repo_root=/home/alpha/Workspace/archmind" --store archmind.db --source gemini --model gemini-2.5-flash
archmind ask-agent --mode pr_review --base main --head HEAD --repo-root /home/alpha/Workspace/archmind --question "Is this PR safe to merge?" --store archmind.db --source gemini --model gemini-2.5-flash
```

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

## Agentic Tooling (MVP)

You can use the unified tool executor to expose query + context tools to an LLM planning loop.

```python
from storage import GraphLoader
from query import QueryEngine
from context.context_builder import ContextBuilder
from agentic import ToolExecutor

loaded = GraphLoader("archmind.db").load()
query = QueryEngine(loaded.graph, repo_root="/home/alpha/Workspace/archmind")
context_builder = ContextBuilder(query)
executor = ToolExecutor(query, context_builder)

print(executor.available_tools())  # includes schemas + cost hints
print(executor.execute("symbol_lookup", {"symbol": "GraphBuilder"}))
print(executor.execute("get_full_implementation", {"symbol": "GraphBuilder"}))
```

## Choosing Between Interfaces

- Use `query` when you know exactly which graph relation you want.
- Use `impact` when you care about likely blast radius.
- Use `stack-trace` when you need static caller/callee chains.
- Use `pr-risk` when reviewing a diff against a known base/head.
- Use `generate_context` when another system or LLM needs structured JSON.
- Use `ask` when you want a single natural-language question answered.
- Use `ask-agent` when you want an LLM to choose multiple tools iteratively.
