# ArchMind Approach

This document explains the current design of ArchMind: what problem it is solving, how the pipeline works, what assumptions it makes, and where its limits are.

## Problem Statement

ArchMind is built for repository-level architecture understanding over source code, with an emphasis on deterministic static analysis that can also feed structured context into LLM workflows.

The core goal is to answer questions such as:

- What symbols and modules does this code depend on?
- What is the likely blast radius if a symbol changes?
- Which parts of the system are affected by a PR?
- What compact context should be provided to an LLM so it can reason about code structure without ingesting an entire repository?

The system is designed to operate without requiring a build, runtime instrumentation, or framework-specific setup.

## Design Principles

- Prefer static, inspectable intermediate representations over opaque end-to-end prompting.
- Separate ingestion, graph construction, query execution, and LLM-facing context generation into distinct layers.
- Keep storage simple and local with SQLite so indexed runs can be inspected, persisted, and reused.
- Make deterministic workflows first-class; LLMs are optional consumers of structured outputs, not the source of truth.
- Expose both low-level primitives and higher-level workflows so the same indexed graph can support multiple interfaces.

## End-to-End Pipeline

ArchMind processes code in a layered pipeline:

1. Repository loading
   Source files are discovered from one or more repository roots while skipping common tool, build, and virtualenv directories.
2. Parsing
   Files are parsed with Tree-sitter into AST representations.
3. Extraction
   The ingestion layer extracts:
   - symbols
   - containment relationships
   - dependencies such as calls, imports, and inheritance
4. Resolution
   Textual dependency targets are resolved to concrete symbol IDs where possible.
5. Graph construction
   The system builds:
   - symbol-level graphs
   - module-level dependency views
6. Persistence
   Indexed runs and derived artifacts are stored in SQLite so they can be queried later without re-indexing.
7. Query and context generation
   Deterministic query APIs and higher-level context builders expose architecture-oriented views over the graph.
8. Optional LLM orchestration
   LLM-backed planners and tool loops can consume those deterministic outputs to answer natural-language questions.

## Main Layers

### Ingestion

The `ingestion/` package turns source files into structured code facts.

Responsibilities:

- enumerate files from repository roots
- select language-specific parsers
- parse source into ASTs
- extract semantic symbols
- extract raw dependencies
- extract containment relationships

This layer is intentionally language-aware but workflow-agnostic: it produces the raw material used by later graph and query layers.

### Storage

The `storage/` package persists indexed state in SQLite.

The current schema stores:

- index runs
- file metadata and content hashes
- AST artifacts
- symbols
- dependencies
- module edges

Why SQLite:

- local and simple to operate
- easy to inspect manually
- suitable for reloading prior runs
- enough structure for deterministic query workflows

### Graph Construction

The `graph/` package turns extracted facts into navigable graph structures.

Responsibilities:

- resolve raw dependency targets to symbol IDs
- build symbol-level dependency graphs
- derive module-level dependency edges
- expose graph traversal helpers

This is the layer where ArchMind moves from raw syntax-derived observations to a more usable semantic representation.

### Query

The `query/` package provides deterministic access patterns on top of the graph.

Examples:

- symbol lookup
- dependency and dependent queries
- callers and callees
- parent/child containment
- module dependencies and dependents
- docstring and source retrieval helpers

The query layer is also where simple natural-language planning currently lives. The planner maps a user question to a deterministic tool sequence and then assembles a context payload for downstream consumption.

### Context Generation

The `context/` package produces compact, structured payloads intended for LLM reasoning and architecture inspection.

Current context families:

- `symbol_context`
- `class_context`
- `module_context`
- `call_chain`
- `impact_context`

These payloads are meant to be:

- smaller than raw repository dumps
- easier for LLMs to consume reliably
- explicit about facts, focus, and warnings

### Agentic Tooling

The `agentic/` package exposes query and context operations as tools that an LLM can call iteratively.

This supports workflows such as:

- PR review assistance
- multi-step architecture investigation
- focused context gathering before final answer generation

The design intent is that the LLM chooses among bounded, inspectable tools instead of reasoning directly over an unconstrained code dump.

## Why This Structure

The current structure reflects a deliberate split between deterministic analysis and optional LLM usage.

Why this is useful:

- The index can be built once and reused many times.
- Query behavior can be tested independently of LLM behavior.
- Structured contexts provide a stable interface between static analysis and language models.
- PR-risk and impact workflows can stay grounded in graph data rather than free-form model guesses.

This approach is especially useful for architecture questions where traceability matters more than fluent but unverifiable summaries.

## Supported Workflows

The repository currently supports several distinct usage modes:

### Deterministic CLI workflows

- `query`
- `impact`
- `stack-trace`
- `pr-risk`
- `generate_context`

These are the core product surface and should remain usable without an external model.

### Natural-language orchestration

- `ask`
- `ask-agent`

These build on deterministic query/context layers. In the default mode, ArchMind can plan heuristically without a model. With Ollama or Gemini enabled, it can use LLMs for intent extraction, tool planning, and final answer generation.

### Programmatic use

The same graph and context objects can be imported directly from Python for custom pipelines and experiments.

## Tradeoffs And Limitations

ArchMind is intentionally pragmatic and lightweight, which means some tradeoffs are explicit.

### Static analysis limits

ArchMind does not execute code. As a result, it can miss or partially model:

- dynamic dispatch patterns
- runtime imports
- reflection
- metaprogramming
- dependency injection frameworks
- convention-heavy framework wiring

### Resolution accuracy

Dependency resolution is heuristic and language-specific. Results are useful for architectural exploration, but not equivalent to a full compiler or runtime-aware whole-program analysis.

### Language support depth

Python is currently the most developed path. Java and C++ support exist, but the extraction and validation depth is lower and needs more testing.

### Incrementality

The system persists runs and file hashes, but the current `update` path is still effectively a full refresh rather than a fully incremental graph maintenance pipeline.

### LLM dependence is optional, not magical

LLMs can make the interface more convenient, but they do not improve the fidelity of the underlying graph by themselves. Final answer quality is bounded by:

- extracted facts
- resolution quality
- chosen context payloads
- model behavior

## Current Direction

Based on the current repository, ArchMind is best understood as an architecture analysis MVP with a solid deterministic core and expanding workflow surface.

The strongest pieces today are:

- repository indexing into a reusable store
- symbol/module graph construction
- deterministic impact and PR-risk workflows
- structured context generation for LLM consumption

Areas that would most improve the system next:

- better evaluation of extraction and resolution accuracy
- more explicit test coverage, especially across languages
- stronger incremental indexing
- clearer architectural component modeling
- more polished docs separating onboarding, design, and CLI reference

## Relationship To README

The root `README.md` should stay focused on:

- what ArchMind is
- how to install it
- how to run the main workflows
- where to find deeper documentation

This document carries the more detailed explanation of the approach so the README can remain usable as the project entry point.
