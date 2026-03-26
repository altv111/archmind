from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import is_dataclass
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="archmind",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  archmind index --repo-list repo_list --store archmind.db\n"
            "  archmind update --repo /repos/serviceA --repo /repos/common-lib --store archmind.db\n"
            "  archmind index --repo /repos/serviceA\n"
            "  archmind reset_store --store archmind.db\n"
            "  archmind explain-symbol GraphBuilder --store archmind.db\n"
            "  archmind impact --symbol GraphBuilder --store archmind.db --depth 3\n"
            "  archmind stack-trace --symbol GraphBuilder --store archmind.db --depth 2\n"
            "  archmind pr-risk --base main --head HEAD --store archmind.db\n"
            "  archmind ask --question \"What is the impact if GraphBuilder changes?\" --store archmind.db\n"
            "  archmind ask-agent --question \"What breaks if I change GraphBuilder?\" --store archmind.db --source ollama\n"
            "  archmind ask --question \"Explain GraphBuilder\" --store archmind.db --source ollama"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_index_like_parser(subparsers, "index", "Index repositories.")
    _add_index_like_parser(
        subparsers,
        "update",
        "Update repository index (currently performs a full refresh).",
    )
    _add_store_reset_parser(subparsers, "reset_store", "Reset (wipe) stored index data.")
    _add_query_parser(subparsers)
    _add_generate_context_parser(subparsers)
    _add_explain_symbol_parser(subparsers)
    _add_impact_parser(subparsers)
    _add_stack_trace_parser(subparsers)
    _add_pr_risk_parser(subparsers)
    _add_ask_parser(subparsers)
    _add_ask_agent_parser(subparsers)

    args = parser.parse_args()
    command = args.command

    if command == "reset_store":
        if not args.store:
            raise SystemExit("--store is required for reset_store.")
        reset_store(args.store)
        print(f"Store reset complete: {args.store}")
        return
    if command == "query":
        run_query(args)
        return
    if command == "generate_context":
        run_generate_context(args)
        return
    if command == "explain-symbol":
        run_explain_symbol(args)
        return
    if command == "ask":
        run_ask(args)
        return
    if command == "impact":
        run_impact(args)
        return
    if command == "stack-trace":
        run_stack_trace(args)
        return
    if command == "pr-risk":
        run_pr_risk(args)
        return
    if command == "ask-agent":
        run_ask_agent(args)
        return

    repo_paths = _resolve_repo_paths(args.repo, args.repo_list)
    if not repo_paths:
        raise SystemExit("No repositories provided. Use --repo or --repo-list.")

    result = run_index_pipeline(repo_paths)

    if args.store:
        persist_to_store(
            store_path=args.store,
            command=command,
            repo_paths=repo_paths,
            source_files=result["source_files"],
            parsed_files=result["parsed_files"],
            symbols=result["symbols"],
            graph_edges=result["graph_edges"],
            module_edges=result["module_edges"],
            directory_edges=result["directory_edges"],
        )
        print(f"Stored index in SQLite: {args.store}")
    else:
        out_dir = write_local_artifacts(result, output_root=args.output_dir)
        print(f"Wrote local artifacts to: {out_dir}")


def _add_index_like_parser(
    subparsers: argparse._SubParsersAction, name: str, help_text: str
) -> None:
    parser = subparsers.add_parser(name, help=help_text)
    parser.add_argument(
        "--repo",
        action="append",
        default=[],
        help="Repository path (repeatable).",
    )
    parser.add_argument(
        "--repo-list",
        default=None,
        help="Path to a file with repository paths (one per line, # comments supported).",
    )
    parser.add_argument(
        "--store",
        default=None,
        help="SQLite database path for persistent storage (e.g., archmind.db).",
    )
    parser.add_argument(
        "--output-dir",
        default=".archmind_runs",
        help="Local artifact output root when --store is not provided.",
    )


def _add_store_reset_parser(
    subparsers: argparse._SubParsersAction, name: str, help_text: str
) -> None:
    parser = subparsers.add_parser(name, help=help_text)
    parser.add_argument(
        "--store",
        required=True,
        help="SQLite database path to reset (e.g., archmind.db).",
    )


def _add_query_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("query", help="Query a persisted graph in SQLite.")
    parser.add_argument("--store", required=True, help="SQLite database path.")
    parser.add_argument("--run-id", type=int, default=None, help="Run ID to load (default: latest completed).")
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "symbol_lookup",
            "dependencies",
            "dependents",
            "callers",
            "callees",
            "children",
            "parent",
            "module_dependencies",
            "module_dependents",
            "module_of_symbol",
        ],
        help="Query mode.",
    )
    parser.add_argument("--symbol", default=None, help="Symbol name or symbol_id.")
    parser.add_argument("--module", default=None, help="Module name.")
    parser.add_argument("--repo-root", default=None, help="Optional repo root for source excerpts.")


def _add_generate_context_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "generate_context",
        help="Generate LLM-ready context payload from SQLite-backed graph.",
    )
    parser.add_argument("--store", required=True, help="SQLite database path.")
    parser.add_argument("--run-id", type=int, default=None, help="Run ID to load (default: latest completed).")
    parser.add_argument(
        "--context",
        required=True,
        choices=["symbol_context", "class_context", "module_context", "call_chain", "impact_context", "all"],
        help="Context type.",
    )
    parser.add_argument(
        "--scope",
        default="symbol",
        choices=["symbol", "module", "all"],
        help="Context generation scope.",
    )
    parser.add_argument("--symbol", default=None, help="Symbol name or symbol_id.")
    parser.add_argument("--module", default=None, help="Module name.")
    parser.add_argument("--depth", type=int, default=2, help="Depth for call_chain/impact_context.")
    parser.add_argument(
        "--direction",
        default="out",
        choices=["out", "in", "both"],
        help="Direction for call_chain.",
    )
    parser.add_argument("--repo-root", default=None, help="Optional repo root for source excerpts.")
    parser.add_argument("--out", default=None, help="Optional output path for JSON.")
    parser.add_argument("--out-dir", default=None, help="Optional output directory for batch context files.")
    parser.add_argument(
        "--kinds",
        default=None,
        help="Comma-separated symbol kinds filter for batch scope (e.g., class,function).",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=200,
        help="Maximum symbols to include for batch generation.",
    )


def _add_explain_symbol_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "explain-symbol",
        help="Print a concise symbol summary (methods, dependencies, callers).",
    )
    parser.add_argument("symbol", help="Symbol name or symbol_id.")
    parser.add_argument(
        "--store",
        default="archmind.db",
        help="SQLite database path (default: archmind.db).",
    )
    parser.add_argument("--run-id", type=int, default=None, help="Run ID to load (default: latest completed).")
    parser.add_argument("--repo-root", default=None, help="Optional repo root for source lookups.")


def _add_ask_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "ask",
        help="Run natural-language query planning/execution on SQLite-backed graph.",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Natural language question.",
    )
    parser.add_argument(
        "--store",
        default="archmind.db",
        help="SQLite database path (default: archmind.db).",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        default=None,
        help="Run ID to load (default: latest completed).",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Optional repo root for richer source snippets in contexts.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output JSON path.",
    )
    parser.add_argument(
        "--source",
        default="archmind",
        choices=["archmind", "ollama", "ollamal", "gemini", "openai"],
        help="Answer source: heuristic-only archmind mode, or Ollama-backed answer generation.",
    )
    parser.add_argument(
        "--model",
        default="llama3:8b",
        help="LLM model name when --source is ollama/ollamal.",
    )
    parser.add_argument(
        "--host",
        default="http://127.0.0.1:11434",
        help="LLM host override (Ollama or Gemini API host).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional API key (used by Gemini/OpenAI sources; otherwise env var can be used).",
    )
    parser.add_argument(
        "--llm-timeout",
        type=int,
        default=600,
        help="LLM HTTP timeout in seconds (default: 600).",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream final Ollama answer tokens to stderr while generating.",
    )


def _add_ask_agent_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "ask-agent",
        help="Agentic loop over tool registry (query + context tools).",
    )
    parser.add_argument("--question", required=True, help="Natural language question.")
    parser.add_argument("--store", default="archmind.db", help="SQLite database path (default: archmind.db).")
    parser.add_argument("--run-id", type=int, default=None, help="Run ID to load (default: latest completed).")
    parser.add_argument("--repo-root", default=None, help="Optional repo root for source lookups.")
    parser.add_argument(
        "--mode",
        default="general",
        choices=["auto", "general", "pr_review"],
        help="Agent mode. `pr_review` uses PR-focused planning/format.",
    )
    parser.add_argument("--base", default="main", help="Default PR base ref for pr_review mode.")
    parser.add_argument("--head", default="HEAD", help="Default PR head ref for pr_review mode.")
    parser.add_argument(
        "--pr-review-strategy",
        default="classic",
        choices=["classic", "windowed"],
        help="PR review strategy: `classic` keeps current flow, `windowed` runs per-symbol deep review windows.",
    )
    parser.add_argument(
        "--source",
        default="ollama",
        choices=["ollama", "ollamal", "gemini", "openai"],
        help="LLM source for agent loop.",
    )
    parser.add_argument("--model", default="llama3:8b", help="LLM model name.")
    parser.add_argument(
        "--host",
        default="http://127.0.0.1:11434",
        help="LLM host override (Ollama or Gemini API host).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional API key (used by Gemini/OpenAI sources; otherwise env var can be used).",
    )
    parser.add_argument("--llm-timeout", type=int, default=600, help="LLM HTTP timeout in seconds.")
    parser.add_argument("--max-steps", type=int, default=6, help="Maximum agent loop steps.")
    parser.add_argument(
        "--budget-chars",
        type=int,
        default=24000,
        help="Maximum cumulative evidence summary chars in loop context.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.75,
        help="Minimum confidence to accept final answer in loop.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print full JSON envelope to stdout (default prints only final answer text).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output JSON path (also writes <out>.answer.txt with plain-text final answer).",
    )


def _add_impact_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "impact",
        help="Cross-repo impact analysis for a symbol.",
    )
    parser.add_argument(
        "--symbol",
        required=True,
        help="Symbol name or symbol_id to analyze.",
    )
    parser.add_argument(
        "--store",
        default="archmind.db",
        help="SQLite database path (default: archmind.db).",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        default=None,
        help="Run ID to load (default: latest completed).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Caller traversal depth for impact propagation.",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Optional repo root for richer source snippets in context.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output JSON path.",
    )


def _add_pr_risk_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "pr-risk",
        help="Analyze local PR diff risk by mapping changed lines to symbols and impact.",
    )
    parser.add_argument("--store", default="archmind.db", help="SQLite database path (default: archmind.db).")
    parser.add_argument("--run-id", type=int, default=None, help="Run ID to load (default: latest completed).")
    parser.add_argument("--base", default="main", help="Base git ref for diff (default: main).")
    parser.add_argument("--head", default="HEAD", help="Head git ref for diff (default: HEAD).")
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root to run git diff from (default: current directory).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Caller traversal depth for impact propagation.",
    )
    parser.add_argument(
        "--top-symbol-contexts",
        type=int,
        default=5,
        help="How many top touched symbols to include symbol_context for.",
    )
    parser.add_argument(
        "--top-module-contexts",
        type=int,
        default=3,
        help="How many top affected modules to include module_context for.",
    )
    parser.add_argument(
        "--format",
        choices=["summary", "full"],
        default="summary",
        help="Output format: summary (compact) or full (verbose).",
    )
    parser.add_argument("--out", default=None, help="Optional output JSON path.")


def _add_stack_trace_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "stack-trace",
        help="Show static caller/callee traces for a symbol.",
    )
    parser.add_argument("--symbol", required=True, help="Symbol name or symbol_id.")
    parser.add_argument("--store", default="archmind.db", help="SQLite database path (default: archmind.db).")
    parser.add_argument("--run-id", type=int, default=None, help="Run ID to load (default: latest completed).")
    parser.add_argument("--depth", type=int, default=2, help="Traversal depth for callers/callees.")
    parser.add_argument("--repo-root", default=None, help="Optional repo root for source excerpts.")
    parser.add_argument("--max-lines", type=int, default=10, help="Max source excerpt lines per symbol detail.")
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format: text (default) or json.",
    )
    parser.add_argument("--out", default=None, help="Optional output path.")


def _resolve_repo_paths(repo_args: list[str], repo_list_path: str | None) -> list[Path]:
    paths: list[Path] = []
    for repo in repo_args:
        paths.append(Path(repo).expanduser().resolve())

    if repo_list_path:
        for line in Path(repo_list_path).expanduser().read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            paths.append(Path(stripped).expanduser().resolve())

    unique: list[Path] = []
    seen = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def run_index_pipeline(repo_paths: list[Path]) -> dict:
    from graph import DirectoryGraphBuilder, GraphBuilder
    from ingestion import CodeParser, DependencyExtractor, SymbolExtractor, iter_repo

    source_files = []
    for repo_path in repo_paths:
        source_files.extend(iter_repo(repo_path))

    parser = CodeParser()
    parsed_files = []
    for source_file in source_files:
        try:
            parsed_files.append(parser.parse(source_file))
        except ValueError:
            # No grammar configured for this language.
            continue

    symbol_extractor = SymbolExtractor()
    dependency_extractor = DependencyExtractor()
    symbols = symbol_extractor.extract_many(parsed_files)
    dependencies = dependency_extractor.extract_many(parsed_files)

    builder = GraphBuilder()
    build_result = builder.build(
        symbols=symbols,
        dependencies=dependencies,
        repo_root=_common_repo_root(repo_paths),
    )
    directory_edges = DirectoryGraphBuilder().build(
        (source_file.repo, source_file.path) for source_file in source_files
    )

    return {
        "source_files": source_files,
        "parsed_files": parsed_files,
        "symbols": symbols,
        "dependencies": dependencies,
        "resolved_dependencies": build_result.resolved_dependencies,
        "containment_edges": build_result.containment_edges,
        "module_edges": build_result.module_edges,
        "directory_edges": directory_edges,
        "graph_nodes": build_result.graph.nodes,
        "graph_edges": build_result.graph.edges,
    }


def persist_to_store(
    *,
    store_path: str,
    command: str,
    repo_paths: list[Path],
    source_files: list,
    parsed_files: list,
    symbols: list,
    graph_edges: list,
    module_edges: list,
    directory_edges: list,
) -> None:
    from storage import IndexStore

    store = IndexStore(store_path)
    run = store.start_run(
        notes=f"{command} {' '.join(str(p) for p in repo_paths)}"
    )
    try:
        parsed_by_repo_path = {
            (parsed.repo, parsed.path): parsed for parsed in parsed_files
        }
        for source_file in source_files:
            file_id, _ = store.upsert_file(
                run_id=run.run_id,
                repo=source_file.repo,
                path=source_file.path,
                language=source_file.language,
                content=source_file.content,
            )
            parsed = parsed_by_repo_path.get((source_file.repo, source_file.path))
            if parsed is not None:
                store.store_ast_json(
                    run_id=run.run_id,
                    file_id=file_id,
                    parser_language=parsed.language,
                    ast_payload={
                        "root_type": parsed.ast.root_node.type,
                        "has_error": bool(parsed.ast.root_node.has_error),
                    },
                )

        store.replace_symbols_for_run(run.run_id, symbols)
        store.replace_dependencies_for_run(run.run_id, graph_edges)
        store.replace_module_edges_for_run(run.run_id, module_edges)
        store.replace_directory_edges_for_run(run.run_id, directory_edges)
        store.complete_run(run.run_id, status="completed")
    except Exception:
        store.complete_run(run.run_id, status="failed")
        raise
    finally:
        store.close()


def reset_store(store_path: str) -> None:
    from storage import IndexStore

    store = IndexStore(store_path)
    try:
        store.reset_store()
    finally:
        store.close()


def run_query(args: argparse.Namespace) -> None:
    from query.query_engine import QueryEngine
    from storage import GraphLoader

    loaded = GraphLoader(args.store).load(run_id=args.run_id)
    query = QueryEngine(loaded.graph, repo_root=args.repo_root)

    if args.mode == "symbol_lookup":
        if not args.symbol:
            raise SystemExit("--symbol is required for symbol_lookup.")
        result = [asdict_symbol(s) for s in query.resolve_symbols(args.symbol)]
    elif args.mode == "dependencies":
        if not args.symbol:
            raise SystemExit("--symbol is required for dependencies.")
        result = [asdict_symbol(s) for s in query.dependencies_of(args.symbol)]
    elif args.mode == "dependents":
        if not args.symbol:
            raise SystemExit("--symbol is required for dependents.")
        result = [asdict_symbol(s) for s in query.dependents_of(args.symbol)]
    elif args.mode == "callers":
        if not args.symbol:
            raise SystemExit("--symbol is required for callers.")
        result = [asdict_symbol(s) for s in query.callers_of(args.symbol)]
    elif args.mode == "callees":
        if not args.symbol:
            raise SystemExit("--symbol is required for callees.")
        result = [asdict_symbol(s) for s in query.callees_of(args.symbol)]
    elif args.mode == "children":
        if not args.symbol:
            raise SystemExit("--symbol is required for children.")
        result = [asdict_symbol(s) for s in query.children_of(args.symbol)]
    elif args.mode == "parent":
        if not args.symbol:
            raise SystemExit("--symbol is required for parent.")
        parent = query.parent_of(args.symbol)
        result = asdict_symbol(parent) if parent is not None else None
    elif args.mode == "module_dependencies":
        if not args.module:
            raise SystemExit("--module is required for module_dependencies.")
        result = [asdict(edge) for edge in query.module_dependencies_of(args.module)]
    elif args.mode == "module_dependents":
        if not args.module:
            raise SystemExit("--module is required for module_dependents.")
        result = [asdict(edge) for edge in query.module_dependents_of(args.module)]
    elif args.mode == "module_of_symbol":
        if not args.symbol:
            raise SystemExit("--symbol is required for module_of_symbol.")
        result = query.module_of_symbol(args.symbol)
    else:
        raise SystemExit(f"Unsupported mode: {args.mode}")

    print(json.dumps({"run_id": loaded.run_id, "mode": args.mode, "result": result}, indent=2))


def run_generate_context(args: argparse.Namespace) -> None:
    from context.context_builder import ContextBuilder
    from query.query_engine import QueryEngine
    from storage import GraphLoader

    loaded = GraphLoader(args.store).load(run_id=args.run_id)
    query = QueryEngine(loaded.graph, repo_root=args.repo_root)
    context_builder = ContextBuilder(query)
    contexts = _build_context_payloads(
        context_builder=context_builder,
        query=query,
        context=args.context,
        scope=args.scope,
        symbol=args.symbol,
        module=args.module,
        depth=args.depth,
        direction=args.direction,
        max_symbols=args.max_symbols,
        kinds=args.kinds,
    )

    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, item in enumerate(contexts):
            label = item.get("label", f"context_{idx}")
            filename = f"{idx:04d}_{_safe_filename(label)}.json"
            (out_dir / filename).write_text(json.dumps(item, indent=2), encoding="utf-8")
        print(f"Wrote {len(contexts)} context file(s) to: {out_dir}")
        return

    envelope = {"run_id": loaded.run_id, "count": len(contexts), "results": contexts}
    content = json.dumps(envelope, indent=2)
    if args.out:
        Path(args.out).write_text(content, encoding="utf-8")
        print(f"Wrote context to: {args.out}")
    else:
        print(content)


def run_explain_symbol(args: argparse.Namespace) -> None:
    from query.query_engine import QueryEngine
    from storage import GraphLoader

    loaded = GraphLoader(args.store).load(run_id=args.run_id)
    query = QueryEngine(loaded.graph, repo_root=args.repo_root)

    matches = query.resolve_symbols(args.symbol)
    if not matches:
        raise SystemExit(f"Symbol not found: {args.symbol}")

    symbol = matches[0]
    methods = [
        child
        for child in query.children_of(symbol.symbol_id)
        if child.kind in {"method", "function"}
    ]

    dependency_names = _collect_component_dependencies(query, symbol)
    caller_names = _collect_component_callers(query, symbol)

    print(symbol.name)
    print("-" * len(symbol.name))
    print(f"file: {symbol.file}")
    print(f"kind: {symbol.kind}")
    print("")
    print("Methods:")
    if methods:
        for method in sorted({m.name for m in methods}):
            print(f" - {method}")
    else:
        print(" - (none)")
    print("")
    print("Dependencies:")
    if dependency_names:
        for name in dependency_names:
            print(f" - {name}")
    else:
        print(" - (none)")
    print("")
    print("Called By:")
    if caller_names:
        for name in caller_names:
            print(f" - {name}")
    else:
        print(" - (none)")


def run_impact(args: argparse.Namespace) -> None:
    from context.context_builder import ContextBuilder
    from query import QueryEngine
    from storage import GraphLoader

    loaded = GraphLoader(args.store).load(run_id=args.run_id)
    query = QueryEngine(loaded.graph, repo_root=args.repo_root)
    context_builder = ContextBuilder(query)

    matches = query.resolve_symbols(args.symbol)
    if not matches:
        raise SystemExit(f"Symbol not found: {args.symbol}")

    focus = matches[0]
    warnings: list[str] = []
    if len(matches) > 1:
        warnings.append(
            f"Multiple symbols matched '{args.symbol}'. Using first match: {focus.symbol_id}"
        )

    impacted_by_level = query.impact_by_level(focus.symbol_id, depth=args.depth)
    impacted_flat: list = []
    seen: set[str] = set()
    for _, symbols in sorted(impacted_by_level.items()):
        for symbol in symbols:
            if symbol.symbol_id in seen:
                continue
            seen.add(symbol.symbol_id)
            impacted_flat.append(symbol)

    direct_callees = query.callees_of(focus.symbol_id)

    all_symbols = [focus] + impacted_flat
    affected_files = sorted({symbol.file for symbol in all_symbols if symbol.file})
    affected_repos = sorted({symbol.repo for symbol in all_symbols if symbol.repo})

    result = {
        "focus_symbol": asdict_symbol(focus),
        "depth": args.depth,
        "summary": {
            "impacted_symbols": len(impacted_flat),
            "affected_files": len(affected_files),
            "affected_repos": len(affected_repos),
            "cross_repo": len(affected_repos) > 1,
            "direct_callees": len(direct_callees),
        },
        "impacted_by_level": {
            str(level): [asdict_symbol(symbol) for symbol in symbols]
            for level, symbols in sorted(impacted_by_level.items())
        },
        "direct_callees": [asdict_symbol(symbol) for symbol in direct_callees],
        "affected_files": affected_files,
        "affected_repos": affected_repos,
        "context": context_builder.impact_context(focus.symbol_id, depth=args.depth),
        "warnings": warnings,
    }

    envelope = {
        "run_id": loaded.run_id,
        "symbol_query": args.symbol,
        "result": result,
    }
    content = json.dumps(_json_ready(envelope), indent=2)
    if args.out:
        Path(args.out).write_text(content, encoding="utf-8")
        print(f"Wrote impact analysis to: {args.out}")
        return
    print(content)


def run_stack_trace(args: argparse.Namespace) -> None:
    from query import QueryEngine
    from storage import GraphLoader

    loaded = GraphLoader(args.store).load(run_id=args.run_id)
    query = QueryEngine(loaded.graph, repo_root=args.repo_root)

    matches = query.resolve_symbols(args.symbol)
    if not matches:
        raise SystemExit(f"Symbol not found: {args.symbol}")
    focus = matches[0]
    warnings: list[str] = []
    if len(matches) > 1:
        warnings.append(
            f"Multiple symbols matched '{args.symbol}'. Using first match: {focus.symbol_id}"
        )

    callers = _trace_callers(query, focus.symbol_id, depth=args.depth)
    callees = _trace_callees(query, focus.symbol_id, depth=args.depth)

    detail_ids: set[str] = {focus.symbol_id}
    detail_ids.update(item["symbol_id"] for item in callers)
    detail_ids.update(item["symbol_id"] for item in callees)
    details = {}
    for symbol_id in sorted(detail_ids):
        symbol = query.resolve_symbol(symbol_id)
        if symbol is None:
            continue
        details[symbol_id] = {
            "symbol": asdict_symbol(symbol),
            "signature": query.get_signature(symbol_id),
            "docstring": query.get_docstring(symbol_id),
            "source_excerpt": query.get_source_excerpt(symbol_id, max_lines=args.max_lines),
        }

    payload = {
        "run_id": loaded.run_id,
        "symbol_query": args.symbol,
        "depth": args.depth,
        "focus_symbol": asdict_symbol(focus),
        "callers": callers,
        "callees": callees,
        "details": details,
        "warnings": warnings,
    }

    if args.format == "json":
        content = json.dumps(_json_ready(payload), indent=2)
    else:
        content = _stack_trace_text(payload)

    if args.out:
        Path(args.out).write_text(content, encoding="utf-8")
        print(f"Wrote stack trace to: {args.out}")
        return
    print(content)


def _trace_callers(query, focus_symbol_id: str, depth: int) -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple[str, str, int]] = set()
    queue: list[tuple[str, int]] = [(focus_symbol_id, 0)]

    while queue:
        current_id, level = queue.pop(0)
        if level >= depth:
            continue
        for edge in query.dependent_edges_of(current_id, kind="calls"):
            caller = edge.source
            key = (caller.symbol_id, current_id, level + 1)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "depth": level + 1,
                    "symbol_id": caller.symbol_id,
                    "name": caller.name,
                    "kind": caller.kind,
                    "file": caller.file,
                    "line": caller.start_line,
                    "calls": current_id,
                }
            )
            queue.append((caller.symbol_id, level + 1))
    return rows


def _trace_callees(query, focus_symbol_id: str, depth: int) -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple[str, str, int]] = set()
    queue: list[tuple[str, int]] = [(focus_symbol_id, 0)]

    while queue:
        current_id, level = queue.pop(0)
        if level >= depth:
            continue
        for edge in query.dependency_edges_of(current_id, kind="calls"):
            callee = edge.target
            key = (current_id, callee.symbol_id, level + 1)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "depth": level + 1,
                    "symbol_id": callee.symbol_id,
                    "name": callee.name,
                    "kind": callee.kind,
                    "file": callee.file,
                    "line": callee.start_line,
                    "called_from": current_id,
                }
            )
            queue.append((callee.symbol_id, level + 1))
    return rows


def _stack_trace_text(payload: dict) -> str:
    focus = payload["focus_symbol"]
    lines = []
    lines.append(f"Static Stack Trace for {focus['name']}")
    lines.append("=" * (24 + len(focus["name"])))
    lines.append(f"Focus: {focus['name']} ({focus['kind']})")
    lines.append(f"File: {focus['file']}:{focus['start_line']}")
    lines.append("")

    lines.append("Callers (all possible static callers):")
    if payload["callers"]:
        for item in payload["callers"]:
            lines.append(
                f"  [depth {item['depth']}] {item['name']} ({item['kind']}) at {item['file']}:{item['line']}"
            )
    else:
        lines.append("  (none)")
    lines.append("")

    lines.append("Callees (all possible static callees):")
    if payload["callees"]:
        for item in payload["callees"]:
            lines.append(
                f"  [depth {item['depth']}] {item['name']} ({item['kind']}) at {item['file']}:{item['line']}"
            )
    else:
        lines.append("  (none)")
    lines.append("")

    lines.append("Symbol Details:")
    for symbol_id, item in payload["details"].items():
        symbol = item["symbol"]
        lines.append(
            f"- {symbol['name']} [{symbol['kind']}] {symbol['file']}:{symbol['start_line']}"
        )
        if item["signature"]:
            lines.append(f"  signature: {item['signature']}")
        if item["docstring"]:
            lines.append(f"  docstring: {item['docstring']}")
        if item["source_excerpt"]:
            lines.append("  source:")
            for src_line in str(item["source_excerpt"]).splitlines():
                lines.append(f"    {src_line}")
    if payload["warnings"]:
        lines.append("")
        lines.append("Warnings:")
        for warning in payload["warnings"]:
            lines.append(f"- {warning}")
    return "\n".join(lines)


def run_pr_risk(args: argparse.Namespace) -> None:
    from context.context_builder import ContextBuilder
    from query import QueryEngine
    from storage import GraphLoader

    loaded = GraphLoader(args.store).load(run_id=args.run_id)
    query = QueryEngine(loaded.graph, repo_root=args.repo_root)
    context_builder = ContextBuilder(query)

    changed_lines = _git_changed_lines(repo_root=args.repo_root, base=args.base, head=args.head)
    primary_touched_symbols, container_symbols, all_touched_symbols = _partition_symbols_touched_by_diff(
        query, changed_lines
    )

    per_symbol: list[dict] = []
    affected_symbol_ids: set[str] = set()
    affected_files: set[str] = set()
    affected_repos: set[str] = set()
    module_hit_counts: dict[str, int] = defaultdict(int)

    for symbol in primary_touched_symbols:
        impacted_by_level = query.impact_by_level(symbol.symbol_id, depth=args.depth)
        call_chain = query.call_chain(symbol.symbol_id, depth=2, direction="both")
        impacted_flat = []
        seen_local: set[str] = set()
        for _, symbols in sorted(impacted_by_level.items()):
            for impacted in symbols:
                if impacted.symbol_id in seen_local:
                    continue
                seen_local.add(impacted.symbol_id)
                impacted_flat.append(impacted)

        total_impacted = len(impacted_flat)
        direct_callers = len(query.callers_of(symbol.symbol_id))
        direct_callees = len(query.callees_of(symbol.symbol_id))
        base_risk_score = total_impacted * 2 + direct_callers + direct_callees
        risk_weight = _pr_risk_weight(symbol)
        risk_score = round(base_risk_score * risk_weight, 2)

        for item in [symbol] + impacted_flat:
            affected_symbol_ids.add(item.symbol_id)
            if item.file:
                affected_files.add(item.file)
                module = query.module_of_symbol(item.symbol_id)
                if module:
                    module_hit_counts[module] += 1
            if item.repo:
                affected_repos.add(item.repo)

        per_symbol.append(
            {
                "symbol": asdict_symbol(symbol),
                "risk_score": risk_score,
                "base_risk_score": base_risk_score,
                "risk_weight": risk_weight,
                "summary": {
                    "impacted_symbols": total_impacted,
                    "direct_callers": direct_callers,
                    "direct_callees": direct_callees,
                },
                "impacted_by_level": {
                    str(level): [asdict_symbol(s) for s in symbols]
                    for level, symbols in sorted(impacted_by_level.items())
                },
                "call_chain": call_chain,
            }
        )

    per_symbol.sort(key=lambda item: item["risk_score"], reverse=True)
    top_touched = per_symbol[: args.top_symbol_contexts]

    top_modules = [
        module for module, _ in sorted(module_hit_counts.items(), key=lambda kv: kv[1], reverse=True)
    ][: args.top_module_contexts]

    symbol_contexts = []
    for item in top_touched:
        symbol_id = item["symbol"]["symbol_id"]
        symbol_contexts.append(
            {
                "symbol_id": symbol_id,
                "context": context_builder.symbol_context(symbol_id),
            }
        )

    module_contexts = []
    for module in top_modules:
        module_contexts.append(
            {
                "module": module,
                "context": context_builder.module_context(module),
            }
        )

    summary = {
        "changed_files": len(changed_lines),
        "touched_symbols": len(primary_touched_symbols),
        "container_symbols": len(container_symbols),
        "affected_symbols": len(affected_symbol_ids),
        "affected_files": len(affected_files),
        "affected_repos": len(affected_repos),
        "cross_repo": len(affected_repos) > 1,
        "risk_level": _pr_risk_level(
            touched_symbols=len(primary_touched_symbols),
            affected_symbols=len(affected_symbol_ids),
            affected_repos=len(affected_repos),
            top_score=per_symbol[0]["risk_score"] if per_symbol else 0,
        ),
    }

    full_result = {
        "base": args.base,
        "head": args.head,
        "repo_root": str(Path(args.repo_root).resolve()),
        "summary": summary,
        "changed_files": {
            file: sorted(lines) for file, lines in sorted(changed_lines.items())
        },
        "touched_symbols": [asdict_symbol(s) for s in primary_touched_symbols],
        "container_symbols": [asdict_symbol(s) for s in container_symbols],
        "all_touched_symbols": [asdict_symbol(s) for s in all_touched_symbols],
        "symbol_impacts": per_symbol,
        "contexts": {
            "top_symbol_contexts": symbol_contexts,
            "top_module_contexts": module_contexts,
        },
    }

    if args.format == "summary":
        top_risky_symbols = [
            {
                "symbol_id": item["symbol"]["symbol_id"],
                "name": item["symbol"]["name"],
                "file": item["symbol"]["file"],
                "risk_score": item["risk_score"],
                "base_risk_score": item["base_risk_score"],
                "risk_weight": item["risk_weight"],
                "impacted_symbols": item["summary"]["impacted_symbols"],
                "direct_callers": item["summary"]["direct_callers"],
                "direct_callees": item["summary"]["direct_callees"],
            }
            for item in per_symbol[:10]
        ]
        top_affected_modules = [
            {"module": module, "hits": hits}
            for module, hits in sorted(module_hit_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
        ]
        result = {
            "base": args.base,
            "head": args.head,
            "repo_root": str(Path(args.repo_root).resolve()),
            "summary": summary,
            "changed_files": sorted(changed_lines),
            "top_risky_symbols": top_risky_symbols,
            "top_affected_modules": top_affected_modules,
            "top_context_symbols": [item["symbol_id"] for item in symbol_contexts],
            "top_context_modules": [item["module"] for item in module_contexts],
        }
    else:
        result = full_result

    envelope = {
        "run_id": loaded.run_id,
        "format": args.format,
        "result": result,
    }
    content = json.dumps(_json_ready(envelope), indent=2)
    if args.out:
        Path(args.out).write_text(content, encoding="utf-8")
        print(f"Wrote PR risk analysis to: {args.out}")
        return
    print(content)


def _git_changed_lines(*, repo_root: str, base: str, head: str) -> dict[str, set[int]]:
    cmd = [
        "git",
        "-C",
        str(Path(repo_root).resolve()),
        "diff",
        "--unified=0",
        "--no-color",
        f"{base}...{head}",
    ]
    try:
        diff_text = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"git diff failed: {exc.output.strip()}")

    changed: dict[str, set[int]] = defaultdict(set)
    current_file: str | None = None
    hunk_re = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")

    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            raw = line[4:].strip()
            if raw == "/dev/null":
                current_file = None
            elif raw.startswith("b/"):
                current_file = raw[2:]
            else:
                current_file = raw
            continue

        if current_file is None:
            continue

        match = hunk_re.match(line)
        if not match:
            continue

        start = int(match.group(1))
        count = int(match.group(2) or "1")
        if count <= 0:
            changed[current_file].add(start)
            continue
        for lineno in range(start, start + count):
            changed[current_file].add(lineno)

    return changed


def _partition_symbols_touched_by_diff(query, changed_lines: dict[str, set[int]]) -> tuple[list, list, list]:
    all_touched = []
    for symbol in query.all_symbols():
        lines = changed_lines.get(symbol.file)
        if not lines:
            continue
        if any(symbol.start_line <= line <= symbol.end_line for line in lines):
            all_touched.append(symbol)

    all_touched.sort(key=lambda s: (s.file, s.start_line, s.end_line, s.symbol_id))
    touched_by_file: dict[str, list] = defaultdict(list)
    by_id: dict[str, object] = {}
    for symbol in all_touched:
        touched_by_file[symbol.file].append(symbol)
        by_id[symbol.symbol_id] = symbol

    primary_ids: set[str] = set()
    for file_path, lines in changed_lines.items():
        candidates = touched_by_file.get(file_path, [])
        if not candidates:
            continue
        for line in lines:
            overlaps = [s for s in candidates if s.start_line <= line <= s.end_line]
            if not overlaps:
                continue
            overlaps.sort(
                key=lambda s: (
                    (s.end_line - s.start_line),
                    -s.start_line,
                    s.symbol_id,
                )
            )
            primary_ids.add(overlaps[0].symbol_id)

    primary = [by_id[sid] for sid in sorted(primary_ids) if sid in by_id]
    primary.sort(key=lambda s: (s.file, s.start_line, s.end_line, s.symbol_id))
    containers = [s for s in all_touched if s.symbol_id not in primary_ids]
    return primary, containers, all_touched


def _pr_risk_weight(symbol: object) -> float:
    path = str(getattr(symbol, "file", "") or "").replace("\\", "/").lower()
    kind = str(getattr(symbol, "kind", "") or "").lower()

    weight = 1.0
    if _is_test_path(path):
        weight *= 0.25
    elif _is_ancillary_path(path):
        weight *= 0.5

    if kind in {"class", "module", "namespace", "interface", "enum"}:
        weight *= 0.55
    return max(weight, 0.1)


def _is_test_path(path: str) -> bool:
    filename = path.split("/")[-1] if path else ""
    if "/tests/" in path or "/test/" in path:
        return True
    if ".spec." in filename or ".test." in filename:
        return True
    if filename.startswith("test_") or filename.endswith("_test.py"):
        return True
    return False


def _is_ancillary_path(path: str) -> bool:
    tokens = ("/docs/", "/doc/", "/examples/", "/example/", "/dev/", "/scripts/", "/bench/", "/benchmark/")
    return any(token in path for token in tokens)


def _pr_risk_level(*, touched_symbols: int, affected_symbols: int, affected_repos: int, top_score: int) -> str:
    if affected_repos > 1 or affected_symbols >= 40 or top_score >= 40:
        return "high"
    if touched_symbols >= 5 or affected_symbols >= 15 or top_score >= 15:
        return "medium"
    return "low"


def run_ask(args: argparse.Namespace) -> None:
    from context.context_builder import ContextBuilder
    from query import QueryEngine, QueryOrchestrator
    from storage import GraphLoader

    loaded = GraphLoader(args.store).load(run_id=args.run_id)
    query = QueryEngine(loaded.graph, repo_root=args.repo_root)
    context_builder = ContextBuilder(query)
    llm = _build_llm(
        args.source,
        model=args.model,
        host=args.host,
        timeout=args.llm_timeout,
        api_key=args.api_key,
    )
    orchestrator = QueryOrchestrator(query, context_builder, llm=llm)

    print("[ask] planning query...", file=sys.stderr)
    plan = orchestrator.planner.plan(args.question)
    print(f"[ask] intent: {plan.get('intent')}", file=sys.stderr)
    if "focus_symbol" in plan:
        print(f"[ask] focus_symbol: {plan['focus_symbol']}", file=sys.stderr)
    if "focus_module" in plan:
        print(f"[ask] focus_module: {plan['focus_module']}", file=sys.stderr)

    print("[ask] executing plan...", file=sys.stderr)
    results = orchestrator.executor.execute(plan)

    print("[ask] building context...", file=sys.stderr)
    context = orchestrator._build_context(plan)

    payload = {
        "question": args.question,
        "plan": plan,
        "results": results,
        "context": context,
    }

    if llm and hasattr(llm, "answer"):
        print("[ask] generating LLM answer...", file=sys.stderr)
        if args.stream and args.source in {"ollama", "ollamal"}:
            print("[ask] streaming answer:", file=sys.stderr)

            def _on_token(token: str) -> None:
                print(token, end="", file=sys.stderr, flush=True)

            payload["llm_answer"] = llm.answer(
                args.question,
                context,
                timeout=args.llm_timeout,
                stream=True,
                on_token=_on_token,
            )
            print("", file=sys.stderr)
        else:
            payload["llm_answer"] = llm.answer(
                args.question,
                context,
                timeout=args.llm_timeout,
            )

    envelope = {
        "run_id": loaded.run_id,
        "question": args.question,
        "result": _json_ready(payload),
    }
    content = json.dumps(envelope, indent=2)
    if args.out:
        Path(args.out).write_text(content, encoding="utf-8")
        print(f"Wrote ask result to: {args.out}")
        return
    print(content)


def run_ask_agent(args: argparse.Namespace) -> None:
    from agentic import AgentConfig, AskAgent, ToolExecutor
    from context.context_builder import ContextBuilder
    from query import QueryEngine
    from storage import GraphLoader

    loaded = GraphLoader(args.store).load(run_id=args.run_id)
    query = QueryEngine(loaded.graph, repo_root=args.repo_root)
    context_builder = ContextBuilder(query)
    executor = ToolExecutor(query_engine=query, context_builder=context_builder)
    llm = _build_llm(
        args.source,
        model=args.model,
        host=args.host,
        timeout=args.llm_timeout,
        api_key=args.api_key,
    )
    if llm is None:
        raise SystemExit(f"Unsupported source for ask-agent: {args.source}")

    print("[ask-agent] starting agent loop...", file=sys.stderr)

    def _on_agent_event(event: str, payload: dict) -> None:
        if event == "mode_selected":
            print(f"[ask-agent] mode: {payload.get('mode')}", file=sys.stderr)
            return
        if event == "tool_execute_start":
            tool = payload.get("tool")
            tool_args = payload.get("args")
            print(
                f"[ask-agent] step {payload.get('step')}: calling tool `{tool}` with args={json.dumps(tool_args)}",
                file=sys.stderr,
            )
            return
        if event == "planner_prompt_stats":
            step = payload.get("step")
            chars = payload.get("prompt_chars")
            est = payload.get("prompt_tokens_est")
            print(
                f"[ask-agent] step {step}: planner prompt size chars={chars} est_tokens~{est}",
                file=sys.stderr,
            )
            return
        if event == "planner_llm_usage":
            step = payload.get("step")
            usage = payload.get("usage") or {}
            provider = usage.get("provider", "llm")
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")
            print(
                "[ask-agent] step "
                f"{step}: {provider} usage prompt={prompt_tokens} completion={completion_tokens} total={total_tokens}",
                file=sys.stderr,
            )
            return
        if event == "planner_repair_prompt_stats":
            step = payload.get("step")
            chars = payload.get("prompt_chars")
            est = payload.get("prompt_tokens_est")
            print(
                f"[ask-agent] step {step}: repair prompt size chars={chars} est_tokens~{est}",
                file=sys.stderr,
            )
            return
        if event == "planner_repair_llm_usage":
            step = payload.get("step")
            usage = payload.get("usage") or {}
            provider = usage.get("provider", "llm")
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")
            print(
                "[ask-agent] step "
                f"{step}: repair {provider} usage prompt={prompt_tokens} completion={completion_tokens} total={total_tokens}",
                file=sys.stderr,
            )
            return
        if event == "tool_execute_done":
            print(
                f"[ask-agent] step {payload.get('step')}: tool `{payload.get('tool')}` done (cost={payload.get('cost')})",
                file=sys.stderr,
            )
            return
        if event == "final_answer":
            accepted = payload.get("accepted")
            conf = payload.get("confidence")
            reason = payload.get("reason")
            status = "accepted" if accepted else "rejected_low_confidence"
            if reason:
                status = f"rejected_{reason}"
            print(
                f"[ask-agent] step {payload.get('step')}: final_answer {status} (confidence={conf})",
                file=sys.stderr,
            )
            return
        if event == "final_answer_quality":
            quality = payload.get("quality") or {}
            print(
                f"[ask-agent] step {payload.get('step')}: quality gate {json.dumps(quality)}",
                file=sys.stderr,
            )
            return
        if event == "planner_duplicate_tool_call":
            print(
                f"[ask-agent] step {payload.get('step')}: duplicate tool call skipped for `{payload.get('tool')}`",
                file=sys.stderr,
            )
            return
        if event == "quality_recovery_tool":
            print(
                f"[ask-agent] step {payload.get('step')}: quality recovery tool `{payload.get('tool')}` with args={json.dumps(payload.get('args') or {})}",
                file=sys.stderr,
            )
            return
        if event == "fallback_start":
            reason = payload.get("reason") or "max_steps_reached"
            if reason == "stopped_early_quality_loop":
                print(
                    "[ask-agent] stopping early due to quality-loop protection, generating fallback answer...",
                    file=sys.stderr,
                )
            else:
                print("[ask-agent] max steps reached, generating fallback answer...", file=sys.stderr)
            return
        if event == "fallback_prompt_stats":
            chars = payload.get("prompt_chars")
            est = payload.get("prompt_tokens_est")
            print(
                f"[ask-agent] fallback prompt size chars={chars} est_tokens~{est}",
                file=sys.stderr,
            )
            return
        if event == "fallback_llm_usage":
            usage = payload.get("usage") or {}
            provider = usage.get("provider", "llm")
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")
            print(
                "[ask-agent] fallback "
                f"{provider} usage prompt={prompt_tokens} completion={completion_tokens} total={total_tokens}",
                file=sys.stderr,
            )
            return
        if event == "llm_usage_totals":
            usage = payload.get("usage") or {}
            print(
                "[ask-agent] llm usage totals: "
                f"calls={usage.get('calls')} "
                f"prompt={usage.get('prompt_tokens')} "
                f"completion={usage.get('completion_tokens')} "
                f"total={usage.get('total_tokens')}",
                file=sys.stderr,
            )
            return
        if event == "planner_invalid_action":
            parsed = payload.get("parsed") or {}
            raw = " ".join(str(payload.get("raw") or "").split())
            if len(raw) > 200:
                raw = raw[:197] + "..."
            print(
                f"[ask-agent] step {payload.get('step')}: planner returned invalid action "
                f"(keys={sorted(parsed.keys()) if isinstance(parsed, dict) else []}) raw={raw}",
                file=sys.stderr,
            )
            return
        if event == "tool_execute_error":
            print(
                f"[ask-agent] step {payload.get('step')}: tool `{payload.get('tool')}` error: {payload.get('error')}",
                file=sys.stderr,
            )

    agent = AskAgent(
        llm=llm,
        executor=executor,
        config=AgentConfig(
            max_steps=args.max_steps,
            budget_chars=args.budget_chars,
            confidence_threshold=args.confidence_threshold,
            timeout=args.llm_timeout,
            temperature=0.0,
            mode=args.mode,
            pr_base=args.base,
            pr_head=args.head,
            pr_repo_root=args.repo_root or ".",
            pr_review_strategy=args.pr_review_strategy,
        ),
        on_event=_on_agent_event,
    )
    result = agent.run(args.question)
    answer = str(result.get("answer") or "").strip()

    envelope = {
        "run_id": loaded.run_id,
        "question": args.question,
        "tools": executor.available_tools(),
        "result": _json_ready(result),
    }
    content = json.dumps(envelope, indent=2)
    if args.out:
        Path(args.out).write_text(content, encoding="utf-8")
        answer_out = Path(f"{args.out}.answer.txt")
        answer_out.write_text((answer + "\n") if answer else "", encoding="utf-8")
        print(f"Wrote ask-agent result to: {args.out}")
        print(f"Wrote ask-agent plain-text answer to: {answer_out}")
        return
    if args.debug:
        print(content)
        return
    if answer:
        print(answer)


def _build_llm(source: str, model: str, host: str, timeout: int, api_key: str | None = None):
    if source in {"ollama", "ollamal"}:
        try:
            from llm.ollama_client import OllamaLLM
        except ModuleNotFoundError:
            # Fallback for environments where entrypoint metadata is stale and
            # new package paths are not yet visible to the installed CLI.
            import importlib.util
            import sys

            module_path = Path(__file__).resolve().parent / "llm" / "ollama_client.py"
            spec = importlib.util.spec_from_file_location("ollama_client", module_path)
            if spec is None or spec.loader is None:
                raise
            module = importlib.util.module_from_spec(spec)
            sys.modules["ollama_client"] = module
            spec.loader.exec_module(module)
            OllamaLLM = module.OllamaLLM

        return OllamaLLM(model=model, host=host, timeout=timeout)
    if source == "gemini":
        try:
            from llm.gemini_client import GeminiLLM
        except ModuleNotFoundError:
            import importlib.util
            import sys

            module_path = Path(__file__).resolve().parent / "llm" / "gemini_client.py"
            spec = importlib.util.spec_from_file_location("gemini_client", module_path)
            if spec is None or spec.loader is None:
                raise
            module = importlib.util.module_from_spec(spec)
            sys.modules["gemini_client"] = module
            spec.loader.exec_module(module)
            GeminiLLM = module.GeminiLLM

        resolved_host = host
        if host == "http://127.0.0.1:11434":
            # Keep CLI ergonomic: if user only switches source to gemini,
            # default to Gemini API host automatically.
            resolved_host = "https://generativelanguage.googleapis.com"
        return GeminiLLM(
            model=model,
            api_key=api_key,
            host=resolved_host,
            timeout=timeout,
        )
    if source == "openai":
        try:
            from llm.openai_client import OpenAILLM
        except ModuleNotFoundError:
            import importlib.util
            import sys

            module_path = Path(__file__).resolve().parent / "llm" / "openai_client.py"
            spec = importlib.util.spec_from_file_location("openai_client", module_path)
            if spec is None or spec.loader is None:
                raise
            module = importlib.util.module_from_spec(spec)
            sys.modules["openai_client"] = module
            spec.loader.exec_module(module)
            OpenAILLM = module.OpenAILLM

        resolved_host = host
        if host == "http://127.0.0.1:11434":
            # Keep CLI ergonomic: if user only switches source to openai,
            # default to OpenAI API host automatically.
            resolved_host = "https://api.openai.com"
        resolved_model = model if model != "llama3:8b" else "gpt-4o-mini"
        return OpenAILLM(
            model=resolved_model,
            api_key=api_key,
            host=resolved_host,
            timeout=timeout,
        )
    return None


def _collect_component_dependencies(query, symbol) -> list[str]:
    names: set[str] = set()
    scope_ids = [symbol.symbol_id] + [child.symbol_id for child in query.children_of(symbol.symbol_id)]
    for symbol_id in scope_ids:
        for dep in query.dependency_edges_of(symbol_id):
            if dep.kind not in {"calls", "imports", "inherits"}:
                continue
            component = _component_name(query, dep.target)
            if not component:
                continue
            if component == symbol.name:
                continue
            names.add(component)
    return sorted(names)


def _collect_component_callers(query, symbol) -> list[str]:
    names: set[str] = set()
    scope_ids = [symbol.symbol_id] + [child.symbol_id for child in query.children_of(symbol.symbol_id)]
    for symbol_id in scope_ids:
        for dep in query.dependent_edges_of(symbol_id, kind="calls"):
            component = _component_name(query, dep.source)
            if not component:
                continue
            if component == symbol.name:
                continue
            names.add(component)
    return sorted(names)


def _component_name(query, symbol) -> str | None:
    if symbol.kind == "external":
        if "." in symbol.name:
            receiver = symbol.name.split(".", 1)[0]
            if receiver and receiver[0].isalpha() and receiver[0].isupper():
                return receiver
            return None
        if symbol.name and symbol.name[0].isalpha() and symbol.name[0].isupper():
            return symbol.name
        return None

    current = symbol
    while current.parent:
        parent = query.resolve_symbol(current.parent)
        if parent is None:
            break
        if parent.kind in {"class", "module", "namespace"}:
            current = parent
            break
        current = parent
    return current.name


def asdict_symbol(symbol) -> dict | None:
    if symbol is None:
        return None
    return {
        "symbol_id": symbol.symbol_id,
        "name": symbol.name,
        "kind": symbol.kind,
        "repo": symbol.repo,
        "file": symbol.file,
        "start_line": symbol.start_line,
        "end_line": symbol.end_line,
        "parent": symbol.parent,
    }


def _build_context_payloads(
    *,
    context_builder,
    query,
    context: str,
    scope: str,
    symbol: str | None,
    module: str | None,
    depth: int,
    direction: str,
    max_symbols: int,
    kinds: str | None,
) -> list[dict]:
    allowed_kinds = None
    if kinds:
        allowed_kinds = {k.strip() for k in kinds.split(",") if k.strip()}

    if scope == "symbol":
        if not symbol:
            raise SystemExit("--symbol is required when --scope symbol.")
        return [
            {
                "label": f"symbol_{symbol}",
                "scope": "symbol",
                "symbol": symbol,
                "context": context,
                "payload": _single_context_payload(
                    context_builder, query, context, symbol, depth, direction
                ),
            }
        ]

    if scope == "module":
        if not module:
            raise SystemExit("--module is required when --scope module.")
        symbols = query.symbols_in_module(module)
        if allowed_kinds:
            symbols = [s for s in symbols if s.kind in allowed_kinds]
        symbols = symbols[:max_symbols]
        items: list[dict] = []
        if context in {"module_context", "all"}:
            items.append(
                {
                    "label": f"module_{module}",
                    "scope": "module",
                    "module": module,
                    "context": "module_context",
                    "payload": context_builder.module_context(module),
                }
            )
        if context in {"symbol_context", "class_context", "call_chain", "impact_context", "all"}:
            for sym in symbols:
                items.append(
                    {
                        "label": f"symbol_{sym.name}_{sym.symbol_id}",
                        "scope": "module_symbol",
                        "module": module,
                        "symbol": sym.symbol_id,
                        "context": context,
                        "payload": _single_context_payload(
                            context_builder,
                            query,
                            context,
                            sym.symbol_id,
                            depth,
                            direction,
                        ),
                    }
                )
        return items

    # scope == "all"
    symbols = query.all_symbols()
    if allowed_kinds:
        symbols = [s for s in symbols if s.kind in allowed_kinds]
    symbols = symbols[:max_symbols]

    items: list[dict] = []
    if context in {"module_context", "all"}:
        modules = sorted({query.module_of_symbol(s.symbol_id) for s in symbols if query.module_of_symbol(s.symbol_id)})
        for mod in modules:
            items.append(
                {
                    "label": f"module_{mod}",
                    "scope": "all_modules",
                    "module": mod,
                    "context": "module_context",
                    "payload": context_builder.module_context(mod),
                }
            )

    if context in {"symbol_context", "class_context", "call_chain", "impact_context", "all"}:
        for sym in symbols:
            items.append(
                {
                    "label": f"symbol_{sym.name}_{sym.symbol_id}",
                    "scope": "all_symbols",
                    "symbol": sym.symbol_id,
                    "context": context,
                    "payload": _single_context_payload(
                        context_builder,
                        query,
                        context,
                        sym.symbol_id,
                        depth,
                        direction,
                    ),
                }
            )
    return items


def _single_context_payload(context_builder, query, context: str, symbol: str, depth: int, direction: str):
    if context == "symbol_context":
        return {"symbol_context": context_builder.symbol_context(symbol)}
    if context == "class_context":
        return {"class_context": context_builder.class_context(symbol)}
    if context == "call_chain":
        return {"call_chain": context_builder.call_chain(symbol, depth=depth, direction=direction)}
    if context == "impact_context":
        return {"impact_context": context_builder.impact_context(symbol, depth=depth)}
    if context == "all":
        module = query.module_of_symbol(symbol)
        payload = {
            "symbol_context": context_builder.symbol_context(symbol),
            "class_context": context_builder.class_context(symbol),
            "call_chain": context_builder.call_chain(symbol, depth=depth, direction=direction),
            "impact_context": context_builder.impact_context(symbol, depth=depth),
        }
        if module:
            payload["module_context"] = context_builder.module_context(module)
        return payload
    raise SystemExit(f"Unsupported context: {context}")


def _safe_filename(value: str) -> str:
    cleaned = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_", "."}:
            cleaned.append(ch)
        else:
            cleaned.append("_")
    return "".join(cleaned)[:180]


def _json_ready(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def write_local_artifacts(result: dict, output_root: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(output_root).expanduser().resolve() / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_json(out_dir / "symbols.json", result["symbols"])
    _write_json(out_dir / "dependencies.json", result["dependencies"])
    _write_json(out_dir / "resolved_dependencies.json", result["resolved_dependencies"])
    _write_json(out_dir / "module_edges.json", result["module_edges"])
    _write_json(out_dir / "directory_edges.json", result["directory_edges"])
    _write_json(out_dir / "graph_nodes.json", result["graph_nodes"])
    _write_json(out_dir / "graph_edges.json", result["graph_edges"])

    summary = {
        "source_files": len(result["source_files"]),
        "parsed_files": len(result["parsed_files"]),
        "symbols": len(result["symbols"]),
        "dependencies": len(result["dependencies"]),
        "module_edges": len(result["module_edges"]),
        "directory_edges": len(result["directory_edges"]),
        "graph_nodes": len(result["graph_nodes"]),
        "graph_edges": len(result["graph_edges"]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_dir


def _write_json(path: Path, data: Iterable) -> None:
    serialized = [asdict(item) for item in data]
    path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")


def _common_repo_root(repo_paths: list[Path]) -> Path:
    if len(repo_paths) == 1:
        return repo_paths[0]
    try:
        return Path(Path.commonpath([str(path) for path in repo_paths]))
    except ValueError:
        return Path.cwd()


if __name__ == "__main__":
    main()
