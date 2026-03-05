from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from context.context_builder import ContextBuilder
from graph import GraphBuilder
from ingestion import (
    CodeParser,
    DependencyExtractor,
    SymbolExtractor,
    iter_repo,
)
from query.query_engine import QueryEngine


def run(repo_path: str) -> None:
    print(f"Repo: {repo_path}")

    print("\n== Layer 1: Source Files ==")
    source_files = list(iter_repo(repo_path))
    language_counts = Counter(sf.language for sf in source_files)
    print(f"Loaded files: {len(source_files)}")
    for language, count in sorted(language_counts.items()):
        print(f"- {language}: {count}")

    print("\n== Layer 2: AST ==")
    parser = CodeParser()
    parsed_files = []
    parse_errors = 0
    parse_error_examples: list[str] = []
    for source_file in source_files:
        try:
            parsed = parser.parse(source_file)
            parsed_files.append(parsed)
            root = parsed.ast.root_node
            print(
                f"- {parsed.path} [{parsed.language}] root={root.type} "
                f"error={root.has_error}"
            )
        except ValueError as exc:
            parse_errors += 1
            if len(parse_error_examples) < 10:
                parse_error_examples.append(
                    f"- {source_file.path} [{source_file.language}] -> {exc}"
                )
    print(f"Parsed files: {len(parsed_files)}")
    if parse_errors:
        print(f"Skipped (no grammar): {parse_errors}")
        for example in parse_error_examples:
            print(example)

    print("\n== Layer 3: Symbols and Dependencies ==")
    symbol_extractor = SymbolExtractor()
    dependency_extractor = DependencyExtractor()

    symbols = symbol_extractor.extract_many(parsed_files)
    dependencies = dependency_extractor.extract_many(parsed_files)

    print(f"Symbols: {len(symbols)}")
    for symbol in symbols:
        print(
            f"- {symbol.kind} {symbol.name} "
            f"({symbol.file}:{symbol.start_line}-{symbol.end_line})"
        )

    print(f"\nDependencies: {len(dependencies)}")
    for dep in dependencies:
        print(
            f"- {dep.source_symbol} -> {dep.kind} -> {dep.target_symbol} "
            f"({dep.file}:{dep.line})"
        )

    print("\n== Layer 4: Symbol Resolution ==")
    builder = GraphBuilder()
    build_result = builder.build(symbols, dependencies, repo_root=repo_path)
    resolved = build_result.resolved_dependencies
    resolved_count = sum(1 for dep in resolved if dep.target_symbol_id is not None)
    unresolved_count = len(resolved) - resolved_count
    print(f"Resolved dependencies: {resolved_count}/{len(resolved)}")
    print(f"Unresolved (external/ambiguous): {unresolved_count}")
    for dep in resolved:
        print(
            f"- {dep.source_symbol} -> {dep.kind} -> {dep.target_symbol} "
            f"=> {dep.target_symbol_id or 'UNRESOLVED'} [{dep.resolution_reason}]"
        )

    print("\n== Layer 5: Graph ==")
    graph = build_result.graph
    print(f"Graph nodes: {len(graph.nodes)}")
    print(f"Graph edges: {len(graph.edges)}")
    print(f"Module edges: {len(build_result.module_edges)}")
    for module_edge in build_result.module_edges:
        print(
            f"- {module_edge.source_module} -> {module_edge.kind} -> "
            f"{module_edge.target_module}"
        )

    unique_symbol_names: list[str] = []
    seen = set()
    for symbol in symbols:
        if symbol.name in seen:
            continue
        seen.add(symbol.name)
        unique_symbol_names.append(symbol.name)

    for name in unique_symbol_names:
        edges = graph.dependencies_of(name)
        if not edges:
            continue
        print(f"- dependencies_of({name}):")
        for edge in edges:
            print(
                f"  {edge.kind} -> {edge.target.name} "
                f"({edge.target.file}:{edge.target.start_line})"
            )

    for name in unique_symbol_names:
        dependents = graph.dependents_of(name)
        if not dependents:
            continue
        print(f"- dependents_of({name}):")
        for dependent in dependents:
            print(f"  {dependent.name} ({dependent.file}:{dependent.start_line})")

    print("\n== Layer 6: ContextBuilder (GraphBuilder) ==")
    query = QueryEngine(graph)
    context_builder = ContextBuilder(query, repo_root=repo_path)
    focus_symbol = "GraphBuilder"
    focus_module = query.module_of_symbol(focus_symbol) or "<root>"

    symbol_ctx = context_builder.symbol_context(focus_symbol)
    class_ctx = context_builder.class_context(focus_symbol)
    call_chain_ctx = context_builder.call_chain(focus_symbol, depth=2, direction="both")
    impact_ctx = context_builder.impact_context(focus_symbol, depth=3)
    module_ctx = context_builder.module_context(focus_module)

    print("\n[symbol_context]")
    print(json.dumps(symbol_ctx, indent=2))
    print("\n[class_context]")
    print(json.dumps(class_ctx, indent=2))
    print("\n[call_chain]")
    print(json.dumps(call_chain_ctx, indent=2))
    print("\n[impact_context]")
    print(json.dumps(impact_ctx, indent=2))
    print("\n[module_context]")
    print(json.dumps(module_ctx, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AST -> symbols/dependencies -> graph pipeline"
    )
    parser.add_argument(
        "repo_path",
        nargs="?",
        default=str(Path(__file__).resolve().parent),
        help="Path to the repository to analyze (default: this repo)",
    )
    args = parser.parse_args()
    run(args.repo_path)


if __name__ == "__main__":
    main()
