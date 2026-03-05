from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from graph.symbol_resolver import ResolvedDependency, SymbolResolver
from ingestion.code_graph import CodeGraph
from ingestion.dependency_extractor import Dependency
from ingestion.symbol_extractor import Symbol


@dataclass(frozen=True)
class BuildResult:
    graph: CodeGraph
    resolved_dependencies: list[ResolvedDependency]


class GraphBuilder:
    def build(
        self,
        symbols: Iterable[Symbol],
        dependencies: Iterable[Dependency],
    ) -> BuildResult:
        symbol_list = list(symbols)
        dependency_list = list(dependencies)

        resolver = SymbolResolver(symbol_list)
        resolved_dependencies = resolver.resolve_many(dependency_list)
        graph_dependencies = resolver.to_graph_dependencies(resolved_dependencies)
        graph = CodeGraph(symbol_list, graph_dependencies)
        return BuildResult(graph=graph, resolved_dependencies=resolved_dependencies)
