from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from graph.module_graph_builder import ModuleDependency, ModuleGraphBuilder
from graph.symbol_resolver import ResolvedDependency, SymbolResolver
from ingestion.code_graph import CodeGraph
from ingestion.containment_extractor import ContainmentExtractor
from ingestion.dependency_extractor import Dependency
from ingestion.symbol_extractor import Symbol


@dataclass(frozen=True)
class BuildResult:
    graph: CodeGraph
    resolved_dependencies: list[ResolvedDependency]
    containment_edges: list[Dependency]
    module_edges: list[ModuleDependency]


class GraphBuilder:
    def build(
        self,
        symbols: Iterable[Symbol],
        dependencies: Iterable[Dependency],
        repo_root: str | Path | None = None,
    ) -> BuildResult:
        symbol_list = list(symbols)
        dependency_list = list(dependencies)

        resolver = SymbolResolver(symbol_list, repo_root=repo_root)
        resolved_dependencies = resolver.resolve_many(dependency_list)
        graph_dependencies = resolver.to_graph_dependencies(resolved_dependencies)

        containment_extractor = ContainmentExtractor()
        containment_edges = containment_extractor.extract(symbol_list)
        symbol_by_id = {symbol.symbol_id: symbol for symbol in symbol_list}
        containment_dependencies: list[Dependency] = []
        for edge in containment_edges:
            child_symbol = symbol_by_id.get(edge.child_symbol)
            if child_symbol is None:
                continue
            containment_dependencies.append(
                Dependency(
                    source_symbol=edge.parent_symbol,
                    target_symbol=edge.child_symbol,
                    kind=edge.kind,
                    file=child_symbol.file,
                    line=child_symbol.start_line,
                )
            )

        all_edges = graph_dependencies + containment_dependencies
        graph = CodeGraph(symbol_list, all_edges)
        module_edges = ModuleGraphBuilder().build(symbol_list, all_edges)
        graph.module_edges = module_edges
        return BuildResult(
            graph=graph,
            resolved_dependencies=resolved_dependencies,
            containment_edges=containment_dependencies,
            module_edges=module_edges,
        )
