from __future__ import annotations

from collections import deque

from graph.module_graph_builder import module_of_file
from query.query_engine import QueryEngine


class ContextBuilder:
    def __init__(self, query_engine: QueryEngine):
        self.query = query_engine
        self.graph = query_engine.graph

    def build_symbol_context(self, symbol_query: str):
        return self.symbol_context(symbol_query)

    def symbol_context(self, symbol_query: str):
        warnings: list[str] = []
        symbol = self._get_symbol(symbol_query, warnings)
        if symbol is None:
            return {
                "focus": {"symbol_query": symbol_query},
                "summary": "Symbol not found.",
                "facts": {},
                "warnings": warnings,
            }

        callers = [
            self._serialize_symbol(s)
            for s in self.query.who_calls(symbol.symbol_id)
        ]
        callees = [
            self._serialize_symbol(s)
            for s in self.query.what_does(symbol.symbol_id)
        ]
        return {
            "focus": self._serialize_symbol(symbol),
            "summary": (
                f"{symbol.name} has {len(callers)} caller(s) "
                f"and {len(callees)} callee(s)."
            ),
            "facts": {
                "callers": callers,
                "callees": callees,
            },
            "warnings": warnings,
        }

    def class_context(self, class_query: str):
        warnings: list[str] = []
        class_symbol = self._get_symbol(class_query, warnings)
        if class_symbol is None:
            return {
                "focus": {"class_query": class_query},
                "summary": "Class not found.",
                "facts": {},
                "warnings": warnings,
            }

        contains = [
            self._serialize_symbol(edge.target)
            for edge in self.graph.dependencies_of(class_symbol.symbol_id)
            if edge.kind == "contains"
        ]
        bases = [
            self._serialize_symbol(edge.target)
            for edge in self.graph.dependencies_of(class_symbol.symbol_id)
            if edge.kind == "inherits"
        ]
        subclasses = [
            self._serialize_symbol(edge.source)
            for edge in self.graph.dependent_edges_of(class_symbol.symbol_id)
            if edge.kind == "inherits"
        ]

        return {
            "focus": self._serialize_symbol(class_symbol),
            "summary": (
                f"{class_symbol.name}: {len(contains)} contained symbol(s), "
                f"{len(bases)} base class(es), {len(subclasses)} subclass(es)."
            ),
            "facts": {
                "contains": contains,
                "inherits": bases,
                "subclasses": subclasses,
            },
            "warnings": warnings,
        }

    def module_context(self, module_name: str):
        symbols = [
            self._serialize_symbol(symbol)
            for symbol in self.graph.nodes
            if module_of_file(symbol.file) == module_name
        ]
        module_edges = getattr(self.graph, "module_edges", [])
        dependencies = [
            {
                "source_module": edge.source_module,
                "target_module": edge.target_module,
                "kind": edge.kind,
            }
            for edge in module_edges
            if edge.source_module == module_name
        ]
        dependents = [
            {
                "source_module": edge.source_module,
                "target_module": edge.target_module,
                "kind": edge.kind,
            }
            for edge in module_edges
            if edge.target_module == module_name
        ]

        return {
            "focus": {"module": module_name},
            "summary": (
                f"{module_name}: {len(symbols)} symbol(s), "
                f"{len(dependencies)} dependency edge(s), "
                f"{len(dependents)} dependent module edge(s)."
            ),
            "facts": {
                "symbols": symbols,
                "depends_on_modules": dependencies,
                "dependent_modules": dependents,
            },
            "warnings": [],
        }

    def call_chain(self, symbol_query: str, depth: int = 2, direction: str = "out"):
        warnings: list[str] = []
        start = self._get_symbol(symbol_query, warnings)
        if start is None:
            return {
                "focus": {"symbol_query": symbol_query},
                "summary": "Symbol not found.",
                "facts": {},
                "warnings": warnings,
            }

        chain_edges: list[dict] = []
        visited: set[tuple[str, int, str]] = set()
        queue = deque([(start.symbol_id, 0)])

        while queue:
            current, level = queue.popleft()
            if level >= depth:
                continue

            if direction in {"out", "both"}:
                for edge in self.graph.dependencies_of(current):
                    if edge.kind != "calls":
                        continue
                    key = (current, level, edge.target.symbol_id)
                    if key in visited:
                        continue
                    visited.add(key)
                    chain_edges.append(
                        {
                            "from": current,
                            "to": edge.target.symbol_id,
                            "kind": edge.kind,
                            "depth": level + 1,
                        }
                    )
                    queue.append((edge.target.symbol_id, level + 1))

            if direction in {"in", "both"}:
                for edge in self.graph.dependent_edges_of(current):
                    if edge.kind != "calls":
                        continue
                    key = (edge.source.symbol_id, level, current)
                    if key in visited:
                        continue
                    visited.add(key)
                    chain_edges.append(
                        {
                            "from": edge.source.symbol_id,
                            "to": current,
                            "kind": edge.kind,
                            "depth": level + 1,
                        }
                    )
                    queue.append((edge.source.symbol_id, level + 1))

        return {
            "focus": self._serialize_symbol(start),
            "summary": (
                f"Collected {len(chain_edges)} call-chain edge(s) "
                f"within depth {depth} ({direction})."
            ),
            "facts": {"edges": chain_edges},
            "warnings": warnings,
        }

    def impact_context(self, symbol_query: str, depth: int = 3):
        warnings: list[str] = []
        start = self._get_symbol(symbol_query, warnings)
        if start is None:
            return {
                "focus": {"symbol_query": symbol_query},
                "summary": "Symbol not found.",
                "facts": {},
                "warnings": warnings,
            }

        levels: dict[int, list[dict]] = {}
        seen: set[str] = {start.symbol_id}
        frontier: list[str] = [start.symbol_id]

        for level in range(1, depth + 1):
            next_frontier: list[str] = []
            impacted_this_level: list[dict] = []
            for symbol_id in frontier:
                for edge in self.graph.dependent_edges_of(symbol_id):
                    if edge.kind != "calls":
                        continue
                    src = edge.source
                    if src.symbol_id in seen:
                        continue
                    seen.add(src.symbol_id)
                    impacted_this_level.append(self._serialize_symbol(src))
                    next_frontier.append(src.symbol_id)
            levels[level] = impacted_this_level
            frontier = next_frontier
            if not frontier:
                break

        total = sum(len(v) for v in levels.values())
        return {
            "focus": self._serialize_symbol(start),
            "summary": f"Found {total} impacted caller symbol(s) up to depth {depth}.",
            "facts": {"impacted_by_level": levels},
            "warnings": warnings,
        }

    def _get_symbol(self, symbol_query: str, warnings: list[str]):
        matches = self.graph.symbol_lookup(symbol_query)
        if not matches:
            warnings.append(f"No symbol match for '{symbol_query}'.")
            return None
        if len(matches) > 1:
            warnings.append(
                f"Multiple symbols matched '{symbol_query}'. Using first match."
            )
        return matches[0]

    @staticmethod
    def _serialize_symbol(symbol):
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
