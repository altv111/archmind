from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from ingestion.dependency_extractor import Dependency
from ingestion.symbol_extractor import Symbol


@dataclass(frozen=True)
class EdgeView:
    kind: str
    target: Symbol


@dataclass(frozen=True)
class ReverseEdgeView:
    kind: str
    source: Symbol


class CodeGraph:
    def __init__(
        self, symbols: Iterable[Symbol], dependencies: Iterable[Dependency]
    ) -> None:
        self.nodes: list[Symbol] = list(symbols)
        self.edges: list[Dependency] = list(dependencies)

        self._symbols_by_id: dict[str, Symbol] = {}
        self._symbols_by_name: dict[str, list[Symbol]] = defaultdict(list)
        self._external_symbols: dict[tuple[str, str, int], Symbol] = {}
        for symbol in self.nodes:
            self._symbols_by_id[symbol.symbol_id] = symbol
            self._symbols_by_name[symbol.name].append(symbol)

        self._forward_by_name: dict[str, list[Dependency]] = defaultdict(list)
        self._reverse_by_name: dict[str, list[Dependency]] = defaultdict(list)
        self._forward_by_id: dict[str, list[Dependency]] = defaultdict(list)
        self._reverse_by_id: dict[str, list[Dependency]] = defaultdict(list)
        for dependency in self.edges:
            self._forward_by_name[dependency.source_symbol].append(dependency)
            self._reverse_by_name[dependency.target_symbol].append(dependency)

            source_ids = self._resolve_symbol_ids_for_source(dependency)
            for source_id in source_ids:
                self._forward_by_id[source_id].append(dependency)

            target_ids = self._resolve_symbol_ids_for_target(dependency)
            for target_id in target_ids:
                self._reverse_by_id[target_id].append(dependency)

    def symbol_lookup(self, query: str) -> list[Symbol]:
        if query in self._symbols_by_id:
            return [self._symbols_by_id[query]]
        return list(self._symbols_by_name.get(query, []))

    def get_symbol_by_name(self, name: str) -> list[Symbol]:
        return list(self._symbols_by_name.get(name, []))

    def dependencies_of(self, symbol_query: str) -> list[EdgeView]:
        dependencies = self.outgoing_dependencies_of(symbol_query)
        return [
            EdgeView(kind=d.kind, target=self._resolve_target_symbol(d))
            for d in dependencies
        ]

    def outgoing_dependencies_of(self, symbol_query: str) -> list[Dependency]:
        out: list[Dependency] = []
        seen: set[tuple[str, str, str, str, int]] = set()
        for dependency in self._dependencies_from_forward(symbol_query):
            key = (
                dependency.source_symbol,
                dependency.target_symbol,
                dependency.kind,
                dependency.file,
                dependency.line,
            )
            if key not in seen:
                seen.add(key)
                out.append(dependency)
        return out

    def _dependencies_from_forward(self, symbol_query: str) -> list[Dependency]:
        if symbol_query in self._symbols_by_id:
            return list(self._forward_by_id.get(symbol_query, []))
        return list(self._forward_by_name.get(symbol_query, []))

    def _dependencies_from_reverse(self, symbol_query: str) -> list[Dependency]:
        if symbol_query in self._symbols_by_id:
            return list(self._reverse_by_id.get(symbol_query, []))
        return list(self._reverse_by_name.get(symbol_query, []))

    def _resolve_symbol_ids_for_source(self, dependency: Dependency) -> list[str]:
        if dependency.source_symbol in self._symbols_by_id:
            return [dependency.source_symbol]

        candidates = self._symbols_by_name.get(dependency.source_symbol, [])
        if not candidates:
            return []

        in_file = [s for s in candidates if s.file == dependency.file]
        if in_file:
            in_range = [
                s
                for s in in_file
                if s.start_line <= dependency.line <= s.end_line
            ]
            if in_range:
                return [s.symbol_id for s in in_range]
            return [s.symbol_id for s in in_file]

        return [s.symbol_id for s in candidates]

    def _resolve_symbol_ids_for_target(self, dependency: Dependency) -> list[str]:
        if dependency.target_symbol in self._symbols_by_id:
            return [dependency.target_symbol]

        candidates = self._symbols_by_name.get(dependency.target_symbol, [])
        if not candidates:
            return []

        in_file = [s for s in candidates if s.file == dependency.file]
        if in_file:
            return [s.symbol_id for s in in_file]

        return [s.symbol_id for s in candidates]

    def _resolve_target_symbol(self, dependency: Dependency) -> Symbol:
        if dependency.target_symbol in self._symbols_by_id:
            return self._symbols_by_id[dependency.target_symbol]

        candidates = self._symbols_by_name.get(dependency.target_symbol, [])
        if candidates:
            in_file = [s for s in candidates if s.file == dependency.file]
            if in_file:
                return in_file[0]
            return candidates[0]

        key = (dependency.target_symbol, dependency.file, dependency.line)
        external = self._external_symbols.get(key)
        if external is not None:
            return external

        external = Symbol(
            symbol_id=f"external:{dependency.target_symbol}:{dependency.file}:{dependency.line}",
            name=dependency.target_symbol,
            kind="external",
            repo="",
            file=dependency.file,
            start_line=dependency.line,
            end_line=dependency.line,
            parent=None,
        )
        self._external_symbols[key] = external
        return external

    def dependents_of(self, symbol_query: str) -> list[Symbol]:
        incoming = self.incoming_dependencies_of(symbol_query)
        out: list[Symbol] = []
        seen: set[str] = set()
        for dependency in incoming:
            source = self._resolve_source_symbol(dependency)
            if source.symbol_id in seen:
                continue
            seen.add(source.symbol_id)
            out.append(source)
        return out

    def dependent_names_of(self, symbol_query: str) -> list[str]:
        return [symbol.name for symbol in self.dependents_of(symbol_query)]

    def dependent_edges_of(self, symbol_query: str) -> list[ReverseEdgeView]:
        incoming = self.incoming_dependencies_of(symbol_query)
        return [
            ReverseEdgeView(kind=dependency.kind, source=self._resolve_source_symbol(dependency))
            for dependency in incoming
        ]

    def incoming_dependencies_of(self, symbol_query: str) -> list[Dependency]:
        out: list[Dependency] = []
        seen: set[tuple[str, str, str, str, int]] = set()
        for dependency in self._dependencies_from_reverse(symbol_query):
            key = (
                dependency.source_symbol,
                dependency.target_symbol,
                dependency.kind,
                dependency.file,
                dependency.line,
            )
            if key not in seen:
                seen.add(key)
                out.append(dependency)
        return out

    def _resolve_source_symbol(self, dependency: Dependency) -> Symbol:
        if dependency.source_symbol in self._symbols_by_id:
            return self._symbols_by_id[dependency.source_symbol]

        candidates = self._symbols_by_name.get(dependency.source_symbol, [])
        if candidates:
            in_file = [s for s in candidates if s.file == dependency.file]
            if in_file:
                in_range = [
                    s for s in in_file if s.start_line <= dependency.line <= s.end_line
                ]
                if in_range:
                    return in_range[0]
                return in_file[0]
            return candidates[0]

        key = (dependency.source_symbol, dependency.file, dependency.line)
        external = self._external_symbols.get(key)
        if external is not None:
            return external

        external = Symbol(
            symbol_id=f"external:{dependency.source_symbol}:{dependency.file}:{dependency.line}",
            name=dependency.source_symbol,
            kind="external",
            repo="",
            file=dependency.file,
            start_line=dependency.line,
            end_line=dependency.line,
            parent=None,
        )
        self._external_symbols[key] = external
        return external
