from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ingestion.dependency_extractor import Dependency
from ingestion.symbol_extractor import Symbol


@dataclass(frozen=True)
class ResolvedDependency:
    source_symbol: str
    target_symbol: str
    kind: str
    file: str
    line: int
    source_symbol_id: str | None
    target_symbol_id: str | None
    resolution_reason: str


class SymbolResolver:
    def __init__(self, symbols: Iterable[Symbol]) -> None:
        self._symbols: list[Symbol] = list(symbols)
        self._by_id: dict[str, Symbol] = {s.symbol_id: s for s in self._symbols}
        self._by_name: dict[str, list[Symbol]] = defaultdict(list)
        self._by_file: dict[str, list[Symbol]] = defaultdict(list)
        for symbol in self._symbols:
            self._by_name[symbol.name].append(symbol)
            self._by_file[symbol.file].append(symbol)

        for file_symbols in self._by_file.values():
            file_symbols.sort(key=lambda s: (s.start_line, -(s.end_line - s.start_line)))

    def resolve_many(self, dependencies: Iterable[Dependency]) -> list[ResolvedDependency]:
        return [self.resolve(dependency) for dependency in dependencies]

    def resolve(self, dependency: Dependency) -> ResolvedDependency:
        source_symbol = self._resolve_source_symbol(dependency)
        target_symbol, reason = self._resolve_target_symbol(dependency, source_symbol)
        return ResolvedDependency(
            source_symbol=dependency.source_symbol,
            target_symbol=dependency.target_symbol,
            kind=dependency.kind,
            file=dependency.file,
            line=dependency.line,
            source_symbol_id=source_symbol.symbol_id if source_symbol else None,
            target_symbol_id=target_symbol.symbol_id if target_symbol else None,
            resolution_reason=reason,
        )

    def to_graph_dependencies(
        self, resolved_dependencies: Iterable[ResolvedDependency]
    ) -> list[Dependency]:
        graph_deps: list[Dependency] = []
        for resolved in resolved_dependencies:
            graph_deps.append(
                Dependency(
                    source_symbol=resolved.source_symbol_id or resolved.source_symbol,
                    target_symbol=resolved.target_symbol_id or resolved.target_symbol,
                    kind=resolved.kind,
                    file=resolved.file,
                    line=resolved.line,
                )
            )
        return graph_deps

    def _resolve_source_symbol(self, dependency: Dependency) -> Symbol | None:
        if dependency.source_symbol in self._by_id:
            return self._by_id[dependency.source_symbol]

        candidates = self._by_name.get(dependency.source_symbol, [])
        if not candidates:
            return self._enclosing_symbol(dependency.file, dependency.line)

        same_file = [s for s in candidates if s.file == dependency.file]
        if same_file:
            in_range = [s for s in same_file if s.start_line <= dependency.line <= s.end_line]
            if in_range:
                return in_range[0]
            return same_file[0]

        if len(candidates) == 1:
            return candidates[0]
        return None

    def _resolve_target_symbol(
        self, dependency: Dependency, source_symbol: Symbol | None
    ) -> tuple[Symbol | None, str]:
        if dependency.target_symbol in self._by_id:
            return self._by_id[dependency.target_symbol], "exact_symbol_id"

        exact_candidates = self._by_name.get(dependency.target_symbol, [])
        if exact_candidates:
            chosen, reason = self._pick_candidate(
                exact_candidates, dependency, source_symbol, "exact_name"
            )
            if chosen is not None:
                return chosen, reason

        simple_name = dependency.target_symbol.split(".")[-1]
        if simple_name != dependency.target_symbol:
            dotted_candidates = self._by_name.get(simple_name, [])
            if dotted_candidates:
                chosen, reason = self._pick_candidate(
                    dotted_candidates, dependency, source_symbol, "dotted_name_tail"
                )
                if chosen is not None:
                    return chosen, reason

        return None, "external"

    def _pick_candidate(
        self,
        candidates: list[Symbol],
        dependency: Dependency,
        source_symbol: Symbol | None,
        name_reason: str,
    ) -> tuple[Symbol | None, str]:
        same_file = [s for s in candidates if s.file == dependency.file]
        if same_file:
            return same_file[0], f"{name_reason}:same_file"

        if source_symbol is not None and source_symbol.parent:
            class_scope = self._by_id.get(source_symbol.parent)
            if class_scope is not None:
                same_class = [
                    s for s in candidates if s.parent == class_scope.symbol_id
                ]
                if same_class:
                    return same_class[0], f"{name_reason}:same_class"

        source_module = self._module_name(dependency.file)
        same_module = [
            s for s in candidates if self._module_name(s.file) == source_module
        ]
        if same_module:
            return same_module[0], f"{name_reason}:same_module"

        if len(candidates) == 1:
            return candidates[0], f"{name_reason}:unique"

        return None, f"{name_reason}:ambiguous"

    def _enclosing_symbol(self, file: str, line: int) -> Symbol | None:
        file_symbols = self._by_file.get(file, [])
        in_range = [s for s in file_symbols if s.start_line <= line <= s.end_line]
        if not in_range:
            return None
        in_range.sort(key=lambda s: (s.end_line - s.start_line, s.start_line))
        return in_range[0]

    @staticmethod
    def _module_name(file_path: str) -> str:
        return Path(file_path).stem
