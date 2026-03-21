from dataclasses import dataclass
from typing import Iterable
from collections import defaultdict
from pathlib import Path

from ingestion.symbol_extractor import Symbol
from ingestion.dependency_extractor import Dependency


def module_of_file(path: str, source_roots: set[str] | None = None) -> str:
    normalized = path.replace("\\", "/").strip("/")
    if not normalized:
        return "<root>"
    path_obj = Path(normalized)
    parts = list(path_obj.parts)
    if not parts:
        return "<root>"

    candidate_roots = source_roots or {"src", "lib", "app"}
    if len(parts) > 1 and parts[0] in candidate_roots:
        parts = parts[1:]

    if not parts:
        return "<root>"

    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    elif path_obj.suffix:
        parts[-1] = Path(parts[-1]).stem

    if not parts:
        return "<root>"
    return ".".join(parts)


@dataclass(frozen=True)
class ModuleDependency:
    source_module: str
    target_module: str
    kind: str = "module_depends"


class ModuleGraphBuilder:

    def build(
        self,
        symbols: Iterable[Symbol],
        dependencies: Iterable[Dependency],
    ) -> list[ModuleDependency]:

        symbols_by_id: dict[str, Symbol] = {}
        symbols_by_name = defaultdict(list)

        for symbol in symbols:
            symbols_by_id[symbol.symbol_id] = symbol
            symbols_by_name[symbol.name].append(symbol)

        edges = set()

        source_roots = _detect_source_roots(symbols_by_id.values())

        for dep in dependencies:

            source_module = module_of_file(dep.file, source_roots=source_roots)

            targets = []
            target_by_id = symbols_by_id.get(dep.target_symbol)
            if target_by_id is not None:
                targets = [target_by_id]
            else:
                targets = symbols_by_name.get(dep.target_symbol, [])
            if not targets:
                continue

            for sym in targets:

                target_module = module_of_file(sym.file, source_roots=source_roots)

                if source_module != target_module:
                    edges.add((source_module, target_module))

        return [
            ModuleDependency(src, tgt)
            for src, tgt in edges
        ]


def _detect_source_roots(symbols: Iterable[Symbol]) -> set[str]:
    roots: set[str] = set()
    for symbol in symbols:
        parts = symbol.file.replace("\\", "/").split("/")
        if len(parts) < 2:
            continue
        root = parts[0]
        if root in {"src", "lib", "app"}:
            roots.add(root)
    return roots or {"src", "lib", "app"}
