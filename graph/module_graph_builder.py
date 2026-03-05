from dataclasses import dataclass
from typing import Iterable
from collections import defaultdict

from ingestion.symbol_extractor import Symbol
from ingestion.dependency_extractor import Dependency


def module_of_file(path: str) -> str:
    parts = path.split("/")
    return parts[0] if len(parts) > 1 else "<root>"


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

        for dep in dependencies:

            source_module = module_of_file(dep.file)

            targets = []
            target_by_id = symbols_by_id.get(dep.target_symbol)
            if target_by_id is not None:
                targets = [target_by_id]
            else:
                targets = symbols_by_name.get(dep.target_symbol, [])
            if not targets:
                continue

            for sym in targets:

                target_module = module_of_file(sym.file)

                if source_module != target_module:
                    edges.add((source_module, target_module))

        return [
            ModuleDependency(src, tgt)
            for src, tgt in edges
        ]
