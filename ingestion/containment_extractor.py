from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ingestion.symbol_extractor import Symbol


@dataclass(frozen=True)
class Containment:
    parent_symbol: str
    child_symbol: str
    kind: str = "contains"


class ContainmentExtractor:
    def extract(self, symbols: Iterable[Symbol]) -> list[Containment]:
        edges: list[Containment] = []

        for symbol in symbols:

            # file contains symbol
            file_symbol = f"file:{symbol.file}"

            edges.append(
                Containment(
                    parent_symbol=file_symbol,
                    child_symbol=symbol.symbol_id,
                )
            )

            # symbol contains nested symbol
            if symbol.parent:
                edges.append(
                    Containment(
                        parent_symbol=symbol.parent,
                        child_symbol=symbol.symbol_id,
                    )
                )

        return edges