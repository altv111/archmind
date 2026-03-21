from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DirectoryEdge:
    repo: str
    source_node: str
    target_node: str
    kind: str


class DirectoryGraphBuilder:
    def build(self, files: Iterable[tuple[str, str]]) -> list[DirectoryEdge]:
        edges: set[tuple[str, str, str, str]] = set()

        for repo, relative_path in files:
            normalized = relative_path.replace("\\", "/").strip("/")
            if not normalized:
                continue

            path = Path(normalized)
            parts = list(path.parts)
            if not parts:
                continue

            parent = "<root>"
            for part in parts[:-1]:
                current = part if parent == "<root>" else f"{parent}/{part}"
                edges.add((repo, parent, current, "contains_dir"))
                parent = current

            edges.add((repo, parent, normalized, "contains_file"))

        return [
            DirectoryEdge(repo=repo, source_node=source, target_node=target, kind=kind)
            for repo, source, target, kind in sorted(edges)
        ]
