from __future__ import annotations

import ast
from pathlib import Path
import re

from graph.code_graph import CodeGraph
from graph.module_graph_builder import module_of_file
from ingestion.symbol_extractor import Symbol


class QueryEngine:

    def __init__(self, graph: CodeGraph, repo_root: str | Path | None = None):
        self.graph = graph
        self.repo_root = Path(repo_root).resolve() if repo_root is not None else None
        self._file_lines_cache: dict[str, list[str]] = {}
        self._python_ast_cache: dict[str, ast.AST | None] = {}

    def set_repo_root(self, repo_root: str | Path | None) -> None:
        self.repo_root = Path(repo_root).resolve() if repo_root is not None else None
        self._file_lines_cache.clear()
        self._python_ast_cache.clear()

    def all_symbols(self) -> list[Symbol]:
        return list(self.graph.nodes)

    def resolve_symbol(self, symbol_query: str) -> Symbol | None:
        matches = self.graph.symbol_lookup(symbol_query)
        if not matches:
            return None
        return matches[0]

    def resolve_symbols(self, symbol_query: str) -> list[Symbol]:
        return self.graph.symbol_lookup(symbol_query)

    def find_symbols_like(
        self,
        keyword: str,
        kinds: set[str] | None = None,
        limit: int = 100,
        match_mode: str = "any",
        return_match_info: bool = False,
    ) -> list[Symbol]:
        needle = keyword.strip().lower()
        if not needle:
            return []
        mode = match_mode.lower().strip()
        if mode not in {"any", "all", "phrase"}:
            mode = "any"

        tokens = _tokenize_search_query(needle)
        if mode == "phrase":
            tokens = [needle]
        if not tokens:
            tokens = [needle]

        scored: list[tuple[int, int, Symbol, list[str], list[str]]] = []
        for idx, symbol in enumerate(self.graph.nodes):
            if kinds is not None and symbol.kind not in kinds:
                continue

            fields = {
                "name": symbol.name.lower(),
                "file": symbol.file.lower(),
                "kind": symbol.kind.lower(),
            }

            matched_tokens: list[str] = []
            matched_fields: set[str] = set()
            for token in tokens:
                token_hit = False
                for field_name, field_value in fields.items():
                    if token in field_value:
                        token_hit = True
                        matched_fields.add(field_name)
                if token_hit:
                    matched_tokens.append(token)

            if mode == "all" and len(matched_tokens) != len(tokens):
                continue
            if mode in {"any", "phrase"} and not matched_tokens:
                continue

            # Weighted score: token coverage + field coverage.
            score = len(matched_tokens) * 10 + len(matched_fields)
            scored.append((score, idx, symbol, matched_tokens, sorted(matched_fields)))

        # Higher score first, stable by symbol iteration order.
        scored.sort(key=lambda item: (-item[0], item[1]))
        scored = scored[:limit]

        if return_match_info:
            return [
                {
                    "symbol": item[2],
                    "score": item[0],
                    "matched_tokens": item[3],
                    "matched_fields": item[4],
                }
                for item in scored
            ]

        return [item[2] for item in scored]

    def dependency_edges_of(self, symbol_query: str, kind: str | None = None):
        edges = self.graph.dependencies_of(symbol_query)
        if kind is None:
            return edges
        return [edge for edge in edges if edge.kind == kind]

    def dependent_edges_of(self, symbol_query: str, kind: str | None = None):
        edges = self.graph.dependent_edges_of(symbol_query)
        if kind is None:
            return edges
        return [edge for edge in edges if edge.kind == kind]

    def dependencies_of(self, symbol_query: str, kind: str | None = None) -> list[Symbol]:
        edges = self.dependency_edges_of(symbol_query, kind=kind)
        out: list[Symbol] = []
        seen: set[str] = set()
        for edge in edges:
            if edge.target.symbol_id in seen:
                continue
            seen.add(edge.target.symbol_id)
            out.append(edge.target)
        return out

    def dependents_of(self, symbol_query: str, kind: str | None = None) -> list[Symbol]:
        edges = self.dependent_edges_of(symbol_query, kind=kind)
        out: list[Symbol] = []
        seen: set[str] = set()
        for edge in edges:
            if edge.source.symbol_id in seen:
                continue
            seen.add(edge.source.symbol_id)
            out.append(edge.source)
        return out

    def who_calls(self, symbol_query: str):
        return self.callers_of(symbol_query)

    def what_does(self, symbol_query: str):
        return self.callees_of(symbol_query)

    def callers_of(self, symbol_query: str) -> list[Symbol]:
        return self.dependents_of(symbol_query, kind="calls")

    def callees_of(self, symbol_query: str) -> list[Symbol]:
        return self.dependencies_of(symbol_query, kind="calls")

    def call_chain(
        self, symbol_query: str, depth: int = 2, direction: str = "out"
    ) -> list[dict]:
        start = self.resolve_symbol(symbol_query)
        if start is None:
            return []

        edges: list[dict] = []
        visited: set[tuple[str, str, int]] = set()
        queue: list[tuple[str, int]] = [(start.symbol_id, 0)]

        while queue:
            current_id, level = queue.pop(0)
            if level >= depth:
                continue

            if direction in {"out", "both"}:
                for edge in self.dependency_edges_of(current_id, kind="calls"):
                    key = (current_id, edge.target.symbol_id, level + 1)
                    if key in visited:
                        continue
                    visited.add(key)
                    edges.append(
                        {
                            "from": current_id,
                            "to": edge.target.symbol_id,
                            "kind": edge.kind,
                            "depth": level + 1,
                        }
                    )
                    queue.append((edge.target.symbol_id, level + 1))

            if direction in {"in", "both"}:
                for edge in self.dependent_edges_of(current_id, kind="calls"):
                    key = (edge.source.symbol_id, current_id, level + 1)
                    if key in visited:
                        continue
                    visited.add(key)
                    edges.append(
                        {
                            "from": edge.source.symbol_id,
                            "to": current_id,
                            "kind": edge.kind,
                            "depth": level + 1,
                        }
                    )
                    queue.append((edge.source.symbol_id, level + 1))

        return edges

    def impact_of(self, symbol_query: str, depth: int = 3):
        levels = self.impact_by_level(symbol_query, depth=depth)
        impacted: list[Symbol] = []
        for _, symbols in sorted(levels.items()):
            impacted.extend(symbols)
        return impacted

    def impact_by_level(self, symbol_query: str, depth: int = 3) -> dict[int, list[Symbol]]:
        start = self.resolve_symbol(symbol_query)
        if start is None:
            return {}

        levels: dict[int, list[Symbol]] = {}
        seen: set[str] = {start.symbol_id}
        frontier: list[str] = [start.symbol_id]

        for level in range(1, depth + 1):
            next_frontier: list[str] = []
            impacted_this_level: list[Symbol] = []
            for symbol_id in frontier:
                for caller in self.dependents_of(symbol_id, kind="calls"):
                    if caller.symbol_id in seen:
                        continue
                    seen.add(caller.symbol_id)
                    impacted_this_level.append(caller)
                    next_frontier.append(caller.symbol_id)
            levels[level] = impacted_this_level
            frontier = next_frontier
            if not frontier:
                break
        return levels

    def module_dependencies(self):
        return getattr(self.graph, "module_edges", [])

    def children_of(self, symbol_query: str):
        return self.dependencies_of(symbol_query, kind="contains")

    def parent_of(self, symbol_query: str):
        incoming = self.dependent_edges_of(symbol_query, kind="contains")
        for edge in incoming:
            return edge.source
        return None

    def module_of_symbol(self, symbol_query: str) -> str | None:
        symbol = self.resolve_symbol(symbol_query)
        if symbol is None:
            return None
        return module_of_file(symbol.file)

    def symbols_in_module(self, module_name: str) -> list[Symbol]:
        return [s for s in self.graph.nodes if module_of_file(s.file) == module_name]

    def module_dependencies_of(self, module_name: str):
        return [
            edge
            for edge in self.module_dependencies()
            if edge.source_module == module_name
        ]

    def module_dependents_of(self, module_name: str):
        return [
            edge
            for edge in self.module_dependencies()
            if edge.target_module == module_name
        ]

    def get_source_excerpt(self, symbol_query: str, max_lines: int = 10) -> str | None:
        symbol = self.resolve_symbol(symbol_query)
        if symbol is None:
            return None
        lines = self._file_lines(symbol.file)
        if not lines:
            return None
        start = max(symbol.start_line - 1, 0)
        end = min(start + max_lines, len(lines), symbol.end_line)
        if start >= end:
            return None
        excerpt = "\n".join(lines[start:end]).strip()
        return excerpt or None

    def get_full_implementation(self, symbol_query: str) -> str | None:
        symbol = self.resolve_symbol(symbol_query)
        if symbol is None:
            return None
        lines = self._file_lines(symbol.file)
        if not lines:
            return None
        start = max(symbol.start_line - 1, 0)
        end = min(symbol.end_line, len(lines))
        if start >= end:
            return None
        implementation = "\n".join(lines[start:end]).strip()
        return implementation or None

    def get_signature(self, symbol_query: str) -> str | None:
        excerpt = self.get_source_excerpt(symbol_query, max_lines=10)
        if not excerpt:
            return None
        for line in excerpt.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
        return None

    def get_docstring(self, symbol_query: str) -> str | None:
        symbol = self.resolve_symbol(symbol_query)
        if symbol is None:
            return None
        if not symbol.file.endswith(".py"):
            return None
        if symbol.kind not in {"function", "class"}:
            return None

        module = self._python_ast(symbol.file)
        if module is None:
            return None

        for node in ast.walk(module):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name != symbol.name:
                    continue
                if getattr(node, "lineno", -1) != symbol.start_line:
                    continue
                return ast.get_docstring(node)
        return None

    def _resolve_path(self, relative_path: str) -> Path | None:
        candidate = Path(relative_path)
        if candidate.is_file():
            return candidate
        if self.repo_root is not None:
            joined = self.repo_root / relative_path
            if joined.is_file():
                return joined
        cwd_joined = Path.cwd() / relative_path
        if cwd_joined.is_file():
            return cwd_joined
        return None

    def _file_lines(self, relative_path: str) -> list[str]:
        cached = self._file_lines_cache.get(relative_path)
        if cached is not None:
            return cached

        path = self._resolve_path(relative_path)
        if path is None:
            self._file_lines_cache[relative_path] = []
            return []

        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            self._file_lines_cache[relative_path] = []
            return []

        lines = text.splitlines()
        self._file_lines_cache[relative_path] = lines
        return lines

    def _python_ast(self, relative_path: str) -> ast.AST | None:
        if relative_path in self._python_ast_cache:
            return self._python_ast_cache[relative_path]

        path = self._resolve_path(relative_path)
        if path is None:
            self._python_ast_cache[relative_path] = None
            return None

        try:
            source = path.read_text(encoding="utf-8")
            module = ast.parse(source)
        except (OSError, SyntaxError):
            self._python_ast_cache[relative_path] = None
            return None

        self._python_ast_cache[relative_path] = module
        return module


def _tokenize_search_query(value: str) -> list[str]:
    tokens = [token for token in re.split(r"[\s,;:/\|]+", value) if token]
    return [token.strip().lower() for token in tokens if token.strip()]
