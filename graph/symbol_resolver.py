from __future__ import annotations

import ast
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


@dataclass(frozen=True)
class _PythonFunctionHints:
    var_types: dict[str, str]


@dataclass(frozen=True)
class _PythonFileHints:
    import_aliases: dict[str, str]
    function_hints: dict[str, _PythonFunctionHints]


class SymbolResolver:
    def __init__(self, symbols: Iterable[Symbol], repo_root: str | Path | None = None) -> None:
        self._symbols: list[Symbol] = list(symbols)
        self._repo_root = Path(repo_root).resolve() if repo_root is not None else None
        self._by_id: dict[str, Symbol] = {s.symbol_id: s for s in self._symbols}
        self._by_name: dict[str, list[Symbol]] = defaultdict(list)
        self._by_file: dict[str, list[Symbol]] = defaultdict(list)
        self._python_hints_by_file: dict[str, _PythonFileHints] = {}
        for symbol in self._symbols:
            self._by_name[symbol.name].append(symbol)
            self._by_file[symbol.file].append(symbol)

        for file_symbols in self._by_file.values():
            file_symbols.sort(key=lambda s: (s.start_line, -(s.end_line - s.start_line)))

        self._build_python_hints()

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

        canonical_target = self._canonical_target_name(dependency, source_symbol)

        exact_candidates = self._by_name.get(canonical_target, [])
        if exact_candidates:
            chosen, reason = self._pick_candidate(
                exact_candidates, dependency, source_symbol, "exact_name"
            )
            if chosen is not None:
                return chosen, reason

        # Option 1: local variable type inference.
        # If we see `builder.build`, use earlier assignment hints like
        # `builder = GraphBuilder()` to resolve to `GraphBuilder.build`.
        inferred = self._resolve_dotted_with_var_type(
            canonical_target, dependency, source_symbol
        )
        if inferred is not None:
            return inferred, "var_type_inference"

        simple_name = canonical_target.split(".")[-1]
        if simple_name != canonical_target:
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

    def _resolve_dotted_with_var_type(
        self,
        target_symbol: str,
        dependency: Dependency,
        source_symbol: Symbol | None,
    ) -> Symbol | None:
        if "." not in target_symbol:
            return None
        receiver, member = target_symbol.split(".", 1)
        if not receiver or not member:
            return None

        hints = self._python_hints_by_file.get(dependency.file)
        if hints is None:
            return None

        scope_name = source_symbol.name if source_symbol is not None else dependency.source_symbol
        scope_hints = hints.function_hints.get(scope_name)
        if scope_hints is None:
            return None

        receiver_type = scope_hints.var_types.get(receiver)
        if receiver_type is None:
            return None

        class_candidates = self._by_name.get(receiver_type, [])
        for class_symbol in class_candidates:
            method_candidates = self._by_name.get(member, [])
            for method_symbol in method_candidates:
                if method_symbol.parent == class_symbol.symbol_id:
                    return method_symbol
        return None

    def _canonical_target_name(
        self, dependency: Dependency, source_symbol: Symbol | None
    ) -> str:
        target = dependency.target_symbol
        hints = self._python_hints_by_file.get(dependency.file)
        if hints is None:
            return target

        # Option 2: import-aware alias resolution.
        # If code imports `GraphBuilder as GB`, rewrite `GB` calls to `GraphBuilder`
        # before symbol matching.
        if "." in target:
            left, right = target.split(".", 1)
            resolved_left = hints.import_aliases.get(left, left).split(".")[-1]
            if left != resolved_left:
                return f"{resolved_left}.{right}"
            return target

        return hints.import_aliases.get(target, target).split(".")[-1]

    def _build_python_hints(self) -> None:
        python_files = [path for path in self._by_file if path.endswith(".py")]
        for relative_path in python_files:
            absolute_path = self._resolve_file_path(relative_path)
            if absolute_path is None or not absolute_path.is_file():
                continue
            try:
                source = absolute_path.read_text(encoding="utf-8")
                module = ast.parse(source)
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue

            import_aliases: dict[str, str] = {}
            for node in module.body:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        import_aliases[alias.asname or alias.name] = alias.name
                elif isinstance(node, ast.ImportFrom):
                    module_name = node.module or ""
                    for alias in node.names:
                        if alias.name == "*":
                            continue
                        fq_name = (
                            f"{module_name}.{alias.name}"
                            if module_name
                            else alias.name
                        )
                        import_aliases[alias.asname or alias.name] = fq_name

            function_hints: dict[str, _PythonFunctionHints] = {}
            for node in module.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    function_hints[node.name] = _PythonFunctionHints(
                        var_types=self._infer_var_types(node, import_aliases)
                    )

            self._python_hints_by_file[relative_path] = _PythonFileHints(
                import_aliases=import_aliases,
                function_hints=function_hints,
            )

    def _infer_var_types(
        self, function_node: ast.FunctionDef | ast.AsyncFunctionDef, aliases: dict[str, str]
    ) -> dict[str, str]:
        var_types: dict[str, str] = {}

        for node in ast.walk(function_node):
            if isinstance(node, ast.Assign):
                value_type = self._callable_name(node.value, aliases)
                if value_type is None:
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_types[target.id] = value_type
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    ann_name = self._annotation_name(node.annotation)
                    if ann_name:
                        var_types[node.target.id] = ann_name
        return var_types

    def _callable_name(self, node: ast.AST, aliases: dict[str, str]) -> str | None:
        if not isinstance(node, ast.Call):
            return None
        func = node.func
        if isinstance(func, ast.Name):
            return aliases.get(func.id, func.id).split(".")[-1]
        if isinstance(func, ast.Attribute):
            attr = self._attribute_name(func)
            if not attr:
                return None
            left = attr.split(".", 1)[0]
            if left in aliases:
                return aliases[left].split(".")[-1]
            return attr.split(".")[-1]
        return None

    def _annotation_name(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            attr = self._attribute_name(node)
            return attr.split(".")[-1] if attr else None
        if isinstance(node, ast.Subscript):
            return self._annotation_name(node.value)
        return None

    def _attribute_name(self, node: ast.Attribute) -> str | None:
        parts: list[str] = [node.attr]
        current = node.value
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
            return ".".join(reversed(parts))
        return None

    def _resolve_file_path(self, relative_path: str) -> Path | None:
        if self._repo_root is not None:
            return self._repo_root / relative_path
        return Path(relative_path)
