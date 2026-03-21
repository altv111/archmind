from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
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


@dataclass(frozen=True)
class _JavaFileHints:
    package_name: str | None
    class_imports: dict[str, str]
    wildcard_import_packages: list[str]
    static_member_imports: dict[str, str]
    static_wildcard_classes: list[str]


class SymbolResolver:
    def __init__(self, symbols: Iterable[Symbol], repo_root: str | Path | None = None) -> None:
        self._symbols: list[Symbol] = list(symbols)
        self._repo_root = Path(repo_root).resolve() if repo_root is not None else None
        self._by_id: dict[str, Symbol] = {s.symbol_id: s for s in self._symbols}
        self._by_name: dict[str, list[Symbol]] = defaultdict(list)
        self._by_file: dict[str, list[Symbol]] = defaultdict(list)
        self._python_hints_by_file: dict[str, _PythonFileHints] = {}
        self._java_hints_by_file: dict[str, _JavaFileHints] = {}
        for symbol in self._symbols:
            self._by_name[symbol.name].append(symbol)
            self._by_file[symbol.file].append(symbol)

        for file_symbols in self._by_file.values():
            file_symbols.sort(key=lambda s: (s.start_line, -(s.end_line - s.start_line)))

        self._build_python_hints()
        self._build_java_hints()

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

        if dependency.file.endswith(".java") and dependency.kind in {"imports", "inherits"}:
            inferred = self._resolve_java_symbol_by_fqn(
                target=canonical_target,
                dependency=dependency,
                source_symbol=source_symbol,
            )
            if inferred is not None:
                return inferred, "java_fqn"

        # Option 1: local variable type inference.
        # If we see `builder.build`, use earlier assignment hints like
        # `builder = GraphBuilder()` to resolve to `GraphBuilder.build`.
        inferred = self._resolve_dotted_with_var_type(
            canonical_target, dependency, source_symbol
        )
        if inferred is not None:
            return inferred, "var_type_inference"

        # Option 3: import alias aware dotted resolution.
        # Handles patterns like:
        #   from airflow.utils import yaml
        #   yaml.dump(...)
        inferred = self._resolve_dotted_with_import_alias(
            dependency=dependency,
            source_symbol=source_symbol,
        )
        if inferred is not None:
            return inferred, "import_alias_dotted"

        # Option 4: direct import alias to symbol.
        # Handles patterns like:
        #   from airflow.utils.yaml import dump
        #   dump(...)
        inferred = self._resolve_direct_import_symbol(
            dependency=dependency,
            source_symbol=source_symbol,
        )
        if inferred is not None:
            return inferred, "import_alias_direct"

        inferred = self._resolve_java_dotted_with_imports(
            dependency=dependency,
            source_symbol=source_symbol,
        )
        if inferred is not None:
            return inferred, "java_import_dotted"

        inferred = self._resolve_java_static_import_symbol(
            dependency=dependency,
            source_symbol=source_symbol,
        )
        if inferred is not None:
            return inferred, "java_static_import"

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

    def _resolve_dotted_with_import_alias(
        self,
        *,
        dependency: Dependency,
        source_symbol: Symbol | None,
    ) -> Symbol | None:
        target = dependency.target_symbol
        if "." not in target:
            return None

        receiver, member = target.split(".", 1)
        if not receiver or not member:
            return None

        hints = self._python_hints_by_file.get(dependency.file)
        if hints is None:
            return None
        import_target = hints.import_aliases.get(receiver)
        if not import_target:
            return None

        # import_target examples:
        # - airflow.utils.yaml (from airflow.utils import yaml)
        # - airflow.utils.yaml.dump (rare, symbol-like target)
        module_paths = [import_target]
        if "." in import_target:
            module_paths.append(import_target.rsplit(".", 1)[0])

        candidates = self._by_name.get(member, [])
        filtered: list[Symbol] = []
        for symbol in candidates:
            if any(self._symbol_module_matches(symbol, module_path) for module_path in module_paths):
                filtered.append(symbol)
        if filtered:
            chosen, _ = self._pick_candidate(
                filtered, dependency, source_symbol, "import_alias_dotted"
            )
            return chosen
        return None

    def _resolve_direct_import_symbol(
        self,
        *,
        dependency: Dependency,
        source_symbol: Symbol | None,
    ) -> Symbol | None:
        target = dependency.target_symbol
        if "." in target:
            return None

        hints = self._python_hints_by_file.get(dependency.file)
        if hints is None:
            return None
        alias_target = hints.import_aliases.get(target)
        if not alias_target:
            return None
        if "." not in alias_target:
            return None

        module_path, member = alias_target.rsplit(".", 1)
        candidates = self._by_name.get(member, [])
        filtered = [
            symbol
            for symbol in candidates
            if self._symbol_module_matches(symbol, module_path)
        ]
        if filtered:
            chosen, _ = self._pick_candidate(
                filtered, dependency, source_symbol, "import_alias_direct"
            )
            return chosen
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

    def _build_java_hints(self) -> None:
        java_files = [path for path in self._by_file if path.endswith(".java")]
        package_re = re.compile(r"^\s*package\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*;")
        import_re = re.compile(
            r"^\s*import\s+(static\s+)?([A-Za-z_]\w*(?:\.[A-Za-z_]\w*|\.\*)*)\s*;"
        )

        for relative_path in java_files:
            absolute_path = self._resolve_file_path(relative_path)
            if absolute_path is None or not absolute_path.is_file():
                continue
            try:
                source = absolute_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            package_name: str | None = None
            class_imports: dict[str, str] = {}
            wildcard_import_packages: list[str] = []
            static_member_imports: dict[str, str] = {}
            static_wildcard_classes: list[str] = []

            for line in source.splitlines():
                package_match = package_re.match(line)
                if package_match:
                    package_name = package_match.group(1)
                    continue

                import_match = import_re.match(line)
                if not import_match:
                    continue

                is_static = bool(import_match.group(1))
                imported = import_match.group(2).strip()
                if not imported:
                    continue

                if is_static:
                    if imported.endswith(".*"):
                        static_wildcard_classes.append(imported[:-2])
                        continue
                    if "." not in imported:
                        continue
                    member_name = imported.rsplit(".", 1)[1]
                    static_member_imports[member_name] = imported
                    continue

                if imported.endswith(".*"):
                    wildcard_import_packages.append(imported[:-2])
                    continue

                simple_name = imported.rsplit(".", 1)[-1]
                class_imports[simple_name] = imported

            self._java_hints_by_file[relative_path] = _JavaFileHints(
                package_name=package_name,
                class_imports=class_imports,
                wildcard_import_packages=wildcard_import_packages,
                static_member_imports=static_member_imports,
                static_wildcard_classes=static_wildcard_classes,
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

    def _symbol_module_matches(self, symbol: Symbol, module_path: str) -> bool:
        symbol_module = _python_module_from_path(symbol.file)
        if not symbol_module:
            return False
        return _module_suffix_match(symbol_module, module_path)

    def _resolve_java_symbol_by_fqn(
        self,
        *,
        target: str,
        dependency: Dependency,
        source_symbol: Symbol | None,
    ) -> Symbol | None:
        if "." not in target:
            return None
        class_symbol = self._resolve_java_class_symbol_by_fqn(
            class_fqn=target,
            dependency=dependency,
            source_symbol=source_symbol,
        )
        if class_symbol is not None:
            return class_symbol
        if target.count(".") < 2:
            return None
        class_fqn, member = target.rsplit(".", 1)
        class_symbol = self._resolve_java_class_symbol_by_fqn(
            class_fqn=class_fqn,
            dependency=dependency,
            source_symbol=source_symbol,
        )
        if class_symbol is None:
            return None
        method_candidates = self._by_name.get(member, [])
        methods = [m for m in method_candidates if m.parent == class_symbol.symbol_id]
        if methods:
            chosen, _ = self._pick_candidate(
                methods, dependency, source_symbol, "java_fqn_member"
            )
            return chosen
        return None

    def _resolve_java_dotted_with_imports(
        self,
        *,
        dependency: Dependency,
        source_symbol: Symbol | None,
    ) -> Symbol | None:
        if not dependency.file.endswith(".java"):
            return None
        target = dependency.target_symbol
        if "." not in target:
            return None
        receiver, member = target.split(".", 1)
        if not receiver or not member:
            return None

        hints = self._java_hints_by_file.get(dependency.file)
        class_candidates: list[Symbol] = []

        if hints is not None:
            imported_fqn = hints.class_imports.get(receiver)
            if imported_fqn:
                resolved = self._resolve_java_class_symbol_by_fqn(
                    class_fqn=imported_fqn,
                    dependency=dependency,
                    source_symbol=source_symbol,
                )
                if resolved is not None:
                    class_candidates.append(resolved)

            if hints.package_name:
                package_resolved = self._resolve_java_class_symbol_by_fqn(
                    class_fqn=f"{hints.package_name}.{receiver}",
                    dependency=dependency,
                    source_symbol=source_symbol,
                )
                if package_resolved is not None:
                    class_candidates.append(package_resolved)

            for wildcard_package in hints.wildcard_import_packages:
                wildcard_resolved = self._resolve_java_class_symbol_by_fqn(
                    class_fqn=f"{wildcard_package}.{receiver}",
                    dependency=dependency,
                    source_symbol=source_symbol,
                )
                if wildcard_resolved is not None:
                    class_candidates.append(wildcard_resolved)

        same_name_classes = [
            s
            for s in self._by_name.get(receiver, [])
            if s.kind in {"class", "interface", "enum"}
        ]
        class_candidates.extend(same_name_classes)

        seen_ids: set[str] = set()
        deduped_classes: list[Symbol] = []
        for candidate in class_candidates:
            if candidate.symbol_id in seen_ids:
                continue
            seen_ids.add(candidate.symbol_id)
            deduped_classes.append(candidate)

        method_candidates = self._by_name.get(member, [])
        scoped_methods: list[Symbol] = []
        class_ids = {c.symbol_id for c in deduped_classes}
        for method_symbol in method_candidates:
            if method_symbol.parent in class_ids:
                scoped_methods.append(method_symbol)

        if scoped_methods:
            chosen, _ = self._pick_candidate(
                scoped_methods, dependency, source_symbol, "java_import_dotted"
            )
            return chosen
        return None

    def _resolve_java_static_import_symbol(
        self,
        *,
        dependency: Dependency,
        source_symbol: Symbol | None,
    ) -> Symbol | None:
        if not dependency.file.endswith(".java"):
            return None
        target = dependency.target_symbol
        if "." in target:
            return None

        hints = self._java_hints_by_file.get(dependency.file)
        if hints is None:
            return None

        imported_member = hints.static_member_imports.get(target)
        if imported_member:
            class_fqn, member_name = imported_member.rsplit(".", 1)
            class_symbol = self._resolve_java_class_symbol_by_fqn(
                class_fqn=class_fqn,
                dependency=dependency,
                source_symbol=source_symbol,
            )
            if class_symbol is not None:
                methods = [
                    m
                    for m in self._by_name.get(member_name, [])
                    if m.parent == class_symbol.symbol_id
                ]
                if methods:
                    chosen, _ = self._pick_candidate(
                        methods, dependency, source_symbol, "java_static_import"
                    )
                    return chosen

        wildcard_methods: list[Symbol] = []
        for class_fqn in hints.static_wildcard_classes:
            class_symbol = self._resolve_java_class_symbol_by_fqn(
                class_fqn=class_fqn,
                dependency=dependency,
                source_symbol=source_symbol,
            )
            if class_symbol is None:
                continue
            wildcard_methods.extend(
                [
                    m
                    for m in self._by_name.get(target, [])
                    if m.parent == class_symbol.symbol_id
                ]
            )

        if wildcard_methods:
            chosen, _ = self._pick_candidate(
                wildcard_methods, dependency, source_symbol, "java_static_wildcard"
            )
            return chosen
        return None

    def _resolve_java_class_symbol_by_fqn(
        self,
        *,
        class_fqn: str,
        dependency: Dependency,
        source_symbol: Symbol | None,
    ) -> Symbol | None:
        class_name = class_fqn.rsplit(".", 1)[-1]
        candidates = [
            s
            for s in self._by_name.get(class_name, [])
            if s.kind in {"class", "interface", "enum"}
        ]
        if not candidates:
            return None

        filtered = [
            symbol
            for symbol in candidates
            if _java_fqn_suffix_match(_java_symbol_fqn(symbol, self._by_id), class_fqn)
        ]
        if not filtered:
            return None
        chosen, _ = self._pick_candidate(
            filtered, dependency, source_symbol, "java_class_fqn"
        )
        return chosen


def _python_module_from_path(path: str) -> str:
    normalized = path.replace("\\", "/").strip("/")
    if not normalized:
        return ""
    parts = normalized.split("/")
    # Trim common source roots while preserving package module path.
    if parts and parts[0] in {"src", "lib", "app"}:
        parts = parts[1:]
    if not parts:
        return ""
    filename = parts[-1]
    if filename == "__init__.py":
        parts = parts[:-1]
    elif filename.endswith(".py"):
        parts[-1] = filename[:-3]
    else:
        # Not a python module file.
        return ""
    return ".".join(part for part in parts if part)


def _module_suffix_match(symbol_module: str, import_module: str) -> bool:
    # Exact match or dotted suffix match.
    if symbol_module == import_module:
        return True
    return bool(re.search(rf"(?:^|\.){re.escape(import_module)}$", symbol_module))


def _java_fqn_suffix_match(symbol_fqn: str, import_fqn: str) -> bool:
    if not symbol_fqn or not import_fqn:
        return False
    if symbol_fqn == import_fqn:
        return True
    return bool(re.search(rf"(?:^|\.){re.escape(import_fqn)}$", symbol_fqn))


def _java_package_from_path(path: str) -> str:
    normalized = path.replace("\\", "/").strip("/")
    if not normalized:
        return ""

    parts = normalized.split("/")
    if not parts:
        return ""

    start_index = 0
    for marker in ("java", "kotlin", "groovy"):
        if marker in parts:
            start_index = parts.index(marker) + 1
            break
    else:
        if parts[0] in {"src", "lib", "app"}:
            start_index = 1

    if not parts[-1].endswith(".java"):
        return ""

    package_parts = parts[start_index:-1]
    if not package_parts:
        return ""
    return ".".join(package_parts)


def _java_symbol_fqn(symbol: Symbol, by_id: dict[str, Symbol]) -> str:
    if symbol.kind not in {"class", "interface", "enum"}:
        return ""

    package_name = _java_package_from_path(symbol.file)
    class_parts = [symbol.name]
    parent_id = symbol.parent
    while parent_id:
        parent = by_id.get(parent_id)
        if parent is None or parent.kind not in {"class", "interface", "enum"}:
            break
        class_parts.insert(0, parent.name)
        parent_id = parent.parent

    class_name = ".".join(part for part in class_parts if part)
    if not class_name:
        return package_name
    if not package_name:
        return class_name
    return f"{package_name}.{class_name}"
