from __future__ import annotations

from collections import Counter
from pathlib import Path

from query.query_engine import QueryEngine


class ContextBuilder:
    def __init__(self, query_engine: QueryEngine, repo_root=None):
        self.query = query_engine
        if repo_root is not None:
            self.query.set_repo_root(repo_root)

    def build_symbol_context(self, symbol_query: str):
        return self.symbol_context(symbol_query)

    def symbol_contexts(self, symbols: list[str]):
        contexts = []
        for symbol_query in symbols:
            contexts.append(
                {
                    "symbol_query": symbol_query,
                    "context": self.symbol_context(symbol_query),
                }
            )
        return {
            "focus": {"symbols": symbols},
            "summary": f"Built {len(contexts)} symbol context payload(s).",
            "facts": {"contexts": contexts},
            "warnings": [],
        }

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
                "architecture_graph": self._architecture_graph(symbol),
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
            self._serialize_symbol(symbol)
            for symbol in self.query.children_of(class_symbol.symbol_id)
        ]
        bases = [
            self._serialize_symbol(symbol)
            for symbol in self.query.dependencies_of(
                class_symbol.symbol_id, kind="inherits"
            )
        ]
        subclasses = [
            self._serialize_symbol(symbol)
            for symbol in self.query.dependents_of(
                class_symbol.symbol_id, kind="inherits"
            )
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
                "architecture_graph": self._architecture_graph(class_symbol),
            },
            "warnings": warnings,
        }

    def module_context(self, module_name: str):
        symbols = [
            self._serialize_symbol(symbol)
            for symbol in self.query.symbols_in_module(module_name)
        ]
        dependencies = [
            {
                "source_module": edge.source_module,
                "target_module": edge.target_module,
                "kind": edge.kind,
            }
            for edge in self.query.module_dependencies_of(module_name)
        ]
        dependents = [
            {
                "source_module": edge.source_module,
                "target_module": edge.target_module,
                "kind": edge.kind,
            }
            for edge in self.query.module_dependents_of(module_name)
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

    def module_contexts(self, modules: list[str]):
        contexts = []
        for module_name in modules:
            contexts.append(
                {
                    "module": module_name,
                    "context": self.module_context(module_name),
                }
            )
        return {
            "focus": {"modules": modules},
            "summary": f"Built {len(contexts)} module context payload(s).",
            "facts": {"contexts": contexts},
            "warnings": [],
        }

    def directory_context(
        self,
        directory: str,
        *,
        recursive: bool = True,
        max_files: int = 200,
    ) -> dict:
        children = self.query.directory_children(directory)
        files = self.query.files_in_directory(directory, recursive=recursive)
        symbols = self.query.symbols_in_directory(directory, recursive=recursive)

        module_counts: Counter[str] = Counter()
        for symbol in symbols:
            module = self.query.module_of_symbol(symbol.symbol_id)
            if module:
                module_counts[module] += 1

        return {
            "focus": {"directory": children["directory"], "recursive": recursive},
            "summary": (
                f"{children['directory']}: {len(children['directories'])} child directory(ies), "
                f"{len(children['files'])} direct file(s), {len(files)} indexed file(s) "
                f"and {len(symbols)} symbol(s) in scope."
            ),
            "facts": {
                "child_directories": children["directories"],
                "direct_files": children["files"],
                "indexed_files": files[:max_files],
                "total_indexed_files": len(files),
                "total_symbols": len(symbols),
                "top_modules": [
                    {"module": name, "symbol_count": count}
                    for name, count in module_counts.most_common(10)
                ],
            },
            "warnings": [],
        }

    def directory_contexts(
        self,
        directories: list[str],
        *,
        recursive: bool = True,
        max_files: int = 200,
    ) -> dict:
        contexts = []
        for directory in directories:
            contexts.append(
                {
                    "directory": directory,
                    "context": self.directory_context(
                        directory,
                        recursive=recursive,
                        max_files=max_files,
                    ),
                }
            )
        return {
            "focus": {"directories": directories, "recursive": recursive},
            "summary": f"Built {len(contexts)} directory context payload(s).",
            "facts": {"contexts": contexts},
            "warnings": [],
        }

    def repo_context(
        self,
        *,
        max_entries: int = 20,
        readme_max_lines: int = 40,
        top_modules: int = 10,
    ) -> dict:
        repo_root = (self.query.repo_root or Path.cwd()).resolve()
        entries = self._top_level_entries(repo_root, limit=max_entries)
        readme_path = self._find_root_readme(repo_root)
        readme = self._read_readme_summary(readme_path, max_lines=readme_max_lines)

        symbols = self.query.all_symbols()
        indexed_files = sorted({symbol.file for symbol in symbols if symbol.file})
        module_counts: Counter[str] = Counter()
        top_level_rollup: Counter[str] = Counter()
        language_counts: Counter[str] = Counter()

        for relative_path in indexed_files:
            module_name = _module_name_from_path(relative_path)
            if module_name:
                module_counts[module_name] += 1
                top_level_rollup[module_name.split(".", 1)[0]] += 1
            language = _language_from_path(relative_path)
            if language:
                language_counts[language] += 1

        key_files = []
        for candidate in (
            "README.md",
            "README.rst",
            "pyproject.toml",
            "setup.py",
            "package.json",
            "Cargo.toml",
            "go.mod",
            "Makefile",
            "docker-compose.yml",
            "Dockerfile",
        ):
            path = repo_root / candidate
            if path.exists():
                key_files.append(candidate)

        return {
            "repo_root": str(repo_root),
            "repo_name": repo_root.name,
            "readme": readme,
            "top_level_entries": entries,
            "key_files": key_files,
            "stats": {
                "indexed_symbols": len(symbols),
                "indexed_files": len(indexed_files),
                "indexed_modules": len(module_counts),
                "languages": dict(language_counts.most_common()),
            },
            "top_level_modules": [
                {"module": name, "indexed_files": count}
                for name, count in top_level_rollup.most_common(top_modules)
            ],
            "top_modules": [
                {"module": name, "indexed_files": count}
                for name, count in module_counts.most_common(top_modules)
            ],
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

        chain_edges = self.query.call_chain(
            start.symbol_id, depth=depth, direction=direction
        )

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

        levels = {
            level: [self._serialize_symbol(symbol) for symbol in symbols]
            for level, symbols in self.query.impact_by_level(
                start.symbol_id, depth=depth
            ).items()
        }

        total = sum(len(v) for v in levels.values())
        return {
            "focus": self._serialize_symbol(start),
            "summary": f"Found {total} impacted caller symbol(s) up to depth {depth}.",
            "facts": {"impacted_by_level": levels},
            "warnings": warnings,
        }

    def _get_symbol(self, symbol_query: str, warnings: list[str]):
        matches = self.query.resolve_symbols(symbol_query)
        if not matches:
            warnings.append(f"No symbol match for '{symbol_query}'.")
            return None
        if len(matches) > 1:
            warnings.append(
                f"Multiple symbols matched '{symbol_query}'. Using first match."
            )
        return matches[0]

    def _serialize_symbol(self, symbol):
        return self._serialize_symbol_with_source(symbol)

    def _serialize_symbol_with_source(self, symbol):
        source_excerpt = self.query.get_source_excerpt(symbol.symbol_id, max_lines=10)
        signature = self.query.get_signature(symbol.symbol_id)
        docstring = self.query.get_docstring(symbol.symbol_id)
        return {
            "symbol_id": symbol.symbol_id,
            "name": symbol.name,
            "kind": symbol.kind,
            "repo": symbol.repo,
            "file": symbol.file,
            "start_line": symbol.start_line,
            "end_line": symbol.end_line,
            "parent": symbol.parent,
            "signature": signature,
            "docstring": docstring,
            "source_excerpt": source_excerpt,
        }

    def _architecture_graph(self, focus_symbol):
        # Build a compact architecture view for diagram use.
        # We collapse method-level edges into component-level links
        # (e.g., GraphBuilder.build -> SymbolResolver.resolve_many becomes
        # GraphBuilder -> SymbolResolver).
        source_component = self._architecture_component(focus_symbol)
        source_name = (
            source_component["name"] if source_component is not None else focus_symbol.name
        )
        scope_ids = [focus_symbol.symbol_id] + [
            child.symbol_id for child in self.query.children_of(focus_symbol.symbol_id)
        ]

        node_map: dict[str, dict] = {
            source_name: {
                "name": source_name,
                "type": source_component["type"] if source_component else "component",
                "module": source_component["module"] if source_component else "external",
            }
        }
        edge_kinds: dict[tuple[str, str], str] = {}

        for symbol_id in scope_ids:
            for edge in self.query.dependency_edges_of(symbol_id):
                if edge.kind not in {"calls", "imports", "inherits"}:
                    continue
                target_component = self._architecture_component(edge.target)
                if not target_component:
                    continue
                target_name = target_component["name"]
                if target_name == source_name:
                    continue
                node_map.setdefault(
                    target_name,
                    {
                        "name": target_name,
                        "type": target_component["type"],
                        "module": target_component["module"],
                    },
                )

                classified_kind = self._classify_architecture_edge(
                    source_name=source_name,
                    target_name=target_name,
                    raw_kind=edge.kind,
                    target_symbol=edge.target,
                )
                key = (source_name, target_name)
                edge_kinds[key] = self._merge_edge_kind(
                    edge_kinds.get(key), classified_kind
                )

        edges = [
            {"source": src, "target": tgt, "kind": kind}
            for (src, tgt), kind in sorted(edge_kinds.items())
        ]
        return {
            "nodes": [node for _, node in sorted(node_map.items())],
            "edges": edges,
        }

    def _architecture_component(self, symbol) -> dict | None:
        repo_components = self._repo_components()
        repo_component_names = set(repo_components)

        if symbol.kind == "external":
            # Keep only external dotted targets whose receiver is a known component.
            # This drops noise like `tmp.append` while preserving cross-repo-style
            # component calls that can still be mapped by name.
            if "." in symbol.name:
                receiver = symbol.name.split(".", 1)[0]
                if receiver in repo_component_names:
                    component = repo_components.get(receiver, {})
                    return {
                        "name": receiver,
                        "type": component.get("type", "component"),
                        "module": component.get("module", "external"),
                    }
            if symbol.name in repo_component_names:
                component = repo_components.get(symbol.name, {})
                return {
                    "name": symbol.name,
                    "type": component.get("type", "component"),
                    "module": component.get("module", "external"),
                }
            return None

        current = symbol
        while current.parent:
            parent = self.query.resolve_symbol(current.parent)
            if parent is None:
                break
            if parent.kind in {"class", "module", "namespace"}:
                current = parent
                break
            current = parent

        if current.kind in {"class", "module", "namespace", "enum"}:
            return {
                "name": current.name,
                "type": current.kind,
                "module": self.query.module_of_symbol(current.symbol_id) or "external",
            }
        if current.kind == "function" and current.parent is None:
            return {
                "name": current.name,
                "type": "function",
                "module": self.query.module_of_symbol(current.symbol_id) or "external",
            }
        return None

    def _repo_components(self) -> dict[str, dict]:
        components: dict[str, dict] = {}
        for symbol in self.query.all_symbols():
            component = self._architecture_component_no_external(symbol)
            if component:
                components.setdefault(component["name"], component)
        return components

    def _architecture_component_no_external(self, symbol) -> dict | None:
        if symbol.kind == "external":
            return None
        current = symbol
        while current.parent:
            parent = self.query.resolve_symbol(current.parent)
            if parent is None:
                break
            if parent.kind in {"class", "module", "namespace"}:
                current = parent
                break
            current = parent
        if current.kind in {"class", "module", "namespace", "enum"}:
            return {
                "name": current.name,
                "type": current.kind,
                "module": self.query.module_of_symbol(current.symbol_id) or "external",
            }
        if current.kind == "function" and current.parent is None:
            return {
                "name": current.name,
                "type": "function",
                "module": self.query.module_of_symbol(current.symbol_id) or "external",
            }
        return None

    @staticmethod
    def _classify_architecture_edge(
        source_name: str,
        target_name: str,
        raw_kind: str,
        target_symbol,
    ) -> str:
        if raw_kind in {"imports", "inherits"}:
            return "depends_on"

        if raw_kind != "calls":
            return raw_kind

        if target_symbol.kind in {"class", "dataclass"}:
            if target_name.endswith(("Result", "Response", "Output")):
                return "produces"
            return "constructs"

        if target_symbol.parent:
            return "uses"
        return "uses"

    @staticmethod
    def _merge_edge_kind(current_kind: str | None, new_kind: str) -> str:
        if current_kind is None:
            return new_kind
        priority = {
            "depends_on": 4,
            "produces": 3,
            "uses": 2,
            "constructs": 1,
        }
        return new_kind if priority.get(new_kind, 0) > priority.get(current_kind, 0) else current_kind

    def _top_level_entries(self, repo_root: Path, *, limit: int) -> list[dict]:
        rows = []
        try:
            children = sorted(repo_root.iterdir(), key=lambda path: (path.is_file(), path.name.lower()))
        except OSError:
            return rows

        for child in children:
            if _is_hidden_name(child.name):
                continue
            if _is_ignored_top_level_name(child.name):
                continue
            rows.append(
                {
                    "name": child.name,
                    "type": _top_level_entry_type(child),
                }
            )
            if len(rows) >= limit:
                break
        return rows

    @staticmethod
    def _find_root_readme(repo_root: Path) -> Path | None:
        for candidate in ("README.md", "README.rst", "README.txt", "README"):
            path = repo_root / candidate
            if path.is_file():
                return path
        return None

    @staticmethod
    def _read_readme_summary(path: Path | None, *, max_lines: int) -> dict | None:
        if path is None:
            return None
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return None

        lines = text.splitlines()
        excerpt_lines: list[str] = []
        in_code_block = False
        for raw_line in lines:
            line = raw_line.rstrip()
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                continue
            if not line.strip() and not excerpt_lines:
                continue
            excerpt_lines.append(line)
            if len(excerpt_lines) >= max_lines:
                break

        excerpt = "\n".join(excerpt_lines).strip()
        first_paragraph = excerpt.split("\n\n", 1)[0].strip() if excerpt else ""
        return {
            "path": path.name,
            "excerpt": excerpt or None,
            "summary": first_paragraph or None,
        }


def _top_level_entry_type(path: Path) -> str:
    if path.is_dir():
        return "directory"
    if path.is_file():
        return "file"
    return "other"


def _is_hidden_name(name: str) -> bool:
    return name.startswith(".")


def _is_ignored_top_level_name(name: str) -> bool:
    ignored = {
        "__pycache__",
        "node_modules",
        "dist",
        "build",
        "env",
        "venv",
        ".venv",
    }
    if name in ignored:
        return True
    return name.endswith(".egg-info")


def _module_name_from_path(relative_path: str) -> str | None:
    path = Path(relative_path)
    if not path.parts:
        return None
    parts = list(path.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    elif path.suffix == ".py":
        parts[-1] = path.stem
    elif len(parts) == 1:
        parts[-1] = path.stem
    if not parts:
        return None
    return ".".join(part for part in parts if part)


def _language_from_path(relative_path: str) -> str | None:
    suffix = Path(relative_path).suffix.lower()
    return {
        ".py": "python",
        ".java": "java",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".c": "c",
        ".h": "c-family",
        ".hpp": "c-family",
    }.get(suffix)
