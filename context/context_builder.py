from __future__ import annotations

from query.query_engine import QueryEngine


class ContextBuilder:
    def __init__(self, query_engine: QueryEngine, repo_root=None):
        self.query = query_engine
        if repo_root is not None:
            self.query.set_repo_root(repo_root)

    def build_symbol_context(self, symbol_query: str):
        return self.symbol_context(symbol_query)

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
