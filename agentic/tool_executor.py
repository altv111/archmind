from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
import re
import subprocess
from typing import Any

from agentic.tool_registry import ToolRegistry, ToolSpec
from context.context_builder import ContextBuilder
from query.query_engine import QueryEngine


class ToolExecutor:
    """Unified executor for query tools and context tools."""

    def __init__(
        self,
        query_engine: QueryEngine,
        context_builder: ContextBuilder | None = None,
        registry: ToolRegistry | None = None,
    ) -> None:
        self.query = query_engine
        self.context = context_builder or ContextBuilder(query_engine)
        self.registry = registry or self._default_registry()

    def available_tools(self) -> list[dict]:
        return [_json_ready(asdict(spec)) for spec in self.registry.specs()]

    def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        tool = self.registry.get(tool_name)
        if tool is None:
            raise ValueError(f"Unknown tool: {tool_name}")
        result = tool.fn(**args)
        return {
            "tool": tool_name,
            "cost": tool.spec.cost,
            "result": _json_ready(result),
        }

    def _default_registry(self) -> ToolRegistry:
        registry = ToolRegistry()

        self._register(
            registry,
            ToolSpec(
                name="inspect_repo",
                description=(
                    "Inspect repository-level context: root README, top-level structure, "
                    "indexed modules, key files, and basic stats."
                ),
                input_schema={
                    "max_entries": "int",
                    "readme_max_lines": "int",
                    "top_modules": "int",
                },
                output_schema={"repo": "dict"},
                cost=3,
                tags=("discovery", "context"),
            ),
            lambda max_entries=20, readme_max_lines=40, top_modules=10: self.context.repo_context(
                max_entries=int(max_entries),
                readme_max_lines=int(readme_max_lines),
                top_modules=int(top_modules),
            ),
        )
        self._register(
            registry,
            ToolSpec(
                name="symbol_lookup",
                description="Find symbol matches by name or symbol_id.",
                input_schema={"symbol": "str"},
                output_schema={"symbols": "list[Symbol]"},
                cost=1,
                tags=("query",),
            ),
            lambda symbol: [self._asdict_symbol(s) for s in self.query.resolve_symbols(symbol)],
        )
        self._register(
            registry,
            ToolSpec(
                name="find_symbol_like",
                description=(
                    "Find symbols by keyword match across symbol name/file/kind. "
                    "Useful when exact symbol is unknown."
                ),
                input_schema={
                    "keyword": "str",
                    "kinds": "list[str]|None",
                    "limit": "int",
                    "match_mode": "str(any|all|phrase)",
                },
                output_schema={
                    "matches": "list[{symbol,score,matched_tokens,matched_fields}]"
                },
                cost=1,
                tags=("query", "discovery"),
            ),
            lambda keyword, kinds=None, limit=100, match_mode="any": [
                {
                    "symbol": self._asdict_symbol(item["symbol"]),
                    "score": item["score"],
                    "matched_tokens": item["matched_tokens"],
                    "matched_fields": item["matched_fields"],
                }
                for item in self.query.find_symbols_like(
                    keyword=keyword,
                    kinds={k for k in (kinds or []) if isinstance(k, str)} or None,
                    limit=int(limit),
                    match_mode=match_mode,
                    return_match_info=True,
                )
            ],
        )
        self._register(
            registry,
            ToolSpec(
                name="dependencies",
                description="Get outgoing dependencies for a symbol.",
                input_schema={"symbol": "str", "kind": "str|None"},
                output_schema={"symbols": "list[Symbol]"},
                cost=2,
                tags=("query",),
            ),
            lambda symbol, kind=None: [self._asdict_symbol(s) for s in self.query.dependencies_of(symbol, kind=kind)],
        )
        self._register(
            registry,
            ToolSpec(
                name="dependents",
                description="Get incoming dependents for a symbol.",
                input_schema={"symbol": "str", "kind": "str|None"},
                output_schema={"symbols": "list[Symbol]"},
                cost=2,
                tags=("query",),
            ),
            lambda symbol, kind=None: [self._asdict_symbol(s) for s in self.query.dependents_of(symbol, kind=kind)],
        )
        self._register(
            registry,
            ToolSpec(
                name="call_chain",
                description="Get depth-bounded call chain edges.",
                input_schema={"symbol": "str", "depth": "int", "direction": "str"},
                output_schema={"edges": "list[dict]"},
                cost=3,
                tags=("query",),
            ),
            lambda symbol, depth=2, direction="both": self.query.call_chain(symbol, depth=depth, direction=direction),
        )
        self._register(
            registry,
            ToolSpec(
                name="impact",
                description="Get impacted callers by level for a symbol.",
                input_schema={"symbol": "str", "depth": "int"},
                output_schema={"impacted_by_level": "dict[level,list[Symbol]]"},
                cost=4,
                tags=("query", "risk"),
            ),
            lambda symbol, depth=3: {
                str(level): [self._asdict_symbol(s) for s in symbols]
                for level, symbols in self.query.impact_by_level(symbol, depth=depth).items()
            },
        )
        self._register(
            registry,
            ToolSpec(
                name="children",
                description="Get contains-children for a symbol.",
                input_schema={"symbol": "str"},
                output_schema={"symbols": "list[Symbol]"},
                cost=2,
                tags=("query",),
            ),
            lambda symbol: [self._asdict_symbol(s) for s in self.query.children_of(symbol)],
        )
        self._register(
            registry,
            ToolSpec(
                name="parent",
                description="Get contains-parent for a symbol.",
                input_schema={"symbol": "str"},
                output_schema={"symbol": "Symbol|None"},
                cost=2,
                tags=("query",),
            ),
            lambda symbol: self._asdict_symbol(self.query.parent_of(symbol)) if self.query.parent_of(symbol) else None,
        )
        self._register(
            registry,
            ToolSpec(
                name="module_dependencies",
                description="Get module-level dependencies.",
                input_schema={"module": "str"},
                output_schema={"edges": "list[ModuleDependency]"},
                cost=2,
                tags=("query",),
            ),
            lambda module: self._module_dependencies_with_fallback(str(module)),
        )
        self._register(
            registry,
            ToolSpec(
                name="module_dependencies_ranked",
                description=(
                    "Get ranked module dependencies for architecture analysis "
                    "(high-signal first; ancillary edges optionally included)."
                ),
                input_schema={
                    "module": "str",
                    "max_edges": "int",
                    "include_ancillary": "bool",
                },
                output_schema={"edges": "list[dict]"},
                cost=3,
                tags=("query", "architecture"),
            ),
            lambda module, max_edges=20, include_ancillary=False: self._module_dependencies_ranked(
                str(module),
                max_edges=int(max_edges),
                include_ancillary=bool(include_ancillary),
            ),
        )
        self._register(
            registry,
            ToolSpec(
                name="get_source_excerpt",
                description="Get top lines from symbol implementation.",
                input_schema={"symbol": "str", "max_lines": "int"},
                output_schema={"source_excerpt": "str|None"},
                cost=3,
                tags=("query", "source"),
            ),
            lambda symbol, max_lines=10: self.query.get_source_excerpt(symbol, max_lines=max_lines),
        )
        self._register(
            registry,
            ToolSpec(
                name="get_full_implementation",
                description="Get full implementation body for a symbol.",
                input_schema={"symbol": "str"},
                output_schema={"implementation": "str|None"},
                cost=8,
                tags=("query", "source", "expensive"),
            ),
            lambda symbol: self.query.get_full_implementation(symbol),
        )

        self._register(
            registry,
            ToolSpec(
                name="symbol_context",
                description="Generate rich symbol context for LLM reasoning.",
                input_schema={"symbol": "str"},
                output_schema={"context": "dict"},
                cost=5,
                tags=("context",),
            ),
            lambda symbol: self.context.symbol_context(symbol),
        )
        self._register(
            registry,
            ToolSpec(
                name="module_context",
                description="Generate module-level context for LLM reasoning.",
                input_schema={"module": "str"},
                output_schema={"context": "dict"},
                cost=5,
                tags=("context",),
            ),
            lambda module: self._module_context_with_fallback(str(module)),
        )
        self._register(
            registry,
            ToolSpec(
                name="module_or_directory_context",
                description=(
                    "Resolve input as module first, otherwise as directory; "
                    "for directories, return contained modules and their contexts."
                ),
                input_schema={"name": "str", "recursive": "bool", "max_modules": "int"},
                output_schema={"context": "dict"},
                cost=6,
                tags=("context", "discovery", "topdown"),
            ),
            lambda name, recursive=True, max_modules=20: self.context.module_or_directory_context(
                name,
                recursive=bool(recursive),
                max_modules=int(max_modules),
            ),
        )
        self._register(
            registry,
            ToolSpec(
                name="directory_context",
                description="Generate directory-level context for top-down architecture reasoning.",
                input_schema={"directory": "str", "recursive": "bool", "max_files": "int"},
                output_schema={"context": "dict"},
                cost=5,
                tags=("context", "discovery", "topdown"),
            ),
            lambda directory="<root>", recursive=True, max_files=200: self.context.directory_context(
                directory,
                recursive=bool(recursive),
                max_files=int(max_files),
            ),
        )
        self._register(
            registry,
            ToolSpec(
                name="symbol_contexts",
                description="Generate symbol contexts for multiple symbols.",
                input_schema={"symbols": "list[str]"},
                output_schema={"context": "dict"},
                cost=7,
                tags=("context", "batch"),
            ),
            lambda symbols: self.context.symbol_contexts(_to_string_list(symbols)),
        )
        self._register(
            registry,
            ToolSpec(
                name="directory_contexts",
                description="Generate directory contexts for multiple directories.",
                input_schema={"directories": "list[str]", "recursive": "bool", "max_files": "int"},
                output_schema={"context": "dict"},
                cost=7,
                tags=("context", "batch", "topdown"),
            ),
            lambda directories, recursive=True, max_files=200: self.context.directory_contexts(
                _to_string_list(directories),
                recursive=bool(recursive),
                max_files=int(max_files),
            ),
        )
        self._register(
            registry,
            ToolSpec(
                name="module_contexts",
                description="Generate module contexts for multiple modules.",
                input_schema={"modules": "list[str]"},
                output_schema={"context": "dict"},
                cost=7,
                tags=("context", "batch"),
            ),
            lambda modules: self._module_contexts_with_fallback(_to_string_list(modules)),
        )
        self._register(
            registry,
            ToolSpec(
                name="module_dependents",
                description="Get module-level dependents.",
                input_schema={"module": "str"},
                output_schema={"edges": "list[ModuleDependency]"},
                cost=2,
                tags=("query",),
            ),
            lambda module: self._module_dependents_with_fallback(str(module)),
        )
        self._register(
            registry,
            ToolSpec(
                name="impact_context",
                description="Generate impact context payload.",
                input_schema={"symbol": "str", "depth": "int"},
                output_schema={"context": "dict"},
                cost=6,
                tags=("context", "risk"),
            ),
            lambda symbol, depth=3: self.context.impact_context(symbol, depth=depth),
        )
        self._register(
            registry,
            ToolSpec(
                name="stack_trace",
                description="Static caller/callee trace with symbol details.",
                input_schema={"symbol": "str", "depth": "int", "max_lines": "int"},
                output_schema={"trace": "dict"},
                cost=6,
                tags=("context", "trace"),
            ),
            lambda symbol, depth=2, max_lines=10: self._stack_trace(symbol, depth=depth, max_lines=max_lines),
        )
        self._register(
            registry,
            ToolSpec(
                name="pr_diff_context",
                description=(
                    "Analyze git diff (base...head): changed files/lines, touched symbols, "
                    "impact rollup, and PR risk summary."
                ),
                input_schema={
                    "base": "str",
                    "head": "str",
                    "repo_root": "str",
                    "depth": "int",
                    "format": "str(summary|full)",
                    "top_symbol_contexts": "int",
                    "top_module_contexts": "int",
                },
                output_schema={"pr_context": "dict"},
                cost=7,
                tags=("query", "context", "risk", "pr"),
            ),
            lambda base="main", head="HEAD", repo_root=".", depth=3, format="summary", top_symbol_contexts=5, top_module_contexts=3: self._pr_diff_context(
                base=base,
                head=head,
                repo_root=repo_root,
                depth=int(depth),
                format=format,
                top_symbol_contexts=int(top_symbol_contexts),
                top_module_contexts=int(top_module_contexts),
            ),
        )
        return registry

    @staticmethod
    def _register(registry: ToolRegistry, spec: ToolSpec, fn) -> None:
        registry.register(spec, fn)

    def _stack_trace(self, symbol: str, depth: int, max_lines: int) -> dict:
        focus = self.query.resolve_symbol(symbol)
        if focus is None:
            return {"symbol_query": symbol, "summary": "Symbol not found.", "callers": [], "callees": [], "details": {}}

        callers = _trace_callers(self.query, focus.symbol_id, depth=depth)
        callees = _trace_callees(self.query, focus.symbol_id, depth=depth)
        detail_ids = {focus.symbol_id}
        detail_ids.update(item["symbol_id"] for item in callers)
        detail_ids.update(item["symbol_id"] for item in callees)

        details: dict[str, dict] = {}
        for symbol_id in sorted(detail_ids):
            current = self.query.resolve_symbol(symbol_id)
            if current is None:
                continue
            details[symbol_id] = {
                "symbol": self._asdict_symbol(current),
                "signature": self.query.get_signature(symbol_id),
                "docstring": self.query.get_docstring(symbol_id),
                "source_excerpt": self.query.get_source_excerpt(symbol_id, max_lines=max_lines),
            }

        return {
            "focus_symbol": self._asdict_symbol(focus),
            "depth": depth,
            "callers": callers,
            "callees": callees,
            "details": details,
        }

    def _pr_diff_context(
        self,
        *,
        base: str,
        head: str,
        repo_root: str,
        depth: int,
        format: str,
        top_symbol_contexts: int,
        top_module_contexts: int,
    ) -> dict:
        changed_lines = _git_changed_lines(repo_root=repo_root, base=base, head=head)
        touched_symbols = _symbols_touched_by_diff(self.query, changed_lines)

        per_symbol: list[dict] = []
        affected_symbol_ids: set[str] = set()
        affected_files: set[str] = set()
        affected_repos: set[str] = set()
        module_hit_counts: dict[str, int] = defaultdict(int)

        for symbol in touched_symbols:
            impacted_by_level = self.query.impact_by_level(symbol.symbol_id, depth=depth)
            call_chain = self.query.call_chain(symbol.symbol_id, depth=2, direction="both")
            impacted_flat = []
            seen_local: set[str] = set()
            for _, symbols in sorted(impacted_by_level.items()):
                for impacted in symbols:
                    if impacted.symbol_id in seen_local:
                        continue
                    seen_local.add(impacted.symbol_id)
                    impacted_flat.append(impacted)

            total_impacted = len(impacted_flat)
            direct_callers = len(self.query.callers_of(symbol.symbol_id))
            direct_callees = len(self.query.callees_of(symbol.symbol_id))
            risk_score = total_impacted * 2 + direct_callers + direct_callees

            for item in [symbol] + impacted_flat:
                affected_symbol_ids.add(item.symbol_id)
                if item.file:
                    affected_files.add(item.file)
                    module = self.query.module_of_symbol(item.symbol_id)
                    if module:
                        module_hit_counts[module] += 1
                if item.repo:
                    affected_repos.add(item.repo)

            per_symbol.append(
                {
                    "symbol": self._asdict_symbol(symbol),
                    "risk_score": risk_score,
                    "summary": {
                        "impacted_symbols": total_impacted,
                        "direct_callers": direct_callers,
                        "direct_callees": direct_callees,
                    },
                    "impacted_by_level": {
                        str(level): [self._asdict_symbol(s) for s in symbols]
                        for level, symbols in sorted(impacted_by_level.items())
                    },
                    "call_chain": call_chain,
                }
            )

        per_symbol.sort(key=lambda item: item["risk_score"], reverse=True)
        top_touched = per_symbol[: top_symbol_contexts]

        top_modules = [
            module
            for module, _ in sorted(module_hit_counts.items(), key=lambda kv: kv[1], reverse=True)
        ][: top_module_contexts]

        symbol_contexts = []
        for item in top_touched:
            symbol_id = item["symbol"]["symbol_id"]
            symbol_contexts.append(
                {
                    "symbol_id": symbol_id,
                    "context": self.context.symbol_context(symbol_id),
                }
            )

        module_contexts = []
        for module in top_modules:
            module_contexts.append(
                {
                    "module": module,
                    "context": self.context.module_context(module),
                }
            )

        summary = {
            "changed_files": len(changed_lines),
            "touched_symbols": len(touched_symbols),
            "affected_symbols": len(affected_symbol_ids),
            "affected_files": len(affected_files),
            "affected_repos": len(affected_repos),
            "cross_repo": len(affected_repos) > 1,
            "risk_level": _pr_risk_level(
                touched_symbols=len(touched_symbols),
                affected_symbols=len(affected_symbol_ids),
                affected_repos=len(affected_repos),
                top_score=per_symbol[0]["risk_score"] if per_symbol else 0,
            ),
        }

        full_result = {
            "base": base,
            "head": head,
            "repo_root": str(Path(repo_root).resolve()),
            "summary": summary,
            "changed_files": {
                file: sorted(lines) for file, lines in sorted(changed_lines.items())
            },
            "touched_symbols": [self._asdict_symbol(s) for s in touched_symbols],
            "symbol_impacts": per_symbol,
            "contexts": {
                "top_symbol_contexts": symbol_contexts,
                "top_module_contexts": module_contexts,
            },
        }

        if format == "full":
            return full_result

        top_risky_symbols = [
            {
                "symbol_id": item["symbol"]["symbol_id"],
                "name": item["symbol"]["name"],
                "file": item["symbol"]["file"],
                "risk_score": item["risk_score"],
                "impacted_symbols": item["summary"]["impacted_symbols"],
                "direct_callers": item["summary"]["direct_callers"],
                "direct_callees": item["summary"]["direct_callees"],
            }
            for item in per_symbol[:10]
        ]
        top_affected_modules = [
            {"module": module, "hits": hits}
            for module, hits in sorted(module_hit_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
        ]
        return {
            "base": base,
            "head": head,
            "repo_root": str(Path(repo_root).resolve()),
            "summary": summary,
            "changed_files": sorted(changed_lines),
            "top_risky_symbols": top_risky_symbols,
            "top_affected_modules": top_affected_modules,
            "top_context_symbols": [item["symbol_id"] for item in symbol_contexts],
            "top_context_modules": [item["module"] for item in module_contexts],
        }

    @staticmethod
    def _asdict_symbol(symbol) -> dict:
        return {
            "symbol_id": symbol.symbol_id,
            "name": symbol.name,
            "kind": symbol.kind,
            "repo": symbol.repo,
            "file": symbol.file,
            "start_line": symbol.start_line,
            "end_line": symbol.end_line,
            "parent": symbol.parent,
        }

    def _module_context_with_fallback(self, module_name: str) -> dict:
        primary = self.context.module_context(module_name)
        if not _is_empty_module_context(primary):
            return primary

        fallback = self.context.module_or_directory_context(module_name)
        return {
            "focus": {"module": module_name, "fallback_used": True},
            "summary": (
                f"No direct module match for '{module_name}'. "
                "Used module_or_directory_context fallback."
            ),
            "facts": {
                "module_context": primary,
                "fallback_context": fallback,
            },
            "warnings": [
                f"module_context('{module_name}') was empty; fallback applied.",
            ],
        }

    def _module_contexts_with_fallback(self, modules: list[str]) -> dict:
        contexts: list[dict[str, Any]] = []
        fallback_count = 0
        for module_name in modules:
            context = self._module_context_with_fallback(module_name)
            if context.get("focus", {}).get("fallback_used"):
                fallback_count += 1
            contexts.append({"module": module_name, "context": context})

        summary = f"Built {len(contexts)} module context payload(s)."
        if fallback_count:
            summary += f" Applied fallback for {fallback_count} module name(s)."
        return {
            "focus": {"modules": modules},
            "summary": summary,
            "facts": {"contexts": contexts},
            "warnings": [],
        }

    def _module_dependencies_with_fallback(self, module_name: str) -> list[dict]:
        direct = list(self.query.module_dependencies_of(module_name))
        if direct:
            return [asdict(edge) for edge in direct]
        if module_name in set(self.query.modules()):
            return []

        return self._module_edges_via_directory_fallback(
            module_name,
            direction="out",
        )

    def _module_dependencies_ranked(
        self,
        module_name: str,
        *,
        max_edges: int = 20,
        include_ancillary: bool = False,
    ) -> list[dict]:
        edges = self._module_dependencies_with_fallback(module_name)
        aggregated: dict[tuple[str, str], dict[str, Any]] = {}
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            source_full = str(edge.get("source_module") or "")
            target_full = str(edge.get("target_module") or "")
            if _is_implementation_noise(source_full) or _is_implementation_noise(target_full):
                continue

            source = _architecture_component_name(source_full)
            target = _architecture_component_name(target_full)
            if source == target:
                continue

            score, reasons = _architecture_edge_score(source_full, target_full)
            ancillary = _is_ancillary_module(target) or _is_ancillary_module(source)
            if ancillary and not include_ancillary:
                continue

            key = (source, target)
            current = aggregated.get(key)
            candidate = {
                "source_module": source,
                "target_module": target,
                "kind": edge.get("kind", "module_depends"),
                "score": score,
                "ancillary": ancillary,
                "reasons": reasons,
                "source_example": source_full,
                "target_example": target_full,
            }
            if current is None or int(candidate["score"]) > int(current["score"]):
                aggregated[key] = candidate

        ranked = list(aggregated.values())
        ranked.sort(
            key=lambda row: (
                -int(row.get("score", 0)),
                str(row.get("source_module", "")),
                str(row.get("target_module", "")),
            )
        )
        if max_edges <= 0:
            return ranked
        return ranked[:max_edges]

    def _module_dependents_with_fallback(self, module_name: str) -> list[dict]:
        direct = list(self.query.module_dependents_of(module_name))
        if direct:
            return [asdict(edge) for edge in direct]
        if module_name in set(self.query.modules()):
            return []

        return self._module_edges_via_directory_fallback(
            module_name,
            direction="in",
        )

    def _module_edges_via_directory_fallback(self, name: str, direction: str) -> list[dict]:
        resolved = self.context.module_or_directory_context(name, recursive=True, max_modules=200)
        focus = resolved.get("focus", {})
        if focus.get("resolved_as") != "directory":
            return []

        facts = resolved.get("facts", {})
        modules = facts.get("modules", [])
        if not isinstance(modules, list):
            return []

        edges: list[dict] = []
        seen: set[tuple[str, str, str]] = set()
        for module_name in modules:
            if direction == "out":
                module_edges = self.query.module_dependencies_of(module_name)
            else:
                module_edges = self.query.module_dependents_of(module_name)
            for edge in module_edges:
                key = (edge.source_module, edge.target_module, edge.kind)
                if key in seen:
                    continue
                seen.add(key)
                edges.append(asdict(edge))
        return edges


def _trace_callers(query: QueryEngine, focus_symbol_id: str, depth: int) -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple[str, str, int]] = set()
    queue: list[tuple[str, int]] = [(focus_symbol_id, 0)]

    while queue:
        current_id, level = queue.pop(0)
        if level >= depth:
            continue
        for edge in query.dependent_edges_of(current_id, kind="calls"):
            caller = edge.source
            key = (caller.symbol_id, current_id, level + 1)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "depth": level + 1,
                    "symbol_id": caller.symbol_id,
                    "name": caller.name,
                    "kind": caller.kind,
                    "file": caller.file,
                    "line": caller.start_line,
                    "calls": current_id,
                }
            )
            queue.append((caller.symbol_id, level + 1))
    return rows


def _trace_callees(query: QueryEngine, focus_symbol_id: str, depth: int) -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple[str, str, int]] = set()
    queue: list[tuple[str, int]] = [(focus_symbol_id, 0)]

    while queue:
        current_id, level = queue.pop(0)
        if level >= depth:
            continue
        for edge in query.dependency_edges_of(current_id, kind="calls"):
            callee = edge.target
            key = (current_id, callee.symbol_id, level + 1)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "depth": level + 1,
                    "symbol_id": callee.symbol_id,
                    "name": callee.name,
                    "kind": callee.kind,
                    "file": callee.file,
                    "line": callee.start_line,
                    "called_from": current_id,
                }
            )
            queue.append((callee.symbol_id, level + 1))
    return rows


def _json_ready(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _to_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def _is_empty_module_context(context: Any) -> bool:
    if not isinstance(context, dict):
        return True
    facts = context.get("facts")
    if not isinstance(facts, dict):
        return True
    symbols = facts.get("symbols")
    depends = facts.get("depends_on_modules")
    dependents = facts.get("dependent_modules")
    return (
        isinstance(symbols, list)
        and isinstance(depends, list)
        and isinstance(dependents, list)
        and len(symbols) == 0
        and len(depends) == 0
        and len(dependents) == 0
    )


def _is_ancillary_module(name: str) -> bool:
    lowered = name.lower()
    patterns = (
        ".docs",
        ".doc",
        ".tests",
        ".test",
        ".dev",
        ".example",
        ".examples",
        ".benchmark",
        ".bench",
    )
    if any(token in lowered for token in patterns):
        return True
    leaves = ("docs", "doc", "tests", "test", "dev", "examples", "benchmark", "bench")
    return lowered.split(".")[-1] in leaves


def _architecture_edge_score(source_module: str, target_module: str) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []

    if source_module and target_module:
        score += 2
        reasons.append("base")

    source_root = source_module.split(".", 1)[0] if source_module else ""
    target_root = target_module.split(".", 1)[0] if target_module else ""
    if source_root and target_root and source_root != target_root:
        score += 4
        reasons.append("cross_root")

    if any(token in target_module for token in ("providers", "task-sdk", "airflow-ctl", "airflow_core", "airflow.core", "core")):
        score += 3
        reasons.append("domain_signal")

    depth = target_module.count(".")
    if depth <= 3:
        score += 2
        reasons.append("shallow_target")
    elif depth >= 7:
        score -= 2
        reasons.append("deep_target")

    if _is_ancillary_module(source_module) or _is_ancillary_module(target_module):
        score -= 5
        reasons.append("ancillary_penalty")

    return score, reasons


def _architecture_component_name(module_name: str) -> str:
    parts = [part for part in module_name.split(".") if part]
    if not parts:
        return module_name
    # Typical normalized module roots in indexed repos.
    roots = {
        "airflow-core",
        "airflow-ctl",
        "airflow-ctl-tests",
        "airflow-e2e-tests",
        "task-sdk",
        "providers",
        "dev",
        "devel-common",
        "clients",
        "chart",
    }
    if parts[0] in roots:
        return parts[0]
    return parts[0]


def _is_implementation_noise(name: str) -> bool:
    lowered = name.lower()
    noisy_tokens = (
        ".hatch_build",
        ".setup",
        ".conftest",
        ".docs.conf",
        ".empty_plugin",
        ".scripts.",
        ".script.",
    )
    return any(token in lowered for token in noisy_tokens)


def _git_changed_lines(*, repo_root: str, base: str, head: str) -> dict[str, set[int]]:
    cmd = [
        "git",
        "-C",
        str(Path(repo_root).resolve()),
        "diff",
        "--unified=0",
        "--no-color",
        f"{base}...{head}",
    ]
    diff_text = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)

    changed: dict[str, set[int]] = defaultdict(set)
    current_file: str | None = None
    hunk_re = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")

    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            raw = line[4:].strip()
            if raw == "/dev/null":
                current_file = None
            elif raw.startswith("b/"):
                current_file = raw[2:]
            else:
                current_file = raw
            continue

        if current_file is None:
            continue

        match = hunk_re.match(line)
        if not match:
            continue

        start = int(match.group(1))
        count = int(match.group(2) or "1")
        if count <= 0:
            changed[current_file].add(start)
            continue
        for lineno in range(start, start + count):
            changed[current_file].add(lineno)

    return changed


def _symbols_touched_by_diff(query: QueryEngine, changed_lines: dict[str, set[int]]) -> list:
    touched = []
    for symbol in query.all_symbols():
        lines = changed_lines.get(symbol.file)
        if not lines:
            continue
        if any(symbol.start_line <= line <= symbol.end_line for line in lines):
            touched.append(symbol)
    touched.sort(key=lambda s: (s.file, s.start_line, s.end_line, s.symbol_id))
    return touched


def _pr_risk_level(*, touched_symbols: int, affected_symbols: int, affected_repos: int, top_score: int) -> str:
    if affected_repos > 1 or affected_symbols >= 40 or top_score >= 40:
        return "high"
    if touched_symbols >= 5 or affected_symbols >= 15 or top_score >= 15:
        return "medium"
    return "low"
