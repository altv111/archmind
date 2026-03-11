from __future__ import annotations

from collections import Counter
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
            lambda max_entries=20, readme_max_lines=40, top_modules=10: self._inspect_repo(
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
            lambda module: [asdict(edge) for edge in self.query.module_dependencies_of(module)],
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
            lambda module: self.context.module_context(module),
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

    def _inspect_repo(self, *, max_entries: int, readme_max_lines: int, top_modules: int) -> dict:
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
