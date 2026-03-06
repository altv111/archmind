from __future__ import annotations

from dataclasses import dataclass
import re
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from ingestion.code_parser import ParsedFile
    from tree_sitter import Node
else:
    ParsedFile = Any
    Node = Any


@dataclass(frozen=True)
class Dependency:
    source_symbol: str
    target_symbol: str
    kind: str
    file: str
    line: int


class BaseLanguageDependencyExtractor:
    SCOPE_NODE_TYPES: set[str] = set()
    IMPORT_NODE_TYPES: set[str] = set()
    CALL_NODE_TYPES: set[str] = set()
    INHERIT_NODE_TYPES: set[str] = set()

    def extract(self, parsed_file: ParsedFile) -> list[Dependency]:
        source_bytes = parsed_file.source_bytes
        file_symbol = parsed_file.path
        file_path = parsed_file.path
        out: list[Dependency] = []
        seen: set[tuple[str, str, str, str, int]] = set()

        stack: list[tuple[Node, str | None]] = [(parsed_file.ast.root_node, None)]
        while stack:
            node, scope_symbol = stack.pop()
            current_scope = scope_symbol

            next_scope = self._scope_name(node, source_bytes) or current_scope
            source_for_calls = current_scope or file_symbol

            if node.type in self.IMPORT_NODE_TYPES:
                line = int(node.start_point[0]) + 1
                for target in self._import_targets(node, source_bytes):
                    self._append_dependency(
                        out,
                        seen,
                        Dependency(
                            source_symbol=file_symbol,
                            target_symbol=target,
                            kind="imports",
                            file=file_path,
                            line=line,
                        ),
                    )

            if node.type in self.INHERIT_NODE_TYPES:
                inherit_data = self._inheritance(node, source_bytes)
                if inherit_data is not None:
                    line = int(node.start_point[0]) + 1
                    source_symbol, targets = inherit_data
                    for target in targets:
                        self._append_dependency(
                            out,
                            seen,
                            Dependency(
                                source_symbol=source_symbol,
                                target_symbol=target,
                                kind="inherits",
                                file=file_path,
                                line=line,
                            ),
                        )

            if node.type in self.CALL_NODE_TYPES:
                target_symbol = self._call_target(node, source_bytes)
                if target_symbol and self._should_emit_call_target(target_symbol):
                    line = int(node.start_point[0]) + 1
                    self._append_dependency(
                        out,
                        seen,
                        Dependency(
                            source_symbol=source_for_calls,
                            target_symbol=target_symbol,
                            kind="calls",
                            file=file_path,
                            line=line,
                        ),
                    )

            for child in reversed(node.children):
                stack.append((child, next_scope))

        return out

    def _scope_name(self, node: Node, source_bytes: bytes) -> str | None:
        if node.type not in self.SCOPE_NODE_TYPES:
            return None
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return None
        return self._clean_symbol(self._node_text(name_node, source_bytes))

    def _import_targets(self, node: Node, source_bytes: bytes) -> list[str]:
        return []

    def _call_target(self, node: Node, source_bytes: bytes) -> str | None:
        return None

    def _should_emit_call_target(self, target_symbol: str) -> bool:
        return True

    def _inheritance(
        self, node: Node, source_bytes: bytes
    ) -> tuple[str, list[str]] | None:
        return None

    @staticmethod
    def _append_dependency(
        out: list[Dependency],
        seen: set[tuple[str, str, str, str, int]],
        dependency: Dependency,
    ) -> None:
        if not dependency.source_symbol or not dependency.target_symbol:
            return
        key = (
            dependency.source_symbol,
            dependency.target_symbol,
            dependency.kind,
            dependency.file,
            dependency.line,
        )
        if key in seen:
            return
        seen.add(key)
        out.append(dependency)

    @staticmethod
    def _node_text(node: Node, source_bytes: bytes) -> str:
        return source_bytes[node.start_byte : node.end_byte].decode(
            "utf-8", errors="ignore"
        ).strip()

    @staticmethod
    def _clean_symbol(symbol: str) -> str:
        return symbol.strip().strip("()")

    @staticmethod    
    def _normalize_call_target(target: str) -> str:
        normalized = target.strip()
        normalized = re.sub(r"\s+", "", normalized)
        normalized = re.sub(r"\(.*\)$", "", normalized)

        # remove common instance prefixes
        for prefix in ("self.", "this.", "cls."):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]        

        return normalized    

    def _first_descendant_text_by_types(
        self, node: Node, source_bytes: bytes, types: set[str]
    ) -> str | None:
        stack = [node]
        while stack:
            current = stack.pop()
            if current.type in types:
                text = self._node_text(current, source_bytes)
                if text:
                    return text
            for child in reversed(current.children):
                stack.append(child)
        return None

    def _collect_descendant_texts_by_types(
        self, node: Node, source_bytes: bytes, types: set[str]
    ) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        stack = [node]
        while stack:
            current = stack.pop()
            if current.type in types:
                text = self._clean_symbol(self._node_text(current, source_bytes))
                if text and text not in seen:
                    seen.add(text)
                    out.append(text)
            for child in reversed(current.children):
                stack.append(child)
        return out


class PythonDependencyExtractor(BaseLanguageDependencyExtractor):
    SCOPE_NODE_TYPES = {"function_definition", "class_definition"}
    IMPORT_NODE_TYPES = {"import_statement", "import_from_statement"}
    CALL_NODE_TYPES = {"call"}
    INHERIT_NODE_TYPES = {"class_definition"}
    BUILTIN_FUNCTIONS = {
        "print",
        "len",
        "list",
        "set",
        "dict",
        "tuple",
        "str",
        "int",
        "float",
        "bool",
        "sum",
        "min",
        "max",
        "sorted",
        "range",
        "enumerate",
        "zip",
        "map",
        "filter",
        "any",
        "all",
        "isinstance",
    }

    def _import_targets(self, node: Node, source_bytes: bytes) -> list[str]:
        if node.type == "import_from_statement":
            module_node = node.child_by_field_name("module_name")
            if module_node is not None:
                module = self._clean_symbol(self._node_text(module_node, source_bytes))
                if module:
                    return [module]

            dotted = self._first_descendant_text_by_types(
                node, source_bytes, {"dotted_name", "identifier"}
            )
            if dotted:
                return [self._clean_symbol(dotted)]

            text = self._node_text(node, source_bytes)
            match = re.match(r"from\s+([A-Za-z_][\w\.]*)\s+import\s+", text)
            return [match.group(1)] if match else []

        targets: list[str] = []
        if node.type == "import_statement":
            targets = self._collect_descendant_texts_by_types(
                node, source_bytes, {"dotted_name"}
            )
            if targets:
                return targets

            text = self._node_text(node, source_bytes)
            if text.startswith("import "):
                body = text[len("import ") :]
                for part in body.split(","):
                    module = part.strip().split(" as ")[0].strip()
                    if module:
                        targets.append(module)
        return targets

    def _call_target(self, node: Node, source_bytes: bytes) -> str | None:
        func_node = node.child_by_field_name("function")
        if func_node is None:
            return None

        if func_node.type == "attribute":
            object_node = func_node.child_by_field_name("object")
            attr_node = func_node.child_by_field_name("attribute")
            if attr_node is None:
                return None

            method = self._clean_symbol(self._node_text(attr_node, source_bytes))
            if object_node is None:
                return method

            obj = self._clean_symbol(self._node_text(object_node, source_bytes))
            return self._normalize_call_target(f"{obj}.{method}")

        if func_node.type == "identifier":
            return self._clean_symbol(self._node_text(func_node, source_bytes))

        return self._normalize_call_target(self._node_text(func_node, source_bytes))

    def _should_emit_call_target(self, target_symbol: str) -> bool:
        if "." in target_symbol:
            return True
        return target_symbol not in self.BUILTIN_FUNCTIONS

    def _inheritance(
        self, node: Node, source_bytes: bytes
    ) -> tuple[str, list[str]] | None:
        class_name = self._scope_name(node, source_bytes)
        if not class_name:
            return None

        superclasses = node.child_by_field_name("superclasses")
        if superclasses is not None:
            bases = [
                self._clean_symbol(self._node_text(child, source_bytes))
                for child in superclasses.named_children
                if self._clean_symbol(self._node_text(child, source_bytes))
            ]
            if bases:
                return (class_name, bases)

        text = self._node_text(node, source_bytes)
        match = re.match(r"class\s+[A-Za-z_][\w]*\(([^)]*)\)\s*:", text)
        if not match:
            return None
        bases = [
            self._clean_symbol(part)
            for part in match.group(1).split(",")
            if self._clean_symbol(part)
        ]
        return (class_name, bases) if bases else None


class JavaDependencyExtractor(BaseLanguageDependencyExtractor):
    SCOPE_NODE_TYPES = {"method_declaration", "constructor_declaration", "class_declaration"}
    IMPORT_NODE_TYPES = {"import_declaration"}
    CALL_NODE_TYPES = {"method_invocation", "object_creation_expression"}
    INHERIT_NODE_TYPES = {"class_declaration", "interface_declaration"}

    def _import_targets(self, node: Node, source_bytes: bytes) -> list[str]:
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            name = self._clean_symbol(self._node_text(name_node, source_bytes))
            if name:
                return [name]

        scoped = self._first_descendant_text_by_types(
            node, source_bytes, {"scoped_identifier", "identifier"}
        )
        if scoped:
            return [self._clean_symbol(scoped)]

        text = self._node_text(node, source_bytes)
        match = re.match(r"import\s+(static\s+)?([^;]+);", text)
        return [match.group(2).strip()] if match else []

    def _call_target(self, node: Node, source_bytes: bytes) -> str | None:
        if node.type == "object_creation_expression":
            type_node = node.child_by_field_name("type")
            if type_node is not None:
                return self._normalize_call_target(self._node_text(type_node, source_bytes))

            text = self._node_text(node, source_bytes)
            match_new = re.match(r"new\s+([A-Za-z_][\w\.]*)\s*\(", text)
            return self._normalize_call_target(match_new.group(1)) if match_new else None

        object_node = node.child_by_field_name("object")
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            name_text = self._clean_symbol(self._node_text(name_node, source_bytes))
            if object_node is not None:
                object_text = self._clean_symbol(self._node_text(object_node, source_bytes))
                return self._normalize_call_target(f"{object_text}.{name_text}")
            return self._normalize_call_target(name_text)

        text = self._node_text(node, source_bytes)
        match_method = re.match(r"([A-Za-z_][\w\.]*)\s*\(", text)
        return self._normalize_call_target(match_method.group(1)) if match_method else None

    def _inheritance(
        self, node: Node, source_bytes: bytes
    ) -> tuple[str, list[str]] | None:
        source = self._scope_name(node, source_bytes)
        if not source:
            return None

        targets: list[str] = []
        for field_name in (
            "superclass",
            "interfaces",
            "super_interfaces",
            "extends_interfaces",
        ):
            field_node = node.child_by_field_name(field_name)
            if field_node is None:
                continue
            targets.extend(
                self._collect_descendant_texts_by_types(
                    field_node,
                    source_bytes,
                    {"type_identifier", "identifier", "scoped_identifier"},
                )
            )

        if not targets:
            text = self._node_text(node, source_bytes)
            extends_match = re.search(r"\bextends\s+([A-Za-z_][\w\.]*)", text)
            if extends_match:
                targets.append(extends_match.group(1))

            implements_match = re.search(r"\bimplements\s+([A-Za-z0-9_,\s\.]+)\{", text)
            if implements_match:
                interfaces = [
                    self._clean_symbol(x)
                    for x in implements_match.group(1).split(",")
                    if self._clean_symbol(x)
                ]
                targets.extend(interfaces)

        return (source, targets) if targets else None


class CppDependencyExtractor(BaseLanguageDependencyExtractor):
    SCOPE_NODE_TYPES = {"function_definition", "class_specifier", "struct_specifier"}
    IMPORT_NODE_TYPES = {"preproc_include"}
    CALL_NODE_TYPES = {"call_expression"}
    INHERIT_NODE_TYPES = {"class_specifier", "struct_specifier"}

    def _scope_name(self, node: Node, source_bytes: bytes) -> str | None:
        if node.type == "function_definition":
            declarator = node.child_by_field_name("declarator")
            if declarator is not None:
                text = self._node_text(declarator, source_bytes)
                return self._normalize_call_target(text.split("(")[0])

        if node.type in {"class_specifier", "struct_specifier"}:
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                return self._clean_symbol(self._node_text(name_node, source_bytes))

        return None

    def _import_targets(self, node: Node, source_bytes: bytes) -> list[str]:
        path_node = node.child_by_field_name("path")
        if path_node is not None:
            path_text = self._node_text(path_node, source_bytes).strip("<>\"")
            if path_text:
                return [path_text]

        for child in node.named_children:
            if child.type in {"system_lib_string", "string_literal"}:
                path_text = self._node_text(child, source_bytes).strip("<>\"")
                if path_text:
                    return [path_text]

        text = self._node_text(node, source_bytes)
        match = re.search(r'#include\s*[<"]([^>"]+)[>"]', text)
        return [match.group(1)] if match else []

    def _call_target(self, node: Node, source_bytes: bytes) -> str | None:
        function_node = node.child_by_field_name("function")
        if function_node is None:
            text = self._node_text(node, source_bytes)
            text = text.split("(", 1)[0]
            return self._normalize_call_target(text)
        return self._normalize_call_target(self._node_text(function_node, source_bytes))

    def _inheritance(
        self, node: Node, source_bytes: bytes
    ) -> tuple[str, list[str]] | None:
        source = self._scope_name(node, source_bytes)
        if not source:
            return None

        bases: list[str] = []
        base_clause = next((c for c in node.children if c.type == "base_class_clause"), None)
        if base_clause is not None:
            bases = self._collect_descendant_texts_by_types(
                base_clause,
                source_bytes,
                {
                    "type_identifier",
                    "identifier",
                    "qualified_identifier",
                    "namespace_identifier",
                    "scoped_identifier",
                },
            )
            bases = [
                b
                for b in bases
                if b not in {"public", "protected", "private", "virtual"}
            ]

        if not bases:
            text = self._node_text(node, source_bytes)
            match = re.search(r":\s*([^\\{]+)\{", text)
            if not match:
                return None
            for part in match.group(1).split(","):
                cleaned = re.sub(
                    r"\b(public|protected|private|virtual)\b", "", part
                ).strip()
                base = cleaned.split()[-1] if cleaned else ""
                if base:
                    bases.append(base)
        return (source, bases) if bases else None


class DependencyExtractor:
    def __init__(self) -> None:
        self._extractors: dict[str, BaseLanguageDependencyExtractor] = {
            "python": PythonDependencyExtractor(),
            "java": JavaDependencyExtractor(),
            "cpp": CppDependencyExtractor(),
            "c": CppDependencyExtractor(),
        }

    def register_extractor(
        self, language: str, extractor: BaseLanguageDependencyExtractor
    ) -> None:
        self._extractors[language] = extractor

    def extract(self, parsed_file: ParsedFile) -> list[Dependency]:
        extractor = self._extractors.get(parsed_file.language)
        if extractor is None:
            return []
        return extractor.extract(parsed_file)

    def extract_many(self, parsed_files: Iterable[ParsedFile]) -> list[Dependency]:
        out: list[Dependency] = []
        for parsed_file in parsed_files:
            out.extend(self.extract(parsed_file))
        return out
