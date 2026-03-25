from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from ingestion.code_parser import ParsedFile
    from tree_sitter import Node
else:
    ParsedFile = Any
    Node = Any


@dataclass(frozen=True)
class Symbol:
    symbol_id: str
    name: str
    kind: str
    repo: str
    file: str
    start_line: int
    end_line: int
    parent: str | None


class BaseLanguageSymbolExtractor:
    SYMBOL_NODE_KIND: dict[str, str] = {}
    NAME_FIELD_CANDIDATES: tuple[str, ...] = (
        "name",
        "declarator",
        "declaration",
        "type",
        "function",
        "property",
        "field",
    )
    IDENTIFIER_TYPES: set[str] = {
        "identifier",
        "type_identifier",
        "field_identifier",
        "property_identifier",
        "scoped_identifier",
        "qualified_identifier",
        "namespace_identifier",
        "name",
    }

    def extract(self, parsed_file: ParsedFile) -> list[Symbol]:
        root = parsed_file.ast.root_node
        source_bytes = parsed_file.source_bytes
        out: list[Symbol] = []
        seen: set[str] = set()

        stack: list[tuple[Node, str | None]] = [(root, None)]
        while stack:
            node, parent_id = stack.pop()
            current_parent_id = parent_id
            symbol_kind = self._symbol_kind_for_node(node, source_bytes)
            if symbol_kind is not None:
                symbol_name = self._extract_name(node, source_bytes)
                if symbol_name:
                    start_line = int(node.start_point[0]) + 1
                    end_line = int(node.end_point[0]) + 1
                    symbol_id = self._make_symbol_id(
                        parsed_file.repo,
                        parsed_file.path,
                        symbol_kind,
                        symbol_name,
                        start_line,
                        end_line,
                    )
                    if symbol_id not in seen:
                        seen.add(symbol_id)
                        out.append(
                            Symbol(
                                symbol_id=symbol_id,
                                name=symbol_name,
                                kind=symbol_kind,
                                repo=parsed_file.repo,
                                file=parsed_file.path,
                                start_line=start_line,
                                end_line=end_line,
                                parent=parent_id,
                            )
                        )
                        current_parent_id = symbol_id

            for child in reversed(node.children):
                stack.append((child, current_parent_id))

        return out

    def _symbol_kind_for_node(self, node: Node, source_bytes: bytes) -> str | None:
        del source_bytes
        return self.SYMBOL_NODE_KIND.get(node.type)

    def _extract_name(self, node: Node, source_bytes: bytes) -> str | None:
        for field_name in self.NAME_FIELD_CANDIDATES:
            field_node = node.child_by_field_name(field_name)
            if field_node is None:
                continue
            identifier = self._identifier_from_node(field_node, source_bytes)
            if identifier:
                return identifier
        return self._identifier_from_node(node, source_bytes)

    def _identifier_from_node(self, node: Node, source_bytes: bytes) -> str | None:
        if node.type in self.IDENTIFIER_TYPES:
            node_text = self._node_text(node, source_bytes)
            if node_text:
                return node_text

        queue = deque([node])
        while queue:
            current = queue.popleft()
            if current.type in self.IDENTIFIER_TYPES:
                node_text = self._node_text(current, source_bytes)
                if node_text:
                    return node_text
            queue.extend(current.children)
        return None

    @staticmethod
    def _node_text(node: Node, source_bytes: bytes) -> str:
        return source_bytes[node.start_byte : node.end_byte].decode(
            "utf-8", errors="ignore"
        ).strip()

    @staticmethod
    def _make_symbol_id(
        repo: str,
        file: str,
        kind: str,
        name: str,
        start_line: int,
        end_line: int,
    ) -> str:
        return f"{repo}:{file}:{kind}:{name}:{start_line}:{end_line}"


class CppSymbolExtractor(BaseLanguageSymbolExtractor):
    SYMBOL_NODE_KIND = {
        "namespace_definition": "namespace",
        "class_specifier": "class",
        "struct_specifier": "class",
        "enum_specifier": "enum",
        "function_definition": "function",
        "function_declarator": "function",
        "method_definition": "method",
    }


class PythonSymbolExtractor(BaseLanguageSymbolExtractor):
    SYMBOL_NODE_KIND = {
        "class_definition": "class",
        "function_definition": "function",
    }


class JavaSymbolExtractor(BaseLanguageSymbolExtractor):
    SYMBOL_NODE_KIND = {
        "class_declaration": "class",
        "interface_declaration": "interface",
        "enum_declaration": "enum",
        "record_declaration": "class",
        "method_declaration": "method",
        "constructor_declaration": "method",
    }


class JavaScriptSymbolExtractor(BaseLanguageSymbolExtractor):
    SYMBOL_NODE_KIND = {
        "class_declaration": "class",
        "function_declaration": "function",
        "method_definition": "method",
        "generator_function_declaration": "function",
        "variable_declarator": "function",
    }

    def _symbol_kind_for_node(self, node: Node, source_bytes: bytes) -> str | None:
        if node.type != "variable_declarator":
            return super()._symbol_kind_for_node(node, source_bytes)

        value_node = node.child_by_field_name("value")
        if value_node is None:
            return None
        if value_node.type in {
            "arrow_function",
            "function",
            "function_expression",
            "generator_function",
            "generator_function_declaration",
        }:
            return "function"
        return None


class TypeScriptSymbolExtractor(JavaScriptSymbolExtractor):
    SYMBOL_NODE_KIND = {
        **JavaScriptSymbolExtractor.SYMBOL_NODE_KIND,
        "interface_declaration": "interface",
        "type_alias_declaration": "type_alias",
        "enum_declaration": "enum",
        "abstract_class_declaration": "class",
    }


class SymbolExtractor:
    def __init__(self) -> None:
        self._extractors: dict[str, BaseLanguageSymbolExtractor] = {
            "cpp": CppSymbolExtractor(),
            "c": CppSymbolExtractor(),
            "python": PythonSymbolExtractor(),
            "java": JavaSymbolExtractor(),
            "javascript": JavaScriptSymbolExtractor(),
            "typescript": TypeScriptSymbolExtractor(),
        }

    def register_extractor(
        self, language: str, extractor: BaseLanguageSymbolExtractor
    ) -> None:
        self._extractors[language] = extractor

    def extract(self, parsed_file: ParsedFile) -> list[Symbol]:
        extractor = self._extractors.get(parsed_file.language)
        if extractor is None:
            return []
        return extractor.extract(parsed_file)

    def extract_many(self, parsed_files: Iterable[ParsedFile]) -> list[Symbol]:
        symbols: list[Symbol] = []
        for parsed_file in parsed_files:
            symbols.extend(self.extract(parsed_file))
        return symbols
