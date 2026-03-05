from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterable

from ingestion.repo_loader import SourceFile

try:
    from tree_sitter import Parser
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "tree_sitter is required. Install with: pip install tree-sitter"
    ) from exc

if TYPE_CHECKING:
    from tree_sitter import Language, Tree
else:
    Language = Any
    Tree = Any


@dataclass(frozen=True)
class ParsedFile:
    repo: str
    path: str
    language: str
    source_bytes: bytes
    ast: Tree


class CodeParser:
    def __init__(
        self,
        language_registry: dict[str, Language] | None = None,
        language_loader: Callable[[str], Language] | None = None,
    ) -> None:
        self._language_registry: dict[str, Language] = dict(language_registry or {})
        self._language_loader = language_loader
        self._parser_cache: dict[str, Parser] = {}

    def parse(self, source_file: SourceFile) -> ParsedFile:
        parser = self._get_parser(source_file.language)
        source_bytes = source_file.content.encode("utf-8")
        ast = parser.parse(source_bytes)
        return ParsedFile(
            repo=source_file.repo,
            path=source_file.path,
            language=source_file.language,
            source_bytes=source_bytes,
            ast=ast,
        )

    def parse_many(self, source_files: Iterable[SourceFile]) -> list[ParsedFile]:
        return [self.parse(source_file) for source_file in source_files]

    def register_language(self, language_name: str, language: Language) -> None:
        self._language_registry[language_name] = language
        self._parser_cache.pop(language_name, None)

    def _get_parser(self, language_name: str) -> Parser:
        parser = self._parser_cache.get(language_name)
        if parser is not None:
            return parser

        parser = self._get_default_parser(language_name)
        if parser is not None:
            self._parser_cache[language_name] = parser
            return parser

        language = self._get_language(language_name)
        parser = Parser()
        self._set_parser_language(parser, language)
        self._parser_cache[language_name] = parser
        return parser

    def _get_default_parser(self, language_name: str) -> Parser | None:
        if language_name in self._language_registry or self._language_loader is not None:
            return None

        try:
            from tree_sitter_languages import get_parser
        except ImportError:
            return None

        try:
            parser = get_parser(language_name)
        except Exception:
            return None

        if parser is None:
            return None
        return parser

    def _get_language(self, language_name: str) -> Language:
        if language_name in self._language_registry:
            return self._language_registry[language_name]

        if self._language_loader is not None:
            language = self._language_loader(language_name)
            self._language_registry[language_name] = language
            return language

        try:
            from tree_sitter_languages import get_language
        except ImportError as exc:
            raise ValueError(
                f"No Tree-sitter grammar configured for '{language_name}'. "
                "Register languages via `register_language(...)`, pass a "
                "`language_loader`, or install tree_sitter_languages."
            ) from exc

        try:
            language = get_language(language_name)
        except Exception as exc:
            raise ValueError(
                f"Tree-sitter grammar not found for language '{language_name}'."
            ) from exc

        self._language_registry[language_name] = language
        return language

    @staticmethod
    def _set_parser_language(parser: Parser, language: Language) -> None:
        if hasattr(parser, "set_language"):
            parser.set_language(language)
            return
        parser.language = language
