try:
    # Keep DB-only commands usable even if parser runtime deps (tree_sitter)
    # are not installed in the active environment.
    from .code_parser import CodeParser, ParsedFile
except ImportError:  # pragma: no cover - runtime environment dependent
    CodeParser = None  # type: ignore[assignment]
    ParsedFile = None  # type: ignore[assignment]
from .containment_extractor import Containment, ContainmentExtractor
from .dependency_extractor import (
    BaseLanguageDependencyExtractor,
    CppDependencyExtractor,
    Dependency,
    DependencyExtractor,
    JavaScriptDependencyExtractor,
    JavaDependencyExtractor,
    PythonDependencyExtractor,
    TypeScriptDependencyExtractor,
)
from .repo_loader import SourceFile, iter_repo, iter_repos, load_repos
from .symbol_extractor import (
    BaseLanguageSymbolExtractor,
    CppSymbolExtractor,
    JavaScriptSymbolExtractor,
    JavaSymbolExtractor,
    PythonSymbolExtractor,
    Symbol,
    SymbolExtractor,
    TypeScriptSymbolExtractor,
)

__all__ = [
    "CodeParser",
    "ParsedFile",
    "SourceFile",
    "Containment",
    "ContainmentExtractor",
    "Dependency",
    "DependencyExtractor",
    "BaseLanguageDependencyExtractor",
    "CppDependencyExtractor",
    "PythonDependencyExtractor",
    "JavaDependencyExtractor",
    "JavaScriptDependencyExtractor",
    "TypeScriptDependencyExtractor",
    "BaseLanguageSymbolExtractor",
    "CppSymbolExtractor",
    "PythonSymbolExtractor",
    "JavaSymbolExtractor",
    "JavaScriptSymbolExtractor",
    "TypeScriptSymbolExtractor",
    "Symbol",
    "SymbolExtractor",
    "iter_repo",
    "iter_repos",
    "load_repos",
]
