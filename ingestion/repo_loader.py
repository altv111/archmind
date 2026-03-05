from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class SourceFile:
    repo: str
    path: str
    language: str
    content: str


_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "env",
    "build",
    "dist",
    "target",
}

_EXTENSION_LANGUAGE_MAP = {
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".swift": "swift",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".scala": "scala",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".sql": "sql",
    ".xml": "xml",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
}

_SPECIAL_FILENAMES = {
    "dockerfile": "dockerfile",
    "makefile": "makefile",
    "cmakelists.txt": "cmake",
}


def detect_language(path: Path) -> str:
    name = path.name.lower()
    if name in _SPECIAL_FILENAMES:
        return _SPECIAL_FILENAMES[name]

    return _EXTENSION_LANGUAGE_MAP.get(path.suffix.lower(), "text")


def _is_virtualenv_dir(path: Path) -> bool:
    if (path / "pyvenv.cfg").is_file():
        return True

    activate = path / "bin" / "activate"
    if not activate.is_file():
        return False

    lib_dir = path / "lib"
    if not lib_dir.is_dir():
        return False

    for candidate in lib_dir.iterdir():
        if candidate.is_dir() and candidate.name.startswith("python"):
            if (candidate / "site-packages").is_dir():
                return True
    return False


def _iter_text_files(repo_root: Path, skip_dirs: set[str]) -> Iterator[Path]:
    for current_root, dir_names, file_names in os.walk(repo_root):
        current_root_path = Path(current_root)
        filtered: list[str] = []
        for dir_name in dir_names:
            if dir_name in skip_dirs:
                continue
            if _is_virtualenv_dir(current_root_path / dir_name):
                continue
            filtered.append(dir_name)
        dir_names[:] = filtered

        for file_name in file_names:
            file_path = current_root_path / file_name
            if file_path.is_symlink():
                continue
            yield file_path


def _read_text(path: Path) -> str | None:
    try:
        raw = path.read_bytes()
    except OSError:
        return None

    if b"\x00" in raw:
        return None

    for encoding in ("utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def iter_repo(
    repo_path: str | Path, skip_dirs: Iterable[str] | None = None
) -> Iterator[SourceFile]:
    repo_root = Path(repo_path).expanduser().resolve()
    if not repo_root.exists() or not repo_root.is_dir():
        raise ValueError(f"Invalid repository path: {repo_path}")

    effective_skip_dirs = set(_SKIP_DIRS)
    if skip_dirs:
        effective_skip_dirs.update(skip_dirs)

    repo_name = repo_root.name
    for file_path in _iter_text_files(repo_root, effective_skip_dirs):
        content = _read_text(file_path)
        if content is None:
            continue

        yield SourceFile(
            repo=repo_name,
            path=file_path.relative_to(repo_root).as_posix(),
            language=detect_language(file_path),
            content=content,
        )


def iter_repos(
    repo_paths: str | Path | Iterable[str | Path],
    skip_dirs: Iterable[str] | None = None,
) -> Iterator[SourceFile]:
    if isinstance(repo_paths, (str, Path)):
        paths: list[str | Path] = [repo_paths]
    else:
        paths = list(repo_paths)

    for repo_path in paths:
        yield from iter_repo(repo_path, skip_dirs=skip_dirs)


def load_repos(
    repo_paths: str | Path | Iterable[str | Path],
    skip_dirs: Iterable[str] | None = None,
) -> list[SourceFile]:
    return list(iter_repos(repo_paths, skip_dirs=skip_dirs))
