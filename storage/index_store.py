from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Iterable

from graph.directory_graph_builder import DirectoryEdge
from graph.module_graph_builder import ModuleDependency
from ingestion.dependency_extractor import Dependency
from ingestion.symbol_extractor import Symbol


@dataclass(frozen=True)
class IndexRun:
    run_id: int
    started_at: str
    status: str
    commit: str | None
    notes: str | None


class IndexStore:
    """SQLite-backed persistence for incremental repository indexing."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._enable_pragmas()
        self.init_schema()

    def close(self) -> None:
        self.conn.close()

    def reset_store(self) -> None:
        """Delete all indexed data while keeping schema."""
        self.conn.executescript(
            """
            DELETE FROM directory_edges;
            DELETE FROM module_edges;
            DELETE FROM dependencies;
            DELETE FROM symbols;
            DELETE FROM ast_artifacts;
            DELETE FROM files;
            DELETE FROM index_runs;
            """
        )
        self.conn.commit()

    def _enable_pragmas(self) -> None:
        # WAL gives better concurrent read/write behavior for local workloads.
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")

    def init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS index_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                status TEXT NOT NULL,
                commit_hash TEXT,
                notes TEXT
            );

            CREATE TABLE IF NOT EXISTS files (
                file_id INTEGER PRIMARY KEY AUTOINCREMENT,
                repo TEXT NOT NULL,
                path TEXT NOT NULL,
                language TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                last_run_id INTEGER,
                indexed_at TEXT NOT NULL,
                UNIQUE(repo, path),
                FOREIGN KEY(last_run_id) REFERENCES index_runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS ast_artifacts (
                ast_id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                run_id INTEGER NOT NULL,
                parser_language TEXT NOT NULL,
                parser_version TEXT,
                ast_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(file_id) REFERENCES files(file_id) ON DELETE CASCADE,
                FOREIGN KEY(run_id) REFERENCES index_runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS symbols (
                symbol_id TEXT PRIMARY KEY,
                run_id INTEGER NOT NULL,
                repo TEXT NOT NULL,
                file TEXT NOT NULL,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                parent_symbol_id TEXT,
                FOREIGN KEY(run_id) REFERENCES index_runs(run_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file);
            CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);

            CREATE TABLE IF NOT EXISTS dependencies (
                dep_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                source_symbol TEXT NOT NULL,
                target_symbol TEXT NOT NULL,
                kind TEXT NOT NULL,
                file TEXT NOT NULL,
                line INTEGER NOT NULL,
                FOREIGN KEY(run_id) REFERENCES index_runs(run_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_deps_source ON dependencies(source_symbol);
            CREATE INDEX IF NOT EXISTS idx_deps_target ON dependencies(target_symbol);
            CREATE INDEX IF NOT EXISTS idx_deps_file ON dependencies(file);

            CREATE TABLE IF NOT EXISTS module_edges (
                edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                source_module TEXT NOT NULL,
                target_module TEXT NOT NULL,
                kind TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES index_runs(run_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_module_edges_src ON module_edges(source_module);
            CREATE INDEX IF NOT EXISTS idx_module_edges_tgt ON module_edges(target_module);

            CREATE TABLE IF NOT EXISTS directory_edges (
                edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                repo TEXT NOT NULL,
                source_node TEXT NOT NULL,
                target_node TEXT NOT NULL,
                kind TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES index_runs(run_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_directory_edges_repo_src
                ON directory_edges(repo, source_node);
            CREATE INDEX IF NOT EXISTS idx_directory_edges_repo_tgt
                ON directory_edges(repo, target_node);
            """
        )
        self._migrate_symbols_schema_if_needed()
        self.conn.commit()

    def _migrate_symbols_schema_if_needed(self) -> None:
        """
        Ensure symbols are unique per run, not globally by symbol_id.
        Older schemas used `symbol_id` as PRIMARY KEY, which breaks re-index
        across runs for unchanged files/symbol IDs.
        """
        columns = self.conn.execute("PRAGMA table_info(symbols)").fetchall()
        if not columns:
            return

        # New schema should have a composite primary key:
        # PRIMARY KEY (run_id, symbol_id)
        pk_by_name = {str(row["name"]): int(row["pk"]) for row in columns}
        is_new_schema = pk_by_name.get("run_id") == 1 and pk_by_name.get("symbol_id") == 2
        if is_new_schema:
            return

        self.conn.executescript(
            """
            ALTER TABLE symbols RENAME TO symbols_old;

            CREATE TABLE symbols (
                run_id INTEGER NOT NULL,
                symbol_id TEXT NOT NULL,
                repo TEXT NOT NULL,
                file TEXT NOT NULL,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                parent_symbol_id TEXT,
                PRIMARY KEY (run_id, symbol_id),
                FOREIGN KEY(run_id) REFERENCES index_runs(run_id) ON DELETE CASCADE
            );

            INSERT OR REPLACE INTO symbols (
                run_id, symbol_id, repo, file, name, kind, start_line, end_line, parent_symbol_id
            )
            SELECT
                run_id, symbol_id, repo, file, name, kind, start_line, end_line, parent_symbol_id
            FROM symbols_old;

            DROP TABLE symbols_old;

            CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file);
            CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
            """
        )

    def start_run(self, commit: str | None = None, notes: str | None = None) -> IndexRun:
        now = _utc_now()
        cur = self.conn.execute(
            """
            INSERT INTO index_runs (started_at, status, commit_hash, notes)
            VALUES (?, 'running', ?, ?)
            """,
            (now, commit, notes),
        )
        self.conn.commit()
        run_id = int(cur.lastrowid)
        return IndexRun(run_id=run_id, started_at=now, status="running", commit=commit, notes=notes)

    def complete_run(self, run_id: int, status: str = "completed") -> None:
        self.conn.execute(
            """
            UPDATE index_runs
            SET completed_at = ?, status = ?
            WHERE run_id = ?
            """,
            (_utc_now(), status, run_id),
        )
        self.conn.commit()

    def upsert_file(
        self,
        *,
        run_id: int,
        repo: str,
        path: str,
        language: str,
        content: str,
    ) -> tuple[int, bool]:
        """Upsert file row, returning (file_id, changed_since_last_index)."""
        content_hash = _sha256_text(content)
        now = _utc_now()

        row = self.conn.execute(
            "SELECT file_id, content_hash FROM files WHERE repo = ? AND path = ?",
            (repo, path),
        ).fetchone()

        if row is None:
            cur = self.conn.execute(
                """
                INSERT INTO files (repo, path, language, content_hash, last_run_id, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (repo, path, language, content_hash, run_id, now),
            )
            self.conn.commit()
            return int(cur.lastrowid), True

        file_id = int(row["file_id"])
        changed = row["content_hash"] != content_hash
        self.conn.execute(
            """
            UPDATE files
            SET language = ?, content_hash = ?, last_run_id = ?, indexed_at = ?
            WHERE file_id = ?
            """,
            (language, content_hash, run_id, now, file_id),
        )
        self.conn.commit()
        return file_id, changed

    def store_ast_json(
        self,
        *,
        run_id: int,
        file_id: int,
        parser_language: str,
        ast_payload: dict | list,
        parser_version: str | None = None,
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO ast_artifacts (
                file_id, run_id, parser_language, parser_version, ast_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                file_id,
                run_id,
                parser_language,
                parser_version,
                json.dumps(ast_payload, separators=(",", ":")),
                _utc_now(),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def replace_symbols_for_run(self, run_id: int, symbols: Iterable[Symbol]) -> None:
        self.conn.execute("DELETE FROM symbols WHERE run_id = ?", (run_id,))
        rows = [
            (
                s.symbol_id,
                run_id,
                s.repo,
                s.file,
                s.name,
                s.kind,
                s.start_line,
                s.end_line,
                s.parent,
            )
            for s in symbols
        ]
        self.conn.executemany(
            """
            INSERT INTO symbols (
                symbol_id, run_id, repo, file, name, kind, start_line, end_line, parent_symbol_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()

    def replace_dependencies_for_run(self, run_id: int, dependencies: Iterable[Dependency]) -> None:
        self.conn.execute("DELETE FROM dependencies WHERE run_id = ?", (run_id,))
        rows = [
            (
                run_id,
                dep.source_symbol,
                dep.target_symbol,
                dep.kind,
                dep.file,
                dep.line,
            )
            for dep in dependencies
        ]
        self.conn.executemany(
            """
            INSERT INTO dependencies (
                run_id, source_symbol, target_symbol, kind, file, line
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()

    def replace_module_edges_for_run(self, run_id: int, module_edges: Iterable[ModuleDependency]) -> None:
        self.conn.execute("DELETE FROM module_edges WHERE run_id = ?", (run_id,))
        rows = [
            (run_id, edge.source_module, edge.target_module, edge.kind)
            for edge in module_edges
        ]
        self.conn.executemany(
            """
            INSERT INTO module_edges (run_id, source_module, target_module, kind)
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()

    def replace_directory_edges_for_run(self, run_id: int, directory_edges: Iterable[DirectoryEdge]) -> None:
        self.conn.execute("DELETE FROM directory_edges WHERE run_id = ?", (run_id,))
        rows = [
            (run_id, edge.repo, edge.source_node, edge.target_node, edge.kind)
            for edge in directory_edges
        ]
        self.conn.executemany(
            """
            INSERT INTO directory_edges (run_id, repo, source_node, target_node, kind)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()

    def files_changed_since_last_run(self, repo: str, file_contents: dict[str, str]) -> list[str]:
        changed: list[str] = []
        for path, content in file_contents.items():
            row = self.conn.execute(
                "SELECT content_hash FROM files WHERE repo = ? AND path = ?",
                (repo, path),
            ).fetchone()
            content_hash = _sha256_text(content)
            if row is None or row["content_hash"] != content_hash:
                changed.append(path)
        return changed


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
