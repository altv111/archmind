from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3

from graph.code_graph import CodeGraph
from graph.directory_graph_builder import DirectoryEdge
from graph.module_graph_builder import ModuleDependency
from ingestion.dependency_extractor import Dependency
from ingestion.symbol_extractor import Symbol


@dataclass(frozen=True)
class LoadedGraph:
    run_id: int
    graph: CodeGraph


class GraphLoader:
    """Load persisted graph artifacts from SQLite into an in-memory CodeGraph."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)

    def load(self, run_id: int | None = None) -> LoadedGraph:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            resolved_run_id = run_id if run_id is not None else self._latest_completed_run_id(conn)
            if resolved_run_id is None:
                raise ValueError("No completed index run found in store.")

            symbols = self._load_symbols(conn, resolved_run_id)
            dependencies = self._load_dependencies(conn, resolved_run_id)
            module_edges = self._load_module_edges(conn, resolved_run_id)
            directory_edges = self._load_directory_edges(conn, resolved_run_id)

            graph = CodeGraph(symbols, dependencies)
            graph.module_edges = module_edges
            graph.directory_edges = directory_edges
            return LoadedGraph(run_id=resolved_run_id, graph=graph)
        finally:
            conn.close()

    def _latest_completed_run_id(self, conn: sqlite3.Connection) -> int | None:
        row = conn.execute(
            """
            SELECT run_id
            FROM index_runs
            WHERE status = 'completed'
            ORDER BY run_id DESC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None
        return int(row["run_id"])

    def _load_symbols(self, conn: sqlite3.Connection, run_id: int) -> list[Symbol]:
        rows = conn.execute(
            """
            SELECT symbol_id, repo, file, name, kind, start_line, end_line, parent_symbol_id
            FROM symbols
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchall()
        return [
            Symbol(
                symbol_id=row["symbol_id"],
                repo=row["repo"],
                file=row["file"],
                name=row["name"],
                kind=row["kind"],
                start_line=int(row["start_line"]),
                end_line=int(row["end_line"]),
                parent=row["parent_symbol_id"],
            )
            for row in rows
        ]

    def _load_dependencies(self, conn: sqlite3.Connection, run_id: int) -> list[Dependency]:
        rows = conn.execute(
            """
            SELECT source_symbol, target_symbol, kind, file, line
            FROM dependencies
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchall()
        return [
            Dependency(
                source_symbol=row["source_symbol"],
                target_symbol=row["target_symbol"],
                kind=row["kind"],
                file=row["file"],
                line=int(row["line"]),
            )
            for row in rows
        ]

    def _load_module_edges(self, conn: sqlite3.Connection, run_id: int) -> list[ModuleDependency]:
        rows = conn.execute(
            """
            SELECT source_module, target_module, kind
            FROM module_edges
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchall()
        return [
            ModuleDependency(
                source_module=row["source_module"],
                target_module=row["target_module"],
                kind=row["kind"],
            )
            for row in rows
        ]

    def _load_directory_edges(self, conn: sqlite3.Connection, run_id: int) -> list[DirectoryEdge]:
        try:
            rows = conn.execute(
                """
                SELECT repo, source_node, target_node, kind
                FROM directory_edges
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchall()
        except sqlite3.OperationalError:
            # Backward compatibility: older DBs may not have directory_edges yet.
            return []
        return [
            DirectoryEdge(
                repo=row["repo"],
                source_node=row["source_node"],
                target_node=row["target_node"],
                kind=row["kind"],
            )
            for row in rows
        ]
