from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import db
from config import _use_postgres


CORE_TABLES = [
    "users",
    "customers",
    "glossary_terms",
    "glossary_change_requests",
    "term_candidates",
    "translation_jobs",
    "translation_candidate_occurrences",
]
RUNTIME_TABLES = ["translation_workers"]


def _require_company_admin(user: dict | None) -> None:
    if not user or user.get("role") != "company_admin":
        raise PermissionError("Only company_admin can create SQLite snapshots.")


def _connect_sqlite(path: Path, *, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _validate_snapshot(path: Path) -> dict[str, Any]:
    with _connect_sqlite(path, readonly=True) as conn:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        existing_tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        missing_tables = [table for table in CORE_TABLES if table not in existing_tables]
        row_counts = {}
        for table in CORE_TABLES + RUNTIME_TABLES:
            if table in existing_tables:
                row_counts[table] = int(conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"])
    return {
        "integrity_check": integrity,
        "missing_tables": missing_tables,
        "row_counts": row_counts,
        "valid": integrity == "ok" and not missing_tables,
    }


def create_sqlite_snapshot(user: dict | None = None, *, require_company_admin: bool = True) -> dict[str, Any]:
    """Create a consistent SQLite snapshot using sqlite3.Connection.backup().

    The returned dict contains snapshot bytes and metadata only. No snapshot file
    is persisted after this function returns.
    """
    if require_company_admin:
        _require_company_admin(user)

    if _use_postgres():
        return {
            "status": "postgres_active",
            "database_backend": "postgres",
            "snapshot_bytes": None,
            "metadata": {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "database_backend": "postgres",
                "message": "PostgreSQL is active; SQLite snapshot is not applicable.",
            },
        }

    source_path = Path(db.DB_PATH)
    if not source_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {source_path}")

    created_at = datetime.now().isoformat(timespec="seconds")
    tmp_path = None
    try:
        fd, tmp_name = tempfile.mkstemp(prefix="pdf_project_snapshot_", suffix=".db")
        os.close(fd)
        tmp_path = Path(tmp_name)
        with _connect_sqlite(source_path, readonly=True) as source, _connect_sqlite(tmp_path) as destination:
            source.backup(destination)

        validation = _validate_snapshot(tmp_path)
        snapshot_bytes = tmp_path.read_bytes()
        metadata = {
            "created_at": created_at,
            "database_backend": "sqlite",
            "size_bytes": len(snapshot_bytes),
            "source_size_bytes": source_path.stat().st_size,
            "integrity_check": validation["integrity_check"],
            "missing_tables": validation["missing_tables"],
            **validation["row_counts"],
        }
        return {
            "status": "completed" if validation["valid"] else "failed",
            "database_backend": "sqlite",
            "snapshot_bytes": snapshot_bytes,
            "metadata": metadata,
        }
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass


def snapshot_download_names(created_at: str | None = None) -> tuple[str, str]:
    stamp = (created_at or datetime.now().isoformat(timespec="seconds")).replace("-", "").replace(":", "").replace("T", "_")
    return (
        f"pdf_project_production_snapshot_{stamp}.db",
        f"pdf_project_production_snapshot_{stamp}.json",
    )


def snapshot_metadata_bytes(metadata: dict[str, Any]) -> bytes:
    return json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8")


__all__ = [
    "CORE_TABLES",
    "RUNTIME_TABLES",
    "create_sqlite_snapshot",
    "snapshot_download_names",
    "snapshot_metadata_bytes",
]
