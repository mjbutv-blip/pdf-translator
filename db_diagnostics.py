from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import db
from config import _database_url, _use_postgres
from translation_jobs import get_worker_health


TABLES = [
    "users",
    "customers",
    "glossary_terms",
    "glossary_change_requests",
    "term_candidates",
    "translation_jobs",
    "translation_candidate_occurrences",
    "translation_workers",
]

REQUIRED_TRANSLATION_JOB_COLUMNS = [
    "execution_mode",
    "worker_id",
    "heartbeat_at",
    "attempt_count",
]

LOCAL_OLD_DRY_RUN_COUNTS = {
    "users": 5,
    "customers": 5,
    "glossary_terms": 55,
    "glossary_change_requests": 4,
    "term_candidates": 63,
    "translation_jobs": 2,
}


def _require_company_admin(user: dict | None) -> None:
    if not user or user.get("role") != "company_admin":
        raise PermissionError("Only company_admin can view database diagnostics.")


def _safe_database_identity() -> dict[str, str | None]:
    if not _use_postgres():
        return {
            "database_backend": "SQLite",
            "host": None,
            "database_name": Path(db.DB_PATH).name,
        }
    parsed = urlparse(_database_url())
    database_name = parsed.path.lstrip("/") or None
    host = parsed.hostname or "local socket"
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return {
        "database_backend": "PostgreSQL",
        "host": host,
        "database_name": database_name,
    }


def _row_counts(conn) -> dict[str, int | None]:
    counts: dict[str, int | None] = {}
    for table in TABLES:
        try:
            counts[table] = int(conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"])
        except Exception:
            counts[table] = None
    return counts


def _active_glossary_by_customer(conn) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT customer_id, COUNT(*) AS active_glossary_count
        FROM glossary_terms
        WHERE status = 'active'
        GROUP BY customer_id
        ORDER BY customer_id
        """
    ).fetchall()
    return [dict(row) for row in rows]


def _candidate_count_by_status(conn) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT status, COUNT(*) AS candidate_count
        FROM term_candidates
        GROUP BY status
        ORDER BY status
        """
    ).fetchall()
    return [dict(row) for row in rows]


def _translation_job_distribution(conn) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT job_type, status, COALESCE(execution_mode, '') AS execution_mode, COUNT(*) AS job_count
        FROM translation_jobs
        GROUP BY job_type, status, COALESCE(execution_mode, '')
        ORDER BY job_type, status, COALESCE(execution_mode, '')
        """
    ).fetchall()
    return [dict(row) for row in rows]


def _postgres_server_info(conn) -> dict[str, Any]:
    if not conn.is_postgres:
        return {
            "server_version": None,
            "database_time_utc": datetime.now(timezone.utc).isoformat(),
        }
    row = conn.execute(
        """
        SELECT version() AS server_version,
               (CURRENT_TIMESTAMP AT TIME ZONE 'UTC')::text AS database_time_utc
        """
    ).fetchone()
    return dict(row)


def _schema_readiness(conn) -> dict[str, Any]:
    if conn.is_postgres:
        rows = conn.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'translation_jobs'
            """
        ).fetchall()
        translation_job_columns = {row["column_name"] for row in rows}
        worker_table = conn.execute(
            """
            SELECT 1 AS table_exists
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name = 'translation_workers'
            """
        ).fetchone()
    else:
        rows = conn.execute("PRAGMA table_info(translation_jobs)").fetchall()
        translation_job_columns = {row["name"] for row in rows}
        worker_table = conn.execute(
            """
            SELECT 1 AS table_exists
            FROM sqlite_master
            WHERE type = 'table'
              AND name = 'translation_workers'
            """
        ).fetchone()
    return {
        "translation_jobs_columns": {
            column: column in translation_job_columns
            for column in REQUIRED_TRANSLATION_JOB_COLUMNS
        },
        "translation_workers_table": bool(worker_table),
    }


def _compare_to_local_old_counts(counts: dict[str, int | None]) -> list[dict[str, int | str | None]]:
    rows = []
    for table, old_count in LOCAL_OLD_DRY_RUN_COUNTS.items():
        production_count = counts.get(table)
        rows.append({
            "table": table,
            "production_count": production_count,
            "local_old_count": old_count,
            "difference": None if production_count is None else production_count - old_count,
        })
    return rows


def get_safe_database_diagnostics(user: dict | None = None, *, require_company_admin: bool = True) -> dict[str, Any]:
    """Return safe, read-only database diagnostics without secrets or document content."""
    if require_company_admin:
        _require_company_admin(user)

    identity = _safe_database_identity()
    with db.get_db_connection() as conn:
        counts = _row_counts(conn)
        result = {
            **identity,
            **_postgres_server_info(conn),
            "row_counts": counts,
            "active_glossary_by_customer": _active_glossary_by_customer(conn),
            "candidate_count_by_status": _candidate_count_by_status(conn),
            "translation_job_distribution": _translation_job_distribution(conn),
            "schema_readiness": _schema_readiness(conn),
            "worker_health": get_worker_health(),
            "compare_to_local_old_dry_run": _compare_to_local_old_counts(counts),
        }
    return result


def database_diagnostics_verdict(diagnostics: dict[str, Any]) -> str:
    schema = diagnostics.get("schema_readiness") or {}
    columns = schema.get("translation_jobs_columns") or {}
    schema_ready = bool(schema.get("translation_workers_table")) and all(columns.values())
    required_counts = diagnostics.get("row_counts") or {}
    data_present = all((required_counts.get(table) or 0) > 0 for table in ("users", "customers", "glossary_terms"))
    if diagnostics.get("database_backend") != "PostgreSQL":
        return "unable to verify safely"
    if schema_ready and data_present:
        return "production Postgres appears complete and ready for worker"
    return "production Postgres exists but data/schema needs investigation"


__all__ = [
    "TABLES",
    "REQUIRED_TRANSLATION_JOB_COLUMNS",
    "LOCAL_OLD_DRY_RUN_COUNTS",
    "get_safe_database_diagnostics",
    "database_diagnostics_verdict",
]
