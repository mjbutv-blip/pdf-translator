from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import psycopg
from psycopg.rows import dict_row

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


BUSINESS_TABLES = [
    "users",
    "customers",
    "glossary_terms",
    "glossary_change_requests",
    "term_candidates",
    "translation_jobs",
    "translation_candidate_occurrences",
]

RUNTIME_TABLES = ["translation_workers"]

IDENTITY_COLUMNS = {
    "glossary_terms": "glossary_id",
    "glossary_change_requests": "request_id",
    "term_candidates": "candidate_id",
    "translation_candidate_occurrences": "occurrence_id",
}

BLOB_COLUMNS = {
    "translation_jobs": ["input_bytes", "aux_bytes", "result_file", "result_report"],
}


def _mask_url(url: str) -> str:
    parsed = urlparse(url)
    netloc = parsed.hostname or ""
    if parsed.port:
        netloc += f":{parsed.port}"
    if parsed.username:
        netloc = f"{parsed.username}:***@{netloc}"
    return parsed._replace(netloc=netloc, query="", fragment="").geturl()


def _looks_like_test_url(url: str) -> bool:
    parsed = urlparse(url)
    haystack = " ".join(
        str(part or "").lower()
        for part in [parsed.hostname, parsed.username, parsed.path]
    )
    return any(marker in haystack for marker in ("localhost", "127.0.0.1", "test", "dev", "staging"))


def _sqlite_connect_readonly(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _row_count(conn, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"])


def _pg_row_count(conn, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"])


def _sqlite_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def _target_is_empty(pg_conn) -> bool:
    for table in BUSINESS_TABLES:
        if _pg_row_count(pg_conn, table) != 0:
            return False
    return True


def _copy_table(sqlite_conn: sqlite3.Connection, pg_conn, table: str) -> None:
    columns = _sqlite_columns(sqlite_conn, table)
    column_list = ", ".join(columns)
    placeholders = ", ".join(["%s"] * len(columns))
    rows = sqlite_conn.execute(f"SELECT {column_list} FROM {table}").fetchall()
    if not rows:
        return
    values = [tuple(row[col] for col in columns) for row in rows]
    with pg_conn.cursor() as cur:
        cur.executemany(
            f"INSERT INTO {table} ({column_list}) VALUES ({placeholders})",
            values,
        )


def _calibrate_identity_sequences(pg_conn) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for table, column in IDENTITY_COLUMNS.items():
        row = pg_conn.execute(
            "SELECT pg_get_serial_sequence(%s, %s) AS seq",
            (table, column),
        ).fetchone()
        seq_name = row["seq"] if row else None
        max_row = pg_conn.execute(f"SELECT MAX({column}) AS max_id FROM {table}").fetchone()
        max_id = max_row["max_id"] if max_row else None
        if seq_name:
            if max_id is None:
                pg_conn.execute("SELECT setval(%s, 1, false)", (seq_name,))
            else:
                pg_conn.execute("SELECT setval(%s, %s, true)", (seq_name, int(max_id)))
        result[table] = {"column": column, "sequence": seq_name, "max_id": max_id}
    return result


def _sha256_bytes(value) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(bytes(value)).hexdigest()


def _blob_validation(sqlite_conn: sqlite3.Connection, pg_conn) -> list[dict]:
    rows: list[dict] = []
    for table, columns in BLOB_COLUMNS.items():
        key = "job_id"
        sqlite_rows = sqlite_conn.execute(
            f"SELECT {key}, {', '.join(columns)} FROM {table} ORDER BY {key}"
        ).fetchall()
        for src in sqlite_rows:
            dst = pg_conn.execute(
                f"SELECT {', '.join(columns)} FROM {table} WHERE {key} = %s",
                (src[key],),
            ).fetchone()
            for col in columns:
                src_value = src[col]
                dst_value = dst[col] if dst else None
                src_len = len(src_value) if src_value is not None else None
                dst_len = len(dst_value) if dst_value is not None else None
                rows.append({
                    "table": table,
                    "key": src[key],
                    "column": col,
                    "sqlite_length": src_len,
                    "postgres_length": dst_len,
                    "sha256_match": _sha256_bytes(src_value) == _sha256_bytes(dst_value),
                })
    return rows


def _distribution(conn, sql: str) -> dict[str, int]:
    rows = conn.execute(sql).fetchall()
    result: dict[str, int] = {}
    for row in rows:
        values = list(row)
        result["|".join(str(value) for value in values[:-1])] = int(values[-1])
    return result


def _pg_distribution(conn, sql: str) -> dict[str, int]:
    rows = conn.execute(sql).fetchall()
    result: dict[str, int] = {}
    for row in rows:
        values = list(row.values())
        result["|".join(str(value) for value in values[:-1])] = int(values[-1])
    return result


def _business_validation(sqlite_conn: sqlite3.Connection, pg_conn) -> dict:
    occurrence_orphans = pg_conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM translation_candidate_occurrences tco
        LEFT JOIN translation_jobs tj ON tj.job_id = tco.translation_job_id
        LEFT JOIN term_candidates tc ON tc.candidate_id = tco.candidate_id
        WHERE tj.job_id IS NULL OR tc.candidate_id IS NULL
        """
    ).fetchone()["n"]
    return {
        "active_glossary_by_customer_match": _distribution(
            sqlite_conn,
            """
            SELECT customer_id, COUNT(*) AS n
            FROM glossary_terms
            WHERE status = 'active'
            GROUP BY customer_id
            ORDER BY customer_id
            """,
        ) == _pg_distribution(
            pg_conn,
            """
            SELECT customer_id, COUNT(*) AS n
            FROM glossary_terms
            WHERE status = 'active'
            GROUP BY customer_id
            ORDER BY customer_id
            """,
        ),
        "candidate_status_match": _distribution(
            sqlite_conn,
            "SELECT status, COUNT(*) AS n FROM term_candidates GROUP BY status ORDER BY status",
        ) == _pg_distribution(
            pg_conn,
            "SELECT status, COUNT(*) AS n FROM term_candidates GROUP BY status ORDER BY status",
        ),
        "job_distribution_match": _distribution(
            sqlite_conn,
            """
            SELECT job_type, status, COALESCE(execution_mode, ''), COUNT(*) AS n
            FROM translation_jobs
            GROUP BY job_type, status, COALESCE(execution_mode, '')
            ORDER BY job_type, status, COALESCE(execution_mode, '')
            """,
        ) == _pg_distribution(
            pg_conn,
            """
            SELECT job_type, status, COALESCE(execution_mode, '') AS execution_mode, COUNT(*) AS n
            FROM translation_jobs
            GROUP BY job_type, status, COALESCE(execution_mode, '')
            ORDER BY job_type, status, COALESCE(execution_mode, '')
            """,
        ),
        "occurrence_orphans": int(occurrence_orphans),
    }


def _row_count_validation(sqlite_conn: sqlite3.Connection, pg_conn) -> list[dict]:
    rows = []
    for table in BUSINESS_TABLES:
        sqlite_count = _row_count(sqlite_conn, table)
        pg_count = _pg_row_count(pg_conn, table)
        rows.append({
            "table": table,
            "sqlite": sqlite_count,
            "postgres": pg_count,
            "match": sqlite_count == pg_count,
        })
    for table in RUNTIME_TABLES:
        rows.append({
            "table": table,
            "sqlite": _row_count(sqlite_conn, table),
            "postgres": _pg_row_count(pg_conn, table),
            "match": True,
            "note": "runtime table intentionally not migrated",
        })
    return rows


def _run_init_db(target_url: str) -> None:
    os.environ["DATABASE_URL"] = target_url
    from translation_core import init_db

    init_db()


def migrate(source_sqlite: Path, target_url: str, *, allow_nonempty_target: bool, confirm_test_target: bool) -> dict:
    if not source_sqlite.exists():
        raise FileNotFoundError(source_sqlite)
    if not confirm_test_target and not _looks_like_test_url(target_url):
        raise RuntimeError("Target DATABASE_URL does not look like a test database. Use --confirm-test-target to override.")

    started_at = datetime.now().isoformat()
    _run_init_db(target_url)
    with _sqlite_connect_readonly(source_sqlite) as sqlite_conn, psycopg.connect(target_url, row_factory=dict_row) as pg_conn:
        if not allow_nonempty_target and not _target_is_empty(pg_conn):
            raise RuntimeError("Target PostgreSQL already contains business rows. Refusing by default.")
        with pg_conn.transaction():
            for table in BUSINESS_TABLES:
                _copy_table(sqlite_conn, pg_conn, table)
            sequences = _calibrate_identity_sequences(pg_conn)
            counts = _row_count_validation(sqlite_conn, pg_conn)
            business = _business_validation(sqlite_conn, pg_conn)
            blobs = _blob_validation(sqlite_conn, pg_conn)
            if not all(row["match"] for row in counts if row["table"] in BUSINESS_TABLES):
                raise RuntimeError("Row count validation failed")
            if not all(
                value
                for key, value in business.items()
                if key.endswith("_match")
            ):
                raise RuntimeError("Business distribution validation failed")
            if business["occurrence_orphans"] != 0:
                raise RuntimeError("Occurrence relationship validation failed")
            if not all(row["sha256_match"] for row in blobs):
                raise RuntimeError("BLOB validation failed")
        return {
            "source_database": str(source_sqlite),
            "target_database": _mask_url(target_url),
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(),
            "tables": counts,
            "sequence_calibration": sequences,
            "business_validation": business,
            "blob_validation": blobs,
            "overall": "PASS",
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dry-run migrate PDF Project SQLite data to a test PostgreSQL database.")
    parser.add_argument("--source-sqlite", required=True, type=Path)
    parser.add_argument("--target-database-url", default=os.environ.get("TEST_POSTGRES_DATABASE_URL", ""))
    parser.add_argument("--allow-nonempty-target", action="store_true")
    parser.add_argument("--confirm-test-target", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON report.")
    args = parser.parse_args(argv)

    if not args.target_database_url:
        print("Missing --target-database-url or TEST_POSTGRES_DATABASE_URL.", file=sys.stderr)
        return 2
    try:
        report = migrate(
            args.source_sqlite,
            args.target_database_url,
            allow_nonempty_target=args.allow_nonempty_target,
            confirm_test_target=args.confirm_test_target,
        )
    except Exception as exc:
        print(f"Migration dry run failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"Migration dry run {report['overall']}: {report['source_database']} -> {report['target_database']}")
        for row in report["tables"]:
            note = f" ({row['note']})" if row.get("note") else ""
            print(f"{row['table']}: sqlite={row['sqlite']} postgres={row['postgres']} match={row['match']}{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
