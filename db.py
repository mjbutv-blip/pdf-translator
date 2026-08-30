from __future__ import annotations

import random
import re
import sqlite3
import time

from config import (
    DB_PATH,
    SQLITE_BUSY_TIMEOUT_MS,
    SQLITE_LOCK_RETRY_ATTEMPTS,
    SQLITE_LOCK_RETRY_BASE_SECONDS,
    SQLITE_TIMEOUT_SECONDS,
    _database_url,
    _use_postgres,
)


def _pg_sql(sql: str) -> str:
    converted = sql.replace("?", "%s")
    if "INSERT OR IGNORE INTO" in converted.upper():
        converted = re.sub(
            r"INSERT\s+OR\s+IGNORE\s+INTO",
            "INSERT INTO",
            converted,
            count=1,
            flags=re.IGNORECASE,
        ).rstrip()
        if not converted.upper().endswith("ON CONFLICT DO NOTHING"):
            converted += " ON CONFLICT DO NOTHING"
    return converted


class DbConnection:
    def __init__(self):
        self.is_postgres = _use_postgres()
        if self.is_postgres:
            try:
                import psycopg
                from psycopg.rows import dict_row
            except ImportError as exc:
                raise RuntimeError("缺少 Postgres 驱动，请确认 requirements.txt 已安装 psycopg[binary]") from exc
            url = _database_url()
            if url.startswith("postgres://"):
                url = "postgresql://" + url[len("postgres://"):]
            self.conn = psycopg.connect(url, row_factory=dict_row)
        else:
            self.conn = sqlite3.connect(DB_PATH, timeout=SQLITE_TIMEOUT_SECONDS)
            self.conn.row_factory = sqlite3.Row
            self._configure_sqlite()

    @staticmethod
    def _is_sqlite_lock_error(exc: sqlite3.OperationalError) -> bool:
        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "database is locked",
                "database table is locked",
                "database schema is locked",
                "busy",
            )
        )

    def _execute_sqlite_with_lock_retry(self, operation: str, fn):
        attempt = 0
        while True:
            try:
                return fn()
            except sqlite3.OperationalError as exc:
                if not self._is_sqlite_lock_error(exc) or attempt >= SQLITE_LOCK_RETRY_ATTEMPTS:
                    raise
                attempt += 1
                sleep_seconds = SQLITE_LOCK_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                sleep_seconds += random.uniform(0, SQLITE_LOCK_RETRY_BASE_SECONDS)
                print(
                    f"event=db_lock_retry operation={operation} attempt={attempt}",
                    flush=True,
                )
                time.sleep(sleep_seconds)

    def _configure_sqlite(self) -> None:
        self._execute_sqlite_with_lock_retry(
            "sqlite_pragma_busy_timeout",
            lambda: self.conn.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}"),
        )
        self._execute_sqlite_with_lock_retry(
            "sqlite_pragma_foreign_keys",
            lambda: self.conn.execute("PRAGMA foreign_keys = ON"),
        )
        self._execute_sqlite_with_lock_retry(
            "sqlite_pragma_journal_mode",
            lambda: self.conn.execute("PRAGMA journal_mode = WAL"),
        )

    def __enter__(self):
        self.conn.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self.conn.__exit__(exc_type, exc, tb)

    def execute(self, sql: str, params=()):
        if self.is_postgres:
            return self.conn.execute(_pg_sql(sql), params)
        return self._execute_sqlite_with_lock_retry(
            "execute",
            lambda: self.conn.execute(sql, params),
        )

    def executemany(self, sql: str, seq_of_params):
        if not self.is_postgres:
            return self._execute_sqlite_with_lock_retry(
                "executemany",
                lambda: self.conn.executemany(sql, seq_of_params),
            )
        with self.conn.cursor() as cur:
            return cur.executemany(_pg_sql(sql), seq_of_params)

    def executescript(self, sql: str) -> None:
        if not self.is_postgres:
            self._execute_sqlite_with_lock_retry(
                "executescript",
                lambda: self.conn.executescript(sql),
            )
            return
        with self.conn.cursor() as cur:
            for statement in sql.split(";"):
                if statement.strip():
                    cur.execute(statement)


def get_db_connection() -> DbConnection:
    return DbConnection()


__all__ = ["DbConnection", "get_db_connection"]
