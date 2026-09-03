from __future__ import annotations

import random
import re
import sqlite3
import time
import os
import threading

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
            attempt = 0
            while True:
                self._pool_context = _postgres_pool(url, dict_row).connection(
                    timeout=POSTGRES_POOL_ACQUIRE_TIMEOUT_SECONDS
                )
                try:
                    self.conn = self._pool_context.__enter__()
                    break
                except psycopg.OperationalError:
                    if attempt >= POSTGRES_CONNECT_RETRY_ATTEMPTS:
                        raise
                    attempt += 1
                    print(f"event=postgres_connect_retry attempt={attempt}", flush=True)
                    time.sleep(POSTGRES_CONNECT_RETRY_BASE_SECONDS * (2 ** (attempt - 1)))
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
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.is_postgres:
            return self._pool_context.__exit__(exc_type, exc, tb)
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


POSTGRES_CONNECT_TIMEOUT_SECONDS = int(os.getenv("POSTGRES_CONNECT_TIMEOUT_SECONDS", "5"))
POSTGRES_CONNECT_RETRY_ATTEMPTS = int(os.getenv("POSTGRES_CONNECT_RETRY_ATTEMPTS", "2"))
POSTGRES_CONNECT_RETRY_BASE_SECONDS = float(os.getenv("POSTGRES_CONNECT_RETRY_BASE_SECONDS", "0.2"))
POSTGRES_POOL_MIN_SIZE = int(os.getenv("POSTGRES_POOL_MIN_SIZE", "0"))
POSTGRES_POOL_MAX_SIZE = int(os.getenv("POSTGRES_POOL_MAX_SIZE", "4"))
POSTGRES_POOL_ACQUIRE_TIMEOUT_SECONDS = float(os.getenv("POSTGRES_POOL_ACQUIRE_TIMEOUT_SECONDS", "5"))
_POSTGRES_POOL = None
_POSTGRES_POOL_KEY = None
_POSTGRES_POOL_LOCK = threading.Lock()


def _postgres_pool(url: str, row_factory):
    global _POSTGRES_POOL, _POSTGRES_POOL_KEY
    key = (os.getpid(), url)
    with _POSTGRES_POOL_LOCK:
        if _POSTGRES_POOL is None or _POSTGRES_POOL_KEY != key:
            from psycopg_pool import ConnectionPool

            _POSTGRES_POOL = ConnectionPool(
                conninfo=url,
                min_size=max(0, POSTGRES_POOL_MIN_SIZE),
                max_size=max(1, POSTGRES_POOL_MAX_SIZE),
                open=True,
                kwargs={"row_factory": row_factory, "connect_timeout": POSTGRES_CONNECT_TIMEOUT_SECONDS},
                check=ConnectionPool.check_connection,
                name=f"pdf-project-{os.getpid()}",
            )
            _POSTGRES_POOL_KEY = key
        return _POSTGRES_POOL


__all__ = ["DbConnection", "get_db_connection"]
