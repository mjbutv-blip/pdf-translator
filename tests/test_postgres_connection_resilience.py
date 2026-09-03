from __future__ import annotations

import sys
import types

import db


class TemporaryOperationalError(Exception):
    pass


class _Lease:
    def __init__(self, failures, calls, pool=None):
        self.failures = failures
        self.calls = calls
        self.pool = pool

    def __enter__(self):
        self.calls.append(1)
        if len(self.calls) <= self.failures:
            raise TemporaryOperationalError("temporary connect failure")
        return object()

    def __exit__(self, *_args):
        return False


class _Pool:
    def __init__(self, failures, calls):
        self.failures = failures
        self.calls = calls

    def connection(self, **_kwargs):
        return _Lease(self.failures, self.calls, self)


def _fake_psycopg_modules(monkeypatch):
    psycopg = types.ModuleType("psycopg")
    psycopg.OperationalError = TemporaryOperationalError
    rows = types.ModuleType("psycopg.rows")
    rows.dict_row = object()
    monkeypatch.setitem(sys.modules, "psycopg", psycopg)
    monkeypatch.setitem(sys.modules, "psycopg.rows", rows)


def test_postgres_transient_connect_failure_recovers_with_finite_retry(monkeypatch):
    calls = []
    _fake_psycopg_modules(monkeypatch)
    monkeypatch.setattr(db, "_use_postgres", lambda: True)
    monkeypatch.setattr(db, "_database_url", lambda: "postgresql://example/db")
    monkeypatch.setattr(db, "_postgres_pool", lambda *_args: _Pool(1, calls))
    monkeypatch.setattr(db.time, "sleep", lambda _seconds: None)
    conn = db.DbConnection()
    assert conn.is_postgres
    assert len(calls) == 2


def test_postgres_connect_retry_is_bounded(monkeypatch):
    calls = []
    _fake_psycopg_modules(monkeypatch)
    monkeypatch.setattr(db, "_use_postgres", lambda: True)
    monkeypatch.setattr(db, "_database_url", lambda: "postgresql://example/db")
    monkeypatch.setattr(db, "_postgres_pool", lambda *_args: _Pool(999, calls))
    monkeypatch.setattr(db.time, "sleep", lambda _seconds: None)
    try:
        db.DbConnection()
        assert False, "expected bounded retry failure"
    except TemporaryOperationalError:
        pass
    assert len(calls) == db.POSTGRES_CONNECT_RETRY_ATTEMPTS + 1


def test_repeated_get_db_connection_uses_same_process_pool(monkeypatch):
    calls = []
    _fake_psycopg_modules(monkeypatch)
    pool_module = types.ModuleType("psycopg_pool")

    class FakeConnectionPool(_Pool):
        check_connection = staticmethod(lambda _conn: None)

        def __init__(self, **kwargs):
            super().__init__(0, calls)
            self.kwargs = kwargs

    pool_module.ConnectionPool = FakeConnectionPool
    monkeypatch.setitem(sys.modules, "psycopg_pool", pool_module)
    monkeypatch.setattr(db, "_use_postgres", lambda: True)
    monkeypatch.setattr(db, "_database_url", lambda: "postgresql://example/db")
    monkeypatch.setattr(db, "_POSTGRES_POOL", None)
    monkeypatch.setattr(db, "_POSTGRES_POOL_KEY", None)

    first = db.get_db_connection()
    second = db.get_db_connection()
    try:
        assert first._pool_context.pool is second._pool_context.pool is db._POSTGRES_POOL
        assert db._POSTGRES_POOL.kwargs["max_size"] == 4
    finally:
        first.__exit__(None, None, None)
        second.__exit__(None, None, None)
