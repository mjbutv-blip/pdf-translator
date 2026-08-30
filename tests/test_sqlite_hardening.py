from __future__ import annotations

import sqlite3
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import db


def _prepare_temp_db(tmpdir: str):
    db.DB_PATH = Path(tmpdir) / "sqlite_hardening.sqlite"
    db.SQLITE_TIMEOUT_SECONDS = 0.05
    db.SQLITE_BUSY_TIMEOUT_MS = 50
    db.SQLITE_LOCK_RETRY_ATTEMPTS = 8
    db.SQLITE_LOCK_RETRY_BASE_SECONDS = 0.02
    import translation_core

    translation_core.init_db()
    with db.get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO customers (
                customer_id, customer_name, customer_code, group_name,
                assigned_staff_username, created_at
            )
            VALUES ('TEST-CUST', 'SQLite Test Customer', 'TEST-CUST', 'Test', 'worker_test', ?)
            """,
            (translation_core._now_iso(),),
        )
        conn.execute("CREATE TABLE IF NOT EXISTS lock_test (id INTEGER PRIMARY KEY, value INTEGER NOT NULL)")
        conn.execute("INSERT INTO lock_test (id, value) VALUES (1, 0)")
    return translation_core


def _make_job(filename: str = "test.pdf") -> str:
    import translation_jobs

    return translation_jobs.create_translation_job(
        job_type="PDF",
        username="sqlite_worker_test",
        customer_id="TEST-CUST",
        source_file_name=filename,
        input_bytes=b"%PDF-1.4\n%fake input\n",
        aux_bytes=None,
        config={"scope_mode": "all", "selected_pages": None, "scope_detection": [], "scope_cfg": {"total_pages": 1}},
    )


def _row(job_id: str) -> dict:
    import translation_jobs

    row = translation_jobs.get_translation_job_by_id(job_id)
    assert row is not None
    return row


def test_db_config() -> None:
    with db.get_db_connection() as conn:
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        busy_timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        foreign_keys = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    assert journal_mode.lower() == "wal"
    assert busy_timeout > 0
    assert foreign_keys == 1


def test_concurrent_read_write() -> None:
    errors: list[BaseException] = []

    def writer(iterations: int) -> None:
        try:
            for _ in range(iterations):
                with db.get_db_connection() as conn:
                    conn.execute("UPDATE lock_test SET value = value + 1 WHERE id = 1")
        except BaseException as exc:
            errors.append(exc)

    def reader(iterations: int) -> None:
        try:
            for _ in range(iterations):
                with db.get_db_connection() as conn:
                    conn.execute("SELECT value FROM lock_test WHERE id = 1").fetchone()
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=writer, args=(20,)),
        threading.Thread(target=writer, args=(20,)),
        threading.Thread(target=reader, args=(40,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    with db.get_db_connection() as conn:
        value = conn.execute("SELECT value FROM lock_test WHERE id = 1").fetchone()[0]
    assert value == 40


def test_lock_retry_success() -> None:
    retry_conn = db.get_db_connection()
    retry_conn.__enter__()
    locker = sqlite3.connect(db.DB_PATH, timeout=0.05, check_same_thread=False)
    try:
        locker.execute("BEGIN IMMEDIATE")
        locker.execute("UPDATE lock_test SET value = value + 1 WHERE id = 1")

        def release_lock() -> None:
            time.sleep(0.15)
            locker.commit()

        releaser = threading.Thread(target=release_lock)
        releaser.start()
        retry_conn.execute("UPDATE lock_test SET value = value + 1 WHERE id = 1")
        releaser.join()
    finally:
        retry_conn.__exit__(None, None, None)
        locker.close()


def test_lock_retry_limit() -> None:
    original_attempts = db.SQLITE_LOCK_RETRY_ATTEMPTS
    db.SQLITE_LOCK_RETRY_ATTEMPTS = 1
    retry_conn = db.get_db_connection()
    retry_conn.__enter__()
    locker = sqlite3.connect(db.DB_PATH, timeout=0.05)
    try:
        locker.execute("BEGIN IMMEDIATE")
        locker.execute("UPDATE lock_test SET value = value + 1 WHERE id = 1")
        try:
            retry_conn.execute("UPDATE lock_test SET value = value + 1 WHERE id = 1")
            raise AssertionError("expected database lock failure")
        except sqlite3.OperationalError as exc:
            assert "locked" in str(exc).lower() or "busy" in str(exc).lower()
    finally:
        locker.rollback()
        locker.close()
        retry_conn.__exit__(None, None, None)
        db.SQLITE_LOCK_RETRY_ATTEMPTS = original_attempts


def test_no_retry_programming_error() -> None:
    start = time.monotonic()
    with db.get_db_connection() as conn:
        try:
            conn.execute("UPDATE missing_table SET value = 1")
            raise AssertionError("expected programming error")
        except sqlite3.OperationalError as exc:
            assert "no such table" in str(exc).lower()
    assert time.monotonic() - start < 0.5


def test_atomic_claim_regression() -> None:
    import translation_jobs

    job_id = _make_job("atomic-regression.pdf")
    barrier = threading.Barrier(2)
    results = []

    def claim(worker_id: str) -> None:
        barrier.wait()
        results.append(translation_jobs.claim_next_pdf_job(worker_id))

    threads = [
        threading.Thread(target=claim, args=("worker-A",)),
        threading.Thread(target=claim, args=("worker-B",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    claimed = [job for job in results if job]
    assert len(claimed) == 1
    assert claimed[0]["job_id"] == job_id


def test_ownership_regression() -> None:
    import translation_jobs

    job_id = _make_job("ownership-regression.pdf")
    job = translation_jobs.claim_next_pdf_job("worker-A")
    assert job and job["job_id"] == job_id
    translation_jobs.update_translation_job(job_id, status="queued", worker_id=None, heartbeat_at=None)
    reclaimed = translation_jobs.claim_next_pdf_job("worker-B")
    assert reclaimed and reclaimed["job_id"] == job_id
    assert not translation_jobs.update_translation_job_owned(job_id, "worker-A", status="complete")
    row = _row(job_id)
    assert row["status"] == "running"
    assert row["worker_id"] == "worker-B"


def test_independent_heartbeat_during_blocking_translation() -> None:
    import translation_jobs

    original_interval = translation_jobs.PDF_JOB_HEARTBEAT_SECONDS
    translation_jobs.PDF_JOB_HEARTBEAT_SECONDS = 1
    try:
        job_id = _make_job("heartbeat-blocking.pdf")
        job = translation_jobs.claim_next_pdf_job("worker-heartbeat")
        assert job and job["job_id"] == job_id
        old_time = "2000-01-01T00:00:00"
        translation_jobs.update_translation_job_owned(job_id, "worker-heartbeat", heartbeat_at=old_time)

        def blocking_translator(**_kwargs):
            time.sleep(2.3)
            return (b"%PDF-1.4\n%fake output\n", b"", 0, {"n_review_items": 0})

        thread = threading.Thread(
            target=translation_jobs.run_claimed_pdf_job,
            args=(job, "test-key", "worker-heartbeat"),
            kwargs={"translator": blocking_translator},
        )
        thread.start()
        time.sleep(1.4)
        row = _row(job_id)
        assert row["status"] == "running"
        assert row["heartbeat_at"] != old_time
        thread.join()
        assert _row(job_id)["status"] == "complete"
    finally:
        translation_jobs.PDF_JOB_HEARTBEAT_SECONDS = original_interval


def test_stale_recovery_respects_active_heartbeat() -> None:
    import translation_jobs

    job_id = _make_job("active-heartbeat.pdf")
    job = translation_jobs.claim_next_pdf_job("active-worker")
    assert job and job["job_id"] == job_id
    result = translation_jobs.recover_stale_pdf_jobs("recovery-worker")
    assert job_id not in result["recovered"]
    assert _row(job_id)["status"] == "running"

    stale_time = (datetime.now() - timedelta(seconds=translation_jobs.PDF_JOB_STALE_SECONDS + 10)).isoformat()
    translation_jobs.update_translation_job_owned(job_id, "active-worker", heartbeat_at=stale_time)
    result = translation_jobs.recover_stale_pdf_jobs("recovery-worker")
    assert job_id in result["recovered"]
    assert _row(job_id)["status"] == "queued"


def main() -> None:
    start = time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        _prepare_temp_db(tmp)
        test_db_config()
        test_concurrent_read_write()
        test_lock_retry_success()
        test_lock_retry_limit()
        test_no_retry_programming_error()
        test_atomic_claim_regression()
        test_ownership_regression()
        test_independent_heartbeat_during_blocking_translation()
        test_stale_recovery_respects_active_heartbeat()
    print(f"sqlite hardening tests passed in {time.monotonic() - start:.2f}s")


if __name__ == "__main__":
    main()
