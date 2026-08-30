from __future__ import annotations

import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import db


def _use_temp_db(tmpdir: str):
    db.DB_PATH = Path(tmpdir) / "worker_health.sqlite"
    import translation_core

    translation_core.init_db()
    with db.get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO customers (
                customer_id, customer_name, customer_code, group_name,
                assigned_staff_username, created_at
            )
            VALUES ('TEST-CUST', 'Worker Health Customer', 'TEST-CUST', 'Test', 'worker_test', ?)
            """,
            (translation_core._now_iso(),),
        )
    return translation_core


def _make_job(
    *,
    job_type: str = "PDF",
    status: str = "queued",
    execution_mode: str = "external",
    filename: str = "test.pdf",
) -> str:
    import translation_jobs

    return translation_jobs.create_translation_job(
        job_type=job_type,
        username="worker_health",
        customer_id="TEST-CUST",
        source_file_name=filename,
        input_bytes=b"fake input",
        aux_bytes=None,
        config={},
        execution_mode=execution_mode,
        initial_status=status,
    )


def test_worker_registration() -> None:
    import translation_jobs

    translation_jobs.register_translation_worker("worker-A")
    health = translation_jobs.get_worker_health()
    assert health["live_worker_count"] == 1
    assert health["stale_worker_count"] == 0
    assert health["latest_heartbeat_at"]


def test_idle_heartbeat_updates() -> None:
    import translation_jobs
    import worker

    original_interval = worker.WORKER_HEARTBEAT_SECONDS
    worker.WORKER_HEARTBEAT_SECONDS = 1
    try:
        translation_jobs.register_translation_worker("idle-worker")
        before = translation_jobs.get_worker_health(include_workers=True)["workers"][0]["heartbeat_at"]
        shutdown = worker.ShutdownRequested()
        thread = worker._start_worker_heartbeat("idle-worker", shutdown)
        time.sleep(1.2)
        shutdown.value = True
        thread.join(timeout=2)
        after = translation_jobs.get_worker_health(include_workers=True)["workers"][0]["heartbeat_at"]
        assert after > before
    finally:
        worker.WORKER_HEARTBEAT_SECONDS = original_interval


def test_stale_worker_not_live() -> None:
    import translation_jobs

    old_now = datetime.now() - timedelta(seconds=translation_jobs.WORKER_STALE_SECONDS + 10)
    with db.get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO translation_workers (
                worker_id, started_at, heartbeat_at, status, stopped_at
            )
            VALUES ('stale-worker', ?, ?, 'running', NULL)
            """,
            (old_now.isoformat(), old_now.isoformat()),
        )
    health = translation_jobs.get_worker_health()
    assert health["live_worker_count"] == 0
    assert health["stale_worker_count"] == 1


def test_graceful_stop() -> None:
    import translation_jobs

    translation_jobs.register_translation_worker("stop-worker")
    translation_jobs.stop_translation_worker("stop-worker")
    health = translation_jobs.get_worker_health(include_workers=True)
    assert health["live_worker_count"] == 0
    assert health["workers"][0]["status"] == "stopped"


def test_crash_simulation() -> None:
    import translation_jobs

    now = datetime.now()
    translation_jobs.register_translation_worker("crashed-worker")
    assert translation_jobs.get_worker_health(now=now)["live_worker_count"] == 1
    later = now + timedelta(seconds=translation_jobs.WORKER_STALE_SECONDS + 5)
    health = translation_jobs.get_worker_health(now=later)
    assert health["live_worker_count"] == 0
    assert health["stale_worker_count"] == 1


def test_multiple_workers() -> None:
    import translation_jobs

    now = datetime.now()
    stale = now - timedelta(seconds=translation_jobs.WORKER_STALE_SECONDS + 5)
    with db.get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO translation_workers (worker_id, started_at, heartbeat_at, status, stopped_at)
            VALUES ('worker-A', ?, ?, 'running', NULL)
            """,
            (now.isoformat(), now.isoformat()),
        )
        conn.execute(
            """
            INSERT INTO translation_workers (worker_id, started_at, heartbeat_at, status, stopped_at)
            VALUES ('worker-B', ?, ?, 'running', NULL)
            """,
            (now.isoformat(), now.isoformat()),
        )
        conn.execute(
            """
            INSERT INTO translation_workers (worker_id, started_at, heartbeat_at, status, stopped_at)
            VALUES ('worker-C', ?, ?, 'running', NULL)
            """,
            (stale.isoformat(), stale.isoformat()),
        )
    health = translation_jobs.get_worker_health(now=now)
    assert health["live_worker_count"] == 2
    assert health["stale_worker_count"] == 1


def test_queue_summary_counts_only_external_jobs() -> None:
    import translation_jobs

    _make_job(job_type="PDF", status="queued", execution_mode="external", filename="q1.pdf")
    _make_job(job_type="PDF", status="queued", execution_mode="external", filename="q2.pdf")
    _make_job(job_type="Excel", status="queued", execution_mode="external", filename="q1.xlsx")
    _make_job(job_type="Excel", status="queued", execution_mode="external", filename="q2.xlsx")
    _make_job(job_type="Excel", status="queued", execution_mode="external", filename="q3.xlsx")
    _make_job(job_type="PDF", status="running", execution_mode="external", filename="r1.pdf")
    _make_job(job_type="PDF", status="running", execution_mode="sync", filename="sync.pdf")
    summary = translation_jobs.get_translation_queue_health()
    assert summary["queued_count"] == 5
    assert summary["running_count"] == 1
    assert summary["queued_pdf_count"] == 2
    assert summary["queued_excel_count"] == 3
    assert summary["running_pdf_count"] == 1
    assert summary["running_excel_count"] == 0
    assert summary["oldest_queued_created_at"]
    assert summary["oldest_queue_age_seconds"] is not None


def test_no_worker_with_queue_flag() -> None:
    import translation_jobs

    _make_job(job_type="PDF", status="queued", execution_mode="external", filename="waiting.pdf")
    health = translation_jobs.get_worker_queue_health()
    assert health["live_worker_count"] == 0
    assert health["queued_count"] == 1
    assert health["queue_waiting_without_worker"] is True


def test_worker_with_queue_not_flagged() -> None:
    import translation_jobs

    translation_jobs.register_translation_worker("live-worker")
    _make_job(job_type="PDF", status="queued", execution_mode="external", filename="waiting.pdf")
    _make_job(job_type="Excel", status="running", execution_mode="external", filename="running.xlsx")
    health = translation_jobs.get_worker_queue_health()
    assert health["live_worker_count"] == 1
    assert health["queued_count"] == 1
    assert health["running_count"] == 1
    assert health["queue_waiting_without_worker"] is False


def test_once_mode_marks_worker_stopped() -> None:
    import translation_jobs
    import worker

    assert worker.run_worker(once=True, worker_id="health-once", api_key="test-key") == 0
    health = translation_jobs.get_worker_health(include_workers=True)
    rows = {row["worker_id"]: row for row in health["workers"]}
    assert rows["health-once"]["status"] == "stopped"
    assert health["live_worker_count"] == 0


def test_imports_do_not_load_streamlit() -> None:
    import translation_jobs  # noqa: F401
    import worker  # noqa: F401

    assert "streamlit" not in sys.modules


def main() -> None:
    start = time.monotonic()
    tests = [
        test_worker_registration,
        test_idle_heartbeat_updates,
        test_stale_worker_not_live,
        test_graceful_stop,
        test_crash_simulation,
        test_multiple_workers,
        test_queue_summary_counts_only_external_jobs,
        test_no_worker_with_queue_flag,
        test_worker_with_queue_not_flagged,
        test_once_mode_marks_worker_stopped,
        test_imports_do_not_load_streamlit,
    ]
    for test in tests:
        with tempfile.TemporaryDirectory() as tmp:
            _use_temp_db(tmp)
            test()
    print(f"worker health tests passed in {time.monotonic() - start:.2f}s")


if __name__ == "__main__":
    main()
