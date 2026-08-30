from __future__ import annotations

import tempfile
import threading
import time
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import db


def _use_temp_db(tmpdir: str):
    db.DB_PATH = Path(tmpdir) / "worker_lifecycle.sqlite"
    import translation_core

    translation_core.init_db()
    with db.get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO customers (
                customer_id, customer_name, customer_code, group_name,
                assigned_staff_username, created_at
            )
            VALUES ('TEST-CUST', 'Worker Test Customer', 'TEST-CUST', 'Test', 'worker_test', ?)
            """,
            (translation_core._now_iso(),),
        )
    return translation_core


def _make_job(filename: str = "test.pdf", username: str = "worker_test") -> str:
    import translation_jobs

    return translation_jobs.create_translation_job(
        job_type="PDF",
        username=username,
        customer_id="TEST-CUST",
        source_file_name=filename,
        input_bytes=b"%PDF-1.4\n%fake input\n",
        aux_bytes=None,
        config={"scope_mode": "all", "selected_pages": None, "scope_detection": [], "scope_cfg": {"total_pages": 1}},
    )


def _fake_translator(**kwargs):
    kwargs["on_page"](0, 1, 1)
    kwargs["on_block"]("fake block")
    kwargs["on_progress"](1.0)
    return (
        b"%PDF-1.4\n%fake output\n",
        b"fake report",
        2,
        {
            "n_review_items": 1,
            "summary_text": "fake summary",
            "unrecorded_terms": [{"original_term": "fake"}],
        },
    )


def _row(job_id: str) -> dict:
    import translation_jobs

    row = translation_jobs.get_translation_job_by_id(job_id)
    assert row is not None
    return row


def test_atomic_claim() -> None:
    job_id = _make_job("atomic.pdf")
    barrier = threading.Barrier(2)
    results = []

    def claim(worker_id: str) -> None:
        import translation_jobs

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
    assert len(claimed) == 1, results
    assert claimed[0]["job_id"] == job_id
    assert _row(job_id)["status"] == "running"


def test_complete_lifecycle() -> None:
    import translation_jobs

    job_id = _make_job("complete.pdf")
    job = translation_jobs.claim_next_pdf_job("worker-complete")
    assert job and job["job_id"] == job_id
    assert translation_jobs.run_claimed_pdf_job(job, "test-key", "worker-complete", translator=_fake_translator) == "complete"
    row = _row(job_id)
    assert row["status"] == "complete"
    assert row["worker_id"] == "worker-complete"
    assert row["attempt_count"] == 1
    assert row["heartbeat_at"]
    assert row["result_file"] == b"%PDF-1.4\n%fake output\n"


def test_stale_recovery_and_max_attempts() -> None:
    import translation_core
    import translation_jobs

    job_id = _make_job("stale.pdf")
    stale_time = (datetime.now() - timedelta(seconds=translation_jobs.PDF_JOB_STALE_SECONDS + 10)).isoformat()
    translation_jobs.update_translation_job(
        job_id,
        status="running",
        worker_id="dead-worker",
        heartbeat_at=stale_time,
        attempt_count=1,
    )
    result = translation_jobs.recover_stale_pdf_jobs("recovery-worker")
    assert result["recovered"] == [job_id]
    row = _row(job_id)
    assert row["status"] == "queued"
    assert row["worker_id"] is None
    assert row["error"] == translation_jobs.WORKER_LOST_ERROR

    max_job_id = _make_job("max-attempts.pdf")
    translation_jobs.update_translation_job(
        max_job_id,
        status="running",
        worker_id="dead-worker",
        heartbeat_at=stale_time,
        attempt_count=translation_jobs.PDF_JOB_MAX_ATTEMPTS,
    )
    result = translation_jobs.recover_stale_pdf_jobs("recovery-worker")
    assert result["failed"] == [max_job_id]
    row = _row(max_job_id)
    assert row["status"] == "failed"
    assert row["error"] == translation_jobs.MAX_ATTEMPTS_EXCEEDED_ERROR
    assert translation_core._now_iso()


def test_ownership_protection() -> None:
    import translation_jobs

    job_id = _make_job("ownership.pdf")
    job = translation_jobs.claim_next_pdf_job("worker-A")
    assert job and job["job_id"] == job_id
    translation_jobs.update_translation_job(job_id, status="queued", worker_id=None, heartbeat_at=None)
    reclaimed = translation_jobs.claim_next_pdf_job("worker-B")
    assert reclaimed and reclaimed["job_id"] == job_id
    assert not translation_jobs.update_translation_job_owned(
        job_id,
        "worker-A",
        status="complete",
        result_file=b"stale result",
    )
    row = _row(job_id)
    assert row["status"] == "running"
    assert row["worker_id"] == "worker-B"
    assert row["result_file"] is None


def test_cancellation() -> None:
    import translation_jobs

    queued_job_id = _make_job("cancel-queued.pdf", username="cancel_user")
    result = translation_jobs.cancel_pdf_translation_jobs("cancel_user")
    assert result["cancelled_count"] == 1
    row = _row(queued_job_id)
    assert row["status"] == "failed"
    assert row["error"] == translation_jobs.PDF_JOB_CANCELLED_ERROR

    running_job_id = _make_job("cancel-running.pdf", username="cancel_running_user")
    job = translation_jobs.claim_next_pdf_job("worker-cancel")
    assert job and job["job_id"] == running_job_id
    result = translation_jobs.cancel_pdf_translation_jobs("cancel_running_user")
    assert result["cancelled_count"] == 1
    assert translation_jobs.run_claimed_pdf_job(job, "test-key", "worker-cancel", translator=_fake_translator) == "cancelled"
    row = _row(running_job_id)
    assert row["status"] == "failed"
    assert row["error"] == translation_jobs.PDF_JOB_CANCELLED_ERROR
    assert row["result_file"] is None


def main() -> None:
    start = time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        _use_temp_db(tmp)
        test_atomic_claim()
    with tempfile.TemporaryDirectory() as tmp:
        _use_temp_db(tmp)
        test_complete_lifecycle()
    with tempfile.TemporaryDirectory() as tmp:
        _use_temp_db(tmp)
        test_stale_recovery_and_max_attempts()
    with tempfile.TemporaryDirectory() as tmp:
        _use_temp_db(tmp)
        test_ownership_protection()
    with tempfile.TemporaryDirectory() as tmp:
        _use_temp_db(tmp)
        test_cancellation()
    print(f"worker lifecycle tests passed in {time.monotonic() - start:.2f}s")


if __name__ == "__main__":
    main()
