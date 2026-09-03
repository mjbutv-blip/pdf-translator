from __future__ import annotations

import json
import threading

import pytest

import db
import translation_core
import translation_jobs


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "retry.sqlite")
    translation_core.init_db()
    with db.get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO customers (
                customer_id, customer_name, customer_code, group_name,
                assigned_staff_username, created_at
            ) VALUES ('RETRY-CUST', 'Retry Customer', 'RETRY', 'Test', 'retry_user', ?)
            """,
            (translation_core._now_iso(),),
        )


def _job(job_type="PDF", status="queued", *, aux=b"font", config=None):
    return translation_jobs.create_translation_job(
        job_type=job_type,
        username="retry_user",
        customer_id="RETRY-CUST",
        source_file_name="source.pdf" if job_type == "PDF" else "source.xlsx",
        input_bytes=b"input-bytes",
        aux_bytes=aux,
        config=config or {"scope_mode": "all"},
        execution_mode="external",
        initial_status=status,
    )


@pytest.mark.parametrize("job_type", ["PDF", "Excel"])
def test_running_retry_stops_old_owner_and_creates_new_external_job(job_type):
    old_id = _job(job_type, config={"scope_mode": "selected", "translate_images": job_type == "Excel"})
    old = translation_jobs.claim_next_external_translation_job("old-worker")
    assert old["job_id"] == old_id

    result = translation_jobs.retry_translation_job(old_id, "retry_user")
    new = translation_jobs.get_translation_job_by_id(result["new_job_id"])
    stopped = translation_jobs.get_translation_job_by_id(old_id)

    assert result["created"] is True
    assert new["job_id"] != old_id
    assert new["status"] == "queued"
    assert new["execution_mode"] == "external"
    assert new["worker_id"] is None
    assert new["heartbeat_at"] is None
    assert new["attempt_count"] == 0
    cloned_config = json.loads(new["config"])
    assert cloned_config["scope_mode"] == "selected"
    assert cloned_config["translate_images"] is (job_type == "Excel")
    assert cloned_config["retry_of_job_id"] == old_id
    assert stopped["status"] == "failed"
    assert stopped["error"] == translation_jobs.USER_RETRY_REQUESTED_ERROR
    assert not translation_jobs.update_translation_job_owned(
        old_id, "old-worker", status="complete", result_file=b"unsafe"
    )


def test_failed_pdf_retry_preserves_inputs_and_config():
    config = {"scope_mode": "selected", "selected_pages": [1, 3]}
    old_id = _job("PDF", "failed", aux=b"font-bytes", config=config)
    result = translation_jobs.retry_translation_job(old_id, "retry_user")
    new = translation_jobs.get_translation_job_by_id(result["new_job_id"])

    assert new["input_bytes"] == b"input-bytes"
    assert new["aux_bytes"] == b"font-bytes"
    assert new["customer_id"] == "RETRY-CUST"
    assert new["source_file_name"] == "source.pdf"
    new_config = json.loads(new["config"])
    assert new_config["scope_mode"] == "selected"
    assert new_config["selected_pages"] == [1, 3]
    assert new_config["retry_of_job_id"] == old_id


@pytest.mark.parametrize("status", ["queued", "complete"])
def test_queued_and_complete_retry_are_rejected_without_clone(status):
    old_id = _job("PDF", status)
    with pytest.raises(ValueError):
        translation_jobs.retry_translation_job(old_id, "retry_user")
    with db.get_db_connection() as conn:
        count = conn.execute("SELECT COUNT(*) AS n FROM translation_jobs").fetchone()["n"]
    assert count == 1


def test_retry_authorization_requires_matching_username():
    old_id = _job("PDF", "failed")
    with pytest.raises(PermissionError):
        translation_jobs.retry_translation_job(old_id, "another_user")


def test_double_click_returns_one_retry_job():
    old_id = _job("Excel", "queued", config={"translate_images": True})
    claimed = translation_jobs.claim_next_external_translation_job("old-worker")
    assert claimed["job_id"] == old_id
    barrier = threading.Barrier(2)
    results = []

    def retry():
        barrier.wait()
        results.append(translation_jobs.retry_translation_job(old_id, "retry_user"))

    threads = [threading.Thread(target=retry), threading.Thread(target=retry)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert {item["new_job_id"] for item in results} == {results[0]["new_job_id"]}
    assert sum(bool(item["created"]) for item in results) == 1
    with db.get_db_connection() as conn:
        retries = conn.execute(
            "SELECT COUNT(*) AS n FROM translation_jobs WHERE job_id != ?", (old_id,)
        ).fetchone()["n"]
    assert retries == 1


def test_ui_offers_retry_only_for_running_and_failed_jobs():
    source = open(translation_jobs.__file__.replace("translation_jobs.py", "app.py"), encoding="utf-8").read()
    assert '_render_active_job(job, current_user["username"], "pdf")' in source
    assert '_render_active_job(job, current_user["username"], "excel")' in source
    assert 'if job["status"] == "failed":\n                _render_retry_action' in source
    assert 'if job["status"] == "complete":\n                _render_retry_action' not in source
