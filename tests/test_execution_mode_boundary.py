from __future__ import annotations

import io
import sys
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path

import fitz
import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import db


def _prepare_temp_db(tmpdir: str):
    db.DB_PATH = Path(tmpdir) / "execution_mode.sqlite"
    import translation_core

    translation_core.init_db()
    with db.get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO customers (
                customer_id, customer_name, customer_code, group_name,
                assigned_staff_username, created_at
            )
            VALUES ('TEST-CUST', 'Execution Test Customer', 'TEST-CUST', 'Test', 'worker_test', ?)
            """,
            (translation_core._now_iso(),),
        )
    return translation_core


def _pdf_bytes() -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello")
    data = doc.tobytes()
    doc.close()
    return data


def _xlsx_bytes() -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws["A1"] = "Hello"
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _make_job(
    filename: str,
    *,
    job_type: str = "PDF",
    execution_mode: str = "external",
    initial_status: str = "queued",
    username: str = "execution_test",
) -> str:
    import translation_jobs

    payload = _xlsx_bytes() if job_type == "Excel" else _pdf_bytes()
    return translation_jobs.create_translation_job(
        job_type=job_type,
        username=username,
        customer_id="TEST-CUST",
        source_file_name=filename,
        input_bytes=payload,
        aux_bytes=None,
        config={"scope_mode": "all", "selected_pages": None, "scope_detection": [], "scope_cfg": {"total_pages": 1}},
        execution_mode=execution_mode,
        initial_status=initial_status,
        initial_progress=0.01 if initial_status == "running" else 0.0,
        initial_message="running" if initial_status == "running" else "queued",
    )


def _row(job_id: str) -> dict:
    import translation_jobs

    row = translation_jobs.get_translation_job_by_id(job_id)
    assert row is not None
    return row


def test_service_pdf_sync_job_not_claimable() -> None:
    import translation_jobs
    import translation_service

    pdf = _pdf_bytes()
    original_translator = translation_service.run_pdf_translation

    def fake_pdf_translator(**kwargs):
        assert translation_jobs.claim_next_pdf_job("external-worker") is None
        kwargs["on_progress"](0.5)
        return (pdf, b"", 0, {"n_review_items": 0, "unrecorded_terms": []})

    translation_service.run_pdf_translation = fake_pdf_translator
    try:
        result = translation_service.translate_pdf_document(
            customer_id="TEST-CUST",
            filename="service.pdf",
            file_bytes=pdf,
            api_key="test-key",
            created_by="service_user",
        )
    finally:
        translation_service.run_pdf_translation = original_translator

    assert result["status"] == "completed"
    row = _row(result["translation_job_id"])
    assert row["execution_mode"] == "sync"
    assert row["status"] == "complete"


def test_service_excel_sync_job_marked_sync() -> None:
    import translation_service

    original_translator = translation_service.run_excel_translation

    def fake_excel_translator(**_kwargs):
        return (_xlsx_bytes(), 1, 0, [], {"n_review_items": 0, "unrecorded_terms": [], "unrecorded_term_count": 0})

    translation_service.run_excel_translation = fake_excel_translator
    try:
        result = translation_service.translate_excel_document(
            customer_id="TEST-CUST",
            filename="service.xlsx",
            file_bytes=_xlsx_bytes(),
            api_key="test-key",
            created_by="service_user",
        )
    finally:
        translation_service.run_excel_translation = original_translator

    assert result["status"] == "completed"
    row = _row(result["translation_job_id"])
    assert row["execution_mode"] == "sync"
    assert row["status"] == "complete"


def test_external_claim_and_mixed_queue() -> None:
    import translation_jobs

    external_id = _make_job("external.pdf", execution_mode="external", initial_status="queued")
    embedded_id = _make_job("embedded.pdf", execution_mode="embedded", initial_status="queued")
    sync_pdf_id = _make_job("sync.pdf", execution_mode="sync", initial_status="running")
    sync_excel_id = _make_job("sync.xlsx", job_type="Excel", execution_mode="sync", initial_status="running")

    claimed = translation_jobs.claim_next_pdf_job("external-worker")
    assert claimed and claimed["job_id"] == external_id
    assert translation_jobs.claim_next_pdf_job("external-worker-2") is None
    assert _row(embedded_id)["status"] == "queued"
    assert _row(sync_pdf_id)["status"] == "running"
    assert _row(sync_excel_id)["status"] == "running"


def test_stale_sync_and_embedded_are_not_recovered() -> None:
    import translation_jobs

    stale_time = (datetime.now() - timedelta(seconds=translation_jobs.PDF_JOB_STALE_SECONDS + 10)).isoformat()
    sync_id = _make_job("stale-sync.pdf", execution_mode="sync", initial_status="running")
    embedded_id = _make_job("stale-embedded.pdf", execution_mode="embedded", initial_status="running")
    translation_jobs.update_translation_job(sync_id, worker_id=None, heartbeat_at=stale_time, updated_at=stale_time)
    translation_jobs.update_translation_job(embedded_id, worker_id="embedded-worker", heartbeat_at=stale_time, updated_at=stale_time)

    result = translation_jobs.recover_stale_pdf_jobs("external-recovery")
    assert sync_id not in result["recovered"]
    assert embedded_id not in result["recovered"]
    assert _row(sync_id)["status"] == "running"
    assert _row(embedded_id)["status"] == "running"


def test_stale_external_recovery() -> None:
    import translation_jobs

    stale_time = (datetime.now() - timedelta(seconds=translation_jobs.PDF_JOB_STALE_SECONDS + 10)).isoformat()
    external_id = _make_job("stale-external.pdf", execution_mode="external", initial_status="running")
    translation_jobs.update_translation_job(external_id, worker_id="worker-A", heartbeat_at=stale_time, updated_at=stale_time, attempt_count=1)

    result = translation_jobs.recover_stale_pdf_jobs("external-recovery")
    assert external_id in result["recovered"]
    row = _row(external_id)
    assert row["status"] == "queued"
    assert row["execution_mode"] == "external"


def test_external_worker_does_not_claim_embedded() -> None:
    import translation_jobs

    embedded_id = _make_job("embedded-only.pdf", execution_mode="embedded", initial_status="queued")
    assert translation_jobs.claim_next_pdf_job("external-worker") is None
    assert _row(embedded_id)["status"] == "queued"


def test_ownership_and_atomic_regression() -> None:
    import translation_jobs

    ownership_id = _make_job("ownership.pdf", execution_mode="external", initial_status="queued")
    job = translation_jobs.claim_next_pdf_job("worker-A")
    assert job and job["job_id"] == ownership_id
    translation_jobs.update_translation_job(ownership_id, status="queued", worker_id=None, heartbeat_at=None)
    reclaimed = translation_jobs.claim_next_pdf_job("worker-B")
    assert reclaimed and reclaimed["job_id"] == ownership_id
    assert not translation_jobs.update_translation_job_owned(ownership_id, "worker-A", status="complete")
    assert _row(ownership_id)["worker_id"] == "worker-B"

    atomic_id = _make_job("atomic.pdf", execution_mode="external", initial_status="queued")
    barrier = threading.Barrier(2)
    results = []

    def claim(worker_id: str) -> None:
        barrier.wait()
        results.append(translation_jobs.claim_next_pdf_job(worker_id))

    threads = [
        threading.Thread(target=claim, args=("atomic-A",)),
        threading.Thread(target=claim, args=("atomic-B",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    claimed = [job for job in results if job]
    assert len(claimed) == 1
    assert claimed[0]["job_id"] == atomic_id


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _prepare_temp_db(tmp)
        test_service_pdf_sync_job_not_claimable()
    with tempfile.TemporaryDirectory() as tmp:
        _prepare_temp_db(tmp)
        test_service_excel_sync_job_marked_sync()
    with tempfile.TemporaryDirectory() as tmp:
        _prepare_temp_db(tmp)
        test_external_claim_and_mixed_queue()
    with tempfile.TemporaryDirectory() as tmp:
        _prepare_temp_db(tmp)
        test_stale_sync_and_embedded_are_not_recovered()
    with tempfile.TemporaryDirectory() as tmp:
        _prepare_temp_db(tmp)
        test_stale_external_recovery()
    with tempfile.TemporaryDirectory() as tmp:
        _prepare_temp_db(tmp)
        test_external_worker_does_not_claim_embedded()
    with tempfile.TemporaryDirectory() as tmp:
        _prepare_temp_db(tmp)
        test_ownership_and_atomic_regression()
    print("execution mode boundary tests passed")


if __name__ == "__main__":
    main()
