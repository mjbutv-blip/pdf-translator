from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _test_database_url() -> str:
    return os.environ.get("TEST_POSTGRES_DATABASE_URL", "").strip()


def _looks_like_test_url(url: str) -> bool:
    parsed = urlparse(url)
    haystack = " ".join(
        str(part or "").lower()
        for part in [parsed.hostname, parsed.username, parsed.path]
    )
    return any(marker in haystack for marker in ("localhost", "127.0.0.1", "test", "dev", "staging"))


def _skip_without_test_postgres() -> bool:
    url = _test_database_url()
    if not url:
        print("SKIP: TEST_POSTGRES_DATABASE_URL is not set")
        return True
    if not _looks_like_test_url(url) and os.environ.get("ALLOW_POSTGRES_INTEGRATION_TESTS") != "1":
        print("SKIP: TEST_POSTGRES_DATABASE_URL does not look like a test database")
        return True
    os.environ["DATABASE_URL"] = url
    return False


def _now() -> str:
    import translation_core

    return translation_core._now_iso()


def _cleanup(prefix: str) -> None:
    from db import get_db_connection

    with get_db_connection() as conn:
        conn.execute(
            """
            DELETE FROM translation_candidate_occurrences
            WHERE translation_job_id IN (
                SELECT job_id FROM translation_jobs WHERE customer_id LIKE ?
            )
            """,
            (f"{prefix}%",),
        )
        conn.execute("DELETE FROM translation_jobs WHERE customer_id LIKE ?", (f"{prefix}%",))
        conn.execute("DELETE FROM glossary_change_requests WHERE customer_id LIKE ?", (f"{prefix}%",))
        conn.execute("DELETE FROM term_candidates WHERE customer_id LIKE ?", (f"{prefix}%",))
        conn.execute("DELETE FROM glossary_terms WHERE customer_id LIKE ?", (f"{prefix}%",))
        conn.execute("DELETE FROM customers WHERE customer_id LIKE ?", (f"{prefix}%",))
        conn.execute("DELETE FROM translation_workers WHERE worker_id LIKE ?", (f"{prefix}%",))


def _insert_customer(customer_id: str) -> None:
    from db import get_db_connection

    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO customers (
                customer_id, customer_name, customer_code, group_name,
                assigned_staff_username, note, created_at
            )
            VALUES (?, ?, ?, 'PG Test', 'pg_test', '', ?)
            """,
            (customer_id, customer_id, customer_id, _now()),
        )


def _insert_glossary_term(customer_id: str) -> int:
    from db import get_db_connection

    with get_db_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO glossary_terms (
                customer_id, english_term, chinese_translation, normalized_key,
                note, created_by, updated_by, updated_at, status
            )
            VALUES (?, 'lining', '里布', 'lining', '', 'pg_test', 'pg_test', ?, 'active')
            RETURNING glossary_id
            """,
            (customer_id, _now()),
        ).fetchone()
    return int(row["glossary_id"])


def _insert_candidate(customer_id: str) -> int:
    from db import get_db_connection

    with get_db_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO term_candidates (
                customer_id, source_file_name, source_type, page_or_sheet,
                cell_coordinate, original_term, normalized_term, variants,
                ai_suggested_translation, final_translation, context_sentence,
                frequency, confidence, status, created_by, created_at
            )
            VALUES (
                ?, 'pg-test.pdf', 'PDF', '1', '', 'gusset lining', 'gusset lining',
                '["gusset lining"]', '裆部内衬', '', 'sample context',
                1, 'medium', 'draft', 'pg_test', ?
            )
            RETURNING candidate_id
            """,
            (customer_id, _now()),
        ).fetchone()
    return int(row["candidate_id"])


def _create_job(customer_id: str, *, job_type: str = "PDF", execution_mode: str = "external", status: str = "queued") -> str:
    import translation_jobs

    return translation_jobs.create_translation_job(
        job_type=job_type,
        username="pg_test",
        customer_id=customer_id,
        source_file_name="pg-test.pdf" if job_type == "PDF" else "pg-test.xlsx",
        input_bytes=b"pg input bytes",
        aux_bytes=None,
        config={"scope_mode": "all"},
        execution_mode=execution_mode,
        initial_status=status,
    )


def test_postgres_init_and_core_paths() -> None:
    if _skip_without_test_postgres():
        return

    import translation_core
    import translation_jobs
    import translation_service
    from db import get_db_connection

    prefix = f"PGIT-{uuid4().hex[:8]}"
    customer_id = f"{prefix}-CUST"
    worker_id = f"{prefix}-worker"
    _cleanup(prefix)
    try:
        translation_core.init_db()
        with get_db_connection() as conn:
            tables = {
                row["tablename"]
                for row in conn.execute(
                    """
                    SELECT tablename
                    FROM pg_catalog.pg_tables
                    WHERE schemaname = 'public'
                    """
                ).fetchall()
            }
        for table in {
            "users",
            "customers",
            "glossary_terms",
            "glossary_change_requests",
            "term_candidates",
            "translation_jobs",
            "translation_candidate_occurrences",
            "translation_workers",
        }:
            assert table in tables

        _insert_customer(customer_id)
        assert any(row["customer_id"] == customer_id for row in translation_service.list_customers())

        glossary_id = _insert_glossary_term(customer_id)
        with get_db_connection() as conn:
            term = conn.execute(
                "SELECT glossary_id, customer_id, english_term FROM glossary_terms WHERE glossary_id = ?",
                (glossary_id,),
            ).fetchone()
        assert term["customer_id"] == customer_id

        candidate_id = _insert_candidate(customer_id)
        with get_db_connection() as conn:
            conn.execute(
                "UPDATE term_candidates SET final_translation = ? WHERE candidate_id = ?",
                ("裆衬", candidate_id),
            )
            updated = conn.execute(
                "SELECT final_translation FROM term_candidates WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
        assert updated["final_translation"] == "裆衬"

        job_id = _create_job(customer_id)
        claimed = translation_jobs.claim_next_external_translation_job(worker_id)
        assert claimed and claimed["job_id"] == job_id
        assert translation_jobs.update_translation_job_owned(
            job_id,
            worker_id,
            heartbeat_at=_now(),
            result_file=b"translated bytes",
            result_report=b"report bytes",
            result_meta=json.dumps({"postgres": True}),
        )
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO translation_candidate_occurrences (
                    translation_job_id, candidate_id, occurrence_count,
                    representative_page_or_sheet, representative_cell_coordinate,
                    original_term_snapshot, normalized_term_snapshot,
                    context_sentence_snapshot, created_at
                )
                VALUES (?, ?, 2, '1', '', 'gusset lining', 'gusset lining', 'sample context', ?)
                """,
                (job_id, candidate_id, _now()),
            )
            persisted = conn.execute(
                """
                SELECT result_file, result_report
                FROM translation_jobs
                WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()
        assert bytes(persisted["result_file"]) == b"translated bytes"
        assert bytes(persisted["result_report"]) == b"report bytes"
        assert len(translation_jobs.list_term_candidates_for_job(job_id)) == 1

        sync_job_id = _create_job(customer_id, job_type="Excel", execution_mode="sync", status="running")
        assert translation_jobs.claim_next_external_translation_job(f"{prefix}-worker-2") is None
        assert translation_jobs.get_translation_job_by_id(sync_job_id)["status"] == "running"

        translation_jobs.register_translation_worker(worker_id)
        assert translation_jobs.heartbeat_translation_worker(worker_id)
        assert translation_jobs.get_worker_health()["live_worker_count"] >= 1
        translation_jobs.stop_translation_worker(worker_id)
    finally:
        _cleanup(prefix)


def test_postgres_atomic_claim_concurrent_connections() -> None:
    if _skip_without_test_postgres():
        return

    import translation_core
    import translation_jobs

    prefix = f"PGCLAIM-{uuid4().hex[:8]}"
    customer_id = f"{prefix}-CUST"
    _cleanup(prefix)
    try:
        translation_core.init_db()
        _insert_customer(customer_id)
        job_id = _create_job(customer_id)
        barrier = threading.Barrier(2)
        results: list[dict | None] = []

        def claim(worker_id: str) -> None:
            barrier.wait()
            results.append(translation_jobs.claim_next_external_translation_job(worker_id))

        threads = [
            threading.Thread(target=claim, args=(f"{prefix}-worker-A",)),
            threading.Thread(target=claim, args=(f"{prefix}-worker-B",)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        claimed = [row for row in results if row is not None]
        assert len(claimed) == 1
        assert claimed[0]["job_id"] == job_id
    finally:
        _cleanup(prefix)


def test_postgres_worker_once_smoke() -> None:
    if _skip_without_test_postgres():
        return

    import translation_core
    import worker
    from db import get_db_connection

    prefix = f"PGWORKER-{uuid4().hex[:8]}"
    worker_id = f"{prefix}-once"
    _cleanup(prefix)
    try:
        translation_core.init_db()
        assert worker.run_worker(once=True, worker_id=worker_id, api_key="test-key") == 0
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT status FROM translation_workers WHERE worker_id = ?",
                (worker_id,),
            ).fetchone()
        assert row and row["status"] == "stopped"
    finally:
        _cleanup(prefix)


if __name__ == "__main__":
    test_postgres_init_and_core_paths()
    test_postgres_atomic_claim_concurrent_connections()
    test_postgres_worker_once_smoke()
