from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import db
import db_diagnostics


def _prepare_temp_db(tmpdir: str) -> None:
    db.DB_PATH = Path(tmpdir) / "diagnostics.sqlite"
    import translation_core

    translation_core.init_db()
    with db.get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO customers (
                customer_id, customer_name, customer_code, group_name,
                assigned_staff_username, created_at
            )
            VALUES ('DIAG-CUST', 'Diagnostics Customer', 'DIAG-CUST', 'Test', 'diag', ?)
            """,
            (translation_core._now_iso(),),
        )
        conn.execute(
            """
            INSERT INTO glossary_terms (
                customer_id, english_term, chinese_translation, normalized_key,
                updated_at, status
            )
            VALUES ('DIAG-CUST', 'strap', '肩带', 'strap', ?, 'active')
            """,
            (translation_core._now_iso(),),
        )


def test_database_diagnostics_sqlite_read_only_summary() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        _prepare_temp_db(tmpdir)
        result = db_diagnostics.get_safe_database_diagnostics(
            {"role": "company_admin", "username": "admin"}
        )
    assert result["database_backend"] == "SQLite"
    assert result["host"] is None
    assert result["database_name"] == "diagnostics.sqlite"
    assert result["row_counts"]["customers"] == 1
    assert result["row_counts"]["glossary_terms"] == 1
    assert result["schema_readiness"]["translation_workers_table"] is True
    assert all(result["schema_readiness"]["translation_jobs_columns"].values())
    assert result["active_glossary_by_customer"] == [
        {"customer_id": "DIAG-CUST", "active_glossary_count": 1}
    ]
    assert result["candidate_count_by_status"] == []
    assert result["translation_job_distribution"] == []
    assert db_diagnostics.database_diagnostics_verdict(result) == "unable to verify safely"


def test_database_diagnostics_requires_company_admin() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        _prepare_temp_db(tmpdir)
        try:
            db_diagnostics.get_safe_database_diagnostics({"role": "staff", "username": "staff"})
        except PermissionError:
            return
        raise AssertionError("Expected PermissionError")


def test_database_diagnostics_helper_does_not_import_streamlit() -> None:
    assert "streamlit" not in sys.modules


def test_database_diagnostics_ui_is_company_admin_only() -> None:
    source = Path(__file__).resolve().parents[1] / "app.py"
    text = source.read_text(encoding="utf-8")
    tab_pos = text.index('tab_labels.append("🛡️ 数据库备份")')
    admin_guard_pos = text.index("if can_approve_glossary_change(current_user):")
    tabs_pos = text.index("tabs = st.tabs(tab_labels)")
    diagnostics_pos = text.index("get_safe_database_diagnostics(current_user)")
    assert admin_guard_pos < tab_pos < tabs_pos < diagnostics_pos


def test_database_diagnostics_postgres_verdict() -> None:
    diagnostics = {
        "database_backend": "PostgreSQL",
        "row_counts": {"users": 1, "customers": 1, "glossary_terms": 1},
        "schema_readiness": {
            "translation_workers_table": True,
            "translation_jobs_columns": {
                "execution_mode": True,
                "worker_id": True,
                "heartbeat_at": True,
                "attempt_count": True,
            },
        },
    }
    assert (
        db_diagnostics.database_diagnostics_verdict(diagnostics)
        == "production Postgres appears complete and ready for worker"
    )


def test_database_diagnostics_does_not_use_production_url_in_tests() -> None:
    assert not os.environ.get("DATABASE_URL")


if __name__ == "__main__":
    test_database_diagnostics_sqlite_read_only_summary()
    test_database_diagnostics_requires_company_admin()
    test_database_diagnostics_helper_does_not_import_streamlit()
    test_database_diagnostics_ui_is_company_admin_only()
    test_database_diagnostics_postgres_verdict()
    test_database_diagnostics_does_not_use_production_url_in_tests()
    print("database diagnostics tests passed")
