from __future__ import annotations

import hashlib
import os
import sqlite3
import sys
import tempfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import db
import db_backup


def _prepare_temp_db(tmpdir: str) -> Path:
    db.DB_PATH = Path(tmpdir) / "snapshot_source.sqlite"
    import translation_core

    translation_core.init_db()
    with db.get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO customers (
                customer_id, customer_name, customer_code, group_name,
                assigned_staff_username, created_at
            )
            VALUES ('SNAP-CUST', 'Snapshot Customer', 'SNAP-CUST', 'Test', 'snapshot_test', ?)
            """,
            (translation_core._now_iso(),),
        )
    return db.DB_PATH


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_sqlite_snapshot_succeeds_and_validates() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = _prepare_temp_db(tmpdir)
        before_hash = _sha256(source_path)

        result = db_backup.create_sqlite_snapshot({"role": "company_admin", "username": "admin"})

        after_hash = _sha256(source_path)
        assert after_hash == before_hash
        assert result["status"] == "completed"
        assert result["database_backend"] == "sqlite"
        assert result["snapshot_bytes"]
        metadata = result["metadata"]
        assert metadata["integrity_check"] == "ok"
        assert metadata["missing_tables"] == []
        assert metadata["customers"] == 1
        assert metadata["size_bytes"] == len(result["snapshot_bytes"])

        snapshot_path = Path(tmpdir) / "downloaded_snapshot.sqlite"
        snapshot_path.write_bytes(result["snapshot_bytes"])
        with sqlite3.connect(snapshot_path) as conn:
            assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            count = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        assert count == 1


def test_sqlite_snapshot_permission_denied_for_non_admin() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        _prepare_temp_db(tmpdir)
        try:
            db_backup.create_sqlite_snapshot({"role": "staff", "username": "staff"})
        except PermissionError:
            return
        raise AssertionError("Expected PermissionError")


def test_sqlite_snapshot_refuses_when_postgres_active() -> None:
    old_url = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = "postgresql:///pdf_project_test_only"
    try:
        result = db_backup.create_sqlite_snapshot(
            {"role": "company_admin", "username": "admin"},
            require_company_admin=True,
        )
        assert result["status"] == "postgres_active"
        assert result["database_backend"] == "postgres"
        assert result["snapshot_bytes"] is None
    finally:
        if old_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = old_url


def test_sqlite_snapshot_helper_does_not_import_streamlit() -> None:
    assert "streamlit" not in sys.modules


def test_snapshot_download_names() -> None:
    db_name, meta_name = db_backup.snapshot_download_names("2026-08-30T12:34:56")
    assert db_name == "pdf_project_production_snapshot_20260830_123456.db"
    assert meta_name == "pdf_project_production_snapshot_20260830_123456.json"


def test_snapshot_ui_is_company_admin_only() -> None:
    source = Path(__file__).resolve().parents[1] / "app.py"
    app_text = source.read_text(encoding="utf-8")
    backup_tab_pos = app_text.index('tab_labels.append("🛡️ 数据库备份")')
    admin_guard_pos = app_text.index("if can_approve_glossary_change(current_user):")
    tabs_pos = app_text.index("tabs = st.tabs(tab_labels)")
    assert admin_guard_pos < backup_tab_pos < tabs_pos
    assert "create_sqlite_snapshot(current_user)" in app_text


if __name__ == "__main__":
    test_sqlite_snapshot_succeeds_and_validates()
    test_sqlite_snapshot_permission_denied_for_non_admin()
    test_sqlite_snapshot_refuses_when_postgres_active()
    test_sqlite_snapshot_helper_does_not_import_streamlit()
    test_snapshot_download_names()
    test_snapshot_ui_is_company_admin_only()
    print("sqlite snapshot tests passed")
