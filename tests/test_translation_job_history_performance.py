from __future__ import annotations

import inspect

import translation_jobs


class _Cursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _Connection:
    def __init__(self, rows, statements):
        self.rows = rows
        self.statements = statements

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, sql, params=()):
        self.statements.append((sql, params))
        return _Cursor(self.rows)


def test_rendering_35_jobs_uses_one_metadata_query_without_blob_columns(monkeypatch):
    rows = [
        {
            "job_id": f"job-{n}", "source_file_name": f"{n}.pdf", "job_type": "PDF",
            "status": "complete", "progress": 1.0, "message": "done", "error": "",
            "created_at": str(n), "updated_at": str(n), "execution_mode": "external",
            "worker_id": None, "heartbeat_at": None,
        }
        for n in range(35)
    ]
    statements = []
    connections = []

    def connect():
        connections.append(1)
        return _Connection(rows, statements)

    monkeypatch.setattr(translation_jobs, "get_db_connection", connect)
    jobs = translation_jobs.list_translation_jobs("alice", "PDF", 35)

    assert len(jobs) == 35
    assert len(connections) == 1
    assert len(statements) == 1
    select_clause = statements[0][0].lower().split("from translation_jobs", 1)[0]
    for blob_column in ("input_bytes", "aux_bytes", "result_file", "result_report"):
        assert blob_column not in select_clause


def test_result_bytes_have_a_dedicated_single_job_query():
    source = inspect.getsource(translation_jobs.get_translation_job_result).lower()
    assert "result_file" in source
    assert "result_report" in source
    assert "job_id = ?" in source
    assert "username = ?" in source
    assert "status = 'complete'" in source


def test_active_jobs_are_sorted_before_history_in_metadata_query():
    source = inspect.getsource(translation_jobs.list_translation_jobs).lower()
    assert "when 'running' then 0" in source
    assert "when 'queued' then 1" in source


def test_job_panels_only_request_result_after_prepare_download():
    app_source = (translation_jobs.__file__.replace("translation_jobs.py", "app.py"))
    source = open(app_source, encoding="utf-8").read()
    assert 'st.button("准备下载"' in source
    assert "get_translation_job_result(job[\"job_id\"], username)" in source
    assert "get_translation_job(job[\"job_id\"]" not in source
    assert 'st.button(\n            f"准备批量 zip' in source.lower()
