import base64
import io
import json
import os
import re
import sqlite3
import threading
import tempfile
import time
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime
from uuid import uuid4

import anthropic
import fitz
import openpyxl
import pandas as pd
import streamlit as st
from openai import OpenAI
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

DEFAULT_FONT     = Path(__file__).parent / "font.ttf"
DEFAULT_GLOSSARY = Path(__file__).parent / "glossary.xlsx"
DB_PATH          = Path(__file__).parent / "pdf_project.db"
OPENAI_MODEL     = "gpt-5.6-terra"


# ── Local user/customer/glossary access control ────────────────────────────────

def _secret_or_env(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value:
        return value
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


def _int_secret_or_env(name: str, default: int) -> int:
    try:
        return int(_secret_or_env(name, str(default)))
    except (TypeError, ValueError):
        return default


def _database_url() -> str:
    return _secret_or_env("DATABASE_URL", "").strip()


OPENAI_API_KEY = _secret_or_env("OPENAI_API_KEY", "") or _secret_or_env("ANTHROPIC_API_KEY", "")
OPENAI_MODEL = _secret_or_env("OPENAI_MODEL", OPENAI_MODEL)
OPENAI_TIMEOUT_SECONDS = max(20, _int_secret_or_env("OPENAI_TIMEOUT_SECONDS", 45))
OPENAI_REASONING_EFFORT = _secret_or_env("OPENAI_REASONING_EFFORT", "low")
OPENAI_FALLBACK_MODELS = [
    model.strip()
    for model in _secret_or_env(
        "OPENAI_FALLBACK_MODELS",
        "gpt-5.6-luna,gpt-5.4-mini",
    ).split(",")
    if model.strip()
]

# Backward-compatible aliases so the rest of the file can be migrated gradually.
ANTHROPIC_MODEL = OPENAI_MODEL
ANTHROPIC_TIMEOUT_SECONDS = OPENAI_TIMEOUT_SECONDS
ANTHROPIC_FALLBACK_MODELS = OPENAI_FALLBACK_MODELS


def _model_not_found_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    text = str(exc).lower()
    return status_code == 404 and ("model" in text or "not_found" in text)


def _response_output_text(response) -> str:
    text = getattr(response, "output_text", "") or ""
    if text:
        return str(text).strip()
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", "") == "output_text":
                chunks.append(str(getattr(content, "text", "")))
    return "".join(chunks).strip()


def _openai_image_content(image_data: dict) -> dict:
    source = image_data.get("source") or {}
    if source.get("type") == "base64":
        media_type = source.get("media_type") or "image/png"
        data = source.get("data") or ""
        return {
            "type": "input_image",
            "image_url": f"data:{media_type};base64,{data}",
            "detail": "high",
        }
    return {
        "type": "input_image",
        "image_url": image_data.get("image_url") or "",
        "detail": image_data.get("detail") or "high",
    }


def _build_openai_input_messages(system: str, messages: list[dict] | None) -> list[dict]:
    input_messages: list[dict] = []
    if system:
        input_messages.append({
            "type": "message",
            "role": "system",
            "content": system,
        })
    for msg in messages or []:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, str):
            input_messages.append({
                "type": "message",
                "role": role,
                "content": content,
            })
            continue
        mapped_content = []
        for part in content:
            part_type = part.get("type")
            if part_type in {"text", "input_text"}:
                mapped_content.append({
                    "type": "input_text",
                    "text": str(part.get("text", "")),
                })
            elif part_type in {"image", "input_image"}:
                mapped_content.append(_openai_image_content(part))
        input_messages.append({
            "type": "message",
            "role": role,
            "content": mapped_content,
        })
    return input_messages


def _create_anthropic_message(client: OpenAI, **kwargs):
    requested_model = kwargs.pop("model", OPENAI_MODEL)
    timeout = kwargs.pop("timeout", OPENAI_TIMEOUT_SECONDS)
    max_output_tokens = kwargs.pop("max_tokens", kwargs.pop("max_output_tokens", None))
    kwargs.pop("temperature", None)
    system = kwargs.pop("system", "")
    messages = kwargs.pop("messages", None)
    text_config = kwargs.pop("text", None)
    reasoning = kwargs.pop("reasoning", None)
    models_to_try = list(dict.fromkeys([requested_model, *OPENAI_FALLBACK_MODELS]))
    input_messages = _build_openai_input_messages(system, messages)
    create_kwargs = {
        "input": input_messages,
        "timeout": timeout,
    }
    if max_output_tokens is not None:
        create_kwargs["max_output_tokens"] = max_output_tokens
    if text_config is not None:
        create_kwargs["text"] = text_config
    if reasoning is not None:
        create_kwargs["reasoning"] = reasoning
    else:
        create_kwargs["reasoning"] = {"effort": OPENAI_REASONING_EFFORT}
    for model in models_to_try:
        try:
            return client.responses.create(model=model, **create_kwargs)
        except Exception as exc:
            if _model_not_found_error(exc):
                continue
            raise
    raise RuntimeError(
        "所有配置的 OpenAI 模型都不可用。请检查 API key 所属账号可用模型，"
        "或在 Streamlit Secrets 里设置 OPENAI_MODEL。已尝试："
        + "；".join(models_to_try)
    )


def _use_postgres() -> bool:
    return bool(_database_url())


def _pg_sql(sql: str) -> str:
    converted = sql.replace("?", "%s")
    if "INSERT OR IGNORE INTO" in converted.upper():
        converted = re.sub(
            r"INSERT\s+OR\s+IGNORE\s+INTO",
            "INSERT INTO",
            converted,
            count=1,
            flags=re.IGNORECASE,
        ).rstrip()
        if not converted.upper().endswith("ON CONFLICT DO NOTHING"):
            converted += " ON CONFLICT DO NOTHING"
    return converted


class DbConnection:
    def __init__(self):
        self.is_postgres = _use_postgres()
        if self.is_postgres:
            try:
                import psycopg
                from psycopg.rows import dict_row
            except ImportError as exc:
                raise RuntimeError("缺少 Postgres 驱动，请确认 requirements.txt 已安装 psycopg[binary]") from exc
            url = _database_url()
            if url.startswith("postgres://"):
                url = "postgresql://" + url[len("postgres://"):]
            self.conn = psycopg.connect(url, row_factory=dict_row)
        else:
            self.conn = sqlite3.connect(DB_PATH)
            self.conn.row_factory = sqlite3.Row

    def __enter__(self):
        self.conn.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self.conn.__exit__(exc_type, exc, tb)

    def execute(self, sql: str, params=()):
        return self.conn.execute(_pg_sql(sql) if self.is_postgres else sql, params)

    def executemany(self, sql: str, seq_of_params):
        if not self.is_postgres:
            return self.conn.executemany(sql, seq_of_params)
        with self.conn.cursor() as cur:
            return cur.executemany(_pg_sql(sql), seq_of_params)

    def executescript(self, sql: str) -> None:
        if not self.is_postgres:
            self.conn.executescript(sql)
            return
        with self.conn.cursor() as cur:
            for statement in sql.split(";"):
                if statement.strip():
                    cur.execute(statement)


def _postgres_create_schema(conn: DbConnection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('company_admin', 'group_leader', 'staff')),
            group_name TEXT,
            assigned_customer_ids TEXT NOT NULL DEFAULT '[]'
        );

        CREATE TABLE IF NOT EXISTS customers (
            customer_id TEXT PRIMARY KEY,
            customer_name TEXT NOT NULL,
            customer_code TEXT NOT NULL UNIQUE,
            group_name TEXT NOT NULL,
            assigned_staff_username TEXT NOT NULL,
            note TEXT DEFAULT '',
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS glossary_terms (
            glossary_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            customer_id TEXT NOT NULL REFERENCES customers(customer_id),
            english_term TEXT NOT NULL,
            chinese_translation TEXT NOT NULL,
            normalized_key TEXT DEFAULT '',
            note TEXT DEFAULT '',
            created_by TEXT,
            updated_by TEXT,
            updated_at TEXT,
            status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'inactive'))
        );

        CREATE TABLE IF NOT EXISTS glossary_change_requests (
            request_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            customer_id TEXT NOT NULL REFERENCES customers(customer_id),
            action_type TEXT NOT NULL CHECK(action_type IN ('add', 'update', 'delete')),
            candidate_id BIGINT,
            english_term_old TEXT,
            chinese_translation_old TEXT,
            english_term_new TEXT,
            chinese_translation_new TEXT,
            note TEXT DEFAULT '',
            submitted_by TEXT NOT NULL,
            submitted_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'approved', 'rejected')),
            reviewed_by TEXT,
            reviewed_at TEXT,
            review_comment TEXT
        );

        CREATE TABLE IF NOT EXISTS term_candidates (
            candidate_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            customer_id TEXT NOT NULL REFERENCES customers(customer_id),
            source_file_name TEXT NOT NULL,
            source_type TEXT NOT NULL CHECK(source_type IN ('PDF', 'Excel')),
            page_or_sheet TEXT DEFAULT '',
            cell_coordinate TEXT DEFAULT '',
            original_term TEXT NOT NULL,
            normalized_term TEXT NOT NULL,
            variants TEXT DEFAULT '',
            ai_suggested_translation TEXT DEFAULT '',
            final_translation TEXT DEFAULT '',
            context_sentence TEXT DEFAULT '',
            frequency INTEGER NOT NULL DEFAULT 1,
            confidence TEXT DEFAULT 'medium',
            matched_by TEXT DEFAULT 'no_match',
            matched_glossary_term TEXT DEFAULT '',
            conflict_warning TEXT DEFAULT '',
            status TEXT NOT NULL DEFAULT 'draft' CHECK(status IN ('draft', 'selected', 'submitted', 'ignored', 'approved', 'rejected')),
            created_by TEXT,
            created_at TEXT,
            submitted_at TEXT,
            UNIQUE(customer_id, normalized_term, status)
        );

        CREATE TABLE IF NOT EXISTS translation_jobs (
            job_id TEXT PRIMARY KEY,
            job_type TEXT NOT NULL CHECK(job_type IN ('PDF', 'Excel')),
            status TEXT NOT NULL CHECK(status IN ('queued', 'running', 'complete', 'failed')),
            username TEXT NOT NULL,
            customer_id TEXT NOT NULL,
            source_file_name TEXT NOT NULL,
            progress DOUBLE PRECISION NOT NULL DEFAULT 0,
            message TEXT DEFAULT '',
            error TEXT DEFAULT '',
            input_bytes BYTEA NOT NULL,
            aux_bytes BYTEA,
            result_file BYTEA,
            result_report BYTEA,
            result_meta TEXT DEFAULT '{}',
            config TEXT DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    for sql in [
        "ALTER TABLE customers ADD COLUMN IF NOT EXISTS note TEXT DEFAULT ''",
        "ALTER TABLE customers ADD COLUMN IF NOT EXISTS created_at TEXT",
        "ALTER TABLE glossary_terms ADD COLUMN IF NOT EXISTS normalized_key TEXT DEFAULT ''",
        "ALTER TABLE glossary_change_requests ADD COLUMN IF NOT EXISTS candidate_id BIGINT",
        "ALTER TABLE term_candidates ADD COLUMN IF NOT EXISTS variants TEXT DEFAULT ''",
        "ALTER TABLE term_candidates ADD COLUMN IF NOT EXISTS matched_by TEXT DEFAULT 'no_match'",
        "ALTER TABLE term_candidates ADD COLUMN IF NOT EXISTS matched_glossary_term TEXT DEFAULT ''",
        "ALTER TABLE term_candidates ADD COLUMN IF NOT EXISTS conflict_warning TEXT DEFAULT ''",
        "ALTER TABLE translation_jobs ADD COLUMN IF NOT EXISTS aux_bytes BYTEA",
        "ALTER TABLE translation_jobs ADD COLUMN IF NOT EXISTS result_meta TEXT DEFAULT '{}'",
        "ALTER TABLE translation_jobs ADD COLUMN IF NOT EXISTS config TEXT DEFAULT '{}'",
    ]:
        conn.execute(sql)

def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def hash_password(password: str) -> str:
    # Basic local-only password hashing for the first SQLite version.
    import hashlib
    return hashlib.sha256(f"pdf-project::{password}".encode("utf-8")).hexdigest()


def get_db_connection() -> DbConnection:
    return DbConnection()


def init_db() -> None:
    with get_db_connection() as conn:
        if conn.is_postgres:
            _postgres_create_schema(conn)
            rows = conn.execute(
                """
                SELECT glossary_id, english_term
                FROM glossary_terms
                WHERE COALESCE(normalized_key, '') = ''
                """
            ).fetchall()
            for row in rows:
                conn.execute(
                    """
                    UPDATE glossary_terms
                    SET normalized_key = ?
                    WHERE glossary_id = ?
                    """,
                    (normalize_term_key(row["english_term"]), row["glossary_id"]),
                )
            rows = conn.execute(
                """
                SELECT candidate_id, original_term
                FROM term_candidates
                WHERE COALESCE(variants, '') = ''
                """
            ).fetchall()
            for row in rows:
                conn.execute(
                    """
                    UPDATE term_candidates
                    SET variants = ?
                    WHERE candidate_id = ?
                    """,
                    (json.dumps([row["original_term"]], ensure_ascii=False), row["candidate_id"]),
                )
            return

        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('company_admin', 'group_leader', 'staff')),
                group_name TEXT,
                assigned_customer_ids TEXT NOT NULL DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS customers (
                customer_id TEXT PRIMARY KEY,
                customer_name TEXT NOT NULL,
                customer_code TEXT NOT NULL UNIQUE,
                group_name TEXT NOT NULL,
                assigned_staff_username TEXT NOT NULL,
                note TEXT DEFAULT '',
                created_at TEXT
            );

            CREATE TABLE IF NOT EXISTS glossary_terms (
                glossary_id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT NOT NULL,
                english_term TEXT NOT NULL,
                chinese_translation TEXT NOT NULL,
                normalized_key TEXT DEFAULT '',
                note TEXT DEFAULT '',
                created_by TEXT,
                updated_by TEXT,
                updated_at TEXT,
                status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'inactive')),
                FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
            );

            CREATE TABLE IF NOT EXISTS glossary_change_requests (
                request_id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT NOT NULL,
                action_type TEXT NOT NULL CHECK(action_type IN ('add', 'update', 'delete')),
                candidate_id INTEGER,
                english_term_old TEXT,
                chinese_translation_old TEXT,
                english_term_new TEXT,
                chinese_translation_new TEXT,
                note TEXT DEFAULT '',
                submitted_by TEXT NOT NULL,
                submitted_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'approved', 'rejected')),
                reviewed_by TEXT,
                reviewed_at TEXT,
                review_comment TEXT,
                FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
            );

            CREATE TABLE IF NOT EXISTS term_candidates (
                candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT NOT NULL,
                source_file_name TEXT NOT NULL,
                source_type TEXT NOT NULL CHECK(source_type IN ('PDF', 'Excel')),
                page_or_sheet TEXT DEFAULT '',
                cell_coordinate TEXT DEFAULT '',
                original_term TEXT NOT NULL,
                normalized_term TEXT NOT NULL,
                variants TEXT DEFAULT '',
                ai_suggested_translation TEXT DEFAULT '',
                final_translation TEXT DEFAULT '',
                context_sentence TEXT DEFAULT '',
                frequency INTEGER NOT NULL DEFAULT 1,
                confidence TEXT DEFAULT 'medium',
                matched_by TEXT DEFAULT 'no_match',
                matched_glossary_term TEXT DEFAULT '',
                conflict_warning TEXT DEFAULT '',
                status TEXT NOT NULL DEFAULT 'draft' CHECK(status IN ('draft', 'selected', 'submitted', 'ignored', 'approved', 'rejected')),
                created_by TEXT,
                created_at TEXT,
                submitted_at TEXT,
                UNIQUE(customer_id, normalized_term, status)
            );

            CREATE TABLE IF NOT EXISTS translation_jobs (
                job_id TEXT PRIMARY KEY,
                job_type TEXT NOT NULL CHECK(job_type IN ('PDF', 'Excel')),
                status TEXT NOT NULL CHECK(status IN ('queued', 'running', 'complete', 'failed')),
                username TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                source_file_name TEXT NOT NULL,
                progress REAL NOT NULL DEFAULT 0,
                message TEXT DEFAULT '',
                error TEXT DEFAULT '',
                input_bytes BLOB NOT NULL,
                aux_bytes BLOB,
                result_file BLOB,
                result_report BLOB,
                result_meta TEXT DEFAULT '{}',
                config TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        existing_cols = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(customers)").fetchall()
        }
        if "note" not in existing_cols:
            try:
                conn.execute("ALTER TABLE customers ADD COLUMN note TEXT DEFAULT ''")
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise
        if "created_at" not in existing_cols:
            try:
                conn.execute("ALTER TABLE customers ADD COLUMN created_at TEXT")
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise
        glossary_cols = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(glossary_terms)").fetchall()
        }
        if "normalized_key" not in glossary_cols:
            try:
                conn.execute("ALTER TABLE glossary_terms ADD COLUMN normalized_key TEXT DEFAULT ''")
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise
        request_cols = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(glossary_change_requests)").fetchall()
        }
        if "candidate_id" not in request_cols:
            try:
                conn.execute("ALTER TABLE glossary_change_requests ADD COLUMN candidate_id INTEGER")
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise
        candidate_cols = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(term_candidates)").fetchall()
        }
        if "variants" not in candidate_cols:
            try:
                conn.execute("ALTER TABLE term_candidates ADD COLUMN variants TEXT DEFAULT ''")
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise
        for col_name, col_sql in [
            ("matched_by", "ALTER TABLE term_candidates ADD COLUMN matched_by TEXT DEFAULT 'no_match'"),
            ("matched_glossary_term", "ALTER TABLE term_candidates ADD COLUMN matched_glossary_term TEXT DEFAULT ''"),
            ("conflict_warning", "ALTER TABLE term_candidates ADD COLUMN conflict_warning TEXT DEFAULT ''"),
        ]:
            if col_name not in candidate_cols:
                try:
                    conn.execute(col_sql)
                except sqlite3.OperationalError as exc:
                    if "duplicate column name" not in str(exc).lower():
                        raise
        job_cols = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(translation_jobs)").fetchall()
        }
        for col_name, col_sql in [
            ("aux_bytes", "ALTER TABLE translation_jobs ADD COLUMN aux_bytes BLOB"),
            ("result_meta", "ALTER TABLE translation_jobs ADD COLUMN result_meta TEXT DEFAULT '{}'"),
            ("config", "ALTER TABLE translation_jobs ADD COLUMN config TEXT DEFAULT '{}'"),
        ]:
            if col_name not in job_cols:
                try:
                    conn.execute(col_sql)
                except sqlite3.OperationalError as exc:
                    if "duplicate column name" not in str(exc).lower():
                        raise
        rows = conn.execute(
            """
            SELECT glossary_id, english_term
            FROM glossary_terms
            WHERE COALESCE(normalized_key, '') = ''
            """
        ).fetchall()
        for row in rows:
            conn.execute(
                """
                UPDATE glossary_terms
                SET normalized_key = ?
                WHERE glossary_id = ?
                """,
                (normalize_term_key(row["english_term"]), row["glossary_id"]),
            )
        rows = conn.execute(
            """
            SELECT candidate_id, original_term
            FROM term_candidates
            WHERE COALESCE(variants, '') = ''
            """
        ).fetchall()
        for row in rows:
            conn.execute(
                """
                UPDATE term_candidates
                SET variants = ?
                WHERE candidate_id = ?
                """,
                (json.dumps([row["original_term"]], ensure_ascii=False), row["candidate_id"]),
            )


@st.cache_resource(show_spinner=False)
def run_startup_tasks_once(use_postgres_marker: bool, schema_version: int) -> bool:
    init_db()
    seed_demo_data_if_empty()
    sync_staff_customer_assignments()
    return True


def seed_demo_data_if_empty() -> None:
    with get_db_connection() as conn:
        exists = conn.execute("SELECT 1 FROM users LIMIT 1").fetchone()
        if exists:
            return

        users = [
            ("u_admin", "admin", "admin123", "company_admin", "", []),
            ("u_leader_a", "leader_a", "leader123", "group_leader", "A组", []),
            ("u_staff_1", "staff_1", "staff123", "staff", "A组", ["CUST001"]),
            ("u_staff_2", "staff_2", "staff123", "staff", "A组", ["CUST002"]),
            ("u_staff_3", "staff_3", "staff123", "staff", "B组", ["CUST003"]),
        ]
        conn.executemany(
            """
            INSERT INTO users
                (user_id, username, password_hash, role, group_name, assigned_customer_ids)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (uid, username, hash_password(password), role, group, json.dumps(assigned, ensure_ascii=False))
                for uid, username, password, role, group, assigned in users
            ],
        )

        customers = [
            ("CUST001", "客户A", "CUST001", "A组", "staff_1"),
            ("CUST002", "客户B", "CUST002", "A组", "staff_2"),
            ("CUST003", "客户C", "CUST003", "B组", "staff_3"),
        ]
        conn.executemany(
            """
            INSERT INTO customers
                (customer_id, customer_name, customer_code, group_name, assigned_staff_username)
            VALUES (?, ?, ?, ?, ?)
            """,
            customers,
        )

        terms = [
            ("CUST001", "gusset", "裆布", "客户A偏好：裆布", "admin"),
            ("CUST001", "underwire", "钢圈", "", "admin"),
            ("CUST001", "strap", "肩带", "", "admin"),
            ("CUST002", "gusset", "裆衬", "客户B偏好：裆衬", "admin"),
            ("CUST002", "underwire", "托圈", "", "admin"),
            ("CUST002", "strap", "带仔", "", "admin"),
            ("CUST003", "gusset", "裆里", "客户C偏好：裆里", "admin"),
            ("CUST003", "underwire", "钢托", "", "admin"),
            ("CUST003", "strap", "肩袢", "", "admin"),
        ]
        conn.executemany(
            """
            INSERT INTO glossary_terms
                (customer_id, english_term, chinese_translation, normalized_key, note,
                 created_by, updated_by, updated_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active')
            """,
            [
                (cid, en, zh, normalize_term_key(en), note, by, by, _now_iso())
                for cid, en, zh, note, by in terms
            ],
        )


def _user_from_row(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    user = dict(row)
    try:
        user["assigned_customer_ids"] = json.loads(user.get("assigned_customer_ids") or "[]")
    except json.JSONDecodeError:
        user["assigned_customer_ids"] = []
    return user


def authenticate_user(username: str, password: str) -> dict | None:
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ? AND password_hash = ?",
            (username.strip(), hash_password(password)),
        ).fetchone()
    return _user_from_row(row)


def get_customer(customer_id: str) -> dict | None:
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM customers WHERE customer_id = ?", (customer_id,)).fetchone()
    return dict(row) if row else None


def list_users() -> list[dict]:
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT user_id, username, role, group_name, assigned_customer_ids
            FROM users
            ORDER BY username
            """
        ).fetchall()
    return [_user_from_row(r) for r in rows]


def list_customers() -> list[dict]:
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT customer_id, customer_name, customer_code, group_name,
                   assigned_staff_username, note, created_at
            FROM customers
            ORDER BY customer_code
            """
        ).fetchall()
    return [dict(r) for r in rows]


def list_customers_with_term_counts(user: dict) -> list[dict]:
    customers = get_accessible_customers(user)
    if not customers:
        return []
    customer_ids = [c["customer_id"] for c in customers]
    placeholders = ",".join("?" for _ in customer_ids)
    with get_db_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT customer_id, COUNT(*) AS term_count
            FROM glossary_terms
            WHERE status = 'active' AND customer_id IN ({placeholders})
            GROUP BY customer_id
            """,
            customer_ids,
        ).fetchall()
    counts = {r["customer_id"]: r["term_count"] for r in rows}
    return [{**c, "term_count": counts.get(c["customer_id"], 0)} for c in customers]


def get_staff_usernames() -> list[str]:
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT username FROM users WHERE role = 'staff' ORDER BY username"
        ).fetchall()
    return [r["username"] for r in rows]


def sync_staff_customer_assignments() -> None:
    with get_db_connection() as conn:
        staff_rows = conn.execute(
            "SELECT username FROM users WHERE role = 'staff' ORDER BY username"
        ).fetchall()
        for staff in staff_rows:
            customer_rows = conn.execute(
                """
                SELECT customer_id
                FROM customers
                WHERE assigned_staff_username = ?
                ORDER BY customer_code
                """,
                (staff["username"],),
            ).fetchall()
            assigned = [r["customer_id"] for r in customer_rows]
            conn.execute(
                """
                UPDATE users
                SET assigned_customer_ids = ?
                WHERE username = ?
                """,
                (json.dumps(assigned, ensure_ascii=False), staff["username"]),
            )


def create_user(
    username: str,
    password: str,
    role: str,
    group_name: str = "",
) -> None:
    username = username.strip()
    if not username or not password:
        raise ValueError("用户名和密码不能为空")
    if role not in {"company_admin", "group_leader", "staff"}:
        raise ValueError("角色不正确")
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO users (user_id, username, password_hash, role, group_name, assigned_customer_ids)
            VALUES (?, ?, ?, ?, ?, '[]')
            """,
            (f"u_{username}", username, hash_password(password), role, group_name.strip()),
        )
    sync_staff_customer_assignments()


def update_user(
    username: str,
    role: str,
    group_name: str = "",
    new_password: str = "",
) -> None:
    if role not in {"company_admin", "group_leader", "staff"}:
        raise ValueError("角色不正确")
    with get_db_connection() as conn:
        if new_password.strip():
            conn.execute(
                """
                UPDATE users
                SET password_hash = ?, role = ?, group_name = ?
                WHERE username = ?
                """,
                (hash_password(new_password), role, group_name.strip(), username),
            )
        else:
            conn.execute(
                """
                UPDATE users
                SET role = ?, group_name = ?
                WHERE username = ?
                """,
                (role, group_name.strip(), username),
            )
    sync_staff_customer_assignments()


def create_customer(
    customer_id: str,
    customer_name: str,
    customer_code: str,
    group_name: str,
    assigned_staff_username: str,
) -> None:
    values = [customer_id.strip(), customer_name.strip(), customer_code.strip(),
              group_name.strip(), assigned_staff_username.strip()]
    if not all(values):
        raise ValueError("客户 ID、客户名称、客户代码、小组、负责人都不能为空")
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO customers (
                customer_id, customer_name, customer_code, group_name, assigned_staff_username
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            tuple(values),
        )
    sync_staff_customer_assignments()


def create_customer_auto_id(
    customer_name: str,
    customer_code: str,
    group_name: str,
    assigned_staff_username: str,
    note: str = "",
) -> str:
    customer_name = customer_name.strip()
    customer_code = customer_code.strip()
    group_name = group_name.strip()
    assigned_staff_username = assigned_staff_username.strip()
    note = note.strip()
    if not customer_name:
        raise ValueError("客户名称不能为空")
    if not customer_code:
        raise ValueError("客户代码不能为空")
    if not group_name:
        raise ValueError("所属小组不能为空")
    if not assigned_staff_username:
        raise ValueError("负责人不能为空")

    with get_db_connection() as conn:
        staff = conn.execute(
            "SELECT 1 FROM users WHERE username = ? AND role = 'staff'",
            (assigned_staff_username,),
        ).fetchone()
        if staff is None:
            raise ValueError("负责人必须是已有 staff 用户")
        duplicate = conn.execute(
            "SELECT 1 FROM customers WHERE lower(customer_code) = lower(?)",
            (customer_code,),
        ).fetchone()
        if duplicate:
            raise ValueError("客户代码已存在，不能重复创建")

        customer_id = customer_code
        id_duplicate = conn.execute(
            "SELECT 1 FROM customers WHERE customer_id = ?",
            (customer_id,),
        ).fetchone()
        if id_duplicate:
            raise ValueError("客户 ID 已存在，请更换客户代码")

        conn.execute(
            """
            INSERT INTO customers (
                customer_id, customer_name, customer_code, group_name,
                assigned_staff_username, note, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                customer_id,
                customer_name,
                customer_code,
                group_name,
                assigned_staff_username,
                note,
                _now_iso(),
            ),
        )
    sync_staff_customer_assignments()
    return customer_id


def update_customer(
    customer_id: str,
    customer_name: str,
    customer_code: str,
    group_name: str,
    assigned_staff_username: str,
) -> None:
    values = [customer_name.strip(), customer_code.strip(), group_name.strip(),
              assigned_staff_username.strip(), customer_id]
    if not all(values):
        raise ValueError("客户名称、客户代码、小组、负责人都不能为空")
    with get_db_connection() as conn:
        conn.execute(
            """
            UPDATE customers
            SET customer_name = ?, customer_code = ?, group_name = ?, assigned_staff_username = ?
            WHERE customer_id = ?
            """,
            tuple(values),
        )
    sync_staff_customer_assignments()


def get_accessible_customers(user: dict) -> list[dict]:
    if not user:
        return []
    with get_db_connection() as conn:
        if user["role"] == "company_admin":
            rows = conn.execute(
                "SELECT * FROM customers ORDER BY customer_code"
            ).fetchall()
        elif user["role"] == "group_leader":
            rows = conn.execute(
                "SELECT * FROM customers WHERE group_name = ? ORDER BY customer_code",
                (user.get("group_name") or "",),
            ).fetchall()
        else:
            assigned = user.get("assigned_customer_ids") or []
            if not assigned:
                return []
            placeholders = ",".join("?" for _ in assigned)
            rows = conn.execute(
                f"SELECT * FROM customers WHERE customer_id IN ({placeholders}) ORDER BY customer_code",
                assigned,
            ).fetchall()
    return [dict(r) for r in rows]


def can_view_customer_glossary(user: dict, customer_id: str) -> bool:
    if not user or not customer_id:
        return False
    if user["role"] == "company_admin":
        return True
    if user["role"] == "staff":
        return customer_id in (user.get("assigned_customer_ids") or [])
    customer = get_customer(customer_id)
    return bool(customer and customer["group_name"] == user.get("group_name"))


def can_use_customer_glossary(user: dict, customer_id: str) -> bool:
    return can_view_customer_glossary(user, customer_id)


def can_submit_glossary_change(user: dict, customer_id: str) -> bool:
    return can_view_customer_glossary(user, customer_id)


def can_approve_glossary_change(user: dict) -> bool:
    return bool(user and user.get("role") == "company_admin")


def get_customer_glossary_df(customer_id: str) -> pd.DataFrame:
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT english_term, chinese_translation, note, normalized_key
            FROM glossary_terms
            WHERE customer_id = ? AND status = 'active'
            ORDER BY lower(english_term)
            """,
            (customer_id,),
        ).fetchall()
    return pd.DataFrame(
        [
            {
                _GLOSSARY_EN_COL: r["english_term"],
                _GLOSSARY_ZH_COL: r["chinese_translation"],
                _GLOSSARY_NOTE_COL: r["note"] or "",
                _GLOSSARY_CAT_COL: customer_id,
            }
            for r in rows
        ],
        columns=_GLOSSARY_EDIT_COLS,
    )


def get_customer_glossary_bytes_for_translation(user: dict, customer_id: str) -> bytes:
    if not can_use_customer_glossary(user, customer_id):
        raise PermissionError("当前用户无权使用该客户术语库")
    return glossary_df_to_xlsx_bytes(get_customer_glossary_df(customer_id))


def get_active_glossary_terms(customer_id: str) -> list[dict]:
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT glossary_id, customer_id, english_term, chinese_translation, note,
                   normalized_key, created_by, updated_by, updated_at, status
            FROM glossary_terms
            WHERE customer_id = ? AND status = 'active'
            ORDER BY lower(english_term)
            """,
            (customer_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_active_glossary_term(customer_id: str, english_term: str) -> dict | None:
    with get_db_connection() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM glossary_terms
            WHERE customer_id = ? AND lower(english_term) = lower(?) AND status = 'active'
            ORDER BY glossary_id DESC
            LIMIT 1
            """,
            (customer_id, english_term.strip()),
        ).fetchone()
    return dict(row) if row else None


_TERM_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
    "it", "of", "on", "or", "the", "to", "with", "without", "this", "that",
    "option", "season", "style", "supplier", "owner", "collection", "division",
    "description", "comments", "instructions", "page", "date", "plan", "type",
    "designer", "design", "designed", "designs", "fashion", "trend", "trends",
    "sample", "samples", "layout", "artwork", "diagram", "drawing", "view",
    "product", "products", "item", "items", "look", "looks", "brand",
}
_KEY_STOPWORDS = {"of", "the", "a", "an", "for", "to"}
_AMBIGUOUS_DIRECTION_WORDS = {
    "front", "back", "left", "right", "upper", "lower", "inner", "outer",
    "inside", "outside", "top", "bottom",
}
_BRAND_LIKE_TERMS = {"cotton juice", "cotton juice baby", "uneco", "marca"}
_RE_DATE_LIKE = re.compile(r"^(?:\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?|\d{4}[./-]\d{1,2}[./-]\d{1,2})$")
_RE_STYLE_LIKE = re.compile(r"^[A-Z]{1,4}\d{3,}[A-Z0-9\-/ ]*$")
_RE_PANTONE_LIKE = re.compile(
    r"^(?:pantone\s*)?(?:\d{2}-\d{4}|\d{2}\s+\d{4}|\d{4,6})(?:\s*(?:tcx|tpx|tc|tp|c|u|cp|up))?$",
    re.IGNORECASE,
)
_RE_COLOR_CODE_FRAGMENT = re.compile(
    r"\b(?:pantone|pms)?\s*(?:\d{1,2}[-\s]\d{3,4}|\d{4,6})\s*(?:tcx|tpx|tc|tp|c|u|cp|up)?\b",
    re.IGNORECASE,
)
_RE_FILE_PATH_LIKE = re.compile(r"(^/|[A-Za-z]:\\|[/\\][^/\\]+\.(?:pdf|xlsx?)$|\.(?:pdf|xlsx?)$)", re.IGNORECASE)
_RE_TERM_TOKEN = re.compile(r"[A-Za-z][A-Za-z&'’.\-]*")
_RE_COLOR_CODE_NAME = re.compile(
    r"^(?:pantone\s*)?(?:\d{1,2}[-\s]\d{3,4}|\d{4,6})(?:\s*(?:tcx|tpx|tc|tp|c|u|cp|up))?(?:\s+[A-Za-z][A-Za-z&'’.\-]*){0,4}$",
    re.IGNORECASE,
)
_RE_COLOR_CONTEXT = re.compile(
    r"(?:color|colour|pantone|pms|swatch|shade|tone|colorway|colourway|颜色|色号|色卡|色样|色板|色名|色调)",
    re.IGNORECASE,
)
_COLOR_HINT_WORDS = {
    "color", "colour", "pantone", "swatch", "shade", "tone", "colourway", "colour way",
    "colour ways", "colourways", "pms", "colorway", "colorways", "palette",
    "色号", "色卡", "色样", "颜色", "色板", "色调",
}
_COMMON_COLOR_WORDS = {
    "black", "white", "red", "blue", "green", "yellow", "orange", "pink",
    "purple", "brown", "grey", "gray", "beige", "ivory", "navy", "rose",
    "burgundy", "maroon", "wine", "teal", "turquoise", "emerald", "violet",
    "lavender", "lilac", "magenta", "fuchsia", "coral", "mint", "plum",
    "khaki", "olive", "camel", "sand", "nude", "gold", "silver", "bronze",
    "copper", "charcoal", "slate", "sapphire", "aqua", "cream", "chocolate",
    "coffee", "denim", "mustard", "ochre", "apricot", "ruby", "crimson",
}
_NON_WORKMANSHIP_TERM_WORDS = {
    "designer", "design", "designed", "designs", "fashion", "trend", "trends",
    "sample", "samples", "layout", "artwork", "diagram", "drawing", "view",
    "product", "products", "item", "items", "look", "looks", "brand", "season",
    "collection", "collections", "supplier", "owner", "comments", "instructions",
    "description", "page", "date", "type", "plan",
}
_WORKMANSHIP_SIGNAL_WORDS = {
    "seam", "seams", "sewing", "stitch", "stitching", "stitches", "binding",
    "hem", "hems", "cuff", "cuffs", "collar", "collars", "sleeve", "sleeves",
    "neckline", "armhole", "waist", "waistband", "gusset", "panel", "panels",
    "placket", "yoke", "lining", "interlining", "padding", "pad", "elastic",
    "elasticated", "strap", "straps", "bra", "cup", "cups", "hook", "eye",
    "zipper", "zip", "snap", "snaps", "closure", "fabric", "shell", "thread",
    "woven", "knit", "knitted", "knitting", "print", "printing", "printed",
    "embroider", "embroidered", "embroidery", "lace", "mesh", "foam", "bonded",
    "fusing", "fused", "overlock", "interlock", "underband", "support", "elasticity",
    "polyamide", "polyester", "nylon", "elastane", "spandex", "cotton", "modal",
    "viscose", "rayon", "lyocell", "acrylic", "wool", "silk", "bamboo",
}


def normalize_term(term: str) -> str:
    return re.sub(r"\s+", " ", str(term or "").strip()).lower()


def normalize_term_key(text: str) -> str:
    t = str(text or "").strip().lower()
    t = re.sub(r"[-_/,.:;()]+", " ", t)
    t = re.sub(r"\s+", " ", t)
    words = [w for w in re.findall(r"[a-z]+", t) if w not in _KEY_STOPWORDS]
    if len(words) == 1:
        return words[0]
    if not 2 <= len(words) <= 4:
        return ""
    return "|".join(sorted(words))


def normalized_key_low_confidence(text: str) -> bool:
    words = set(re.findall(r"[a-z]+", str(text or "").lower()))
    return bool(words & _AMBIGUOUS_DIRECTION_WORDS)


def _is_color_item(text: str) -> bool:
    t = re.sub(r"\s+", " ", str(text or "").strip())
    if not t:
        return False
    tl = t.lower()
    if _RE_COLOR_CONTEXT.search(t):
        return True
    if _RE_PANTONE_LIKE.match(t) or _RE_COLOR_CODE_FRAGMENT.search(t):
        return True
    if _RE_COLOR_CODE_NAME.match(t):
        return True
    tokens = [tok.lower() for tok in re.findall(r"[A-Za-z]+", t)]
    if len(tokens) <= 4 and any(tok in _COMMON_COLOR_WORDS for tok in tokens):
        return True
    if re.search(r"\bpantone\b", tl) and re.search(r"[A-Za-z]", t):
        return True
    return False


def _should_preserve_color_text(en: str, zh: str = "") -> bool:
    combined = " ".join(
        part for part in [
            re.sub(r"\s+", " ", str(en or "").strip()),
            re.sub(r"\s+", " ", str(zh or "").strip()),
        ]
        if part
    )
    if not combined:
        return False
    if _is_color_item(en) or _is_color_item(zh) or _is_color_item(combined):
        return True
    if re.search(r"\b(?:tcx|tpx|tc|tp|pantone|pms)\b", combined, re.IGNORECASE) and re.search(r"\d", combined):
        return True
    if re.search(r"色|彩|紅|红|蓝|綠|绿|黑|白|灰|紫|棕|金|银|銀|橙|粉|青|褐|葡萄|酒|木", str(zh or "")):
        en_tokens = re.findall(r"[A-Za-z]+", str(en or ""))
        if len(en_tokens) <= 4 or _RE_COLOR_CODE_FRAGMENT.search(str(en or "")):
            return True
    return False


def _is_color_chart_item(item: dict) -> bool:
    en = re.sub(r"\s+", " ", str(item.get("en", "")).strip())
    zh = re.sub(r"\s+", " ", str(item.get("zh", "")).strip())
    combined = " ".join(part for part in [en, zh] if part).strip()
    if not combined:
        return False
    if _should_preserve_color_text(en, zh):
        return True
    if _RE_PANTONE_LIKE.match(en) or _RE_PANTONE_LIKE.match(zh):
        return True
    if re.search(r"\b(tcx|tpx|tc|tp)\b", combined, re.IGNORECASE) and re.search(r"\d", combined):
        return True
    if len(re.findall(r"[A-Za-z]+", en)) <= 4 and len(re.findall(r"[一-鿿]", zh)) >= 2:
        # OCR sometimes returns only the code in `en` and the translated color name in `zh`.
        if re.search(r"色|彩|紅|红|蓝|綠|绿|黑|白|灰|紫|棕|金|银|銀|橙|粉|青|褐|葡萄|酒|木", zh):
            return True
    return False


def _should_skip_color_image_items(items: list[dict]) -> bool:
    if len(items) < 2:
        return False
    color_like = sum(1 for item in items if _is_color_chart_item(item))
    return color_like >= 2 and (color_like / len(items)) >= 0.6


def _is_workmanship_candidate_term(term: str, context: str = "") -> bool:
    t = re.sub(r"\s+", " ", str(term or "").strip())
    if not t or is_noise_term(t) or _is_color_item(t):
        return False
    tokens = [tok.lower() for tok in re.findall(r"[A-Za-z]+", t)]
    if not tokens:
        return False
    if any(tok in _NON_WORKMANSHIP_TERM_WORDS for tok in tokens):
        return False
    if context:
        ctx_tokens = {tok.lower() for tok in re.findall(r"[A-Za-z]+", str(context))}
        if ctx_tokens & _NON_WORKMANSHIP_TERM_WORDS and not (ctx_tokens & _WORKMANSHIP_SIGNAL_WORDS):
            return False
    if len(tokens) == 1:
        return tokens[0] in _WORKMANSHIP_SIGNAL_WORDS
    if any(tok in _WORKMANSHIP_SIGNAL_WORDS for tok in tokens):
        return True
    lowered = t.lower()
    return bool(re.search(
        r"(seam|stitch|bind|hem|cuff|collar|sleeve|neckline|armhole|waist|gusset|placket|yoke|lining|elastic|strap|zip|snap|hook|eye|fabric|thread|print|embroider|lace|mesh|foam|poly|cotton|nylon|modal|viscose|rayon|lyocell|spandex|elastane)",
        lowered,
    ))


def glossary_conflicts_by_key(customer_id: str, normalized_key: str) -> list[dict]:
    if not normalized_key:
        return []
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT glossary_id, english_term, chinese_translation
            FROM glossary_terms
            WHERE customer_id = ? AND normalized_key = ? AND status = 'active'
            ORDER BY glossary_id
            """,
            (customer_id, normalized_key),
        ).fetchall()
    return [dict(r) for r in rows]


def glossary_key_has_translation_conflict(rows: list[dict], new_translation: str = "") -> bool:
    translations = {
        normalize_term(r.get("chinese_translation", ""))
        for r in rows
        if normalize_term(r.get("chinese_translation", ""))
    }
    if new_translation:
        translations.add(normalize_term(new_translation))
    return len(translations) > 1


def glossary_conflict_keys_from_dict(glossary: dict) -> set[str]:
    grouped: dict[str, set[str]] = {}
    for en, zh in glossary.items():
        key = normalize_term_key(en)
        if key:
            grouped.setdefault(key, set()).add(normalize_term(zh))
    return {key for key, translations in grouped.items() if len(translations) > 1}


def is_noise_term(term: str) -> bool:
    t = re.sub(r"\s+", " ", str(term or "").strip())
    if not t:
        return True
    nl = t.lower()
    alpha = re.findall(r"[A-Za-z]", t)
    if not alpha:
        return True
    if nl in _TERM_STOPWORDS or nl in _BRAND_LIKE_TERMS:
        return True
    if _is_color_item(t):
        return True
    if len(t) <= 1:
        return True
    if _RE_FILE_PATH_LIKE.search(t):
        return True
    if _RE_DATE_LIKE.match(t) or _RE_STYLE_LIKE.match(t) or _RE_PANTONE_LIKE.match(t):
        return True
    if _is_passthrough_eligible(t):
        return True
    tokens = re.findall(r"[A-Za-z]+", t)
    if tokens and all(tok.lower() in _TERM_STOPWORDS for tok in tokens):
        return True
    if len(tokens) == 1 and len(tokens[0]) <= 2:
        return True
    return False


def extract_candidate_terms_from_text(text: str, glossary: dict) -> list[str]:
    """Rule-based fallback candidate extraction from already-extracted text.
    It favors short garment/process/material phrases and filters obvious codes.
    """
    cleaned = _clean_extracted_text(text)
    if not cleaned:
        return []
    conflict_keys = glossary_conflict_keys_from_dict(glossary)
    conflict_terms_by_key: dict[str, str] = {}
    for en in glossary:
        k = normalize_term_key(en)
        if k in conflict_keys:
            conflict_terms_by_key.setdefault(k, "")
            conflict_terms_by_key[k] = (
                f"{conflict_terms_by_key[k]}; {en}" if conflict_terms_by_key[k] else en
            )
    active_keys = {
        normalize_term_key(k) or normalize_term(k)
        for k in glossary
        if (normalize_term_key(k) or normalize_term(k)) not in conflict_keys
    }
    terms: list[str] = []

    # Split on separators commonly used in tech packs, then keep compact phrases.
    parts = re.split(r"[\n\r;,，。:：|()（）\[\]{}]+", cleaned)
    for part in parts:
        phrase = re.sub(r"\s+", " ", part).strip(" -_./")
        if not phrase or not _is_workmanship_candidate_term(phrase, cleaned):
            continue
        words = _RE_TERM_TOKEN.findall(phrase)
        if not words:
            continue
        for n in range(min(4, len(words)), 0, -1):
            for i in range(0, len(words) - n + 1):
                cand = " ".join(words[i:i + n]).strip()
                key = normalize_term_key(cand) or normalize_term(cand)
                if key in active_keys or not _is_workmanship_candidate_term(cand, phrase):
                    continue
                if n == 1 and len(cand) < 4:
                    continue
                terms.append(cand)
            if len(words) <= 4:
                break

    return list(dict.fromkeys(terms))[:12]


def suggest_term_candidate_translations(
    client: anthropic.Anthropic,
    candidate_contexts: list[dict],
    glossary: dict,
) -> dict[str, dict]:
    if not candidate_contexts:
        return {}
    rel = relevant_glossary(
        "\n".join(f"{c['term']}\n{c.get('context', '')}" for c in candidate_contexts),
        glossary,
    )
    gloss_block = ""
    if rel:
        lines = "\n".join(f"  {k} → {v}" for k, v in rel.items())
        gloss_block = f"当前客户术语库参考（必须尊重既有译法）：\n{lines}\n\n"
    items = "\n".join(
        json.dumps(
            {
                "term": c["term"],
                "context": c.get("context", ""),
                "source_type": c.get("source_type", ""),
            },
            ensure_ascii=False,
        )
        for c in candidate_contexts[:60]
    )
    prompt = (
        f"{gloss_block}"
        "请为以下服装工艺术语候选给出适合加入客户术语库的中文建议译法。"
        "结合上下文判断，不要只逐词直译；如果不确定，confidence 填 low。\n\n"
        "<term_candidates>\n"
        f"{items}\n"
        "</term_candidates>\n\n"
        "只返回 JSON，格式："
        '{"items":[{"term":"英文术语","translation":"中文建议","confidence":"high|medium|low"}]}'
    )
    msg = _create_anthropic_message(
        client,
        model=ANTHROPIC_MODEL,
        max_tokens=3072,
        temperature=0,
        system=_GARMENT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = _response_output_text(msg)
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        raw = m.group()
    data = json.loads(raw)
    out = {}
    for item in data.get("items", []):
        term = str(item.get("term", "")).strip()
        if not term:
            continue
        out[normalize_term_key(term) or normalize_term(term)] = {
            "translation": str(item.get("translation", "")).strip(),
            "confidence": str(item.get("confidence", "medium")).strip() or "medium",
        }
    return out


def _pending_term_keys(customer_id: str) -> set[str]:
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT english_term_new
            FROM glossary_change_requests
            WHERE customer_id = ? AND status = 'pending' AND action_type IN ('add', 'update')
            """,
            (customer_id,),
        ).fetchall()
    return {
        normalize_term_key(r["english_term_new"]) or normalize_term(r["english_term_new"])
        for r in rows
        if r["english_term_new"]
    }


def save_term_candidates(
    customer_id: str,
    source_file_name: str,
    source_type: str,
    created_by: str,
    candidate_contexts: list[dict],
    glossary: dict,
    client: anthropic.Anthropic | None = None,
) -> int:
    if not customer_id or not candidate_contexts:
        return 0
    merged, _summary = _merge_term_candidate_contexts(
        customer_id,
        candidate_contexts,
        glossary,
        source_type=source_type,
    )
    if not merged:
        return 0
    _persist_term_candidates(
        customer_id=customer_id,
        source_file_name=source_file_name,
        source_type=source_type,
        created_by=created_by,
        merged_candidates=merged,
        glossary=glossary,
        client=client,
    )
    return len(merged)


def _merge_term_candidate_contexts(
    customer_id: str,
    candidate_contexts: list[dict],
    glossary: dict,
    source_type: str = "",
) -> tuple[dict[str, dict], dict]:
    if not customer_id or not candidate_contexts:
        return {}, {
            "total_contexts": 0,
            "filtered_active": 0,
            "filtered_pending": 0,
            "filtered_noise": 0,
            "filtered_scope": 0,
            "workmanship_only": False,
            "candidate_count": 0,
            "review_terms": [],
            "review_locations": [],
        }
    conflict_keys = glossary_conflict_keys_from_dict(glossary)
    conflict_terms_by_key: dict[str, str] = {}
    for en in glossary:
        k = normalize_term_key(en)
        if k in conflict_keys:
            conflict_terms_by_key.setdefault(k, "")
            conflict_terms_by_key[k] = (
                f"{conflict_terms_by_key[k]}; {en}" if conflict_terms_by_key[k] else en
            )
    active_keys = {
        normalize_term_key(k) or normalize_term(k)
        for k in glossary
        if (normalize_term_key(k) or normalize_term(k)) not in conflict_keys
    }
    pending_keys = _pending_term_keys(customer_id)
    workmanship_only = any("is_workmanship_source" in item for item in candidate_contexts)
    merged: dict[str, dict] = {}
    stats = {
        "total_contexts": len(candidate_contexts),
        "filtered_active": 0,
        "filtered_pending": 0,
        "filtered_noise": 0,
        "filtered_scope": 0,
        "workmanship_only": workmanship_only,
    }
    for item in candidate_contexts:
        term = str(item.get("term", "")).strip()
        key = normalize_term_key(term) or normalize_term(term)
        if not key:
            stats["filtered_noise"] += 1
            continue
        if key in active_keys:
            stats["filtered_active"] += 1
            continue
        if key in pending_keys:
            stats["filtered_pending"] += 1
            continue
        if is_noise_term(term):
            stats["filtered_noise"] += 1
            continue
        if workmanship_only and not bool(item.get("is_workmanship_source")):
            stats["filtered_scope"] += 1
            continue
        if key not in merged:
            merged[key] = {
                "term": term,
                "variants": [],
                "context": str(item.get("context", "")).strip(),
                "page_or_sheet": str(item.get("page_or_sheet", "")).strip(),
                "cell_coordinate": str(item.get("cell_coordinate", "")).strip(),
                "frequency": 0,
                "source_type": source_type or str(item.get("source_type", "")).strip(),
            }
        if term not in merged[key]["variants"]:
            merged[key]["variants"].append(term)
        merged[key]["frequency"] += int(item.get("frequency", 1) or 1)
        if item.get("context") and len(str(item["context"])) > len(merged[key]["context"]):
            merged[key]["context"] = str(item["context"]).strip()

    merged_items = sorted(
        merged.values(),
        key=lambda x: (-int(x.get("frequency", 0) or 0), str(x.get("term", "")).lower()),
    )
    review_terms: list[str] = []
    review_locations: list[str] = []
    for item in merged_items:
        if item.get("term") and item["term"] not in review_terms:
            review_terms.append(item["term"])
        loc = " ".join(
            part for part in [
                str(item.get("source_type", "")).strip(),
                str(item.get("page_or_sheet", "")).strip(),
                str(item.get("cell_coordinate", "")).strip(),
            ]
            if part
        ).strip()
        if loc and loc not in review_locations:
            review_locations.append(loc)
        if len(review_terms) >= 5 and len(review_locations) >= 5:
            break
    stats.update({
        "candidate_count": len(merged_items),
        "review_terms": review_terms[:5],
        "review_locations": review_locations[:5],
    })
    return merged, stats


def _persist_term_candidates(
    customer_id: str,
    source_file_name: str,
    source_type: str,
    created_by: str,
    merged_candidates: dict[str, dict],
    glossary: dict,
    client: anthropic.Anthropic | None = None,
) -> int:
    if not merged_candidates:
        return 0

    conflict_keys = glossary_conflict_keys_from_dict(glossary)
    conflict_terms_by_key: dict[str, str] = {}
    for en in glossary:
        k = normalize_term_key(en)
        if k in conflict_keys:
            conflict_terms_by_key.setdefault(k, "")
            conflict_terms_by_key[k] = (
                f"{conflict_terms_by_key[k]}; {en}" if conflict_terms_by_key[k] else en
            )

    suggestions: dict[str, dict] = {}
    if client is not None:
        for batch in _chunk(list(merged_candidates.values()), 40):
            try:
                suggestions.update(suggest_term_candidate_translations(client, batch, glossary))
            except Exception:
                pass

    now = _now_iso()
    saved = 0
    with get_db_connection() as conn:
        for key, item in merged_candidates.items():
            suggestion = suggestions.get(key, {})
            zh = suggestion.get("translation", "")
            confidence = suggestion.get("confidence", "medium")
            if normalized_key_low_confidence(item["term"]):
                confidence = "low"
            existing = conn.execute(
                """
                SELECT candidate_id, frequency, variants
                FROM term_candidates
                WHERE customer_id = ? AND normalized_term = ?
                  AND status IN ('draft', 'selected')
                ORDER BY candidate_id DESC
                LIMIT 1
                """,
                (customer_id, key),
            ).fetchone()
            if existing:
                try:
                    variants = json.loads(existing["variants"] or "[]")
                except json.JSONDecodeError:
                    variants = []
                for v in item["variants"]:
                    if v not in variants:
                        variants.append(v)
                conn.execute(
                    """
                    UPDATE term_candidates
                    SET frequency = frequency + ?, context_sentence = ?,
                        source_file_name = ?, source_type = ?, page_or_sheet = ?,
                        cell_coordinate = ?,
                        variants = ?,
                        matched_by = ?,
                        matched_glossary_term = ?,
                        conflict_warning = ?,
                        ai_suggested_translation = COALESCE(NULLIF(ai_suggested_translation, ''), ?),
                        final_translation = COALESCE(NULLIF(final_translation, ''), ?),
                        confidence = COALESCE(NULLIF(confidence, ''), ?)
                    WHERE candidate_id = ?
                    """,
                    (
                        item["frequency"],
                        item["context"],
                        source_file_name,
                        source_type,
                        item["page_or_sheet"],
                        item["cell_coordinate"],
                        json.dumps(variants, ensure_ascii=False),
                        "normalized_key" if key in conflict_keys else "no_match",
                        conflict_terms_by_key.get(key, ""),
                        "normalized_key 命中多条不同中文翻译，请人工确认" if key in conflict_keys else "",
                        zh,
                        zh,
                        confidence,
                        existing["candidate_id"],
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO term_candidates (
                        customer_id, source_file_name, source_type, page_or_sheet,
                        cell_coordinate, original_term, normalized_term, variants,
                        ai_suggested_translation, final_translation, context_sentence,
                        frequency, confidence, matched_by, matched_glossary_term,
                        conflict_warning, status, created_by, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'draft', ?, ?)
                    """,
                    (
                        customer_id,
                        source_file_name,
                        source_type,
                        item["page_or_sheet"],
                        item["cell_coordinate"],
                        item["term"],
                        key,
                        json.dumps(item["variants"], ensure_ascii=False),
                        zh,
                        zh,
                        item["context"],
                        item["frequency"],
                        confidence,
                        "normalized_key" if key in conflict_keys else "no_match",
                        conflict_terms_by_key.get(key, ""),
                        "normalized_key 命中多条不同中文翻译，请人工确认" if key in conflict_keys else "",
                        created_by,
                        now,
                    ),
                )
            saved += 1
    return saved


def _build_translation_review_summary(
    n_unrecorded_terms: int,
    candidate_stats: dict,
) -> dict:
    review_count = int(candidate_stats.get("candidate_count", 0) or 0)
    review_terms = candidate_stats.get("review_terms") or []
    review_locations = candidate_stats.get("review_locations") or []
    workmanship_only = bool(candidate_stats.get("workmanship_only"))

    if review_count == 0 and n_unrecorded_terms == 0:
        summary_text = "本次翻译全部基于术语库完成，未发现需人工核查项。"
    elif review_count == 0:
        summary_text = f"本次翻译未发现需补充的做工术语，但仍有 {n_unrecorded_terms} 个未收录术语。"
    elif n_unrecorded_terms == 0:
        summary_text = f"本次翻译已基于术语库完成，另有 {review_count} 处做工相关待补充项，建议重点核查。"
    else:
        summary_text = (
            f"本次翻译基于术语库完成，发现 {n_unrecorded_terms} 个未收录术语 "
            f"和 {review_count} 处做工相关待补充项。"
        )

    if review_terms:
        summary_text += f" 重点术语：{'、'.join(review_terms[:3])}。"
    if review_locations:
        summary_text += f" 重点位置：{'、'.join(review_locations[:3])}。"

    return {
        "summary_text": summary_text,
        "n_unrecorded_terms": int(n_unrecorded_terms),
        "n_review_items": review_count,
        "review_terms": review_terms[:5],
        "review_locations": review_locations[:5],
        "workmanship_only": workmanship_only,
        "needs_manual_review": bool(review_count or n_unrecorded_terms),
    }


def list_term_candidates(user: dict, customer_id: str | None = None, status: str | None = None) -> list[dict]:
    where = []
    params: list = []
    accessible = get_accessible_customers(user)
    ids = [c["customer_id"] for c in accessible]
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    where.append(f"tc.customer_id IN ({placeholders})")
    params.extend(ids)
    if customer_id:
        if customer_id not in ids:
            return []
        where.append("tc.customer_id = ?")
        params.append(customer_id)
    if status and status != "全部":
        where.append("tc.status = ?")
        params.append(status)
    sql = f"""
        SELECT tc.*, c.customer_code, c.customer_name
        FROM term_candidates tc
        LEFT JOIN customers c ON c.customer_id = tc.customer_id
        WHERE {' AND '.join(where)}
        ORDER BY tc.created_at DESC, tc.candidate_id DESC
    """
    with get_db_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def submit_term_candidates_for_approval(user: dict, edited_rows: list[dict]) -> tuple[int, list[str]]:
    submitted = 0
    errors: list[str] = []
    pending_keys_cache: dict[str, set[str]] = {}
    active_keys_cache: dict[str, set[str]] = {}
    with get_db_connection() as conn:
        for row in edited_rows:
            candidate_id = int(row.get("candidate_id"))
            customer_id = str(row.get("customer_id", "")).strip()
            final_translation = str(row.get("final_translation", "")).strip()
            note = str(row.get("note", "") or "").strip()
            if not final_translation:
                errors.append(f"#{candidate_id} 缺少人工确认中文")
                continue
            if not can_submit_glossary_change(user, customer_id):
                errors.append(f"#{candidate_id} 无权提交该客户")
                continue
            candidate = conn.execute(
                """
                SELECT *
                FROM term_candidates
                WHERE candidate_id = ? AND customer_id = ? AND status IN ('draft', 'selected')
                """,
                (candidate_id, customer_id),
            ).fetchone()
            if candidate is None:
                errors.append(f"#{candidate_id} 已提交或不存在")
                continue
            if customer_id not in active_keys_cache:
                active_rows = conn.execute(
                    "SELECT english_term FROM glossary_terms WHERE customer_id = ? AND status = 'active'",
                    (customer_id,),
                ).fetchall()
                active_keys_cache[customer_id] = {
                    normalize_term_key(r["english_term"]) or normalize_term(r["english_term"])
                    for r in active_rows
                }
            if customer_id not in pending_keys_cache:
                pending_keys_cache[customer_id] = _pending_term_keys(customer_id)
            key = normalize_term_key(candidate["original_term"]) or normalize_term(candidate["original_term"])
            if key in active_keys_cache[customer_id]:
                conn.execute(
                    "UPDATE term_candidates SET status = 'ignored' WHERE candidate_id = ?",
                    (candidate_id,),
                )
                errors.append(f"{candidate['original_term']} 已在正式术语库中，已忽略")
                continue
            if key in pending_keys_cache[customer_id]:
                errors.append(f"{candidate['original_term']} 已有待审批申请")
                continue
            source_note = (
                f"candidate_id={candidate_id}；来源文件={candidate['source_file_name']}；"
                f"来源位置={candidate['page_or_sheet']} {candidate['cell_coordinate'] or ''}；"
                f"normalized_key={candidate['normalized_term']}；"
                f"variants={candidate['variants'] or candidate['original_term']}；"
                f"出现次数={candidate['frequency']}；置信度={candidate['confidence']}；"
                f"上下文={candidate['context_sentence']}"
            )
            full_note = f"{note}；{source_note}" if note else source_note
            req_id = create_glossary_change_request(
                user,
                customer_id,
                "add",
                english_term_new=candidate["original_term"],
                chinese_translation_new=final_translation,
                note=full_note,
                candidate_id=candidate_id,
            )
            conn.execute(
                """
                UPDATE term_candidates
                SET final_translation = ?, status = 'submitted', submitted_at = ?
                WHERE candidate_id = ?
                """,
                (final_translation, _now_iso(), candidate_id),
            )
            pending_keys_cache[customer_id].add(key)
            submitted += 1
    return submitted, errors


def create_glossary_change_request(
    user: dict,
    customer_id: str,
    action_type: str,
    english_term_old: str = "",
    chinese_translation_old: str = "",
    english_term_new: str = "",
    chinese_translation_new: str = "",
    note: str = "",
    candidate_id: int | None = None,
) -> int:
    if action_type not in {"add", "update", "delete"}:
        raise ValueError("不支持的术语申请类型")
    if not can_submit_glossary_change(user, customer_id):
        raise PermissionError("当前用户无权提交该客户的术语修改申请")

    with get_db_connection() as conn:
        returning = " RETURNING request_id" if conn.is_postgres else ""
        cur = conn.execute(
            f"""
            INSERT INTO glossary_change_requests (
                customer_id, action_type, candidate_id,
                english_term_old, chinese_translation_old,
                english_term_new, chinese_translation_new,
                note, submitted_by, submitted_at, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
            {returning}
            """,
            (
                customer_id,
                action_type,
                candidate_id,
                english_term_old.strip(),
                chinese_translation_old.strip(),
                english_term_new.strip(),
                chinese_translation_new.strip(),
                note.strip(),
                user["username"],
                _now_iso(),
            ),
        )
        if conn.is_postgres:
            row = cur.fetchone()
            return int(row["request_id"])
        return int(cur.lastrowid)


def import_customer_glossary(
    user: dict,
    customer_id: str,
    glossary_df: pd.DataFrame,
    conflict_mode: str,
) -> tuple[dict, list[dict]]:
    """Import parsed glossary rows for one customer.
    conflict_mode: skip_existing, overwrite_existing, pending_request.
    Non-admin users are forced through pending_request.
    """
    if not can_submit_glossary_change(user, customer_id):
        raise PermissionError("当前用户无权为该客户上传术语库")
    if conflict_mode not in {"skip_existing", "overwrite_existing", "pending_request"}:
        raise ValueError("不支持的导入方式")
    if not can_approve_glossary_change(user):
        conflict_mode = "pending_request"

    stats = {
        "success_imported": 0,
        "skipped_existing": 0,
        "overwritten": 0,
        "pending_count": 0,
        "error_rows": 0,
    }
    report_rows: list[dict] = []
    now = _now_iso()

    with get_db_connection() as conn:
        existing_rows = conn.execute(
            """
            SELECT glossary_id, english_term, chinese_translation, normalized_key, note
            FROM glossary_terms
            WHERE customer_id = ? AND status = 'active'
            """,
            (customer_id,),
        ).fetchall()
        existing = {r["english_term"].strip().lower(): dict(r) for r in existing_rows}
        existing_by_key: dict[str, list[dict]] = {}
        for r in existing_rows:
            key = r["normalized_key"] or normalize_term_key(r["english_term"])
            if key:
                existing_by_key.setdefault(key, []).append(dict(r))

        for idx, row in glossary_df.iterrows():
            en = str(row.get(_GLOSSARY_EN_COL, "")).strip()
            zh = str(row.get(_GLOSSARY_ZH_COL, "")).strip()
            normalized_key = normalize_term_key(en)
            note = str(row.get(_GLOSSARY_NOTE_COL, "") or "").strip()
            category = str(row.get(_GLOSSARY_CAT_COL, "") or "").strip()
            if category:
                note = f"{note}｜来源：{category}" if note else f"来源：{category}"
            if not en or not zh:
                stats["error_rows"] += 1
                report_rows.append({
                    "sheet_name": category,
                    "row_number": idx + 2,
                    "english_term": en,
                    "chinese_translation": zh,
                    "status": "error",
                    "reason": "英文术语或中文翻译为空",
                })
                continue

            key = en.lower()
            old = existing.get(key)
            key_matches = existing_by_key.get(normalized_key, []) if normalized_key else []
            if old is None and key_matches and glossary_key_has_translation_conflict(key_matches, zh):
                stats["error_rows"] += 1
                report_rows.append({
                    "sheet_name": category,
                    "row_number": idx + 2,
                    "english_term": en,
                    "chinese_translation": zh,
                    "status": "normalized_key_conflict",
                    "reason": "同一 normalized_key 已存在不同中文翻译，请人工处理",
                })
                continue
            if conflict_mode == "skip_existing" and old:
                stats["skipped_existing"] += 1
                report_rows.append({
                    "sheet_name": category,
                    "row_number": idx + 2,
                    "english_term": en,
                    "chinese_translation": zh,
                    "status": "skipped_existing",
                    "reason": "该客户已存在相同英文术语",
                })
                continue

            if conflict_mode == "pending_request":
                req_id = create_glossary_change_request(
                    user,
                    customer_id,
                    "update" if old else "add",
                    english_term_old=old["english_term"] if old else "",
                    chinese_translation_old=old["chinese_translation"] if old else "",
                    english_term_new=en,
                    chinese_translation_new=zh,
                    note=note,
                )
                stats["pending_count"] += 1
                report_rows.append({
                    "sheet_name": category,
                    "row_number": idx + 2,
                    "english_term": en,
                    "chinese_translation": zh,
                    "status": "pending",
                    "reason": f"已生成待审批申请 #{req_id}",
                })
                continue

            if old:
                conn.execute(
                    """
                    UPDATE glossary_terms
                    SET english_term = ?, chinese_translation = ?, normalized_key = ?, note = ?,
                        updated_by = ?, updated_at = ?, status = 'active'
                    WHERE glossary_id = ?
                    """,
                    (en, zh, normalized_key, note, user["username"], now, old["glossary_id"]),
                )
                stats["overwritten"] += 1
                report_rows.append({
                    "sheet_name": category,
                    "row_number": idx + 2,
                    "english_term": en,
                    "chinese_translation": zh,
                    "status": "overwritten",
                    "reason": "已覆盖已有术语",
                })
            else:
                conn.execute(
                    """
                    INSERT INTO glossary_terms (
                        customer_id, english_term, chinese_translation, normalized_key, note,
                        created_by, updated_by, updated_at, status
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active')
                    """,
                    (customer_id, en, zh, normalized_key, note, user["username"], user["username"], now),
                )
                stats["success_imported"] += 1
                report_rows.append({
                    "sheet_name": category,
                    "row_number": idx + 2,
                    "english_term": en,
                    "chinese_translation": zh,
                    "status": "imported",
                    "reason": "已导入 active 术语库",
                })
                new_existing = {
                    "english_term": en,
                    "chinese_translation": zh,
                    "normalized_key": normalized_key,
                    "note": note,
                }
                existing[key] = new_existing
                if normalized_key:
                    existing_by_key.setdefault(normalized_key, []).append(new_existing)

    return stats, report_rows


def list_glossary_change_requests(
    status: str | None = None,
    customer_id: str | None = None,
    submitted_by: str | None = None,
    action_type: str | None = None,
) -> list[dict]:
    where = []
    params = []
    if status:
        where.append("r.status = ?")
        params.append(status)
    if customer_id:
        where.append("r.customer_id = ?")
        params.append(customer_id)
    if submitted_by:
        where.append("r.submitted_by = ?")
        params.append(submitted_by)
    if action_type:
        where.append("r.action_type = ?")
        params.append(action_type)

    sql = """
        SELECT r.*, c.customer_code, c.customer_name
        FROM glossary_change_requests r
        LEFT JOIN customers c ON c.customer_id = r.customer_id
    """
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY r.submitted_at DESC, r.request_id DESC"

    with get_db_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def _apply_approved_glossary_change(conn: sqlite3.Connection, request_row: sqlite3.Row) -> None:
    action = request_row["action_type"]
    customer_id = request_row["customer_id"]
    reviewer = request_row["reviewed_by"]
    now = request_row["reviewed_at"]

    if action == "add":
        normalized_key = normalize_term_key(request_row["english_term_new"])
        key_matches = glossary_conflicts_by_key(customer_id, normalized_key)
        if glossary_key_has_translation_conflict(key_matches, request_row["chinese_translation_new"]):
            raise ValueError("该术语 normalized_key 已存在不同中文翻译，请先处理术语冲突")
        existing = conn.execute(
            """
            SELECT glossary_id
            FROM glossary_terms
            WHERE customer_id = ? AND lower(english_term) = lower(?) AND status = 'active'
            ORDER BY glossary_id DESC
            LIMIT 1
            """,
            (customer_id, request_row["english_term_new"]),
        ).fetchone()
        if existing:
            conn.execute(
                """
                UPDATE glossary_terms
                SET chinese_translation = ?, normalized_key = ?, note = ?,
                    updated_by = ?, updated_at = ?, status = 'active'
                WHERE glossary_id = ?
                """,
                (
                    request_row["chinese_translation_new"],
                    normalized_key,
                    request_row["note"] or "",
                    reviewer,
                    now,
                    existing["glossary_id"],
                ),
            )
        else:
            conn.execute(
                """
                INSERT INTO glossary_terms (
                    customer_id, english_term, chinese_translation, normalized_key, note,
                    created_by, updated_by, updated_at, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active')
                """,
                (
                    customer_id,
                    request_row["english_term_new"],
                    request_row["chinese_translation_new"],
                    normalized_key,
                    request_row["note"] or "",
                    request_row["submitted_by"],
                    reviewer,
                    now,
                ),
            )
    elif action == "update":
        normalized_key = normalize_term_key(request_row["english_term_new"])
        key_matches = [
            r for r in glossary_conflicts_by_key(customer_id, normalized_key)
            if normalize_term(r["english_term"]) != normalize_term(request_row["english_term_old"])
        ]
        if glossary_key_has_translation_conflict(key_matches, request_row["chinese_translation_new"]):
            raise ValueError("该术语 normalized_key 已存在不同中文翻译，请先处理术语冲突")
        conn.execute(
            """
            UPDATE glossary_terms
            SET english_term = ?, chinese_translation = ?, normalized_key = ?, note = ?,
                updated_by = ?, updated_at = ?, status = 'active'
            WHERE customer_id = ? AND lower(english_term) = lower(?) AND status = 'active'
            """,
            (
                request_row["english_term_new"],
                request_row["chinese_translation_new"],
                normalized_key,
                request_row["note"] or "",
                reviewer,
                now,
                customer_id,
                request_row["english_term_old"],
            ),
        )
    elif action == "delete":
        conn.execute(
            """
            UPDATE glossary_terms
            SET status = 'inactive', updated_by = ?, updated_at = ?
            WHERE customer_id = ? AND lower(english_term) = lower(?) AND status = 'active'
            """,
            (reviewer, now, customer_id, request_row["english_term_old"]),
        )


def approve_glossary_change_request(request_id: int, reviewer: dict, comment: str = "") -> None:
    if not can_approve_glossary_change(reviewer):
        raise PermissionError("当前用户无权审批术语申请")
    with get_db_connection() as conn:
        now = _now_iso()
        conn.execute(
            """
            UPDATE glossary_change_requests
            SET status = 'approved', reviewed_by = ?, reviewed_at = ?, review_comment = ?
            WHERE request_id = ? AND status = 'pending'
            """,
            (reviewer["username"], now, comment.strip(), request_id),
        )
        request_row = conn.execute(
            "SELECT * FROM glossary_change_requests WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if request_row is None or request_row["status"] != "approved":
            raise ValueError("申请不存在或已被处理")
        _apply_approved_glossary_change(conn, request_row)
        try:
            candidate_id = request_row["candidate_id"]
        except (KeyError, IndexError):
            candidate_id = None
        if candidate_id:
            conn.execute(
                """
                UPDATE term_candidates
                SET status = 'approved'
                WHERE candidate_id = ?
                """,
                (candidate_id,),
            )


def reject_glossary_change_request(request_id: int, reviewer: dict, comment: str = "") -> None:
    if not can_approve_glossary_change(reviewer):
        raise PermissionError("当前用户无权审批术语申请")
    with get_db_connection() as conn:
        cur = conn.execute(
            """
            UPDATE glossary_change_requests
            SET status = 'rejected', reviewed_by = ?, reviewed_at = ?, review_comment = ?
            WHERE request_id = ? AND status = 'pending'
            """,
            (reviewer["username"], _now_iso(), comment.strip(), request_id),
        )
        if cur.rowcount != 1:
            raise ValueError("申请不存在或已被处理")
        row = conn.execute(
            "SELECT candidate_id FROM glossary_change_requests WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if row and row["candidate_id"]:
            conn.execute(
                """
                UPDATE term_candidates
                SET status = 'rejected'
                WHERE candidate_id = ?
                """,
                (row["candidate_id"],),
            )


def render_login_panel() -> dict | None:
    current_user = st.session_state.get("current_user")
    if current_user:
        with st.sidebar:
            st.subheader("当前用户")
            st.write(f"用户名：**{current_user['username']}**")
            st.write(f"角色：`{current_user['role']}`")
            if current_user.get("group_name"):
                st.write(f"小组：**{current_user['group_name']}**")

            accessible = get_accessible_customers(current_user)
            st.caption(f"可访问客户：{len(accessible)} 个")
            if accessible:
                st.dataframe(
                    pd.DataFrame(accessible)[
                        ["customer_code", "customer_name", "group_name", "assigned_staff_username"]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

            if st.button("退出登录", use_container_width=True):
                for key in ("current_user", "selected_customer_id"):
                    st.session_state.pop(key, None)
                st.rerun()
        return current_user

    st.subheader("登录")
    st.caption("第一版本地账号，用于验证客户术语库权限。")
    with st.form("login_form"):
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        submitted = st.form_submit_button("登录", use_container_width=True, type="primary")

    if submitted:
        user = authenticate_user(username, password)
        if user:
            st.session_state["current_user"] = user
            st.rerun()
        st.error("用户名或密码不正确")

    st.info(
        "测试账号：admin/admin123，leader_a/leader123，"
        "staff_1/staff123，staff_2/staff123，staff_3/staff123"
    )
    return None


def render_customer_selector(user: dict) -> str | None:
    accessible = get_accessible_customers(user)
    valid_ids = [c["customer_id"] for c in accessible]
    pending_selected = st.session_state.pop("_pending_selected_customer_id", None)
    if pending_selected in valid_ids:
        st.session_state["selected_customer_id"] = pending_selected
    selected = st.session_state.get("selected_customer_id")
    if selected not in valid_ids:
        st.session_state.pop("selected_customer_id", None)
        selected = None

    st.subheader("客户选择")
    if not accessible:
        st.warning("当前用户没有可访问客户。")
        return None

    labels = {
        c["customer_id"]: (
            f"{c['customer_code']} / {c['customer_name']} / "
            f"{c['group_name']} / {c['assigned_staff_username']}"
        )
        for c in accessible
    }
    default_index = valid_ids.index(selected) if selected in valid_ids else 0
    selected_customer_id = st.selectbox(
        "当前客户",
        valid_ids,
        index=default_index,
        format_func=lambda cid: labels.get(cid, cid),
        key="selected_customer_id",
    )
    st.caption("后续 PDF / Excel 翻译会使用这里选择的客户术语库。")
    return selected_customer_id


# ── Glossary helpers ───────────────────────────────────────────────────────────

def load_glossary(data: bytes) -> dict[str, str]:
    wb = openpyxl.load_workbook(io.BytesIO(data))
    ws = wb.active
    headers = [str(c.value).strip() if c.value else "" for c in ws[1]]
    chi_idx = eng_idx = None
    for i, h in enumerate(headers):
        if "中文" in h or h.lower() == "chinese":
            chi_idx = i
        if "英文" in h or h.lower() == "english":
            eng_idx = i
    if chi_idx is None or eng_idx is None:
        chi_idx, eng_idx = 0, 1
    g = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        eng = row[eng_idx] if eng_idx < len(row) else None
        chi = row[chi_idx] if chi_idx < len(row) else None
        if eng and chi:
            g[str(eng).strip()] = str(chi).strip()
    return g


def relevant_glossary(text: str, glossary: dict) -> dict:
    tl = text.lower()
    normalized_map: dict[str, list[tuple[str, str]]] = {}
    for en, zh in glossary.items():
        key = normalize_term_key(en)
        if key:
            normalized_map.setdefault(key, []).append((en, zh))
    conflict_keys = {
        key for key, rows in normalized_map.items()
        if len({normalize_term(zh) for _, zh in rows if normalize_term(zh)}) > 1
    }
    result = {
        k: v for k, v in glossary.items()
        if k.lower() in tl and normalize_term_key(k) not in conflict_keys
    }

    result_terms = {normalize_term(k) for k in result}
    words = re.findall(r"[A-Za-z]+", text)
    for n in range(2, min(4, len(words)) + 1):
        for i in range(0, len(words) - n + 1):
            phrase = " ".join(words[i:i + n]).strip()
            if normalize_term(phrase) in result_terms:
                continue
            key = normalize_term_key(phrase)
            if not key or key not in normalized_map:
                continue
            matches = normalized_map[key]
            translations = {normalize_term(zh) for _, zh in matches if normalize_term(zh)}
            if len(translations) == 1:
                result[phrase] = matches[0][1]
    return result


# ── Glossary management (editable in-session glossary) ─────────────────────────

_GLOSSARY_EN_COL   = "英文术语"
_GLOSSARY_ZH_COL   = "中文翻译"
_GLOSSARY_NOTE_COL = "备注"
_GLOSSARY_CAT_COL  = "分类"
_GLOSSARY_EDIT_COLS = [_GLOSSARY_EN_COL, _GLOSSARY_ZH_COL, _GLOSSARY_NOTE_COL, _GLOSSARY_CAT_COL]


def _empty_glossary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_GLOSSARY_EDIT_COLS)


def parse_glossary_excel(data: bytes) -> tuple[pd.DataFrame | None, list[str]]:
    """Parse an uploaded glossary workbook into the standard 4-column shape.
    Returns (df, missing_columns); df is None if required columns are missing
    or the file can't be read at all — callers must check missing_columns
    instead of letting an exception propagate.
    """
    try:
        wb = openpyxl.load_workbook(io.BytesIO(data))
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
    except Exception:
        return None, ["无法解析该文件，请确认上传的是有效的 Excel (.xlsx) 文件"]

    if not rows:
        return None, ["文件内容为空"]

    headers = [str(h).strip() if h is not None else "" for h in rows[0]]
    en_idx = zh_idx = note_idx = cat_idx = None
    for i, h in enumerate(headers):
        hl = h.lower()
        if en_idx is None and ("英文" in h or "english" in hl):
            en_idx = i
        if zh_idx is None and ("中文" in h or "chinese" in hl):
            zh_idx = i
        if note_idx is None and ("备注" in h or "note" in hl or "remark" in hl):
            note_idx = i
        if cat_idx is None and ("分类" in h or "category" in hl or "来源" in h or "source" in hl):
            cat_idx = i

    missing = []
    if en_idx is None:
        missing.append("英文术语列（表头需包含「英文」或 English）")
    if zh_idx is None:
        missing.append("中文翻译列（表头需包含「中文」或 Chinese）")
    if missing:
        return None, missing

    def cell(row, idx):
        if idx is None or idx >= len(row) or row[idx] is None:
            return ""
        return str(row[idx])

    records = [
        {
            _GLOSSARY_EN_COL: cell(row, en_idx),
            _GLOSSARY_ZH_COL: cell(row, zh_idx),
            _GLOSSARY_NOTE_COL: cell(row, note_idx),
            _GLOSSARY_CAT_COL: cell(row, cat_idx),
        }
        for row in rows[1:]
    ]
    return pd.DataFrame(records, columns=_GLOSSARY_EDIT_COLS), []


_EN_HEADER_ALIASES = {"english", "english_term", "英文", "英文术语", "原文"}
_ZH_HEADER_ALIASES = {"chinese", "chinese_translation", "中文", "中文翻译", "译文"}
_NOTE_HEADER_ALIASES = {"note", "备注", "comment", "comments", "remark", "remarks"}


def _normalize_header(value) -> str:
    return str(value).strip().lower().replace(" ", "_") if value is not None else ""


def _rows_from_dataframe(df: pd.DataFrame) -> list[tuple]:
    return [
        tuple("" if pd.isna(value) else value for value in row)
        for row in df.itertuples(index=False, name=None)
    ]


def _read_excel_sheet_rows(data: bytes, filename: str = "") -> tuple[list[tuple[str, list[tuple]]], str | None]:
    ext = Path(filename or "").suffix.lower()
    if ext == ".xls":
        try:
            sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None, engine="xlrd")
            return [(str(name), _rows_from_dataframe(df)) for name, df in sheets.items()], None
        except Exception:
            return [], "无法解析该文件，请确认上传的是有效的 Excel (.xls) 文件"
    try:
        wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        try:
            return [
                (ws.title, list(ws.iter_rows(values_only=True)))
                for ws in wb.worksheets
            ], None
        finally:
            wb.close()
    except Exception:
        try:
            sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None, engine="xlrd")
            return [(str(name), _rows_from_dataframe(df)) for name, df in sheets.items()], None
        except Exception:
            return [], "无法解析该文件，请确认上传的是有效的 Excel (.xlsx / .xls) 文件"


def _find_glossary_header(rows: list[tuple]) -> tuple[int | None, int | None, int | None, int | None]:
    for row_idx, row in enumerate(rows[:30]):
        normalized = [_normalize_header(v) for v in row]
        en_idx = zh_idx = note_idx = None
        for col_idx, h in enumerate(normalized):
            raw = str(row[col_idx]).strip() if row[col_idx] is not None else ""
            if en_idx is None and (h in _EN_HEADER_ALIASES or raw in _EN_HEADER_ALIASES):
                en_idx = col_idx
            if zh_idx is None and (h in _ZH_HEADER_ALIASES or raw in _ZH_HEADER_ALIASES):
                zh_idx = col_idx
            if note_idx is None and (h in _NOTE_HEADER_ALIASES or raw in _NOTE_HEADER_ALIASES):
                note_idx = col_idx
        if en_idx is not None and zh_idx is not None:
            return row_idx, en_idx, zh_idx, note_idx
    return None, None, None, None


def parse_customer_glossary_excel(data: bytes, filename: str = "") -> tuple[pd.DataFrame | None, list[str], dict, list[dict]]:
    """Parse a customer glossary upload across all sheets.
    Supports common Chinese/English headers and returns a cleaned,
    case-insensitively de-duplicated dataframe plus import diagnostics.
    """
    sheet_rows, parse_error = _read_excel_sheet_rows(data, filename)
    if parse_error:
        return None, [parse_error], {}, []

    records: list[dict] = []
    report_rows: list[dict] = []
    stats = {
        "total_rows": 0,
        "valid_rows": 0,
        "skipped_blank": 0,
        "duplicate_terms": 0,
        "error_rows": 0,
        "sheets_without_header": 0,
    }

    for sheet_name, rows in sheet_rows:
        header_idx, en_idx, zh_idx, note_idx = _find_glossary_header(rows)
        if header_idx is None:
            stats["sheets_without_header"] += 1
            report_rows.append({
                "sheet_name": sheet_name,
                "row_number": "",
                "english_term": "",
                "chinese_translation": "",
                "status": "error",
                "reason": "找不到英文/中文表头",
            })
            continue

        for row_number, row in enumerate(rows[header_idx + 1:], start=header_idx + 2):
            stats["total_rows"] += 1

            def cell(idx: int | None) -> str:
                if idx is None or idx >= len(row) or row[idx] is None:
                    return ""
                return str(row[idx]).strip()

            en = cell(en_idx)
            zh = cell(zh_idx)
            note = cell(note_idx)
            if not en and not zh:
                stats["skipped_blank"] += 1
                continue
            if not en or not zh:
                stats["error_rows"] += 1
                report_rows.append({
                    "sheet_name": ws.title,
                    "row_number": row_number,
                    "english_term": en,
                    "chinese_translation": zh,
                    "status": "error",
                    "reason": "英文术语或中文翻译为空",
                })
                continue
            records.append({
                _GLOSSARY_EN_COL: en,
                _GLOSSARY_ZH_COL: zh,
                _GLOSSARY_NOTE_COL: note,
                _GLOSSARY_CAT_COL: sheet_name,
                "_sheet_name": sheet_name,
                "_row_number": row_number,
            })
            stats["valid_rows"] += 1

    if not records:
        return None, ["没有找到可导入的有效术语行"], stats, report_rows

    deduped: dict[str, dict] = {}
    for r in records:
        key = normalize_term_key(r[_GLOSSARY_EN_COL]) or normalize_term(r[_GLOSSARY_EN_COL])
        if key in deduped:
            stats["duplicate_terms"] += 1
            reason = "同一文件内 normalized_key 重复，采用后出现的记录"
            if normalize_term(deduped[key][_GLOSSARY_ZH_COL]) != normalize_term(r[_GLOSSARY_ZH_COL]):
                reason = "同一文件内 normalized_key 重复且中文不同，请人工确认"
                stats["error_rows"] += 1
            report_rows.append({
                "sheet_name": r["_sheet_name"],
                "row_number": r["_row_number"],
                "english_term": r[_GLOSSARY_EN_COL],
                "chinese_translation": r[_GLOSSARY_ZH_COL],
                "status": "duplicate_in_file",
                "reason": reason,
            })
        deduped[key] = r

    df = pd.DataFrame(
        [
            {
                _GLOSSARY_EN_COL: r[_GLOSSARY_EN_COL],
                _GLOSSARY_ZH_COL: r[_GLOSSARY_ZH_COL],
                _GLOSSARY_NOTE_COL: r[_GLOSSARY_NOTE_COL],
                _GLOSSARY_CAT_COL: r[_GLOSSARY_CAT_COL],
            }
            for r in deduped.values()
        ],
        columns=_GLOSSARY_EDIT_COLS,
    )
    return df, [], stats, report_rows


def build_import_report_xlsx(report_rows: list[dict]) -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "导入报告"
    headers = ["sheet_name", "row_number", "english_term", "chinese_translation", "status", "reason"]
    ws.append(headers)
    for c in ws[1]:
        c.font = openpyxl.styles.Font(bold=True)
    for r in report_rows:
        ws.append([r.get(h, "") for h in headers])
    for col_letter, width in zip("ABCDEF", [24, 12, 34, 34, 18, 42]):
        ws.column_dimensions[col_letter].width = width
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def clean_glossary_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Trim whitespace, drop blank rows, de-dup English terms case-insensitively
    (last occurrence wins). Rows with an empty Chinese translation are kept
    (so the user can finish them later) but excluded from the active glossary
    by glossary_df_to_dict().
    Returns (cleaned_df, conflicts_df, n_incomplete).
    """
    conflicts_cols = ["英文术语", "旧翻译", "新翻译（采用）"]
    if df is None or df.empty:
        return _empty_glossary_df(), pd.DataFrame(columns=conflicts_cols), 0

    work = df.copy()
    for col in _GLOSSARY_EDIT_COLS:
        if col not in work.columns:
            work[col] = ""
        work.loc[:, col] = work[col].fillna("").astype(str).str.strip()

    work = work[work[_GLOSSARY_EN_COL] != ""]  # 去掉英文术语为空的行
    incomplete = work[work[_GLOSSARY_ZH_COL] == ""].copy()
    complete = work[work[_GLOSSARY_ZH_COL] != ""].copy()

    conflicts_rows = []
    if not complete.empty:
        complete = complete.copy()
        complete.loc[:, "_key"] = complete[_GLOSSARY_EN_COL].str.lower()
        deduped = []
        for _, grp in complete.groupby("_key", sort=False):
            if len(grp) > 1:
                zh_values = grp[_GLOSSARY_ZH_COL].tolist()
                if len(set(zh_values)) > 1:
                    conflicts_rows.append({
                        "英文术语": grp.iloc[0][_GLOSSARY_EN_COL],
                        "旧翻译": " / ".join(dict.fromkeys(zh_values[:-1])),
                        "新翻译（采用）": zh_values[-1],
                    })
            deduped.append(grp.iloc[-1])  # 同英文术语取最后一条（最新编辑生效）
        complete = pd.DataFrame(deduped, columns=complete.columns)[_GLOSSARY_EDIT_COLS].reset_index(drop=True)

    cleaned = pd.concat([complete, incomplete[_GLOSSARY_EDIT_COLS]], ignore_index=True)
    return cleaned, pd.DataFrame(conflicts_rows, columns=conflicts_cols), len(incomplete)


def glossary_df_to_dict(df: pd.DataFrame) -> dict[str, str]:
    """Active glossary used for translation — only rows with a non-empty
    Chinese translation are included."""
    if df is None or df.empty:
        return {}
    active = df[df[_GLOSSARY_ZH_COL].fillna("").astype(str).str.strip() != ""]
    return {
        str(r[_GLOSSARY_EN_COL]).strip(): str(r[_GLOSSARY_ZH_COL]).strip()
        for _, r in active.iterrows()
        if str(r[_GLOSSARY_EN_COL]).strip()
    }


def _df_to_simple_xlsx(df: pd.DataFrame, sheet_title: str) -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet_title[:31]
    ws.append(list(df.columns))
    for c in ws[1]:
        c.font = openpyxl.styles.Font(bold=True)
    for _, r in df.iterrows():
        ws.append([r[col] for col in df.columns])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def glossary_df_to_xlsx_bytes(df: pd.DataFrame) -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Glossary"
    ws.append(_GLOSSARY_EDIT_COLS)
    for c in ws[1]:
        c.font = openpyxl.styles.Font(bold=True)
    for _, r in df.iterrows():
        ws.append([r[col] for col in _GLOSSARY_EDIT_COLS])
    for col_letter, width in zip("ABCD", [28, 28, 24, 16]):
        ws.column_dimensions[col_letter].width = width
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def parse_unrecorded_excel(data: bytes) -> tuple[pd.DataFrame | None, list[str]]:
    """Parse a filled-in '未收录术语' file: needs an English column and a
    Chinese-translation column the user added by hand."""
    try:
        wb = openpyxl.load_workbook(io.BytesIO(data))
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
    except Exception:
        return None, ["无法解析该文件，请确认上传的是有效的 Excel (.xlsx) 文件"]

    if not rows:
        return None, ["文件内容为空"]

    headers = [str(h).strip() if h is not None else "" for h in rows[0]]
    en_idx = zh_idx = None
    for i, h in enumerate(headers):
        hl = h.lower()
        if en_idx is None and ("英文" in h or "english" in hl or "未收录" in h):
            en_idx = i
        if zh_idx is None and ("中文" in h or "chinese" in hl):
            zh_idx = i

    missing = []
    if en_idx is None:
        missing.append("英文术语列（表头需包含「英文」「未收录」或 English）")
    if zh_idx is None:
        missing.append("中文翻译列（请在表格中新增一列「中文翻译」并填好后再上传）")
    if missing:
        return None, missing

    records = []
    for row in rows[1:]:
        en = row[en_idx] if en_idx < len(row) else None
        zh = row[zh_idx] if zh_idx < len(row) else None
        en = "" if en is None else str(en).strip()
        zh = "" if zh is None else str(zh).strip()
        if en:
            records.append({_GLOSSARY_EN_COL: en, _GLOSSARY_ZH_COL: zh,
                            _GLOSSARY_NOTE_COL: "", _GLOSSARY_CAT_COL: ""})
    return pd.DataFrame(records, columns=_GLOSSARY_EDIT_COLS), []


def merge_unrecorded_into_glossary(
    glossary_df: pd.DataFrame,
    unrecorded_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int, int]:
    """Merge filled-in unrecorded terms into the glossary. New translations
    overwrite existing ones on conflict (per spec: default to newest wins).
    Returns (merged_df, conflicts_df, n_added, n_overwritten, n_skipped_blank).
    """
    base = glossary_df.copy(deep=True)
    for col in _GLOSSARY_EDIT_COLS:
        if col not in base.columns:
            base[col] = ""
    base.loc[:, "_key"] = base[_GLOSSARY_EN_COL].str.lower()
    lookup = {key: idx for idx, key in base["_key"].items()}

    filled = unrecorded_df[unrecorded_df[_GLOSSARY_ZH_COL].fillna("").astype(str).str.strip() != ""]
    n_skipped_blank = len(unrecorded_df) - len(filled)

    conflicts_rows = []
    new_rows = []
    n_added = n_overwritten = 0

    for _, r in filled.iterrows():
        en = str(r[_GLOSSARY_EN_COL]).strip()
        zh = str(r[_GLOSSARY_ZH_COL]).strip()
        key = en.lower()
        if key in lookup:
            idx = lookup[key]
            old_zh = str(base.loc[idx, _GLOSSARY_ZH_COL]).strip()
            if old_zh and old_zh != zh:
                conflicts_rows.append({"英文术语": en, "旧翻译": old_zh, "新翻译（采用）": zh})
            base.loc[idx, _GLOSSARY_ZH_COL] = zh
            n_overwritten += 1
        else:
            new_rows.append({_GLOSSARY_EN_COL: en, _GLOSSARY_ZH_COL: zh,
                             _GLOSSARY_NOTE_COL: "", _GLOSSARY_CAT_COL: ""})
            lookup[key] = None  # 防止同批次内重复英文术语都被当作新增
            n_added += 1

    base = base.drop(columns=["_key"])
    if new_rows:
        base = pd.concat([base, pd.DataFrame(new_rows)], ignore_index=True)

    conflicts_df = pd.DataFrame(conflicts_rows, columns=["英文术语", "旧翻译", "新翻译（采用）"])
    return base.reset_index(drop=True), conflicts_df, n_added, n_overwritten, n_skipped_blank


def _load_default_glossary_df() -> pd.DataFrame:
    if DEFAULT_GLOSSARY.exists():
        df, missing = parse_glossary_excel(DEFAULT_GLOSSARY.read_bytes())
        if df is not None:
            cleaned, _, _ = clean_glossary_df(df)
            return cleaned
    return _empty_glossary_df()


# ── PDF translation ────────────────────────────────────────────────────────────

_GARMENT_SYSTEM_PROMPT = (
    "你是一名资深内衣（含文胸/塑身贴身衣）、内裤、泳衣三大品类的外贸翻译顾问，"
    "同时具备版师/工艺师的实务经验。"
    "你处理的所有文本都来自这三类产品的工艺单（Tech Pack）、规格表或设计说明书，"
    "绝不会涉及童装、男装、户外服、运动服、牛仔裤、外套、鞋类、箱包等其他任何品类。\n\n"

    "【核心原则一：语境优先于字面，禁止逐词拼接】\n"
    "术语对照表给出的中文译法，是「这个英文词在服装语境下的行业标准称呼」，"
    "不是逐词替换表。当术语表中的多个词共同出现在同一个短语或句子里时，"
    "你必须先理解这句话在工艺单里描述的是哪个部位、哪种工艺、想达成什么效果，"
    "再给出一句通顺、符合行业表达习惯的完整翻译——而不是把每个词各自的译法生硬拼接在一起，"
    "导致中文读起来支离破碎、不像人话。\n\n"

    "【核心原则二：调用服装行业常识做合理推断】\n"
    "英文工艺描述经常很简略，或者同一个词在不同款式里实际工艺做法不同。"
    "遇到这种情况，你要结合上下文（款式类型、相邻的工艺描述、常见行业做法）"
    "推断客人真正想表达的工艺意图，用业内通行的专业说法给出延伸翻译，而不是停留在字面直译。\n"
    "例如：'crotch insert' 单独直译是「裆部插入物」，但结合内衣/泳装工艺单的上下文，"
    "这通常是指裆部内侧加一层裆衬/裆布（用于吸湿、防透），应翻译为「裆部内衬」或「裆衬」，"
    "而不是字面的「裆部插入件」。\n\n"

    "【核心原则三：行业语义优先于通用语义】\n"
    "遇到任何存在多义性的词汇，必须优先选取与内衣、泳衣、泳装、外贸成衣"
    "最贴切的中文行业专业术语，不得选用其他行业或日常含义。"
    "例如：hipster → 平角内裤（而非潮人）；brief → 三角内裤（而非简短）；"
    "cup → 罩杯（而非杯子）；underwire → 钢圈（而非底线）。\n\n"

    "【核心原则四：词汇范围严格限定——禁止输出内衣/内裤/泳衣以外品类的术语】\n"
    "你必须假设当前文本只可能描述内衣、内裤、泳衣这三类产品，不会是童装、男装、"
    "户外服、运动服、牛仔裤、外套、鞋类、箱包等其他任何品类，也不会是建筑、机械、"
    "电子、日用品等其他行业的文本。当一个英文词存在多种行业含义时，只允许选取"
    "内衣/内裤/泳衣语境下对应的中文术语，绝不允许使用其他服装品类或其他行业的常见译法，"
    "哪怕那个译法字面上更常见、更通用。\n"
    "示例（默认按这三类产品的语境理解，不做其他联想）：\n"
    "  'lining' → 里布/内衬（不是西装/外套语境下宽泛的「衬里」）\n"
    "  'strap' → 肩带（不是裤腰带、包带、鞋带）\n"
    "  'wire' → 钢圈（不是电线、钢丝）\n"
    "  'band' → 下扒/底围（不是乐队、橡皮筋的泛称）\n"
    "  'gusset' → 裆布/裆衬\n"
    "  'hook & eye' → 钩眼扣\n"
    "如果遇到术语库未收录、且不确定该词在这三类产品语境下准确译法的词，"
    "仍应给出内衣/内裤/泳衣行业最贴近的合理翻译，但不得套用其他品类或行业的常见说法。\n\n"

    "【参考示例（few-shot，请学习这种基于语境的翻译方式）】\n\n"
    "示例 1：\n"
    "输入：Crotch insert with gusset lining for moisture control\n"
    "✗ 错误的逐词拼接：裆部插入物带衬垫衬里用于湿度控制\n"
    "✓ 正确的语境翻译：裆部加裆衬里布，具备吸湿透气功能\n"
    "说明：'crotch insert' 和 'gusset lining' 在内裤/泳装工艺单里描述的是同一个部位、"
    "同一层结构——裆部内衬，应合并理解为一个完整工艺点，不能拆成两个词分别直译再堆叠。\n\n"
    "示例 2：\n"
    "输入：Underwire channel with cup seam binding\n"
    "✗ 错误的逐词拼接：钢圈通道带罩杯缝合捆绑\n"
    "✓ 正确的语境翻译：钢圈槽配罩杯接缝包边\n"
    "说明：'channel' 在文胸工艺单语境下专指钢圈槽（而非通道），'binding' 在缝份处理语境下"
    "是包边工艺（而非捆绑）；结合上下文给出符合行业习惯的完整短语，而非字面直译再堆叠。\n\n"

    "【核心原则五：强制全量翻译，禁止过度保护短词/指示句】\n"
    "除了纯数字（例如 '170 GSM' 中的 170）之外，文本里出现的所有英文字母、单词、缩写、短语、"
    "操作指示句，全部必须翻译成中文，不允许以'太短'、'像是代码'、'已经够清楚'、"
    "'是常见英文词不需要翻译'等理由跳过翻译、原样返回英文，或留空。"
    "哪怕只有一两个字母组成的词，只要它是一个独立单词而不是型号/尺码代号，也必须翻译。\n"
    "示例：'assortment' 必须译为「搭配/分类」；'pls see Artwork' 必须译为「请见图稿」；"
    "'no' 必须译为「否/无」；'incl. binding' 必须译为「含包边」。\n\n"

    "【核心原则六：正面指令——型号/编码原样输出，但不得连带跳过其前后的英文】\n"
    "所有英文字符均需翻译为中文。如果遇到必须保留的型号、货号、编码（例如 2118875、32A），"
    "请将该型号/编码本身原样输出，但绝不能以\"这是型号/编码\"为借口，"
    "连同它前后紧邻的英文单词、短语一起跳过不翻译——同一段文本里，"
    "编码以外的部分必须照常翻译成中文。\n\n"

    "【核心原则七：免翻译放行规则（Pass-through Rule）】\n"
    "当提取到的文本整体属于以下任一情况时，不需要翻译成中文，原样保留英文即可：\n"
    "  ① 单个英文字母（如表示罩杯/部位的 'D'、'C'）\n"
    "  ② 纯数字、纯标点符号，或数字与标点的组合（如 '2025)'）\n"
    "  ③ 简单型号/尺码代码（如 '32A'、'70B'）\n"
    "  ④ 行业专有缩写，即全大写字母组成的缩写词（如 'NOS'、'GSM'、'USA'）\n"
    "  ⑤ 数字+空格+大写单位的组合（如 '170 GSM'、'32 mm'）\n"
    "  ⑥ 无法确信中文含义的英文代号/色号/款式名（如 'Brick'——如果不确定请原样保留）\n"
    "这类内容本身不是需要翻译的语言文字，强行翻译反而会出错或不知所云。\n"
    "但\"不需要翻译\"绝不代表可以丢弃——你必须在输出的 JSON 中将其【原样】放入 "
    "\"translated\" 字段，与 \"original\" 完全一致，绝不能省略、留空，或跳过这一条不返回。\n"
    "示例：\n"
    '  {"original": "D", "translated": "D"}\n'
    '  {"original": "NOS", "translated": "NOS"}\n'
    '  {"original": "170 GSM", "translated": "170 GSM"}\n'
    '  {"original": "Brick", "translated": "Brick"}\n'
    '  {"original": "2025)", "translated": "2025)"}\n'
    "注意区分：放行规则只适用于上述整体属于代号/缩写/数字/标点的情况；"
    "只要文本里包含真正的英文单词或短语（哪怕很短，如 'no'、'ab'、'assortment'），"
    "仍必须依照核心原则五翻译成中文，不能套用本规则逃避翻译。"
)


_MAX_BATCH_ITEMS = max(3, _int_secret_or_env("PDF_TRANSLATION_BATCH_SIZE", 6))


def _chunk(items: list, n: int) -> list[list]:
    return [items[i:i + n] for i in range(0, len(items), n)]


_RE_TABLE_ARTIFACT = re.compile(r'^\s*the following table\s*:?\s*', re.IGNORECASE)
_RE_REPEAT_PUNCT = re.compile(r'([,，.\-_]){3,}')


def _clean_extracted_text(text: str) -> str:
    """Strip known PyMuPDF extraction artifacts (stray 'The following table:'
    prefixes, runs of repeated commas/dots) before the text ever reaches the
    LLM, so this noise can't distract it from real content."""
    t = _RE_TABLE_ARTIFACT.sub('', text)
    t = _RE_REPEAT_PUNCT.sub(lambda m: m.group(1), t)
    return t.strip()


_PDF_METADATA_PATTERNS = [
    re.compile(r"^creation date\b", re.IGNORECASE),
    re.compile(r"^style[-\s]?no\.?:?\s*\d", re.IGNORECASE),
    re.compile(r"^page\s+\d+\s+of\s+\d+$", re.IGNORECASE),
    re.compile(r"^©\s*NKD Group\b", re.IGNORECASE),
    re.compile(r"\btechnical specification\b", re.IGNORECASE),
]
_HEADER_METADATA_LABELS = {
    "style info",
    "style no",
    "style number",
    "style",
    "product group",
    "product manager",
    "designer",
    "member of qa",
    "qa",
    "department",
    "division",
    "season",
    "collection",
    "supplier",
    "owner",
    "date",
    "page",
    "description",
    "short model text",
}
_TOP_METADATA_FRACTION = 0.28


def _metadata_label_key(text: str) -> str:
    t = str(text or "").strip().lower()
    t = re.sub(r"[:：]+$", "", t).strip()
    t = re.sub(r"[-_/]+", " ", t)
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _looks_like_top_metadata_value(text: str) -> bool:
    t = re.sub(r"\s+", " ", str(text or "").strip())
    if not t or len(t) > 32:
        return False
    if re.search(r"[一-鿿]", t):
        return False
    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", t)
    if not words or len(words) > 3:
        return False
    lowered = {w.lower() for w in words}
    if lowered & _WORKMANSHIP_SIGNAL_WORDS:
        return False
    if re.search(r"\b(?:bra|brief|hipster|bikini|swim|wire|cup|lace|mesh|aop|crochet|push)\b", t, re.IGNORECASE):
        return False
    if re.fullmatch(r"[A-Z]{1,4}", t):
        return True
    if re.search(r"[À-ÖØ-öø-ÿ]", t):
        return True
    if re.fullmatch(r"(?:[A-Z]\.\s*)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}", t):
        return True
    return False


def _is_pdf_metadata_text(text: str) -> bool:
    """Skip PDF headers/footers and document metadata that should remain
    unchanged. These spans often contain English plus dates/codes, which can
    trigger unnecessary retry translation and fail an otherwise valid job."""
    t = " ".join(text.strip().split())
    if not t:
        return False
    return any(pattern.search(t) for pattern in _PDF_METADATA_PATTERNS)


def _is_header_or_metadata_text(
    text: str,
    bbox=None,
    page_height: float | None = None,
) -> bool:
    t = re.sub(r"\s+", " ", str(text or "").strip())
    if not t:
        return False
    key = _metadata_label_key(t)
    if key in _HEADER_METADATA_LABELS:
        return True
    if page_height and bbox is not None:
        try:
            y0 = float(bbox[1])
        except (TypeError, ValueError, IndexError):
            y0 = page_height
        in_top_info_area = y0 <= page_height * _TOP_METADATA_FRACTION
        if in_top_info_area and _looks_like_top_metadata_value(t):
            return True
    return False


def translate_batch(client: anthropic.Anthropic, texts: list[str], glossary: dict) -> tuple[dict[str, str], set[str]]:
    """Translate many spans in a single LLM call. Returns
    ({original_text: translated_text}, unrecorded_terms)."""
    unique_texts = list(dict.fromkeys(t for t in texts if t.strip()))
    if not unique_texts:
        return {}, set()
    passthrough = {t: t for t in unique_texts if _is_color_item(t)}
    unique_texts = [t for t in unique_texts if t not in passthrough]
    if not unique_texts:
        return passthrough, set()

    rel = relevant_glossary("\n".join(unique_texts), glossary)
    gloss_block = ""
    if rel:
        lines = "\n".join(f"  {k} → {v}" for k, v in rel.items())
        gloss_block = f"强制术语对照（务必照搬，不得自行发挥）：\n{lines}\n\n"

    items_xml = "\n".join(
        f"<item>{json.dumps(t, ensure_ascii=False)}</item>" for t in unique_texts
    )

    prompt = (
        f"{gloss_block}"
        "<text_to_translate>\n"
        f"{items_xml}\n"
        "</text_to_translate>\n\n"
        "你必须处理 <text_to_translate> 标签内的【每一个 <item>】，不允许遗漏任何一项。\n\n"
        "规则：\n"
        "1. 强制术语对照中词汇的中文译法必须出现在对应译文里，但要结合整条文本的语境自然融入，"
        "禁止逐词直译后生硬拼接。\n"
        "2. 所有英文字符均需翻译为中文。如果某个 item 里含有必须保留的型号/货号/编码"
        "（如 2118875、32A），请将该编码本身原样输出，但不得以此为借口跳过编码前后的英文单词——"
        "同一个 item 里编码以外的部分仍必须翻译。\n"
        "3. 当英文表述简略或存在多义性时，结合服装款式与工艺常识进行合理推断，"
        "给出业内真正想表达的工艺含义，不要停留在字面直译。\n"
        "4. 识别文本中术语对照表里**未收录**的服装/纺织行业专业英文词汇，统一放入返回对象的 "
        '"unrecorded_terms" 数组（去重）。\n'
        "5. 【免翻译放行】如果某个 item 整体仅是单个英文字母（如 'D'、'C'）、纯数字、"
        "纯标点符号，或简单的型号/尺码代码，不需要翻译——但必须原样保留，"
        '"translated" 与 "original" 完全一致地原样输出，绝不能省略或留空这一项。例如：\n'
        '   {"original": "D", "translated": "D"}\n'
        '   {"original": "2025)", "translated": "2025)"}\n'
        "除上述放行情形外，\"translated\" 不允许为空、不允许与 original 相同——"
        "只要 item 里包含真正的英文单词/短语（哪怕很短），必须翻译成中文，不能套用本条逃避翻译。\n"
        "6. 必须返回与 <item> 数量完全一致的 items 数组，每个元素的 \"original\" 必须与对应 "
        "<item> 内容逐字一致。\n"
        "7. 只返回 JSON，不要任何多余文字或 markdown 代码块。\n\n"
        '返回格式：{"items": [{"original": "原文1", "translated": "译文1"}, '
        '{"original": "原文2", "translated": "译文2"}], "unrecorded_terms": ["term1"]}'
    )

    msg = _create_anthropic_message(
        client,
        model=ANTHROPIC_MODEL,
        max_tokens=4096,
        temperature=0,
        system=_GARMENT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = _response_output_text(msg)
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        raw = m.group()
    data = json.loads(raw)

    mapping: dict[str, str] = {}
    for item in data.get("items", []):
        orig = str(item.get("original", "")).strip()
        trans = str(item.get("translated", "")).strip()
        if orig:
            mapping[orig] = trans

    unrecorded = {t.strip() for t in data.get("unrecorded_terms", []) if t.strip()}
    mapping.update(passthrough)
    return mapping, unrecorded


def _is_retryable_batch_error(exc: Exception) -> bool:
    if isinstance(exc, (json.JSONDecodeError, KeyError, TypeError, ValueError)):
        return True
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True
    err_name = type(exc).__name__.lower()
    return "timeout" in err_name or "connection" in err_name


def translate_batch_resilient(
    client: anthropic.Anthropic,
    texts: list[str],
    glossary: dict,
) -> tuple[dict[str, str], set[str]]:
    """Translate a batch without letting malformed JSON kill the whole file.
    It tries the batch once, recursively splits on retryable failures, and
    finally preserves a single original text if even the fallback call fails.
    """
    texts = [t for t in texts if t.strip()]
    if not texts:
        return {}, set()
    try:
        return translate_batch(client, texts, glossary)
    except Exception as exc:
        if not _is_retryable_batch_error(exc):
            raise
    if len(texts) > 1:
        mid = max(1, len(texts) // 2)
        left_map, left_terms = translate_batch_resilient(client, texts[:mid], glossary)
        right_map, right_terms = translate_batch_resilient(client, texts[mid:], glossary)
        return {**left_map, **right_map}, left_terms | right_terms
    try:
        return {texts[0]: _force_translate(client, texts[0], glossary)}, set()
    except Exception as exc:
        print(f"单条批量翻译失败，保留原文继续：{texts[0][:80]}；错误：{exc}")
        return {texts[0]: texts[0]}, set()


_RE_SIZE_LABEL       = re.compile(r'^(?:X{0,3}[SML]|\d?XL|XXL|XXXL)$', re.IGNORECASE)
_RE_SINGLE_LETTER    = re.compile(r'^[A-Za-z]$')
_RE_PURE_DIGIT_PUNCT = re.compile(r'^[\d\W_]+$')           # digits/punctuation only, no letters
_RE_SIMPLE_CODE      = re.compile(r'^[A-Za-z0-9\-/]{1,8}$')  # short alnum model/size code
_RE_ALL_CAPS_ABBR    = re.compile(r'^[A-Z]{2,6}$')          # industry abbreviations: NOS, GSM, USA
_RE_NUM_UNIT         = re.compile(r'^\d+\s+[A-Z]{1,5}$')   # number + uppercase unit: 170 GSM


def _is_pure_size_code(t: str) -> bool:
    """True only for genuine size labels (S/M/L/XL/XXL/2XL...)."""
    return bool(_RE_SIZE_LABEL.match(t))


def _is_passthrough_eligible(t: str) -> bool:
    """True for content that is not translatable natural language: a lone
    letter, pure digits/punctuation, short alnum codes, all-caps industry
    abbreviations (NOS/GSM), or number+unit combos (170 GSM). These are
    allowed to come back unchanged from the LLM without triggering a
    force-retranslation retry. Ordinary short *words* like 'no'/'ab' are
    intentionally NOT eligible and must still be translated."""
    if _is_pure_size_code(t):
        return True
    if _RE_SINGLE_LETTER.match(t):
        return True
    if _RE_PURE_DIGIT_PUNCT.match(t):
        return True
    if _RE_SIMPLE_CODE.match(t) and bool(re.search(r'\d', t)):
        return True
    if _RE_ALL_CAPS_ABBR.match(t):
        return True
    if _RE_NUM_UNIT.match(t):
        return True
    return False


def _needs_retranslation(original: str, translated: str) -> bool:
    """Detect leftover untranslated English: identical to source, or still
    contains Latin letters with zero Chinese characters (excluding
    pass-through-eligible content, which is intentionally left as-is)."""
    o = original.strip()
    t = translated.strip()
    if not t:
        return True
    if o.lower() == t.lower():
        if _is_passthrough_eligible(o) or len(re.findall(r'[a-zA-Z]', o)) == 0:
            return False
        return True
    has_alpha = bool(re.search(r'[a-zA-Z]', t))
    has_cjk = bool(re.search(r'[一-鿿]', t))
    if has_alpha and not has_cjk:
        return True
    return False


def _force_translate(client: anthropic.Anthropic, text: str, glossary: dict) -> str:
    """Last-resort plain-text retry when translate_batch() returns an
    untranslated/empty/identical entry — used instead of silently keeping
    the English original."""
    if _is_color_item(text):
        return text
    rel = relevant_glossary(text, glossary)
    gloss_block = ""
    if rel:
        lines = "\n".join(f"  {k} → {v}" for k, v in rel.items())
        gloss_block = f"强制术语对照（务必照搬）：\n{lines}\n\n"
    prompt = (
        "上一次翻译时你返回了未翻译的英文原文，这是不允许的。\n"
        f"{gloss_block}"
        "<text_to_translate>\n"
        f"{text}\n"
        "</text_to_translate>\n\n"
        "你必须翻译 <text_to_translate> 标签内的所有内容，不允许遗漏任何一个单词。"
        "所有英文字符均需翻译为中文；如果遇到必须保留的型号/编码，请原样输出该编码本身，"
        "但不得以此为借口跳过其前后的英文单词，也必须原样保留文本中出现的所有数字。"
        "只返回中文翻译结果本身，不要输出 JSON、不要解释。"
    )
    msg = _create_anthropic_message(
        client,
        model=ANTHROPIC_MODEL,
        max_tokens=512,
        system=_GARMENT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return _response_output_text(msg)


def _insert_text(page, bbox, text: str, size: float, font_path: str):
    rect = fitz.Rect(bbox)
    # 浅黄底色覆盖英文（英文仍保留在 PDF 数据层，未被销毁）
    page.draw_rect(rect, color=None, fill=(1.0, 1.0, 0.75))
    # 深红色中文，便于审核时一眼识别译文
    for s in [size, size * 0.85, size * 0.7, 8.0, 7.0]:
        if page.insert_textbox(rect, text, fontname="myfont", fontfile=font_path,
                               fontsize=s, color=(0.7, 0, 0), align=0) >= 0:
            return
    page.insert_textbox(rect, text, fontname="myfont", fontfile=font_path,
                        fontsize=7.0, color=(0.7, 0, 0), align=0)


_WORKMANSHIP_KEYWORDS = {
    "details of design": 6,
    "workmanship": 6,
    "construction": 5,
    "sewing": 5,
    "sketch": 4,
    "stitching": 4,
    "double stitching": 5,
    "seam": 4,
    "shoulder seam": 5,
    "binding": 4,
    "shell fabric binding": 6,
    "snaps": 3,
    "ring snaps": 4,
    "printed label": 2,
    "rubber print": 4,
    "aop": 3,
    "waterprint": 4,
    "soft lacquer print": 5,
    "rib": 3,
    "ground color": 3,
    "quality": 2,
    "width": 2,
    "matching color": 3,
    "thread": 4,
    "shell": 2,
    "fabric": 3,
    "finish": 3,
    "handfeel": 3,
}

_NON_WORKMANSHIP_KEYWORDS = {
    "labels": 6,
    "label": 5,
    "labeling manual": 7,
    "care label": 8,
    "care labels": 8,
    "carelabel": 8,
    "carelabels": 8,
    "printing description": 8,
    "packaging": 6,
    "packing": 5,
    "plastic bag": 7,
    "warning": 7,
    "safety warning": 8,
    "suffocation": 8,
    "suffocation risk": 8,
    "juguete": 7,
    "brinquedo": 7,
    "asfixia": 8,
    "sufocação": 8,
    "sufocacao": 8,
    "bolsa de plástico": 7,
    "bolsa de plastico": 7,
    "saco de plástico": 7,
    "saco de plastico": 7,
    "barcode": 6,
    "ean": 6,
    "eans": 6,
    "washing instructions": 8,
    "wash instructions": 8,
    "wash": 3,
    "washing": 4,
    "sticker": 5,
    "hangtag": 5,
    "manual": 5,
    "size measurement": 4,
    "measurement table": 4,
}


def _keyword_hits(text: str, keywords: dict[str, int]) -> tuple[int, list[str]]:
    lowered = text.lower()
    hits = [
        kw for kw in keywords
        if re.search(rf"(?<![a-z0-9]){re.escape(kw)}(?![a-z0-9])", lowered)
    ]
    return sum(keywords[kw] for kw in hits), hits


def _short_label_score(lines: list[str]) -> int:
    useful = [
        ln for ln in lines
        if re.search(r"[a-zA-Z]", ln) and 2 <= len(ln.split()) <= 8 and len(ln) <= 70
    ]
    if len(useful) >= 18:
        return 3
    if len(useful) >= 10:
        return 2
    if len(useful) >= 5:
        return 1
    return 0


def _is_label_or_packaging_page(text: str, lines: list[str], image_count: int, positive_score: int) -> tuple[bool, str]:
    lowered = " ".join(lines).lower()
    forced_terms = [
        "care label",
        "care labels",
        "carelabel",
        "carelabels",
        "printing description",
        "washing instructions",
        "wash instructions",
        "laundry bag",
        "eans",
        "barcode",
        "etikett",
        "label 1",
        "label 2",
    ]
    for term in forced_terms:
        if term in lowered:
            return True, term

    if positive_score == 0 and (("front" in lowered and "back" in lowered) or ("vorderseite" in lowered and "rückseite" in lowered)):
        return True, "front/back layout"

    if positive_score == 0 and image_count >= 2 and len(lines) <= 18 and _short_label_score(lines) >= 2:
        return True, "多图少正文且短标注过多"

    return False, ""


def detect_workmanship_pages(pdf_bytes: bytes) -> list[dict]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    rows: list[dict] = []
    try:
        for idx, page in enumerate(doc):
            text = page.get_text("text") or ""
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            title_text = "\n".join(lines[:6])
            image_count = sum(1 for b in page.get_text("dict")["blocks"] if b["type"] == 1)
            pos_score, pos_hits = _keyword_hits(text, _WORKMANSHIP_KEYWORDS)
            neg_score, neg_hits = _keyword_hits(text, _NON_WORKMANSHIP_KEYWORDS)
            title_bonus, title_hits = _keyword_hits(title_text, {
                "details of design": 8,
                "workmanship": 8,
                "construction": 6,
                "sewing": 6,
                "sketch": 5,
            })
            label_score = _short_label_score(lines)
            forced_non_workmanship, forced_reason = _is_label_or_packaging_page(text, lines, image_count, pos_score + title_bonus)
            image_bonus = 1 if image_count >= 1 and (pos_score + title_bonus) > 0 else 0
            score = pos_score + title_bonus + label_score + image_bonus - neg_score
            if forced_non_workmanship:
                is_workmanship = False
            else:
                is_workmanship = score >= 3 and not (neg_score >= 7 and pos_score + title_bonus < 6)
            reason_bits = []
            if title_hits:
                reason_bits.append("标题命中：" + "、".join(title_hits))
            if pos_hits:
                reason_bits.append("做工词：" + "、".join(pos_hits[:8]))
            if label_score:
                reason_bits.append(f"短标注较多(+{label_score})")
            if image_bonus:
                reason_bits.append("含图片/图稿")
            if neg_hits:
                reason_bits.append("排除词：" + "、".join(neg_hits[:8]))
            if forced_non_workmanship:
                reason_bits.append(f"版式排除：{forced_reason}")
            rows.append({
                "page_index": idx,
                "page_number": idx + 1,
                "score": score,
                "is_workmanship": is_workmanship,
                "reason": "；".join(reason_bits) if reason_bits else "未命中明显做工特征",
            })
    finally:
        doc.close()
    return rows


def detect_workmanship_sheets(xlsx_bytes: bytes) -> list[dict]:
    wb = openpyxl.load_workbook(io.BytesIO(xlsx_bytes), read_only=True, data_only=True)
    rows: list[dict] = []
    try:
        for ws in wb.worksheets:
            values = []
            for row in ws.iter_rows(max_row=40, values_only=True):
                for value in row:
                    if isinstance(value, str) and value.strip():
                        values.append(value.strip())
            sample_text = f"{ws.title}\n" + "\n".join(values[:300])
            pos_score, pos_hits = _keyword_hits(sample_text, _WORKMANSHIP_KEYWORDS)
            neg_score, neg_hits = _keyword_hits(sample_text, _NON_WORKMANSHIP_KEYWORDS)
            title_bonus, title_hits = _keyword_hits(ws.title, {
                "f. técnica": 6,
                "f. tecnica": 6,
                "técnica": 5,
                "tecnica": 5,
                "workmanship": 7,
                "construction": 6,
                "details": 4,
                "measurements spec": 5,
                "spec": 2,
            })
            score = pos_score + title_bonus - neg_score
            if score >= 4:
                verdict = "做工"
            elif score <= -3:
                verdict = "非做工"
            else:
                verdict = "不确定"
            reason_bits = []
            if title_hits:
                reason_bits.append("Sheet 名命中：" + "、".join(title_hits))
            if pos_hits:
                reason_bits.append("做工词：" + "、".join(pos_hits[:8]))
            if neg_hits:
                reason_bits.append("排除词：" + "、".join(neg_hits[:8]))
            rows.append({
                "sheet_name": ws.title,
                "score": score,
                "verdict": verdict,
                "is_workmanship": verdict == "做工",
                "reason": "；".join(reason_bits) if reason_bits else "未命中明显做工特征",
            })
    finally:
        wb.close()
    return rows


def format_page_ranges(page_numbers: list[int]) -> str:
    nums = sorted(set(n for n in page_numbers if n > 0))
    if not nums:
        return ""
    ranges = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        ranges.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = n
    ranges.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


def parse_page_ranges(text: str, total_pages: int) -> tuple[list[int], list[str]]:
    pages: set[int] = set()
    errors: list[str] = []
    for part in re.split(r"[,\s，]+", text.strip()):
        if not part:
            continue
        if "-" in part:
            bits = part.split("-", 1)
            if len(bits) != 2 or not bits[0].isdigit() or not bits[1].isdigit():
                errors.append(f"无法识别页码范围：{part}")
                continue
            start, end = int(bits[0]), int(bits[1])
            if start > end:
                errors.append(f"页码范围起点大于终点：{part}")
                continue
            pages.update(range(start, end + 1))
        elif part.isdigit():
            pages.add(int(part))
        else:
            errors.append(f"无法识别页码：{part}")
    invalid = sorted(n for n in pages if n < 1 or n > total_pages)
    if invalid:
        errors.append(f"页码超出文件范围：{format_page_ranges(invalid)}")
    valid = sorted(n for n in pages if 1 <= n <= total_pages)
    return [n - 1 for n in valid], errors


def build_scope_report_xlsx(rows: list[dict], title: str = "翻译范围报告") -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = title[:31]
    headers = ["file_name", "source_type", "scope_mode", "item", "selected", "score", "reason"]
    ws.append(headers)
    for cell in ws[1]:
        cell.font = openpyxl.styles.Font(bold=True)
    for r in rows:
        ws.append([r.get(h, "") for h in headers])
    for col_letter, width in zip("ABCDEFG", [28, 12, 18, 20, 10, 10, 80]):
        ws.column_dimensions[col_letter].width = width
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def run_pdf_translation(
    pdf_bytes: bytes,
    glossary_bytes: bytes,
    font_path: str,
    api_key: str,
    on_page,
    on_block,
    on_progress,
    customer_id: str | None = None,
    source_file_name: str = "",
    created_by: str = "",
    selected_pages: list[int] | None = None,
    scope_mode: str = "all",
    scope_detection: list[dict] | None = None,
) -> tuple[bytes, bytes, int, dict]:
    """Returns (pdf_out, xlsx_out, n_unrecorded_terms, review_summary)."""
    glossary = load_glossary(glossary_bytes)
    client = OpenAI(api_key=api_key)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)
    selected_page_set = None if selected_pages is None else {
        pn for pn in selected_pages if 0 <= pn < total_pages
    }
    all_unrecorded: set[str] = set()
    candidate_contexts: list[dict] = []

    # 以 span 为最小单元预扫描：bbox 精确到每一段文字，数字 span 绝不擦除
    page_spans: list[list[dict]] = []
    for pn in range(total_pages):
        if selected_page_set is not None and pn not in selected_page_set:
            page_spans.append([])
            continue
        ts = []
        for blk in doc[pn].get_text("dict")["blocks"]:
            if blk["type"] != 0:
                continue
            for ln in blk.get("lines", []):
                for sp in ln.get("spans", []):
                    text = sp["text"].strip()
                    if not text:
                        continue
                    ts.append({
                        "bbox": sp["bbox"],
                        "text": sp["text"],
                        "size": sp["size"],
                    })
        page_spans.append(ts)

    total_spans = max(sum(len(s) for s in page_spans), 1)
    done = 0
    workmanship_by_page = {
        row["page_number"]: bool(row.get("is_workmanship"))
        for row in (scope_detection or detect_workmanship_pages(pdf_bytes))
        if row.get("page_number")
    }

    for pn, spans in enumerate(page_spans):
        if selected_page_set is not None and pn not in selected_page_set:
            continue
        page = doc[pn]
        on_page(pn, total_pages, len(spans))

        to_translate = []
        for sp in spans:
            _t = _clean_extracted_text(sp["text"])
            alpha     = len(re.findall(r'[a-zA-Z]', _t))
            has_digit = bool(re.search(r'\d', _t))

            # 纯数字/符号，或清洗后已无内容（PDF 解析伪影）——绝不擦除，直接跳过
            if not _t or alpha == 0:
                done += 1; on_progress(done / total_spans); continue
            if _is_color_item(_t):
                done += 1; on_progress(done / total_spans); continue
            # 页眉/页脚/创建日期/款号等 PDF 元数据不是正文，保持原样
            if _is_pdf_metadata_text(_t):
                done += 1; on_progress(done / total_spans); continue
            # 顶部样式信息/人员信息/字段名保持原文，不参与翻译
            if _is_header_or_metadata_text(_t, bbox=sp["bbox"], page_height=page.rect.height):
                done += 1; on_progress(done / total_spans); continue
            # 真正的尺码代号（S/M/L/XL/XXL...）——保持原样，不翻译
            if _is_pure_size_code(_t):
                done += 1; on_progress(done / total_spans); continue
            # 含数字且英文字母极少(≤2)：32A、70B 等尺码代号
            if has_digit and alpha <= 2:
                done += 1; on_progress(done / total_spans); continue

            to_translate.append({**sp, "clean_text": _t})

        results = []
        for batch in _chunk(to_translate, _MAX_BATCH_ITEMS):
            texts = [b["clean_text"] for b in batch]
            on_block(f"批量翻译 {len(texts)} 项…")
            try:
                mapping, unrecorded_batch = translate_batch_resilient(client, texts, glossary)
                for term in unrecorded_batch:
                    context_source = next(
                        (t for t in texts if normalize_term(term) in normalize_term(t)),
                        texts[0] if texts else "",
                    )
                    if not _is_workmanship_candidate_term(term, context_source):
                        continue
                    all_unrecorded.add(term)
                    candidate_contexts.append({
                        "term": term,
                        "context": context_source,
                        "page_or_sheet": f"第 {pn + 1} 页",
                        "cell_coordinate": "",
                        "source_type": "PDF",
                        "is_workmanship_source": workmanship_by_page.get(pn + 1, False),
                    })
            except Exception as exc:
                raise RuntimeError(f"批量翻译 API 调用失败：{exc}") from exc

            for sp in batch:
                for term in extract_candidate_terms_from_text(sp["clean_text"], glossary)[:5]:
                    candidate_contexts.append({
                        "term": term,
                        "context": sp["clean_text"],
                        "page_or_sheet": f"第 {pn + 1} 页",
                        "cell_coordinate": "",
                        "source_type": "PDF",
                        "is_workmanship_source": workmanship_by_page.get(pn + 1, False),
                    })
                translated = mapping.get(sp["clean_text"], "")
                if _needs_retranslation(sp["clean_text"], translated):
                    try:
                        translated = _force_translate(client, sp["clean_text"], glossary)
                    except Exception as exc:
                        print(f"文本重试翻译失败，保留原文继续：{sp['clean_text'][:80]}；错误：{exc}")
                        translated = sp["clean_text"]

                if not translated.strip():
                    translated = sp["clean_text"]

                results.append({**sp, "translated": translated})
                done += 1
                on_progress(done / total_spans)

        # 直接叠加译文，不销毁英文原文
        # 原样放行项（translated == original）：跳过整个叠加步骤——不画背景色块也不重绘文字。
        # 底层英文原文保持裸露可见，避免中文字体无法渲染 Latin 字符导致色块遮挡。
        for r in results:
            if r["translated"].strip().lower() == r["clean_text"].strip().lower():
                continue
            _insert_text(page, r["bbox"], r["translated"], r["size"], font_path)

    on_progress(1.0)

    pdf_buf = io.BytesIO()
    doc.save(pdf_buf, garbage=4, deflate=True)
    doc.close()

    xlsx_buf = io.BytesIO()
    if all_unrecorded:
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "未收录术语"
        ws.append(["未收录英文术语"])
        ws["A1"].font = openpyxl.styles.Font(bold=True)
        for term in sorted(all_unrecorded):
            ws.append([term])
        ws.column_dimensions["A"].width = 45

        wb.save(xlsx_buf)

    if customer_id and source_file_name and created_by:
        save_term_candidates(
            customer_id=customer_id,
            source_file_name=source_file_name,
            source_type="PDF",
            created_by=created_by,
            candidate_contexts=candidate_contexts,
            glossary=glossary,
            client=client,
        )

    _, candidate_stats = _merge_term_candidate_contexts(
        customer_id or "",
        candidate_contexts,
        glossary,
        source_type="PDF",
    )
    review_summary = _build_translation_review_summary(len(all_unrecorded), candidate_stats)
    return pdf_buf.getvalue(), xlsx_buf.getvalue(), len(all_unrecorded), review_summary


_RUNNING_JOB_THREADS: dict[str, threading.Thread] = {}
PDF_JOB_CANCELLED_ERROR = "USER_CANCELLED"


def create_translation_job(
    job_type: str,
    username: str,
    customer_id: str,
    source_file_name: str,
    input_bytes: bytes,
    aux_bytes: bytes | None,
    config: dict,
) -> str:
    job_id = str(uuid4())
    now = _now_iso()
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO translation_jobs (
                job_id, job_type, status, username, customer_id, source_file_name,
                progress, message, input_bytes, aux_bytes, config, created_at, updated_at
            )
            VALUES (?, ?, 'queued', ?, ?, ?, 0, '等待开始', ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                job_type,
                username,
                customer_id,
                source_file_name,
                input_bytes,
                aux_bytes,
                json.dumps(config, ensure_ascii=False),
                now,
                now,
            ),
        )
    return job_id


def update_translation_job(job_id: str, **fields) -> None:
    if not fields:
        return
    fields["updated_at"] = _now_iso()
    assignments = ", ".join(f"{key} = ?" for key in fields)
    values = list(fields.values()) + [job_id]
    with get_db_connection() as conn:
        conn.execute(
            f"UPDATE translation_jobs SET {assignments} WHERE job_id = ?",
            values,
        )


def is_translation_job_cancelled(job_id: str) -> bool:
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT status, error FROM translation_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
    return bool(row is None or (row["status"] == "failed" and row["error"] == PDF_JOB_CANCELLED_ERROR))


def raise_if_translation_job_cancelled(job_id: str) -> None:
    if is_translation_job_cancelled(job_id):
        raise RuntimeError(PDF_JOB_CANCELLED_ERROR)


def cancel_pdf_translation_jobs(username: str) -> dict:
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT job_id, status
            FROM translation_jobs
            WHERE username = ? AND job_type = 'PDF' AND status IN ('queued', 'running')
            """,
            (username,),
        ).fetchall()
        job_ids = [row["job_id"] for row in rows]
        if job_ids:
            placeholders = ",".join("?" for _ in job_ids)
            conn.execute(
                f"""
                DELETE FROM translation_jobs
                WHERE job_id IN ({placeholders})
                """,
                job_ids,
            )
    inactive_threads = [
        job_id for job_id, thread in _RUNNING_JOB_THREADS.items()
        if not thread.is_alive()
    ]
    for job_id in inactive_threads:
        _RUNNING_JOB_THREADS.pop(job_id, None)
    return {
        "cancelled_count": len(job_ids),
        "queued_count": sum(1 for row in rows if row["status"] == "queued"),
        "running_count": sum(1 for row in rows if row["status"] == "running"),
        "running_threads": [
            job_id for job_id in job_ids
            if job_id in _RUNNING_JOB_THREADS and _RUNNING_JOB_THREADS[job_id].is_alive()
        ],
    }


def list_translation_jobs(username: str, job_type: str = "PDF", limit: int = 20) -> list[dict]:
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT job_id, job_type, status, username, customer_id, source_file_name,
                   progress, message, error, result_meta, created_at, updated_at
            FROM translation_jobs
            WHERE username = ? AND job_type = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (username, job_type, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_translation_job(job_id: str, username: str) -> dict | None:
    with get_db_connection() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM translation_jobs
            WHERE job_id = ? AND username = ?
            """,
            (job_id, username),
        ).fetchone()
    return dict(row) if row else None


def delete_translation_job(job_id: str, username: str) -> None:
    with get_db_connection() as conn:
        conn.execute(
            "DELETE FROM translation_jobs WHERE job_id = ? AND username = ?",
            (job_id, username),
        )


def get_next_queued_pdf_job(username: str, exclude_job_id: str = "") -> dict | None:
    with get_db_connection() as conn:
        row = conn.execute(
            """
            SELECT job_id
            FROM translation_jobs
            WHERE username = ?
              AND job_type = 'PDF'
              AND status = 'queued'
              AND job_id <> ?
            ORDER BY created_at ASC
            LIMIT 1
            """,
            (username, exclude_job_id),
        ).fetchone()
    return dict(row) if row else None


def _pdf_scope_report_rows(file_name: str, scope_cfg: dict, selected_pages, scope_detection, scope_mode: str) -> list[dict]:
    selected_page_nums = [
        pn + 1 for pn in (selected_pages or [])
    ] if selected_pages is not None else list(range(1, int(scope_cfg.get("total_pages", 0)) + 1))
    detection_by_page = {
        row.get("page_number"): row for row in (scope_detection or [])
    }
    total_pages = int(scope_cfg.get("total_pages") or len(detection_by_page) or 0)
    rows = []
    if total_pages:
        selected_set = set(selected_page_nums)
        for page_number in range(1, total_pages + 1):
            det = detection_by_page.get(page_number, {})
            rows.append({
                "file_name": file_name,
                "source_type": "PDF",
                "scope_mode": scope_mode,
                "item": f"第 {page_number} 页",
                "selected": page_number in selected_set,
                "score": det.get("score", ""),
                "reason": det.get("reason", ""),
            })
    return rows


def _run_pdf_translation_job(job_id: str, api_key: str, start_next_on_finish: bool = False) -> None:
    job = None
    try:
        with get_db_connection() as conn:
            row = conn.execute("SELECT * FROM translation_jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return
        raise_if_translation_job_cancelled(job_id)
        job = dict(row)
        config = json.loads(job.get("config") or "{}")
        update_translation_job(job_id, status="running", progress=0.01, message="正在准备翻译")
        last_progress_write = {"ts": 0.0, "value": 0.0}

        font_bytes = job.get("aux_bytes")
        with tempfile.TemporaryDirectory() as tmpdir:
            if font_bytes:
                font_path = os.path.join(tmpdir, "font.ttf")
                with open(font_path, "wb") as f:
                    f.write(font_bytes)
            else:
                font_path = str(DEFAULT_FONT)

            glossary_bytes = get_customer_glossary_bytes_for_translation(
                {"username": job["username"], "role": "company_admin"},
                job["customer_id"],
            )

            def on_page(pn, total, n_blocks):
                raise_if_translation_job_cancelled(job_id)
                update_translation_job(
                    job_id,
                    message=f"第 {pn + 1}/{total} 页，{n_blocks} 个文本块",
                )

            def on_block(preview):
                raise_if_translation_job_cancelled(job_id)
                update_translation_job(job_id, message=str(preview)[:180])

            def on_progress(frac):
                raise_if_translation_job_cancelled(job_id)
                progress = max(0.0, min(float(frac), 1.0))
                now = time.monotonic()
                if (
                    progress >= 1.0
                    or progress - last_progress_write["value"] >= 0.01
                    or now - last_progress_write["ts"] >= 1.5
                ):
                    last_progress_write["ts"] = now
                    last_progress_write["value"] = progress
                    update_translation_job(job_id, progress=progress)

            selected_pages = config.get("selected_pages")
            scope_detection = config.get("scope_detection") or []
            scope_cfg = config.get("scope_cfg") or {}
            scope_mode = config.get("scope_mode") or "all"
            pdf_out, xlsx_out, n_terms, review_summary = run_pdf_translation(
                pdf_bytes=job["input_bytes"],
                glossary_bytes=glossary_bytes,
                font_path=font_path,
                api_key=api_key,
                on_page=on_page,
                on_block=on_block,
                on_progress=on_progress,
                customer_id=job["customer_id"],
                source_file_name=job["source_file_name"],
                created_by=job["username"],
                selected_pages=selected_pages,
                scope_mode=scope_mode,
                scope_detection=scope_detection,
            )
            raise_if_translation_job_cancelled(job_id)
            scope_report = _pdf_scope_report_rows(
                job["source_file_name"],
                scope_cfg,
                selected_pages,
                scope_detection,
                scope_mode,
            )
            scope_report_bytes = build_scope_report_xlsx(scope_report) if scope_report else b""
            report_bytes = b""
            report_kind = ""
            if xlsx_out and scope_report_bytes:
                report_zip = io.BytesIO()
                with zipfile.ZipFile(report_zip, "w") as zf:
                    zf.writestr("unrecorded_terms.xlsx", xlsx_out)
                    zf.writestr("scope_report.xlsx", scope_report_bytes)
                report_bytes = report_zip.getvalue()
                report_kind = "zip"
            elif xlsx_out:
                report_bytes = xlsx_out
                report_kind = "unrecorded"
            elif scope_report_bytes:
                report_bytes = scope_report_bytes
                report_kind = "scope"
            update_translation_job(
                job_id,
                status="complete",
                progress=1.0,
                message="翻译完成",
                result_file=pdf_out,
                result_report=report_bytes,
                result_meta=json.dumps(
                    {
                        "n_terms": n_terms,
                        "has_scope_report": bool(scope_report),
                        "has_unrecorded_terms": bool(xlsx_out),
                        "report_kind": report_kind,
                        "review_summary": review_summary,
                    },
                    ensure_ascii=False,
                ),
            )
    except Exception as exc:
        if str(exc) == PDF_JOB_CANCELLED_ERROR:
            update_translation_job(
                job_id,
                status="failed",
                progress=0,
                error=PDF_JOB_CANCELLED_ERROR,
                message="用户已取消，可重新上传文件开始翻译",
            )
        else:
            update_translation_job(
                job_id,
                status="failed",
                error=str(exc),
                message="翻译失败",
            )
    finally:
        _RUNNING_JOB_THREADS.pop(job_id, None)
        if start_next_on_finish and job and not is_translation_job_cancelled(job_id):
            next_job = get_next_queued_pdf_job(job["username"], exclude_job_id=job_id)
            if next_job:
                start_pdf_translation_job(
                    next_job["job_id"],
                    api_key,
                    start_next_on_finish=True,
                )


def start_pdf_translation_job(job_id: str, api_key: str, start_next_on_finish: bool = False) -> bool:
    if job_id in _RUNNING_JOB_THREADS and _RUNNING_JOB_THREADS[job_id].is_alive():
        return False
    thread = threading.Thread(
        target=_run_pdf_translation_job,
        args=(job_id, api_key, start_next_on_finish),
        daemon=True,
    )
    _RUNNING_JOB_THREADS[job_id] = thread
    thread.start()
    return True


@st.fragment(run_every=8)
def render_pdf_jobs_panel(current_user: dict, api_key: str) -> None:
    pdf_jobs = list_translation_jobs(current_user["username"], "PDF") if current_user else []
    cancel_notice = st.session_state.pop("pdf_cancel_notice", "")
    if cancel_notice:
        st.info(cancel_notice)
    if pdf_jobs and api_key:
        has_active_pdf_thread = any(
            thread.is_alive() and not is_translation_job_cancelled(job_id)
            for job_id, thread in _RUNNING_JOB_THREADS.items()
        )
        has_running_pdf_job = any(job["status"] == "running" for job in pdf_jobs)
        queued_pdf_job = next((job for job in reversed(pdf_jobs) if job["status"] == "queued"), None)
        if queued_pdf_job and not has_active_pdf_thread and not has_running_pdf_job:
            start_pdf_translation_job(
                queued_pdf_job["job_id"],
                api_key,
                start_next_on_finish=True,
            )
            st.rerun(scope="fragment")
    if not pdf_jobs:
        return

    st.divider()
    st.subheader("PDF 后台任务")
    refresh_col, cancel_col = st.columns(2)
    with refresh_col:
        if st.button("刷新任务状态", use_container_width=True, key="refresh_pdf_jobs_btn"):
            st.rerun(scope="fragment")
    with cancel_col:
        if st.button("取消等待/翻译中的 PDF", use_container_width=True, key="cancel_running_pdf_jobs_btn"):
            cancel_result = cancel_pdf_translation_jobs(current_user["username"])
            if cancel_result["cancelled_count"]:
                st.session_state["pdf_cancel_notice"] = f"已取消 {cancel_result['cancelled_count']} 个 PDF 任务。可以重新上传文件开始翻译。"
            else:
                st.session_state["pdf_cancel_notice"] = "当前没有等待中或翻译中的 PDF 任务。"
            st.rerun(scope="fragment")
    for job in pdf_jobs:
        label = {
            "queued": "等待中",
            "running": "翻译中",
            "complete": "已完成",
            "failed": "失败",
        }.get(job["status"], job["status"])
        with st.expander(f"{label} · {job['source_file_name']} · {job['updated_at']}", expanded=job["status"] in {"running", "failed"}):
            st.progress(float(job.get("progress") or 0), text=job.get("message") or label)
            if job.get("error") and job.get("error") != PDF_JOB_CANCELLED_ERROR:
                st.error(job["error"])
            if job.get("error") == PDF_JOB_CANCELLED_ERROR:
                st.info("该任务已取消。可以重新上传文件开始翻译。")
            if job["status"] == "failed" and job.get("error") != PDF_JOB_CANCELLED_ERROR:
                delete_translation_job(job["job_id"], current_user["username"])
            if job["status"] == "running":
                st.caption("如果进度长时间不动，可以重新启动这个后台任务。")
                if st.button(
                    "重新启动任务",
                    use_container_width=True,
                    key=f"restart_pdf_job_{job['job_id']}",
                ):
                    restarted = start_pdf_translation_job(
                        job["job_id"],
                        api_key,
                        start_next_on_finish=True,
                    )
                    if restarted:
                        st.success("已重新启动任务。")
                    else:
                        st.info("任务线程仍在运行，请稍后刷新查看。")
                    st.rerun(scope="fragment")
            if job["status"] == "complete":
                full_job = get_translation_job(job["job_id"], current_user["username"])
                if full_job:
                    meta = json.loads(full_job.get("result_meta") or "{}")
                    base = full_job["source_file_name"].rsplit(".", 1)[0]
                    review_summary = meta.get("review_summary") or {}
                    if review_summary.get("summary_text"):
                        st.info(review_summary["summary_text"])
                    else:
                        st.caption(f"未收录术语：{meta.get('n_terms', 0)} 条")
                    if review_summary.get("needs_manual_review") and review_summary.get("review_locations"):
                        st.caption(f"重点核查位置：{'、'.join(review_summary['review_locations'][:3])}")
                    dl_cols = st.columns(2)
                    with dl_cols[0]:
                        st.download_button(
                            "⬇️ 下载中文 PDF",
                            data=full_job["result_file"],
                            file_name=f"{base}_translated.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            key=f"job_pdf_dl_{job['job_id']}",
                        )
                    with dl_cols[1]:
                        if full_job.get("result_report"):
                            report_kind = meta.get("report_kind") or "xlsx"
                            report_ext = "zip" if report_kind == "zip" else "xlsx"
                            report_mime = "application/zip" if report_kind == "zip" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            st.download_button(
                                "⬇️ 下载报告",
                                data=full_job["result_report"],
                                file_name=f"{base}_report.{report_ext}",
                                mime=report_mime,
                                use_container_width=True,
                                key=f"job_pdf_report_dl_{job['job_id']}",
                            )


# ── Excel translation ──────────────────────────────────────────────────────────

def _is_translatable(val) -> bool:
    if not isinstance(val, str):
        return False
    s = val.strip()
    if not s:
        return False
    if s.startswith("="):
        return False
    return bool(re.search(r"[a-zA-Z]", s))


def _is_excel_header_or_metadata_cell(ws, cell) -> bool:
    text = str(cell.value or "").strip()
    if not text:
        return False
    if _is_header_or_metadata_text(text):
        return True
    if cell.row <= 10 and _looks_like_top_metadata_value(text):
        return True
    return False


def translate_cell_text(client: anthropic.Anthropic, text: str, glossary: dict) -> str:
    if _is_color_item(text):
        return text
    rel = relevant_glossary(text, glossary)
    gloss_block = ""
    if rel:
        lines = "\n".join(f"  {k} → {v}" for k, v in rel.items())
        gloss_block = f"强制术语对照（务必照搬）：\n{lines}\n\n"

    prompt = (
        "你是一名专业服装行业翻译，请翻译 <text_to_translate> 标签内的全部内容，"
        "不允许遗漏任何一个单词。\n\n"
        f"{gloss_block}"
        f"<text_to_translate>\n{text}\n</text_to_translate>\n\n"
        "规则：\n"
        "1. 对照表中词汇的中文译法必须出现在译文里，但要结合整句语境自然融入，"
        "禁止逐词直译后生硬拼接。\n"
        "2. 所有英文字符均需翻译为中文。如果遇到必须保留的型号/货号/编码，"
        "请将该编码本身原样输出，但不得以此为借口跳过编码前后的英文单词。\n"
        "3. 当英文表述简略或存在多义性时，结合服装款式与工艺常识进行合理推断，"
        "给出业内真正想表达的工艺含义，不要停留在字面直译。\n"
        "4. 除纯数字、型号、尺码代号外，所有英文单词/短语/指示句都必须翻译成中文，"
        "禁止因为词很短或是操作指示句就原样保留英文，不允许返回结果与原文相同。\n"
        "5. 只返回翻译结果，不要任何解释或多余文字。"
    )
    msg = _create_anthropic_message(
        client,
        model=ANTHROPIC_MODEL,
        max_tokens=512,
        system=_GARMENT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return _response_output_text(msg)


def _cell_merge_anchor(ws, cell) -> tuple[bool, bool]:
    """返回 (is_in_merged_range, is_anchor_cell)。
    非主单元格的 .value 在 openpyxl 里恒为 None，本来就不会被当作可翻译文本，
    这里只是为了在报告里明确标注，不改变跳过逻辑。"""
    for rng in ws.merged_cells.ranges:
        if cell.coordinate in rng:
            anchor = f"{rng.min_row}_{rng.min_col}"
            is_anchor = (cell.row == rng.min_row and cell.column == rng.min_col)
            return True, is_anchor
    return False, False


def _estimate_layout_warning(ws, cell, translated: str) -> bool:
    """粗略估算：译文字符数是否明显超出该列宽能容纳的字符数。
    中文字符按 1.8 倍英文字符宽度估算（粗略经验值，不追求精确）。"""
    col_letter = cell.column_letter
    dim = ws.column_dimensions.get(col_letter)
    width = dim.width if (dim and dim.width) else 8.43  # Excel 默认列宽
    capacity_chars = width / 1.2  # 默认字体下，约每字符占 1.2 个列宽单位
    cjk_weight = sum(1.8 if ord(ch) > 0x2E80 else 1.0 for ch in translated)
    return cjk_weight > capacity_chars * 1.1  # 留 10% 容差，避免假警报太多


def run_excel_translation(
    xlsx_bytes: bytes,
    glossary_bytes: bytes,
    api_key: str,
    on_cell,
    on_progress,
    translate_images: bool = False,
    customer_id: str | None = None,
    source_file_name: str = "",
    created_by: str = "",
    selected_sheets: list[str] | None = None,
    scope_mode: str = "all",
    scope_detection: list[dict] | None = None,
) -> tuple[bytes, int, int, list[dict], dict]:
    """Translate text cells (in place, formulas untouched) across ALL sheets.
    Embedded-image text translation is OFF by default (translate_images=False) —
    it rasterizes Chinese into image pixels, which is not editable and uses
    vision-model-estimated coordinates that can drift from the real layout.
    Returns (xlsx_out, n_cells_translated, n_images_translated, report_rows).
    """
    glossary = load_glossary(glossary_bytes)
    client = OpenAI(api_key=api_key)

    # ── 阶段 1：翻译文字单元格 ──────────────────────────────────────────────────
    # 不用 data_only=True：公式必须保留原样（"="开头的字符串），否则保存时
    # 公式会被永久替换成当时的计算结果，且无法恢复。
    wb = openpyxl.load_workbook(io.BytesIO(xlsx_bytes))
    selected_sheet_set = None if selected_sheets is None else set(selected_sheets)
    skipped_sheets = [
        ws.title for ws in wb.worksheets
        if selected_sheet_set is not None and ws.title not in selected_sheet_set
    ]

    to_translate = []
    for ws in wb.worksheets:
        if selected_sheet_set is not None and ws.title not in selected_sheet_set:
            continue
        for row in ws.iter_rows():
            for cell in row:
                if not _is_translatable(cell.value):
                    continue
                text = str(cell.value)
                if _is_color_item(text) or _is_excel_header_or_metadata_cell(ws, cell):
                    continue
                to_translate.append((ws.title, ws, cell))
    n_cells = len(to_translate)
    total_steps = max(n_cells, 1)
    report_rows: list[dict] = []
    candidate_contexts: list[dict] = []
    detection_by_sheet = {
        row.get("sheet_name"): row
        for row in (scope_detection or [])
        if row.get("sheet_name")
    }
    if not detection_by_sheet:
        detection_by_sheet = {
            row.get("sheet_name"): row
            for row in detect_workmanship_sheets(xlsx_bytes)
            if row.get("sheet_name")
        }

    for sheet_name in skipped_sheets:
        det = detection_by_sheet.get(sheet_name, {})
        report_rows.append({
            "sheet_name": sheet_name,
            "cell_coordinate": "",
            "original_text": "",
            "translated_text": "",
            "status": "skipped_sheet",
            "skip_reason": "未包含在本次翻译范围",
            "is_merged_cell": False,
            "layout_warning": False,
            "scope_mode": scope_mode,
            "selected_sheets": ", ".join(selected_sheets or []),
            "skipped_sheets": ", ".join(skipped_sheets),
            "detection_score": det.get("score", ""),
            "detection_reason": det.get("reason", ""),
        })

    for i, (sheet_name, ws, cell) in enumerate(to_translate):
        on_cell(f"[{sheet_name}] {str(cell.value)[:50].replace(chr(10), ' ')}")
        original = str(cell.value)
        is_merged, is_anchor = _cell_merge_anchor(ws, cell)
        row_report = {
            "sheet_name": sheet_name,
            "cell_coordinate": cell.coordinate,
            "original_text": original,
            "translated_text": "",
            "status": "ok",
            "skip_reason": "",
            "is_merged_cell": is_merged,
            "layout_warning": False,
            "scope_mode": scope_mode,
            "selected_sheets": ", ".join(selected_sheets or []),
            "skipped_sheets": ", ".join(skipped_sheets),
            "detection_score": detection_by_sheet.get(sheet_name, {}).get("score", ""),
            "detection_reason": detection_by_sheet.get(sheet_name, {}).get("reason", ""),
        }

        try:
            translated = translate_cell_text(client, original, glossary)
            if _needs_retranslation(original, translated):
                translated = _force_translate(client, original, glossary)
            cell.value = translated
            row_report["translated_text"] = translated

            # 译文过长：保留原对齐方式的其余设置，只追加开启自动换行；
            # 不动列宽，不写入相邻单元格。仍放不下则只记录 layout_warning。
            if _estimate_layout_warning(ws, cell, translated):
                row_report["layout_warning"] = True
                align = cell.alignment
                cell.alignment = openpyxl.styles.Alignment(
                    horizontal=align.horizontal, vertical=align.vertical,
                    wrap_text=True, text_rotation=align.text_rotation,
                    indent=align.indent,
                )
            for term in extract_candidate_terms_from_text(original, glossary)[:8]:
                candidate_contexts.append({
                    "term": term,
                    "context": original,
                    "page_or_sheet": sheet_name,
                    "cell_coordinate": cell.coordinate,
                    "source_type": "Excel",
                    "is_workmanship_source": bool(detection_by_sheet.get(sheet_name, {}).get("is_workmanship")),
                })
        except Exception as e:
            row_report["status"] = "error"
            row_report["skip_reason"] = str(e)

        report_rows.append(row_report)
        on_progress((i + 1) / total_steps * (0.6 if translate_images else 1.0))

    buf = io.BytesIO()
    wb.save(buf)
    text_done_bytes = buf.getvalue()

    # ── 阶段 2：翻译嵌入图片（默认关闭）────────────────────────────────────────
    # 关闭原因：会把中文像素烧录进图片本身（不可编辑），且依赖视觉模型估算的
    # 坐标（容易跟实际版式错位/串位）。保留函数实现，按需再开启。
    if not translate_images:
        if customer_id and source_file_name and created_by:
            save_term_candidates(
                customer_id=customer_id,
                source_file_name=source_file_name,
                source_type="Excel",
                created_by=created_by,
                candidate_contexts=candidate_contexts,
                glossary=glossary,
                client=client,
            )
        on_progress(1.0)
        _, candidate_stats = _merge_term_candidate_contexts(
            customer_id or "",
            candidate_contexts,
            glossary,
            source_type="Excel",
        )
        review_summary = _build_translation_review_summary(0, candidate_stats)
        return text_done_bytes, n_cells, 0, report_rows, review_summary

    with zipfile.ZipFile(io.BytesIO(text_done_bytes)) as zf:
        n_images = sum(
            1 for n in zf.namelist()
            if n.startswith("xl/media/") and n.rsplit(".", 1)[-1].lower() in _IMAGE_EXTS
        )

    if n_images == 0:
        on_progress(1.0)
        _, candidate_stats = _merge_term_candidate_contexts(
            customer_id or "",
            candidate_contexts,
            glossary,
            source_type="Excel",
        )
        review_summary = _build_translation_review_summary(0, candidate_stats)
        return text_done_bytes, n_cells, 0, report_rows, review_summary

    def on_image(idx, total, fname):
        on_cell(f"图片 {idx}/{total}：{fname}")
        on_progress(0.6 + idx / total * 0.4)   # 后 40% 进度给图片翻译

    final_bytes = translate_images_in_excel(text_done_bytes, client, glossary, on_image)
    if customer_id and source_file_name and created_by:
        save_term_candidates(
            customer_id=customer_id,
            source_file_name=source_file_name,
            source_type="Excel",
            created_by=created_by,
            candidate_contexts=candidate_contexts,
            glossary=glossary,
            client=client,
        )
    on_progress(1.0)
    _, candidate_stats = _merge_term_candidate_contexts(
        customer_id or "",
        candidate_contexts,
        glossary,
        source_type="Excel",
    )
    review_summary = _build_translation_review_summary(0, candidate_stats)
    return final_bytes, n_cells, n_images, report_rows, review_summary


# ── Excel image translation ───────────────────────────────────────────────────

_IMAGE_EXTS = {"jpg", "jpeg", "png", "webp"}
_MIME = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}


def _vision_extract_items(
    client: anthropic.Anthropic,
    img_bytes: bytes,
    ext: str,
    glossary: dict,
) -> list[dict]:
    """让 OpenAI Vision 识别图片里的英文标注，返回
    [{"en":..., "zh":..., "x":0~1, "y":0~1, "size":int}, ...]（大致坐标，非精确测量）。
    失败或无文字时返回 []。"""
    media_type = _MIME.get(ext)
    if not media_type:
        return []

    b64 = base64.b64encode(img_bytes).decode("utf-8")

    gloss_hint = ""
    if glossary:
        sample = "\n".join(f"  {k} → {v}" for k, v in list(glossary.items())[:15])
        gloss_hint = f"\n参考术语对照（部分）：\n{sample}"

    try:
        resp = _create_anthropic_message(
            client,
            model=ANTHROPIC_MODEL,
            max_tokens=2048,
            system=_GARMENT_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                    {"type": "text", "text": (
                        f"这是一张服装工艺/规格图，图中有英文标注。{gloss_hint}\n\n"
                        "请识别图中全部英文文字，翻译成中文，并给出每处文字在图片中的大致坐标。\n"
                        "颜色、色号、Pantone/PMS/TCX/TPX/TC/TP、色卡、色样、shade/tone/color/colour 相关文字不要翻译，"
                        "zh 请返回空字符串。\n"
                        "坐标规则：左上角为(0,0)，右下角为(1,1)，用0~1之间的小数表示。\n"
                        "同时估计该处文字的字号（像素大小）。\n\n"
                        "只返回JSON，格式：\n"
                        '{"items": [{"en": "front strap", "zh": "前肩带", "x": 0.25, "y": 0.18, "size": 11}]}\n'
                        "如图中无英文则返回 {\"items\": []}。"
                    )}
                ],
            }],
        )
        raw = _response_output_text(resp)
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            raw = m.group()
        return json.loads(raw).get("items", [])
    except Exception:
        return []


def _translate_image_bytes(
    client: anthropic.Anthropic,
    img_bytes: bytes,
    ext: str,
    glossary: dict,
) -> bytes:
    """Send one image to OpenAI Vision, overlay Chinese translations, return new bytes."""
    items = _vision_extract_items(client, img_bytes, ext, glossary)
    if not items or _should_skip_color_image_items(items):
        return img_bytes

    # Overlay Chinese text with PIL
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    draw = ImageDraw.Draw(img)
    iw, ih = img.size

    font_path = str(DEFAULT_FONT) if DEFAULT_FONT.exists() else None

    for item in items:
        zh = str(item.get("zh", "")).strip()
        en = str(item.get("en", "")).strip()
        if not zh or _should_preserve_color_text(en, zh):
            continue
        x = int(float(item.get("x", 0.5)) * iw)
        y = int(float(item.get("y", 0.5)) * ih)
        size = max(8, min(int(item.get("size", 12)), 28))

        try:
            font = ImageFont.truetype(font_path, size=size) if font_path else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        bbox = draw.textbbox((x, y), zh, font=font)
        pad = 2
        # 浅黄半透明底色（英文像素仍在图层下方，未被完全覆盖）
        draw.rectangle(
            [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
            fill=(255, 255, 180, 200),
        )
        # 深红色中文，与 PDF 审核模式保持一致
        draw.text((x, y), zh, fill=(180, 0, 0, 255), font=font)

    out = io.BytesIO()
    if ext in ("jpg", "jpeg"):
        img.convert("RGB").save(out, format="JPEG", quality=95)
    else:
        img.save(out, format="PNG")
    return out.getvalue()


def translate_images_in_excel(
    xlsx_bytes: bytes,
    client: anthropic.Anthropic,
    glossary: dict,
    on_image,   # on_image(idx, total, filename)
) -> bytes:
    """Extract all embedded images from xlsx, translate labels, repack."""
    with zipfile.ZipFile(io.BytesIO(xlsx_bytes)) as zf:
        infos = {info.filename: info for info in zf.infolist()}
        all_data = {name: zf.read(name) for name in zf.namelist()}

    media = [
        n for n in all_data
        if n.startswith("xl/media/") and n.rsplit(".", 1)[-1].lower() in _IMAGE_EXTS
    ]

    for idx, name in enumerate(media):
        ext = name.rsplit(".", 1)[-1].lower()
        on_image(idx + 1, len(media), name.split("/")[-1])
        all_data[name] = _translate_image_bytes(client, all_data[name], ext, glossary)

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zf_out:
        for name, data in all_data.items():
            zf_out.writestr(name, data, compress_type=infos[name].compress_type)
    return out.getvalue()


# ── Excel 图片译文：可编辑 TextBox 方案（替代 PIL 烧录像素） ──────────────────────
# 原图片像素完全不动；每条译文生成一个真正的 Excel TextBox（<xdr:sp>），
# 用户可以在 Excel/WPS 里直接点选、双击编辑、拖动——已用真实文件验证过可行。

_XDR_NS = "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing"
_A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
_R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_SML_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_EMU_PER_PT = 12700
_MDW = 7  # 默认字体(Calibri 11)下数字字符的像素宽度，列宽换算用，足够定位精度

ET.register_namespace("xdr", _XDR_NS)
ET.register_namespace("a", _A_NS)
ET.register_namespace("r", _R_NS)


def _drawing_to_sheet_name(all_data: dict) -> dict:
    """drawingN.xml 路径 -> 所在 worksheet 名称（要靠它的列宽/行高换算偏移量）。"""
    wb_xml = ET.fromstring(all_data["xl/workbook.xml"])
    sheets_el = wb_xml.find(f"{{{_SML_NS}}}sheets")
    rid_to_name = {
        sheet_el.get(f"{{{_R_NS}}}id"): sheet_el.get("name")
        for sheet_el in (sheets_el if sheets_el is not None else [])
    }

    wb_rels = ET.fromstring(all_data["xl/_rels/workbook.xml.rels"])
    rid_to_target = {rel.get("Id"): rel.get("Target") for rel in wb_rels}

    result = {}
    for rid, name in rid_to_name.items():
        target = rid_to_target.get(rid)
        if not target:
            continue
        if target.startswith("/"):
            sheet_path = target.lstrip("/")
        else:
            sheet_path = target if target.startswith("xl/") else f"xl/{target}"
        base_dir, fname = sheet_path.rsplit("/", 1)
        rels_data = all_data.get(f"{base_dir}/_rels/{fname}.rels")
        if not rels_data:
            continue
        for rel in ET.fromstring(rels_data):
            if rel.get("Type", "").endswith("/drawing"):
                result[_resolve_media_path(sheet_path, rel.get("Target"))] = name
    return result


def _col_width_emu(ws, col_idx_1based: int) -> int:
    from openpyxl.utils import get_column_letter
    from openpyxl.utils.units import pixels_to_EMU, DEFAULT_COLUMN_WIDTH
    dim = ws.column_dimensions.get(get_column_letter(col_idx_1based))
    width = dim.width if (dim and dim.width) else DEFAULT_COLUMN_WIDTH
    pixels = int(((256 * width + int(128 / _MDW)) / 256) * _MDW)
    return pixels_to_EMU(max(pixels, 1))


def _row_height_emu(ws, row_idx_1based: int) -> int:
    from openpyxl.utils.units import DEFAULT_ROW_HEIGHT
    dim = ws.row_dimensions.get(row_idx_1based)
    height_pt = dim.height if (dim and dim.height) else DEFAULT_ROW_HEIGHT
    return int(height_pt * _EMU_PER_PT)


def _carry_col_offset(ws, start_col_0based: int, offset_emu: int) -> tuple[int, int]:
    """把"从 start_col 起、偏移 offset_emu"换算成真正落在哪一列、列内偏移多少，
    避免单列塞进一个远超其宽度的 offset（很多阅读器会把这种情况渲染错位/堆叠）。"""
    col, off = start_col_0based, max(offset_emu, 0)
    for _ in range(500):
        w = _col_width_emu(ws, col + 1)
        if off < w:
            return col, off
        off -= w
        col += 1
    return col, off


def _carry_row_offset(ws, start_row_0based: int, offset_emu: int) -> tuple[int, int]:
    row, off = start_row_0based, max(offset_emu, 0)
    for _ in range(2000):
        h = _row_height_emu(ws, row + 1)
        if off < h:
            return row, off
        off -= h
        row += 1
    return row, off


def _drawing_rels_map(all_data: dict, drawing_path: str) -> dict:
    """drawingN.xml 里的 rId -> 关系目标路径（如 "../media/image1.png"）。"""
    base_dir, fname = drawing_path.rsplit("/", 1)
    rels_path = f"{base_dir}/_rels/{fname}.rels"
    data = all_data.get(rels_path)
    if not data:
        return {}
    root = ET.fromstring(data)
    return {
        rel.get("Id"): rel.get("Target")
        for rel in root
        if rel.get("Id") and rel.get("Target")
    }


def _resolve_media_path(drawing_path: str, target: str) -> str:
    """把 .rels 里的路径解析成 zip 内的绝对路径。target 可能是相对路径
    （如 "../media/image1.png"，原始 Excel/WPS 常见写法），也可能是包内绝对路径
    （如 "/xl/media/image1.png"，openpyxl 重新保存后常见写法）——两种都要处理。"""
    if target.startswith("/"):
        return target.lstrip("/")
    parts = drawing_path.rsplit("/", 1)[0].split("/")
    for seg in target.split("/"):
        if seg == "..":
            parts.pop()
        elif seg != ".":
            parts.append(seg)
    return "/".join(parts)


def _find_top_level_pics(drawing_root) -> list[tuple]:
    """返回 [(anchor_el, pic_el)]。只取直接挂在 anchor 下的 pic，
    不进 <xdr:grpSp> 分组（分组形状结构更复杂，本轮先跳过、在报告里标注）。"""
    results = []
    for anchor in list(drawing_root):
        tag = anchor.tag.rsplit("}", 1)[-1]
        if tag not in ("twoCellAnchor", "oneCellAnchor"):
            continue
        pic = anchor.find(f"{{{_XDR_NS}}}pic")
        if pic is not None:
            results.append((anchor, pic))
    return results


def _anchor_from(anchor_el) -> tuple[int, int, int, int]:
    frm = anchor_el.find(f"{{{_XDR_NS}}}from")
    return (
        int(frm.find(f"{{{_XDR_NS}}}col").text),
        int(frm.find(f"{{{_XDR_NS}}}colOff").text),
        int(frm.find(f"{{{_XDR_NS}}}row").text),
        int(frm.find(f"{{{_XDR_NS}}}rowOff").text),
    )


def _pic_extent(pic_el):
    """从 <xdr:pic>/<xdr:spPr>/<a:xfrm> 读取图片的局部偏移+宽高（EMU）。
    缺失则返回 None（无法据此换算坐标，调用方应跳过并记录原因）。"""
    xfrm = pic_el.find(f"{{{_XDR_NS}}}spPr/{{{_A_NS}}}xfrm")
    if xfrm is None:
        return None
    ext = xfrm.find(f"{{{_A_NS}}}ext")
    off = xfrm.find(f"{{{_A_NS}}}off")
    if ext is None:
        return None
    cx, cy = int(ext.get("cx")), int(ext.get("cy"))
    ox = int(off.get("x")) if off is not None else 0
    oy = int(off.get("y")) if off is not None else 0
    return ox, oy, cx, cy


def _pic_embed_rid(pic_el):
    blip = pic_el.find(f"{{{_XDR_NS}}}blipFill/{{{_A_NS}}}blip")
    return blip.get(f"{{{_R_NS}}}embed") if blip is not None else None


def _max_shape_id_in_drawing(root) -> int:
    max_id = 0
    for cNvPr in root.iter(f"{{{_XDR_NS}}}cNvPr"):
        try:
            max_id = max(max_id, int(cNvPr.get("id", 0)))
        except (TypeError, ValueError):
            pass
    return max_id


def _make_textbox_anchor(shape_id: int, col: int, colOff: int, row: int, rowOff: int,
                          cx: int, cy: int, zh_text: str, fontsize_pt: int,
                          fill_hex: str = "FFFF99", text_hex: str = "B40000"):
    """构造一个 <xdr:oneCellAnchor><xdr:sp>...黄底中文 TextBox。"""
    anchor = ET.Element(f"{{{_XDR_NS}}}oneCellAnchor")
    frm = ET.SubElement(anchor, f"{{{_XDR_NS}}}from")
    ET.SubElement(frm, f"{{{_XDR_NS}}}col").text = str(col)
    ET.SubElement(frm, f"{{{_XDR_NS}}}colOff").text = str(int(colOff))
    ET.SubElement(frm, f"{{{_XDR_NS}}}row").text = str(row)
    ET.SubElement(frm, f"{{{_XDR_NS}}}rowOff").text = str(int(rowOff))
    ext_el = ET.SubElement(anchor, f"{{{_XDR_NS}}}ext")
    ext_el.set("cx", str(int(cx)))
    ext_el.set("cy", str(int(cy)))

    sp = ET.SubElement(anchor, f"{{{_XDR_NS}}}sp")
    sp.set("macro", "")
    sp.set("textlink", "")
    nvSpPr = ET.SubElement(sp, f"{{{_XDR_NS}}}nvSpPr")
    cNvPr = ET.SubElement(nvSpPr, f"{{{_XDR_NS}}}cNvPr")
    cNvPr.set("id", str(shape_id))
    cNvPr.set("name", f"TextBox_CN_{shape_id}")
    cNvSpPr = ET.SubElement(nvSpPr, f"{{{_XDR_NS}}}cNvSpPr")
    cNvSpPr.set("txBox", "1")

    spPr = ET.SubElement(sp, f"{{{_XDR_NS}}}spPr")
    xfrm = ET.SubElement(spPr, f"{{{_A_NS}}}xfrm")
    off_el = ET.SubElement(xfrm, f"{{{_A_NS}}}off")
    off_el.set("x", "0")
    off_el.set("y", "0")
    ext2 = ET.SubElement(xfrm, f"{{{_A_NS}}}ext")
    ext2.set("cx", str(int(cx)))
    ext2.set("cy", str(int(cy)))
    prstGeom = ET.SubElement(spPr, f"{{{_A_NS}}}prstGeom")
    prstGeom.set("prst", "rect")
    ET.SubElement(prstGeom, f"{{{_A_NS}}}avLst")
    solidFill = ET.SubElement(spPr, f"{{{_A_NS}}}solidFill")
    ET.SubElement(solidFill, f"{{{_A_NS}}}srgbClr").set("val", fill_hex)
    ln = ET.SubElement(spPr, f"{{{_A_NS}}}ln")
    ET.SubElement(ln, f"{{{_A_NS}}}noFill")

    txBody = ET.SubElement(sp, f"{{{_XDR_NS}}}txBody")
    bodyPr = ET.SubElement(txBody, f"{{{_A_NS}}}bodyPr")
    bodyPr.set("wrap", "square")
    bodyPr.set("rtlCol", "0")
    ET.SubElement(bodyPr, f"{{{_A_NS}}}spAutoFit")
    ET.SubElement(txBody, f"{{{_A_NS}}}lstStyle")
    p_el = ET.SubElement(txBody, f"{{{_A_NS}}}p")
    r_el = ET.SubElement(p_el, f"{{{_A_NS}}}r")
    rPr = ET.SubElement(r_el, f"{{{_A_NS}}}rPr")
    rPr.set("lang", "zh-CN")
    rPr.set("sz", str(int(fontsize_pt * 100)))
    sf = ET.SubElement(rPr, f"{{{_A_NS}}}solidFill")
    ET.SubElement(sf, f"{{{_A_NS}}}srgbClr").set("val", text_hex)
    t_el = ET.SubElement(r_el, f"{{{_A_NS}}}t")
    t_el.text = zh_text

    ET.SubElement(anchor, f"{{{_XDR_NS}}}clientData")
    return anchor


def add_translated_textboxes_to_excel(
    xlsx_bytes: bytes,
    client: anthropic.Anthropic,
    glossary: dict,
    on_image,
    selected_sheets: list[str] | None = None,
) -> tuple[bytes, list[dict]]:
    """图片像素完全不动；每条识别出的英文标注生成一个独立、可编辑、可拖动的
    Excel TextBox（黄底+中文），定位在原图片范围内对应的相对坐标处。
    返回 (xlsx_out, report_rows)。"""
    with zipfile.ZipFile(io.BytesIO(xlsx_bytes)) as zf:
        infos = {info.filename: info for info in zf.infolist()}
        all_data = {name: zf.read(name) for name in zf.namelist()}

    # 列宽/行高换算偏移量需要 worksheet 的 column_dimensions/row_dimensions，
    # 这里只读不存，不会重新触发 openpyxl 的整书写出（那个交给 run_excel_translation）。
    dims_wb = openpyxl.load_workbook(io.BytesIO(xlsx_bytes))
    drawing_sheet_name = _drawing_to_sheet_name(all_data)
    selected_sheet_set = None if selected_sheets is None else set(selected_sheets)

    drawing_paths = sorted(
        n for n in all_data if re.match(r"xl/drawings/drawing\d+\.xml$", n)
    )

    parsed = {}
    total_pics = 0
    for dp in drawing_paths:
        root = ET.fromstring(all_data[dp])
        pics = _find_top_level_pics(root)
        parsed[dp] = (root, pics)
        total_pics += len(pics)

    report_rows: list[dict] = []
    done = 0

    for dp, (root, pics) in parsed.items():
        if not pics:
            continue
        rels = _drawing_rels_map(all_data, dp)
        next_id = _max_shape_id_in_drawing(root) + 1

        for anchor_el, pic_el in pics:
            done += 1
            rid = _pic_embed_rid(pic_el)
            target = rels.get(rid) if rid else None
            media_path = _resolve_media_path(dp, target) if target else None
            fname = media_path.split("/")[-1] if media_path else f"(未知图片，{dp})"
            on_image(done, max(total_pics, 1), fname)

            if not media_path or media_path not in all_data:
                report_rows.append({"drawing": dp, "image": fname, "status": "skipped",
                                     "original_text": "", "translated_text": "",
                                     "skip_reason": "找不到图片关系映射"})
                continue

            extent = _pic_extent(pic_el)
            if extent is None:
                report_rows.append({"drawing": dp, "image": fname, "status": "skipped",
                                     "original_text": "", "translated_text": "",
                                     "skip_reason": "图片缺少 xfrm 定位信息"})
                continue

            ext = media_path.rsplit(".", 1)[-1].lower()
            items = _vision_extract_items(client, all_data[media_path], ext, glossary)
            if not items:
                report_rows.append({"drawing": dp, "image": fname, "status": "no_text",
                                     "original_text": "", "translated_text": "", "skip_reason": ""})
                continue
            if _should_skip_color_image_items(items):
                report_rows.append({
                    "drawing": dp,
                    "image": fname,
                    "status": "skipped",
                    "original_text": "",
                    "translated_text": "",
                    "skip_reason": "颜色/色号图，按规则跳过",
                })
                continue

            from_col, from_colOff, from_row, from_rowOff = _anchor_from(anchor_el)
            _ox, _oy, cx, cy = extent  # ox/oy 是图片内部历史遗留的局部坐标，不可信，不用它

            sheet_name = drawing_sheet_name.get(dp)
            if selected_sheet_set is not None and sheet_name not in selected_sheet_set:
                report_rows.append({"drawing": dp, "image": fname, "status": "skipped",
                                     "original_text": "", "translated_text": "",
                                     "skip_reason": "图片所在 Sheet 未包含在本次翻译范围"})
                continue
            ws = dims_wb[sheet_name] if sheet_name in dims_wb.sheetnames else None
            if ws is None:
                report_rows.append({"drawing": dp, "image": fname, "status": "skipped",
                                     "original_text": "", "translated_text": "",
                                     "skip_reason": "找不到对应 worksheet，无法换算列宽/行高"})
                continue

            for item in items:
                zh = str(item.get("zh", "")).strip()
                en = str(item.get("en", "")).strip()
                if not zh or _should_preserve_color_text(en, zh):
                    if _should_preserve_color_text(en, zh):
                        report_rows.append({
                            "drawing": dp,
                            "image": fname,
                            "status": "skipped",
                            "original_text": en,
                            "translated_text": zh,
                            "skip_reason": "颜色/色号文字，按规则保留原文",
                        })
                    continue
                x_norm = max(0.0, min(float(item.get("x", 0.5)), 0.95))
                y_norm = max(0.0, min(float(item.get("y", 0.5)), 0.95))
                fontsize_pt = max(8, min(int(item.get("size", 12)), 24))

                # 绝对偏移量按实际列宽/行高"进位"，换算成真正的目标列/行+列内偏移，
                # 不要把一个远超该列宽度的 offset 硬塞在同一列里（会被部分阅读器
                # 误判/堆叠在一起）。
                target_col, box_colOff = _carry_col_offset(
                    ws, from_col, from_colOff + int(x_norm * cx)
                )
                target_row, box_rowOff = _carry_row_offset(
                    ws, from_row, from_rowOff + int(y_norm * cy)
                )
                box_w = max(int(cx * 0.25), len(zh) * fontsize_pt * _EMU_PER_PT)
                box_h = int(fontsize_pt * 1.6 * _EMU_PER_PT)

                new_anchor = _make_textbox_anchor(
                    next_id, target_col, box_colOff, target_row, box_rowOff,
                    box_w, box_h, zh, fontsize_pt,
                )
                next_id += 1
                root.append(new_anchor)
                report_rows.append({"drawing": dp, "image": fname, "status": "ok",
                                     "original_text": en, "translated_text": zh, "skip_reason": ""})

        all_data[dp] = ET.tostring(root, encoding="utf-8", xml_declaration=True)

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zf_out:
        for name, data in all_data.items():
            zf_out.writestr(name, data, compress_type=infos[name].compress_type)
    return out.getvalue(), report_rows


# ── Streamlit page ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="服装行业翻译引擎",
    page_icon="🧵",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("🧵 服装行业翻译引擎")
st.caption("支持 PDF 与 Excel (.xlsx) 双格式 · 上传文件 + 术语库 · 调用 OpenAI 自动翻译为中文")
st.divider()

run_startup_tasks_once(_use_postgres(), 1)
current_user = render_login_panel()
if not current_user:
    st.stop()
selected_customer_id = render_customer_selector(current_user)
if selected_customer_id:
    selected_customer = get_customer(selected_customer_id)
    if selected_customer:
        st.info(
            f"当前选择客户：**{selected_customer['customer_code']} / "
            f"{selected_customer['customer_name']}**（{selected_customer['group_name']}）"
        )

# ── 会话内术语库初始化 ─────────────────────────────────────────────────────────
if "glossary_df" not in st.session_state:
    st.session_state["glossary_df"] = _load_default_glossary_df()
    st.session_state["glossary_source"] = (
        DEFAULT_GLOSSARY.name if DEFAULT_GLOSSARY.exists() else "（空，未找到默认术语库）"
    )
    st.session_state["glossary_conflicts"] = pd.DataFrame(columns=["英文术语", "旧翻译", "新翻译（采用）"])

# ── API Key（全局共用）───────────────────────────────────────────────────────
api_key = st.text_input(
    "🔑 OpenAI API Key",
    type="password",
    value=os.environ.get("OPENAI_API_KEY", "") or os.environ.get("ANTHROPIC_API_KEY", ""),
    placeholder="sk-...",
)
st.divider()

tab_labels = ["📄 PDF 翻译（支持批量）", "📊 Excel 翻译", "📚 客户术语库", "🧩 术语候选"]
if can_approve_glossary_change(current_user):
    tab_labels.append("✅ 术语审批")
    tab_labels.append("👤 用户管理")
    tab_labels.append("🏢 客户管理")
tabs = st.tabs(tab_labels)
tab_pdf, tab_excel, tab_glossary, tab_candidates = tabs[:4]
tab_approval = tabs[4] if len(tabs) > 4 else None
tab_user_admin = tabs[5] if len(tabs) > 5 else None
tab_customer_admin = tabs[6] if len(tabs) > 6 else None

# ════════════════════════════════════════════════════════════════════════════
#  PDF Tab — batch upload, independent processing per file
# ════════════════════════════════════════════════════════════════════════════
with tab_pdf:
    if "pdf_upload_reset_nonce" not in st.session_state:
        st.session_state["pdf_upload_reset_nonce"] = 0
    if st.button(
        "取消当前 PDF 翻译并重新上传",
        use_container_width=True,
        key="cancel_reset_pdf_jobs_btn",
    ):
        cancel_result = cancel_pdf_translation_jobs(current_user["username"])
        st.session_state.pop("last_pdf_job_ids", None)
        st.session_state.pop("pdf_batch_results", None)
        st.session_state["pdf_upload_reset_nonce"] += 1
        if cancel_result["cancelled_count"]:
            st.session_state["pdf_cancel_notice"] = f"已取消 {cancel_result['cancelled_count']} 个 PDF 任务。可以重新上传文件开始翻译。"
        else:
            st.session_state["pdf_cancel_notice"] = "当前没有等待中或翻译中的 PDF 任务，已重置上传区域。"
        st.rerun()

    pdf_upload_nonce = st.session_state["pdf_upload_reset_nonce"]
    pdf_files = st.file_uploader(
        "上传待翻译 PDF（可一次选择多个文件）",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"pdf_uploader_{pdf_upload_nonce}",
    )
    font_label = (
        f"🔤 上传中文字体 TTF（可选，已检测到默认字体 {DEFAULT_FONT.name}）"
        if DEFAULT_FONT.exists()
        else "🔤 上传中文字体 TTF（必填，未检测到默认字体）"
    )
    font_file = st.file_uploader(font_label, type=["ttf"], key=f"pdf_font_uploader_{pdf_upload_nonce}")
    font_ready = bool(font_file or DEFAULT_FONT.exists())
    selected_glossary_count = len(get_customer_glossary_df(selected_customer_id)) if selected_customer_id else 0
    st.caption(f"📚 当前客户术语库：**{selected_customer_id or '未选择'}** · {selected_glossary_count} 条 active 术语")

    pdf_scope_choice = st.radio(
        "翻译范围",
        ["all", "workmanship_auto", "manual"],
        format_func=lambda v: {
            "all": "全部页面",
            "workmanship_auto": "自动识别做工页",
            "manual": "手动选择页面",
        }[v],
        horizontal=True,
        key="pdf_scope_choice",
    )
    pdf_scope_configs: dict[str, dict] = {}
    pdf_scope_errors: list[str] = []
    if pdf_files and pdf_scope_choice != "all":
        st.caption("页码请按 PDF 阅读器里的页码填写，从 1 开始，例如：1-7,10。")
        for file_idx, pf in enumerate(pdf_files):
            pdf_bytes_for_scope = pf.getvalue()
            detection_rows = []
            try:
                if pdf_scope_choice == "workmanship_auto":
                    detection_rows = detect_workmanship_pages(pdf_bytes_for_scope)
                    total_pages = len(detection_rows)
                    default_pages = [
                        row["page_number"] for row in detection_rows
                        if row["is_workmanship"]
                    ]
                else:
                    doc_for_count = fitz.open(stream=pdf_bytes_for_scope, filetype="pdf")
                    total_pages = len(doc_for_count)
                    doc_for_count.close()
                    default_pages = list(range(1, total_pages + 1))
                default_range = format_page_ranges(default_pages)
                with st.expander(f"翻译范围确认：{pf.name}", expanded=True):
                    if detection_rows:
                        preview_df = pd.DataFrame([{
                            "页码": row["page_number"],
                            "判断": "做工" if row["is_workmanship"] else "跳过",
                            "分数": row["score"],
                            "原因": row["reason"],
                        } for row in detection_rows])
                        st.dataframe(preview_df, use_container_width=True, hide_index=True, height=240)
                    page_range = st.text_input(
                        "本文件实际翻译页码",
                        value=default_range,
                        key=f"pdf_scope_pages_{file_idx}_{pf.name}",
                    )
                selected_pages, page_errors = parse_page_ranges(page_range, total_pages)
                if page_errors:
                    pdf_scope_errors.extend([f"{pf.name}：{err}" for err in page_errors])
                if not selected_pages:
                    pdf_scope_errors.append(f"{pf.name}：至少选择 1 页。")
                pdf_scope_configs[pf.name] = {
                    "selected_pages": selected_pages,
                    "detection": detection_rows,
                    "total_pages": total_pages,
                    "page_range": page_range,
                }
            except Exception as exc:
                pdf_scope_errors.append(f"{pf.name}：范围识别失败：{exc}")
        for err in pdf_scope_errors:
            st.error(err)

    customer_ready = bool(selected_customer_id and can_use_customer_glossary(current_user, selected_customer_id))
    can_start_pdf = bool(pdf_files and api_key and font_ready and customer_ready and not pdf_scope_errors)
    if not can_start_pdf:
        missing = []
        if not pdf_files:
            missing.append("PDF 文件")
        if not api_key:
            missing.append("OpenAI API Key")
        if not font_ready:
            missing.append("中文字体 TTF")
        if not customer_ready:
            missing.append("有权限的客户")
        st.info(f"请先提供：{'、'.join(missing)}")

    start_pdf_btn = st.button(
        "🚀  开始翻译 PDF",
        disabled=not can_start_pdf,
        use_container_width=True,
        type="primary",
        key="start_pdf_btn",
    )

    if start_pdf_btn:
        created_jobs = []
        font_bytes = font_file.getvalue() if font_file else None
        for pf in pdf_files:
            pdf_input_bytes = pf.getvalue()
            scope_cfg = pdf_scope_configs.get(pf.name, {})
            if pdf_scope_choice == "all":
                doc_for_count = fitz.open(stream=pdf_input_bytes, filetype="pdf")
                scope_cfg = {"total_pages": len(doc_for_count), "detection": []}
                doc_for_count.close()
            selected_pages = None if pdf_scope_choice == "all" else scope_cfg.get("selected_pages")
            scope_detection = scope_cfg.get("detection") if pdf_scope_choice != "all" else []
            job_id = create_translation_job(
                job_type="PDF",
                username=current_user["username"],
                customer_id=selected_customer_id,
                source_file_name=pf.name,
                input_bytes=pdf_input_bytes,
                aux_bytes=font_bytes,
                config={
                    "scope_mode": pdf_scope_choice,
                    "selected_pages": selected_pages,
                    "scope_detection": scope_detection,
                    "scope_cfg": scope_cfg,
                },
            )
            created_jobs.append(job_id)
        if created_jobs:
            start_pdf_translation_job(created_jobs[0], api_key, start_next_on_finish=True)
        st.session_state["last_pdf_job_ids"] = created_jobs
        st.success(f"已创建 **{len(created_jobs)}** 个后台翻译任务。可以切换页面，稍后回到本页下载结果。")

    render_pdf_jobs_panel(current_user, api_key)

    batch_results = st.session_state.get("pdf_batch_results", [])
    if batch_results:
        st.divider()
        n_ok = sum(1 for r in batch_results if r["ok"])
        st.success(f"批量处理完成：成功 **{n_ok}** / 失败 **{len(batch_results) - n_ok}**（共 {len(batch_results)} 个文件）")

        # 全部成功结果打包为 ZIP
        ok_results = [r for r in batch_results if r["ok"]]
        if len(ok_results) > 1:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w") as zf:
                for r in ok_results:
                    base = r["name"].rsplit(".", 1)[0]
                    zf.writestr(f"{base}_translated.pdf", r["pdf"])
                    if r["n_terms"] > 0:
                        zf.writestr(f"{base}_unrecorded_terms.xlsx", r["xlsx"])
                    if r.get("scope_report"):
                        zf.writestr(
                            f"{base}_翻译范围报告.xlsx",
                            build_scope_report_xlsx(r["scope_report"]),
                        )
            st.download_button(
                label=f"⬇️  打包下载全部 {len(ok_results)} 个译文（ZIP）",
                data=zip_buf.getvalue(),
                file_name="translated_pdfs.zip",
                mime="application/zip",
                use_container_width=True,
                type="primary",
            )

        for i, r in enumerate(batch_results):
            with st.expander(f"{'✅' if r['ok'] else '❌'} {r['name']}", expanded=not r["ok"]):
                if r["ok"]:
                    base = r["name"].rsplit(".", 1)[0]
                    st.caption(f"未收录术语：{r['n_terms']} 条")
                    dl1, dl2, dl3 = st.columns(3)
                    with dl1:
                        st.download_button(
                            label="⬇️  下载中文 PDF",
                            data=r["pdf"],
                            file_name=f"{base}_translated.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            key=f"pdf_dl_{i}",
                        )
                    with dl2:
                        if r["n_terms"] > 0:
                            st.download_button(
                                label="⬇️  下载未收录术语",
                                data=r["xlsx"],
                                file_name=f"{base}_unrecorded_terms.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True,
                                key=f"xlsx_dl_{i}",
                            )
                    with dl3:
                        if r.get("scope_report"):
                            st.download_button(
                                label="⬇️ 下载范围报告",
                                data=build_scope_report_xlsx(r["scope_report"]),
                                file_name=f"{base}_翻译范围报告.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True,
                                key=f"pdf_scope_report_dl_{i}",
                            )
                else:
                    st.error(r["error"])

# ════════════════════════════════════════════════════════════════════════════
#  Excel Tab
# ════════════════════════════════════════════════════════════════════════════
def _build_excel_report_bytes(report_rows: list[dict], img_report_rows: list[dict] | None = None) -> bytes:
    report_wb = openpyxl.Workbook()
    report_ws = report_wb.active
    report_ws.title = "翻译报告"
    headers = ["sheet_name", "cell_coordinate", "original_text", "translated_text",
               "status", "skip_reason", "is_merged_cell", "layout_warning",
               "scope_mode", "selected_sheets", "skipped_sheets",
               "detection_score", "detection_reason"]
    report_ws.append(headers)
    for cell in report_ws[1]:
        cell.font = openpyxl.styles.Font(bold=True)
    for r in report_rows:
        report_ws.append([r.get(h, "") for h in headers])
    for col_letter, width in zip("ABCDEFGHIJKLM", [12, 14, 35, 35, 12, 24, 14, 14, 18, 35, 35, 12, 70]):
        report_ws.column_dimensions[col_letter].width = width

    if img_report_rows:
        img_ws = report_wb.create_sheet("图片译文")
        img_headers = ["drawing", "image", "status", "original_text", "translated_text", "skip_reason"]
        img_ws.append(img_headers)
        for cell in img_ws[1]:
            cell.font = openpyxl.styles.Font(bold=True)
        for r in img_report_rows:
            img_ws.append([r.get(h, "") for h in img_headers])
        for col_letter, width in zip("ABCDEF", [25, 18, 10, 35, 35, 20]):
            img_ws.column_dimensions[col_letter].width = width

    buf = io.BytesIO()
    report_wb.save(buf)
    return buf.getvalue()


def render_customer_glossary_import_panel(
    user: dict,
    customer_id: str | None,
    key_prefix: str,
) -> None:
    st.subheader("上传/更新客户术语库")
    if not customer_id:
        st.info("请先选择客户。")
        return
    if not can_submit_glossary_change(user, customer_id):
        st.warning("当前用户无权为该客户上传术语库。")
        return

    mode_labels = {
        "skip_existing": "跳过已存在术语",
        "overwrite_existing": "覆盖已有术语",
        "pending_request": "生成待审批申请",
    }
    if can_approve_glossary_change(user):
        conflict_mode = st.radio(
            "遇到同一客户下已存在的英文术语时",
            ["skip_existing", "overwrite_existing", "pending_request"],
            format_func=lambda v: mode_labels[v],
            horizontal=True,
            key=f"{key_prefix}_conflict_mode",
        )
    else:
        conflict_mode = "pending_request"
        st.caption("当前角色上传后会生成 pending 术语申请，等待公司管理员审批。")

    uploaded = st.file_uploader(
        "上传术语库 Excel",
        type=["xlsx", "xls"],
        key=f"{key_prefix}_glossary_upload",
    )
    if not uploaded:
        return

    df, missing, parse_stats, parse_report = parse_customer_glossary_excel(uploaded.getvalue(), uploaded.name)
    if missing:
        st.error("；".join(missing))
        if parse_report:
            st.download_button(
                "⬇️ 下载导入错误报告",
                data=build_import_report_xlsx(parse_report),
                file_name="import_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key=f"{key_prefix}_parse_report_dl",
            )
        return

    st.caption(
        f"读取行数 **{parse_stats.get('total_rows', 0)}** ｜ "
        f"有效行 **{parse_stats.get('valid_rows', 0)}** ｜ "
        f"跳过空行 **{parse_stats.get('skipped_blank', 0)}** ｜ "
        f"文件内重复 **{parse_stats.get('duplicate_terms', 0)}** ｜ "
        f"错误行 **{parse_stats.get('error_rows', 0)}**"
    )
    st.dataframe(df.head(30), use_container_width=True, hide_index=True, height=220)

    if st.button(
        "导入为该客户术语库",
        type="primary",
        use_container_width=True,
        key=f"{key_prefix}_import_btn",
    ):
        try:
            import_stats, import_report = import_customer_glossary(
                user,
                customer_id,
                df,
                conflict_mode,
            )
            combined_report = parse_report + import_report
            combined_stats = {
                **parse_stats,
                **import_stats,
                "error_rows": parse_stats.get("error_rows", 0) + import_stats.get("error_rows", 0),
            }
            st.session_state[f"{key_prefix}_last_import_report"] = combined_report
            st.session_state[f"{key_prefix}_last_import_stats"] = combined_stats
            st.success(
                f"导入完成：新增 active **{import_stats['success_imported']}** 条，"
                f"覆盖 **{import_stats['overwritten']}** 条，"
                f"跳过已有 **{import_stats['skipped_existing']}** 条，"
                f"待审批 **{import_stats['pending_count']}** 条。"
            )
        except Exception as exc:
            st.error(str(exc))

    last_report = st.session_state.get(f"{key_prefix}_last_import_report")
    last_stats = st.session_state.get(f"{key_prefix}_last_import_stats")
    if last_report and last_stats:
        st.caption(
            f"最近一次导入报告：总行数 {last_stats.get('total_rows', 0)}，"
            f"成功导入 {last_stats.get('success_imported', 0)}，"
            f"跳过空行 {last_stats.get('skipped_blank', 0)}，"
            f"重复术语 {last_stats.get('duplicate_terms', 0)}，"
            f"覆盖术语 {last_stats.get('overwritten', 0)}，"
            f"待审批 {last_stats.get('pending_count', 0)}，"
            f"错误行 {last_stats.get('error_rows', 0)}。"
        )
        st.download_button(
            "⬇️ 下载 import_report.xlsx",
            data=build_import_report_xlsx(last_report),
            file_name="import_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key=f"{key_prefix}_import_report_dl",
        )


with tab_excel:
    if "excel_upload_reset_nonce" not in st.session_state:
        st.session_state["excel_upload_reset_nonce"] = 0
    if st.button(
        "清空 Excel 翻译并重新上传",
        use_container_width=True,
        key="reset_excel_translation_btn",
    ):
        st.session_state.pop("excel_batch_results", None)
        st.session_state["excel_upload_reset_nonce"] += 1
        st.info("Excel 上传区域已重置，可以重新上传。")
        st.rerun()

    excel_upload_nonce = st.session_state["excel_upload_reset_nonce"]
    excel_files = st.file_uploader(
        "上传待翻译 Excel（可一次选择多个文件）",
        type=["xlsx"],
        accept_multiple_files=True,
        key=f"excel_uploader_{excel_upload_nonce}",
    )
    selected_glossary_count = len(get_customer_glossary_df(selected_customer_id)) if selected_customer_id else 0
    st.caption(f"📚 当前客户术语库：**{selected_customer_id or '未选择'}** · {selected_glossary_count} 条 active 术语")

    excel_scope_choice = st.radio(
        "翻译范围",
        ["all", "workmanship_auto", "manual"],
        format_func=lambda v: {
            "all": "全部 Sheet",
            "workmanship_auto": "自动识别做工 Sheet",
            "manual": "手动选择 Sheet",
        }[v],
        horizontal=True,
        key="excel_scope_choice",
    )
    excel_scope_configs: dict[str, dict] = {}
    excel_scope_errors: list[str] = []
    if excel_files and excel_scope_choice != "all":
        for file_idx, ef in enumerate(excel_files):
            excel_bytes_for_scope = ef.getvalue()
            detection_rows = []
            try:
                if excel_scope_choice == "workmanship_auto":
                    detection_rows = detect_workmanship_sheets(excel_bytes_for_scope)
                    sheet_names = [row["sheet_name"] for row in detection_rows]
                    default_sheets = [
                        row["sheet_name"] for row in detection_rows
                        if row["is_workmanship"]
                    ]
                else:
                    wb_for_scope = openpyxl.load_workbook(
                        io.BytesIO(excel_bytes_for_scope),
                        read_only=True,
                        data_only=True,
                    )
                    sheet_names = wb_for_scope.sheetnames
                    wb_for_scope.close()
                    default_sheets = sheet_names
                with st.expander(f"Sheet 范围确认：{ef.name}", expanded=True):
                    if detection_rows:
                        preview_df = pd.DataFrame([{
                            "Sheet": row["sheet_name"],
                            "判断": row["verdict"],
                            "分数": row["score"],
                            "原因": row["reason"],
                        } for row in detection_rows])
                        st.dataframe(preview_df, use_container_width=True, hide_index=True, height=240)
                    selected_sheets = st.multiselect(
                        "本文件实际翻译 Sheet",
                        options=sheet_names,
                        default=default_sheets,
                        key=f"excel_scope_sheets_{file_idx}_{ef.name}",
                    )
                if not selected_sheets:
                    excel_scope_errors.append(f"{ef.name}：至少选择 1 个 Sheet。")
                excel_scope_configs[ef.name] = {
                    "selected_sheets": selected_sheets,
                    "detection": detection_rows,
                    "sheet_names": sheet_names,
                }
            except Exception as exc:
                excel_scope_errors.append(f"{ef.name}：范围识别失败：{exc}")
        for err in excel_scope_errors:
            st.error(err)

    customer_ready = bool(selected_customer_id and can_use_customer_glossary(current_user, selected_customer_id))
    can_start_excel = bool(excel_files and api_key and customer_ready and not excel_scope_errors)
    if not can_start_excel:
        missing = []
        if not excel_files:
            missing.append("Excel 文件")
        if not api_key:
            missing.append("OpenAI API Key")
        if not customer_ready:
            missing.append("有权限的客户")
        st.info(f"请先提供：{'、'.join(missing)}")

    translate_images_checked = st.checkbox(
        "🧪 同时翻译图片内文字（实验性：生成可编辑 TextBox，不改图片像素）",
        key="excel_translate_images_checkbox",
    )
    st.caption("勾选后，图片里识别到的英文标注会生成黄底中文 TextBox 叠在图片对应位置，"
               "可在 Excel/WPS 里直接双击编辑、拖动。坐标是视觉模型估算的，密集排版的图"
               "可能不够精确，请翻译后打开核对。不勾选则图片完全原样保留。")

    start_excel_btn = st.button(
        "🚀  开始翻译 Excel",
        disabled=not can_start_excel,
        use_container_width=True,
        type="primary",
        key="start_excel_btn",
    )

    if start_excel_btn:
        st.session_state["excel_batch_results"] = []

        glossary_bytes = get_customer_glossary_bytes_for_translation(
            current_user,
            selected_customer_id,
        )
        glossary_dict = load_glossary(glossary_bytes)
        overall = st.progress(0.0, text=f"准备处理 {len(excel_files)} 个文件…")
        results = []

        for fi, ef in enumerate(excel_files):
            with st.status(f"正在处理：{ef.name}", expanded=True) as status:
                cell_ph = st.empty()
                file_prog = st.progress(0.0)

                def on_cell(preview):
                    cell_ph.caption(f"▶ 正在翻译：{preview}…")

                def on_progress(frac):
                    file_prog.progress(frac, text=f"翻译进度 {frac:.0%}")

                try:
                    scope_cfg = excel_scope_configs.get(ef.name, {})
                    selected_sheets = None if excel_scope_choice == "all" else scope_cfg.get("selected_sheets")
                    scope_detection = scope_cfg.get("detection") if excel_scope_choice != "all" else []
                    excel_out, n_cells, n_images, report_rows, review_summary = run_excel_translation(
                        xlsx_bytes=ef.getvalue(),
                        glossary_bytes=glossary_bytes,
                        api_key=api_key,
                        on_cell=on_cell,
                        on_progress=on_progress,
                        translate_images=False,
                        customer_id=selected_customer_id,
                        source_file_name=ef.name,
                        created_by=current_user["username"],
                        selected_sheets=selected_sheets,
                        scope_mode=excel_scope_choice,
                        scope_detection=scope_detection,
                    )

                    img_report_rows = []
                    if translate_images_checked:
                        img_ph = st.empty()

                        def on_image(i, total, fname):
                            img_ph.caption(f"🖼️ 图片译文 {i}/{total}：{fname}")

                        client = OpenAI(api_key=api_key)
                        excel_out, img_report_rows = add_translated_textboxes_to_excel(
                            excel_out, client, glossary_dict, on_image,
                            selected_sheets=selected_sheets,
                        )
                        img_ph.empty()

                    cell_ph.empty()
                    status.update(label=f"✅ {ef.name} 翻译完成", state="complete")
                    results.append({
                        "name": ef.name,
                        "ok": True,
                        "excel": excel_out,
                        "report": report_rows,
                        "img_report": img_report_rows,
                        "n_cells": n_cells,
                        "review_summary": review_summary,
                    })
                except Exception as e:
                    cell_ph.empty()
                    status.update(label=f"❌ {ef.name} 出错：{e}", state="error")
                    results.append({"name": ef.name, "ok": False, "error": str(e)})

            overall.progress((fi + 1) / len(excel_files), text=f"已完成 {fi + 1}/{len(excel_files)}")

        st.session_state["excel_batch_results"] = results

    batch_results = st.session_state.get("excel_batch_results", [])
    if batch_results:
        st.divider()
        n_ok = sum(1 for r in batch_results if r["ok"])
        img_note = "图片已按勾选生成可编辑 TextBox 译文。" if translate_images_checked else "图片未处理，原样保留。"
        st.success(f"批量处理完成：成功 **{n_ok}** / 失败 **{len(batch_results) - n_ok}**（共 {len(batch_results)} 个文件）。{img_note}")

        ok_results = [r for r in batch_results if r["ok"]]
        if len(ok_results) > 1:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w") as zf:
                for r in ok_results:
                    base = r["name"].rsplit(".", 1)[0]
                    zf.writestr(f"{base}_中文版.xlsx", r["excel"])
                    zf.writestr(f"{base}_翻译报告.xlsx",
                                _build_excel_report_bytes(r["report"], r.get("img_report")))
            st.download_button(
                label=f"⬇️  打包下载全部 {len(ok_results)} 个译文（ZIP）",
                data=zip_buf.getvalue(),
                file_name="translated_excels.zip",
                mime="application/zip",
                use_container_width=True,
                type="primary",
            )

        for i, r in enumerate(batch_results):
            with st.expander(f"{'✅' if r['ok'] else '❌'} {r['name']}", expanded=not r["ok"]):
                if r["ok"]:
                    base = r["name"].rsplit(".", 1)[0]
                    review_summary = r.get("review_summary") or {}
                    if review_summary.get("summary_text"):
                        st.info(review_summary["summary_text"])
                    report_rows = r["report"]
                    img_report_rows = r.get("img_report") or []
                    n_warn = sum(1 for row in report_rows if row["layout_warning"])
                    n_merged = sum(1 for row in report_rows if row["is_merged_cell"])
                    st.caption(
                        f"文字单元格 **{r['n_cells']}** 个"
                        f"（合并单元格内 **{n_merged}** 个，触发换行提醒 **{n_warn}** 个）"
                    )
                    if img_report_rows:
                        n_img_ok = sum(1 for row in img_report_rows if row["status"] == "ok")
                        n_img_skip = sum(1 for row in img_report_rows if row["status"] == "skipped")
                        st.caption(
                            f"图片译文 TextBox **{n_img_ok}** 条"
                            + (f"，跳过 **{n_img_skip}** 处（详见报告）" if n_img_skip else "")
                        )
                    dl1, dl2 = st.columns(2)
                    with dl1:
                        st.download_button(
                            label="⬇️  下载中文 Excel",
                            data=r["excel"],
                            file_name=f"{base}_中文版.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                            key=f"excel_dl_{i}",
                        )
                    with dl2:
                        st.download_button(
                            label="⬇️  下载翻译报告",
                            data=_build_excel_report_bytes(report_rows, img_report_rows),
                            file_name=f"{base}_翻译报告.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                            key=f"report_dl_{i}",
                        )
                else:
                    st.error(r["error"])

# ════════════════════════════════════════════════════════════════════════════
#  Customer Glossary Tab
# ════════════════════════════════════════════════════════════════════════════
with tab_glossary:
    if not selected_customer_id or not can_view_customer_glossary(current_user, selected_customer_id):
        st.error("请先选择一个有权限访问的客户。")
    else:
        customer = get_customer(selected_customer_id)
        st.caption(
            f"当前客户：**{customer['customer_code']} / {customer['customer_name']}** ｜ "
            f"小组：**{customer['group_name']}** ｜ 负责人：**{customer['assigned_staff_username']}**"
        )

        terms = get_active_glossary_terms(selected_customer_id)
        terms_df = pd.DataFrame(terms)
        if terms_df.empty:
            st.info("当前客户还没有 active 术语。")
        else:
            display_df = terms_df[
                ["english_term", "chinese_translation", "normalized_key", "note", "updated_by", "updated_at"]
            ].rename(columns={
                "english_term": "英文术语",
                "chinese_translation": "中文翻译",
                "normalized_key": "normalized_key",
                "note": "备注",
                "updated_by": "最后更新人",
                "updated_at": "最后更新时间",
            })
            st.dataframe(display_df, use_container_width=True, height=260, hide_index=True)

        st.divider()
        st.subheader("提交术语修改申请")
        st.caption("新增、修改、删除只生成 pending 申请；正式术语库不会被直接改动。")

        can_submit = can_submit_glossary_change(current_user, selected_customer_id)
        if not can_submit:
            st.warning("当前用户无权提交该客户的术语修改申请。")
        else:
            add_tab, update_tab, delete_tab = st.tabs(["申请新增术语", "申请修改术语", "申请删除术语"])

            with add_tab:
                with st.form("glossary_add_request_form"):
                    en_new = st.text_input("英文术语", key="add_en")
                    zh_new = st.text_input("中文翻译", key="add_zh")
                    note = st.text_area("备注", key="add_note")
                    submitted = st.form_submit_button("提交新增申请", type="primary", use_container_width=True)
                if submitted:
                    if not en_new.strip() or not zh_new.strip():
                        st.error("英文术语和中文翻译不能为空。")
                    else:
                        req_id = create_glossary_change_request(
                            current_user,
                            selected_customer_id,
                            "add",
                            english_term_new=en_new,
                            chinese_translation_new=zh_new,
                            note=note,
                        )
                        st.success(f"已提交新增申请，申请编号 #{req_id}。等待公司管理员审批。")

            with update_tab:
                if not terms:
                    st.info("当前客户没有可修改的 active 术语。")
                else:
                    term_options = [t["english_term"] for t in terms]
                    selected_term = st.selectbox("选择要修改的术语", term_options, key="update_term_select")
                    old_term = get_active_glossary_term(selected_customer_id, selected_term)
                    with st.form("glossary_update_request_form"):
                        st.text_input("原英文术语", value=old_term["english_term"], disabled=True)
                        st.text_input("原中文翻译", value=old_term["chinese_translation"], disabled=True)
                        en_new = st.text_input("新英文术语", value=old_term["english_term"], key="update_en")
                        zh_new = st.text_input("新中文翻译", value=old_term["chinese_translation"], key="update_zh")
                        note = st.text_area("备注", value=old_term.get("note") or "", key="update_note")
                        submitted = st.form_submit_button("提交修改申请", type="primary", use_container_width=True)
                    if submitted:
                        if not en_new.strip() or not zh_new.strip():
                            st.error("新英文术语和新中文翻译不能为空。")
                        else:
                            req_id = create_glossary_change_request(
                                current_user,
                                selected_customer_id,
                                "update",
                                english_term_old=old_term["english_term"],
                                chinese_translation_old=old_term["chinese_translation"],
                                english_term_new=en_new,
                                chinese_translation_new=zh_new,
                                note=note,
                            )
                            st.success(f"已提交修改申请，申请编号 #{req_id}。等待公司管理员审批。")

            with delete_tab:
                if not terms:
                    st.info("当前客户没有可删除的 active 术语。")
                else:
                    term_options = [t["english_term"] for t in terms]
                    selected_term = st.selectbox("选择要删除的术语", term_options, key="delete_term_select")
                    old_term = get_active_glossary_term(selected_customer_id, selected_term)
                    with st.form("glossary_delete_request_form"):
                        st.text_input("英文术语", value=old_term["english_term"], disabled=True)
                        st.text_input("中文翻译", value=old_term["chinese_translation"], disabled=True)
                        note = st.text_area("删除原因 / 备注", key="delete_note")
                        submitted = st.form_submit_button("提交删除申请", type="primary", use_container_width=True)
                    if submitted:
                        req_id = create_glossary_change_request(
                            current_user,
                            selected_customer_id,
                            "delete",
                            english_term_old=old_term["english_term"],
                            chinese_translation_old=old_term["chinese_translation"],
                            note=note,
                        )
                        st.success(f"已提交删除申请，申请编号 #{req_id}。等待公司管理员审批。")

        st.divider()
        render_customer_glossary_import_panel(
            current_user,
            selected_customer_id,
            "selected_customer_glossary",
        )


# ════════════════════════════════════════════════════════════════════════════
#  Term Candidates Tab
# ════════════════════════════════════════════════════════════════════════════
with tab_candidates:
    accessible_customers = get_accessible_customers(current_user)
    if not accessible_customers:
        st.warning("当前用户没有可访问客户。")
    elif not selected_customer_id:
        st.info("请先在页面顶部选择客户。")
    else:
        candidate_customer_labels = {
            c["customer_id"]: f"{c['customer_code']} / {c['customer_name']}"
            for c in accessible_customers
        }
        current_customer_label = candidate_customer_labels.get(selected_customer_id, selected_customer_id)
        candidate_customer = selected_customer_id
        show_all_candidates = False
        fc, fs = st.columns(2)
        with fc:
            st.caption(f"当前客户：**{current_customer_label}**")
            if can_approve_glossary_change(current_user):
                show_all_candidates = st.toggle(
                    "查看全部客户候选",
                    value=False,
                    key="candidate_show_all_toggle",
                )
        with fs:
            candidate_status = st.selectbox(
                "状态",
                ["draft", "selected", "submitted", "ignored", "approved", "rejected", "全部"],
                key="candidate_status_filter",
            )

        candidates = list_term_candidates(
            current_user,
            customer_id=None if show_all_candidates else candidate_customer,
            status=candidate_status,
        )
        if not candidates:
            st.info("当前没有符合条件的术语候选。翻译 PDF / Excel 后会自动生成候选。")
        else:
            rows = []
            for c in candidates:
                try:
                    variants = "; ".join(json.loads(c.get("variants") or "[]"))
                except json.JSONDecodeError:
                    variants = c.get("variants") or c["original_term"]
                rows.append({
                    "是否提交": False,
                    "忽略": False,
                    "candidate_id": c["candidate_id"],
                    "customer_id": c["customer_id"],
                    "客户": f"{c.get('customer_code') or c['customer_id']} / {c.get('customer_name') or ''}",
                    "normalized_key": c["normalized_term"],
                    "英文术语": c["original_term"],
                    "所有变体": variants or c["original_term"],
                    "AI建议中文": c["ai_suggested_translation"],
                    "人工确认中文": c["final_translation"] or c["ai_suggested_translation"],
                    "出现次数": c["frequency"],
                    "来源文件": c["source_file_name"],
                    "来源类型": c["source_type"],
                    "来源位置": c["page_or_sheet"],
                    "单元格": c["cell_coordinate"],
                    "上下文": c["context_sentence"],
                    "置信度": c["confidence"],
                    "matched_by": c.get("matched_by") or "no_match",
                    "命中术语": c.get("matched_glossary_term") or "",
                    "冲突提示": c.get("conflict_warning") or "",
                    "状态": c["status"],
                    "备注": "",
                })
            editor_df = pd.DataFrame(rows)
            edited_df = st.data_editor(
                editor_df,
                use_container_width=True,
                hide_index=True,
                height=420,
                disabled=[
                    "candidate_id", "customer_id", "客户", "normalized_key",
                    "英文术语", "所有变体", "AI建议中文",
                    "出现次数", "来源文件", "来源类型", "来源位置", "单元格",
                    "上下文", "置信度", "matched_by", "命中术语", "冲突提示", "状态",
                ],
                column_config={
                    "是否提交": st.column_config.CheckboxColumn("是否提交"),
                    "忽略": st.column_config.CheckboxColumn("忽略"),
                    "人工确认中文": st.column_config.TextColumn("人工确认中文", required=False),
                    "备注": st.column_config.TextColumn("备注"),
                },
                key="term_candidates_editor",
            )

            submit_col, ignore_col = st.columns(2)
            with submit_col:
                submit_candidates = st.button(
                    "一键提交审核",
                    type="primary",
                    use_container_width=True,
                    key="submit_term_candidates_btn",
                )
            with ignore_col:
                ignore_candidates = st.button(
                    "忽略勾选候选",
                    use_container_width=True,
                    key="ignore_term_candidates_btn",
                )

            if submit_candidates:
                selected_rows = []
                for _, row in edited_df[edited_df["是否提交"] == True].iterrows():
                    selected_rows.append({
                        "candidate_id": int(row["candidate_id"]),
                        "customer_id": row["customer_id"],
                        "final_translation": row["人工确认中文"],
                        "note": row.get("备注", ""),
                    })
                if not selected_rows:
                    st.warning("请先勾选要提交的候选术语。")
                else:
                    submitted_count, errors = submit_term_candidates_for_approval(current_user, selected_rows)
                    if submitted_count:
                        st.success(f"已提交 **{submitted_count}** 条术语候选，等待管理员审批。")
                    if errors:
                        st.warning("；".join(errors[:10]))
                    st.rerun()

            if ignore_candidates:
                ignore_ids = [
                    int(row["candidate_id"])
                    for _, row in edited_df[edited_df["忽略"] == True].iterrows()
                ]
                if not ignore_ids:
                    st.warning("请先勾选要忽略的候选术语。")
                else:
                    placeholders = ",".join("?" for _ in ignore_ids)
                    with get_db_connection() as conn:
                        conn.execute(
                            f"""
                            UPDATE term_candidates
                            SET status = 'ignored'
                            WHERE candidate_id IN ({placeholders})
                            """,
                            ignore_ids,
                        )
                    st.success(f"已忽略 **{len(ignore_ids)}** 条候选。")
                    st.rerun()


# ════════════════════════════════════════════════════════════════════════════
#  Approval Tab
# ════════════════════════════════════════════════════════════════════════════
if tab_approval is not None:
    with tab_approval:
        st.caption("仅 company_admin 可见。审批通过后才会写入正式客户术语库。")

        all_customers = get_accessible_customers(current_user)
        customer_filter_options = ["全部"] + [c["customer_id"] for c in all_customers]
        customer_labels = {
            c["customer_id"]: f"{c['customer_code']} / {c['customer_name']}"
            for c in all_customers
        }
        fc, fa, fs = st.columns(3)
        with fc:
            customer_filter = st.selectbox(
                "客户筛选",
                customer_filter_options,
                format_func=lambda cid: "全部" if cid == "全部" else customer_labels.get(cid, cid),
                key="approval_customer_filter",
            )
        with fa:
            action_filter = st.selectbox("申请类型", ["全部", "add", "update", "delete"], key="approval_action_filter")
        with fs:
            submitted_by_filter = st.text_input("申请人筛选", key="approval_submitter_filter")

        pending = list_glossary_change_requests(
            status="pending",
            customer_id=None if customer_filter == "全部" else customer_filter,
            submitted_by=submitted_by_filter.strip() or None,
            action_type=None if action_filter == "全部" else action_filter,
        )

        if not pending:
            st.info("当前没有符合条件的 pending 申请。")
        else:
            pending_df = pd.DataFrame(pending)
            st.dataframe(
                pending_df[[
                    "request_id", "candidate_id", "customer_code", "customer_name", "action_type",
                    "english_term_old", "chinese_translation_old",
                    "english_term_new", "chinese_translation_new",
                    "submitted_by", "submitted_at",
                ]],
                use_container_width=True,
                hide_index=True,
                height=260,
            )

            request_ids = [int(r["request_id"]) for r in pending]
            selected_request_id = st.selectbox("选择要审批的申请", request_ids, key="approval_request_select")
            selected_request = next(r for r in pending if int(r["request_id"]) == int(selected_request_id))

            st.markdown("#### 申请详情")
            detail_cols = st.columns(2)
            with detail_cols[0]:
                st.write(f"客户：**{selected_request['customer_code']} / {selected_request['customer_name']}**")
                st.write(f"类型：`{selected_request['action_type']}`")
                if selected_request.get("candidate_id"):
                    st.write(f"候选编号：`{selected_request['candidate_id']}`")
                st.write(f"申请人：**{selected_request['submitted_by']}**")
                st.write(f"提交时间：{selected_request['submitted_at']}")
            with detail_cols[1]:
                st.write(f"旧英文：{selected_request['english_term_old'] or ''}")
                st.write(f"旧中文：{selected_request['chinese_translation_old'] or ''}")
                st.write(f"新英文：{selected_request['english_term_new'] or ''}")
                st.write(f"新中文：{selected_request['chinese_translation_new'] or ''}")
            if selected_request.get("note"):
                st.info(f"备注：{selected_request['note']}")

            with st.form("approval_decision_form"):
                review_comment = st.text_area("审核意见", key="approval_comment")
                approve_col, reject_col = st.columns(2)
                with approve_col:
                    approve_clicked = st.form_submit_button("通过申请", type="primary", use_container_width=True)
                with reject_col:
                    reject_clicked = st.form_submit_button("拒绝申请", use_container_width=True)

            if approve_clicked:
                try:
                    approve_glossary_change_request(selected_request_id, current_user, review_comment)
                    st.success(f"已通过申请 #{selected_request_id}，正式术语库已更新。")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

            if reject_clicked:
                try:
                    reject_glossary_change_request(selected_request_id, current_user, review_comment)
                    st.success(f"已拒绝申请 #{selected_request_id}，正式术语库未改变。")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))


# ════════════════════════════════════════════════════════════════════════════
#  User Admin Tab
# ════════════════════════════════════════════════════════════════════════════
if tab_user_admin is not None:
    with tab_user_admin:
        st.caption("仅 company_admin 可见。客户分配以「客户管理」中的负责人为准。")
        users = list_users()
        users_df = pd.DataFrame(users)
        if not users_df.empty:
            display_users = users_df.copy()
            display_users["assigned_customer_ids"] = display_users["assigned_customer_ids"].apply(
                lambda ids: ", ".join(ids) if ids else ""
            )
            st.dataframe(
                display_users[["username", "role", "group_name", "assigned_customer_ids"]],
                use_container_width=True,
                hide_index=True,
                height=240,
            )

        st.divider()
        add_user_tab, edit_user_tab = st.tabs(["新增账号", "修改账号"])

        with add_user_tab:
            with st.form("create_user_form"):
                username = st.text_input("用户名", key="create_user_username")
                password = st.text_input("初始密码", type="password", key="create_user_password")
                role = st.selectbox(
                    "角色",
                    ["staff", "group_leader", "company_admin"],
                    key="create_user_role",
                )
                group_name = st.text_input("小组", key="create_user_group")
                submitted = st.form_submit_button("创建账号", type="primary", use_container_width=True)
            if submitted:
                try:
                    create_user(username, password, role, group_name)
                    st.success(f"已创建账号 {username.strip()}。")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

        with edit_user_tab:
            if not users:
                st.info("当前没有用户。")
            else:
                usernames = [u["username"] for u in users]
                selected_username = st.selectbox("选择账号", usernames, key="edit_user_select")
                selected_user = next(u for u in users if u["username"] == selected_username)
                with st.form("update_user_form"):
                    role = st.selectbox(
                        "角色",
                        ["staff", "group_leader", "company_admin"],
                        index=["staff", "group_leader", "company_admin"].index(selected_user["role"]),
                        key="edit_user_role",
                    )
                    group_name = st.text_input(
                        "小组",
                        value=selected_user.get("group_name") or "",
                        key="edit_user_group",
                    )
                    new_password = st.text_input(
                        "新密码（留空则不修改）",
                        type="password",
                        key="edit_user_password",
                    )
                    st.caption(
                        "该账号负责客户："
                        + (", ".join(selected_user.get("assigned_customer_ids") or []) or "无")
                    )
                    submitted = st.form_submit_button("保存账号修改", type="primary", use_container_width=True)
                if submitted:
                    try:
                        update_user(selected_username, role, group_name, new_password)
                        if selected_username == current_user["username"]:
                            st.session_state["current_user"] = authenticate_user(
                                selected_username,
                                new_password if new_password.strip() else "",
                            ) or {**current_user, "role": role, "group_name": group_name}
                        st.success(f"已更新账号 {selected_username}。")
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))


# ════════════════════════════════════════════════════════════════════════════
#  Customer Admin Tab
# ════════════════════════════════════════════════════════════════════════════
if tab_customer_admin is not None:
    with tab_customer_admin:
        st.caption("仅 company_admin 可见。创建/修改负责人后会同步 staff 的 assigned_customer_ids。")
        customers = list_customers_with_term_counts(current_user)
        staff_usernames = get_staff_usernames()
        customers_df = pd.DataFrame(customers)
        if not customers_df.empty:
            display_customers = customers_df[[
                "customer_id", "customer_code", "customer_name",
                "group_name", "assigned_staff_username", "term_count",
                "created_at", "note",
            ]].rename(columns={
                "customer_id": "客户ID",
                "customer_code": "客户代码",
                "customer_name": "客户名称",
                "group_name": "所属小组",
                "assigned_staff_username": "负责职员",
                "term_count": "当前术语数量",
                "created_at": "创建时间",
                "note": "备注",
            })
            st.dataframe(display_customers, use_container_width=True, hide_index=True, height=260)
        else:
            st.info("当前没有客户。")

        st.divider()
        add_customer_tab, import_customer_tab, edit_customer_tab = st.tabs([
            "创建新客户",
            "上传/更新术语库",
            "修改客户",
        ])

        with add_customer_tab:
            if not staff_usernames:
                st.warning("请先创建 staff 账号，再新增客户。")
            else:
                with st.form("create_customer_form"):
                    customer_name = st.text_input("客户名称", key="create_customer_name")
                    customer_code = st.text_input("客户代码", placeholder="例如 EL 或 CUST004", key="create_customer_code")
                    group_name = st.text_input("小组", key="create_customer_group")
                    assigned_staff_username = st.selectbox(
                        "负责业务员 / 普通职员",
                        staff_usernames,
                        key="create_customer_staff",
                    )
                    note = st.text_area("备注（可选）", key="create_customer_note")
                    submitted = st.form_submit_button("创建客户", type="primary", use_container_width=True)
                if submitted:
                    try:
                        new_customer_id = create_customer_auto_id(
                            customer_name,
                            customer_code,
                            group_name,
                            assigned_staff_username,
                            note,
                        )
                        st.session_state["_pending_selected_customer_id"] = new_customer_id
                        st.success(f"已创建客户 {customer_code.strip()}，可立即在 PDF / Excel 翻译中选择。")
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

        with import_customer_tab:
            if not customers:
                st.info("请先创建客户。")
            else:
                customer_ids = [c["customer_id"] for c in customers]
                labels = {
                    c["customer_id"]: f"{c['customer_code']} / {c['customer_name']} / {c['group_name']}"
                    for c in customers
                }
                default_customer = st.session_state.get("selected_customer_id")
                default_index = customer_ids.index(default_customer) if default_customer in customer_ids else 0
                import_customer_id = st.selectbox(
                    "选择客户",
                    customer_ids,
                    index=default_index,
                    format_func=lambda cid: labels.get(cid, cid),
                    key="admin_import_customer_select",
                )
                selected_terms = get_customer_glossary_df(import_customer_id)
                st.caption(f"当前 active 术语：**{len(selected_terms)}** 条")
                dl_col, view_col = st.columns(2)
                with dl_col:
                    st.download_button(
                        "⬇️ 下载该客户术语库",
                        data=glossary_df_to_xlsx_bytes(selected_terms),
                        file_name=f"{labels.get(import_customer_id, import_customer_id).split(' / ')[0]}_glossary.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        key="admin_customer_glossary_dl",
                    )
                with view_col:
                    if st.button("查看术语库", use_container_width=True, key="admin_view_customer_terms"):
                        st.session_state["admin_show_customer_terms"] = import_customer_id
                if st.session_state.get("admin_show_customer_terms") == import_customer_id:
                    st.dataframe(selected_terms, use_container_width=True, hide_index=True, height=220)

                st.divider()
                render_customer_glossary_import_panel(
                    current_user,
                    import_customer_id,
                    "admin_customer_glossary",
                )

        with edit_customer_tab:
            if not customers:
                st.info("当前没有客户。")
            elif not staff_usernames:
                st.warning("请先创建 staff 账号。")
            else:
                customer_ids = [c["customer_id"] for c in customers]
                selected_customer = st.selectbox("选择客户", customer_ids, key="edit_customer_select")
                customer = next(c for c in customers if c["customer_id"] == selected_customer)
                default_staff_index = (
                    staff_usernames.index(customer["assigned_staff_username"])
                    if customer["assigned_staff_username"] in staff_usernames
                    else 0
                )
                with st.form("update_customer_form"):
                    st.text_input("客户 ID", value=customer["customer_id"], disabled=True)
                    customer_code = st.text_input(
                        "客户代码",
                        value=customer["customer_code"],
                        key="edit_customer_code",
                    )
                    customer_name = st.text_input(
                        "客户名称",
                        value=customer["customer_name"],
                        key="edit_customer_name",
                    )
                    group_name = st.text_input(
                        "小组",
                        value=customer["group_name"],
                        key="edit_customer_group",
                    )
                    assigned_staff_username = st.selectbox(
                        "负责人",
                        staff_usernames,
                        index=default_staff_index,
                        key="edit_customer_staff",
                    )
                    note = st.text_area(
                        "备注",
                        value=customer.get("note") or "",
                        key="edit_customer_note",
                    )
                    submitted = st.form_submit_button("保存客户修改", type="primary", use_container_width=True)
                if submitted:
                    try:
                        update_customer(
                            selected_customer,
                            customer_name,
                            customer_code,
                            group_name,
                            assigned_staff_username,
                        )
                        with get_db_connection() as conn:
                            conn.execute(
                                "UPDATE customers SET note = ? WHERE customer_id = ?",
                                (note.strip(), selected_customer),
                            )
                        st.success(f"已更新客户 {selected_customer}。")
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))
