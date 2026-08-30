from __future__ import annotations

import os
from pathlib import Path

APP_DIR = Path(__file__).parent
DEFAULT_FONT = APP_DIR / "font.ttf"
DEFAULT_GLOSSARY = APP_DIR / "glossary.xlsx"
DB_PATH = APP_DIR / "pdf_project.db"

_DEFAULT_OPENAI_MODEL = "gpt-5.6-terra"


def _secret_or_env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _int_secret_or_env(name: str, default: int) -> int:
    try:
        return int(_secret_or_env(name, str(default)))
    except (TypeError, ValueError):
        return default


def _float_secret_or_env(name: str, default: float) -> float:
    try:
        return float(_secret_or_env(name, str(default)))
    except (TypeError, ValueError):
        return default


def _database_url() -> str:
    return _secret_or_env("DATABASE_URL", "").strip()


def _use_postgres() -> bool:
    return bool(_database_url())


OPENAI_API_KEY = _secret_or_env("OPENAI_API_KEY", "") or _secret_or_env("ANTHROPIC_API_KEY", "")
OPENAI_MODEL = _secret_or_env("OPENAI_MODEL", _DEFAULT_OPENAI_MODEL)
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
PDF_WORKER_POLL_SECONDS = max(1, _int_secret_or_env("PDF_WORKER_POLL_SECONDS", 5))
PDF_JOB_STALE_SECONDS = max(30, _int_secret_or_env("PDF_JOB_STALE_SECONDS", 300))
PDF_JOB_MAX_ATTEMPTS = max(1, _int_secret_or_env("PDF_JOB_MAX_ATTEMPTS", 3))
PDF_JOB_HEARTBEAT_SECONDS = max(5, _int_secret_or_env("PDF_JOB_HEARTBEAT_SECONDS", 15))
SQLITE_TIMEOUT_SECONDS = max(1, _int_secret_or_env("SQLITE_TIMEOUT_SECONDS", 10))
SQLITE_BUSY_TIMEOUT_MS = max(1000, _int_secret_or_env("SQLITE_BUSY_TIMEOUT_MS", 10000))
SQLITE_LOCK_RETRY_ATTEMPTS = max(0, _int_secret_or_env("SQLITE_LOCK_RETRY_ATTEMPTS", 4))
SQLITE_LOCK_RETRY_BASE_SECONDS = max(
    0.01,
    _float_secret_or_env("SQLITE_LOCK_RETRY_BASE_SECONDS", 0.05),
)
AI_RETRY_ATTEMPTS = max(0, _int_secret_or_env("AI_RETRY_ATTEMPTS", 2))
AI_RETRY_BASE_SECONDS = max(0.01, _float_secret_or_env("AI_RETRY_BASE_SECONDS", 0.5))
AI_RETRY_MAX_SECONDS = max(AI_RETRY_BASE_SECONDS, _float_secret_or_env("AI_RETRY_MAX_SECONDS", 4.0))

# Backward-compatible aliases so the rest of the file can be migrated gradually.
ANTHROPIC_MODEL = OPENAI_MODEL
ANTHROPIC_TIMEOUT_SECONDS = OPENAI_TIMEOUT_SECONDS
ANTHROPIC_FALLBACK_MODELS = OPENAI_FALLBACK_MODELS

__all__ = [
    "APP_DIR",
    "DEFAULT_FONT",
    "DEFAULT_GLOSSARY",
    "DB_PATH",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_TIMEOUT_SECONDS",
    "OPENAI_REASONING_EFFORT",
    "OPENAI_FALLBACK_MODELS",
    "PDF_WORKER_POLL_SECONDS",
    "PDF_JOB_STALE_SECONDS",
    "PDF_JOB_MAX_ATTEMPTS",
    "PDF_JOB_HEARTBEAT_SECONDS",
    "SQLITE_TIMEOUT_SECONDS",
    "SQLITE_BUSY_TIMEOUT_MS",
    "SQLITE_LOCK_RETRY_ATTEMPTS",
    "SQLITE_LOCK_RETRY_BASE_SECONDS",
    "AI_RETRY_ATTEMPTS",
    "AI_RETRY_BASE_SECONDS",
    "AI_RETRY_MAX_SECONDS",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_TIMEOUT_SECONDS",
    "ANTHROPIC_FALLBACK_MODELS",
    "_secret_or_env",
    "_int_secret_or_env",
    "_float_secret_or_env",
    "_database_url",
    "_use_postgres",
]
