from __future__ import annotations

import json
import random
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable, Iterator

from config import AI_RETRY_ATTEMPTS, AI_RETRY_BASE_SECONDS, AI_RETRY_MAX_SECONDS


class AIError(RuntimeError):
    error_code = "AI_ERROR"
    retryable = False

    def __init__(self, message: str = "", *, cause: Exception | None = None, attempt: int | None = None):
        super().__init__(message or self.error_code)
        self.cause = cause
        self.attempt = attempt


class AIRetryableError(AIError):
    retryable = True


class AITimeoutError(AIRetryableError):
    error_code = "AI_TIMEOUT"


class AIRateLimitError(AIRetryableError):
    error_code = "AI_RATE_LIMIT"


class AINetworkError(AIRetryableError):
    error_code = "AI_NETWORK"


class AIServerError(AIRetryableError):
    error_code = "AI_SERVER_ERROR"


class AIInvalidResponseError(AIError):
    error_code = "AI_INVALID_RESPONSE"


class AIAuthError(AIError):
    error_code = "AI_AUTH"


class AIModelUnavailableError(AIError):
    error_code = "AI_MODEL_UNAVAILABLE"


class AIInvalidRequestError(AIError):
    error_code = "AI_INVALID_REQUEST"


class AICancelledError(AIError):
    error_code = "AI_CANCELLED"


@dataclass(frozen=True)
class AIContext:
    job_id: str = ""
    job_type: str = ""
    step: str = "ai_request"
    cancel_check: Callable[[], bool] | None = None


_AI_CONTEXT: ContextVar[AIContext] = ContextVar("ai_context", default=AIContext())


@contextmanager
def ai_call_context(
    *,
    job_id: str = "",
    job_type: str = "",
    step: str = "ai_request",
    cancel_check: Callable[[], bool] | None = None,
) -> Iterator[None]:
    token = _AI_CONTEXT.set(AIContext(job_id=job_id, job_type=job_type, step=step, cancel_check=cancel_check))
    try:
        yield
    finally:
        _AI_CONTEXT.reset(token)


def _status_code(exc: Exception) -> int | None:
    code = getattr(exc, "status_code", None)
    if code is None:
        code = getattr(exc, "code", None)
    try:
        return int(code) if code is not None else None
    except (TypeError, ValueError):
        return None


def classify_ai_exception(exc: Exception, *, attempt: int | None = None) -> AIError:
    if isinstance(exc, AIError):
        exc.attempt = attempt if attempt is not None else exc.attempt
        return exc
    if isinstance(exc, json.JSONDecodeError):
        return AIInvalidResponseError("AI response is invalid JSON", cause=exc, attempt=attempt)

    status = _status_code(exc)
    name = type(exc).__name__.lower()
    message = str(exc)
    lowered = message.lower()

    if status in {401, 403} or "authentication" in name or "permissiondenied" in name:
        return AIAuthError("AI authentication failed", cause=exc, attempt=attempt)
    if status == 404 and ("model" in lowered or "not_found" in lowered):
        return AIModelUnavailableError("AI model unavailable", cause=exc, attempt=attempt)
    if status == 429 or "ratelimit" in name or "rate limit" in lowered:
        return AIRateLimitError("AI rate limit", cause=exc, attempt=attempt)
    if status in {408, 409, 500, 502, 503, 504}:
        return AIServerError(f"AI server error {status}", cause=exc, attempt=attempt)
    if status == 400 or "badrequest" in name or "invalidrequest" in name:
        return AIInvalidRequestError("AI request is invalid", cause=exc, attempt=attempt)
    if "timeout" in name or "timed out" in lowered or "timeout" in lowered:
        return AITimeoutError("AI request timed out", cause=exc, attempt=attempt)
    if any(marker in name for marker in ("connection", "network", "apierror")):
        return AINetworkError("AI network error", cause=exc, attempt=attempt)
    if any(marker in lowered for marker in ("connection", "network", "temporarily unavailable")):
        return AINetworkError("AI network error", cause=exc, attempt=attempt)
    return AIError(message or type(exc).__name__, cause=exc, attempt=attempt)


def _safe_ai_log(event: str, **fields) -> None:
    payload = {
        "event": event,
        **{k: v for k, v in fields.items() if v not in (None, "")},
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def _raise_if_cancelled(context: AIContext) -> None:
    if context.cancel_check and context.cancel_check():
        raise AICancelledError("AI request cancelled before retry")


def execute_ai_call(callable_, *, step: str = "ai_request", operation: str = "responses.create"):
    context = _AI_CONTEXT.get()
    effective_step = context.step if step == "ai_request" and context.step != "ai_request" else (step or context.step)
    total_attempts = AI_RETRY_ATTEMPTS + 1
    last_error: AIError | None = None
    for index in range(total_attempts):
        attempt = index + 1
        _raise_if_cancelled(context)
        started = time.monotonic()
        try:
            result = callable_()
            _safe_ai_log(
                "ai_call_completed",
                job_id=context.job_id,
                job_type=context.job_type,
                step=effective_step,
                operation=operation,
                attempt=attempt,
                duration_ms=int((time.monotonic() - started) * 1000),
            )
            return result
        except Exception as exc:
            classified = classify_ai_exception(exc, attempt=attempt)
            last_error = classified
            duration_ms = int((time.monotonic() - started) * 1000)
            _safe_ai_log(
                "ai_call_failed",
                job_id=context.job_id,
                job_type=context.job_type,
                step=effective_step,
                operation=operation,
                attempt=attempt,
                duration_ms=duration_ms,
                error_code=classified.error_code,
                retryable=classified.retryable,
            )
            if not classified.retryable or attempt >= total_attempts:
                raise classified from exc
            _raise_if_cancelled(context)
            sleep_seconds = min(AI_RETRY_MAX_SECONDS, AI_RETRY_BASE_SECONDS * (2 ** (attempt - 1)))
            sleep_seconds += random.uniform(0, AI_RETRY_BASE_SECONDS)
            _safe_ai_log(
                "ai_call_retry",
                job_id=context.job_id,
                job_type=context.job_type,
                step=effective_step,
                operation=operation,
                attempt=attempt + 1,
                error_code=classified.error_code,
            )
            time.sleep(sleep_seconds)
    if last_error:
        raise last_error
    raise AIError("AI call failed")


def error_metadata(exc: Exception, *, step: str = "") -> dict:
    classified = classify_ai_exception(exc)
    return {
        "error_code": classified.error_code,
        "error_step": step or _AI_CONTEXT.get().step,
        "retryable": classified.retryable,
        "error_message": str(classified),
        "ai_attempt": classified.attempt,
    }


__all__ = [
    "AIError",
    "AIRetryableError",
    "AITimeoutError",
    "AIRateLimitError",
    "AINetworkError",
    "AIServerError",
    "AIInvalidResponseError",
    "AIAuthError",
    "AIModelUnavailableError",
    "AIInvalidRequestError",
    "AICancelledError",
    "AIContext",
    "ai_call_context",
    "classify_ai_exception",
    "execute_ai_call",
    "error_metadata",
]
