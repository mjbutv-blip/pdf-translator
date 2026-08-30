from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ai_client


class FakeHTTPError(Exception):
    def __init__(self, status_code: int, message: str = ""):
        super().__init__(message or f"HTTP {status_code}")
        self.status_code = status_code


class FakeTimeout(Exception):
    pass


class FakeConnectionError(Exception):
    pass


def _patch_fast_retry():
    original_attempts = ai_client.AI_RETRY_ATTEMPTS
    original_base = ai_client.AI_RETRY_BASE_SECONDS
    original_max = ai_client.AI_RETRY_MAX_SECONDS
    original_sleep = ai_client.time.sleep
    original_uniform = ai_client.random.uniform
    ai_client.AI_RETRY_ATTEMPTS = 2
    ai_client.AI_RETRY_BASE_SECONDS = 0.01
    ai_client.AI_RETRY_MAX_SECONDS = 0.01
    ai_client.time.sleep = lambda _seconds: None
    ai_client.random.uniform = lambda _a, _b: 0

    def restore():
        ai_client.AI_RETRY_ATTEMPTS = original_attempts
        ai_client.AI_RETRY_BASE_SECONDS = original_base
        ai_client.AI_RETRY_MAX_SECONDS = original_max
        ai_client.time.sleep = original_sleep
        ai_client.random.uniform = original_uniform

    return restore


def test_retry_then_success(exc: Exception, expected_code: str) -> None:
    restore = _patch_fast_retry()
    calls = {"n": 0}
    try:
        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise exc
            return "ok"

        assert ai_client.execute_ai_call(flaky, step="translate_text") == "ok"
        assert calls["n"] == 3
        assert ai_client.classify_ai_exception(exc).error_code == expected_code
    finally:
        restore()


def test_no_retry(exc: Exception, expected_code: str) -> None:
    restore = _patch_fast_retry()
    calls = {"n": 0}
    try:
        def bad():
            calls["n"] += 1
            raise exc

        try:
            ai_client.execute_ai_call(bad, step="translate_text")
            raise AssertionError("expected structured AI error")
        except ai_client.AIError as err:
            assert err.error_code == expected_code
            assert err.retryable is False
        assert calls["n"] == 1
    finally:
        restore()


def test_retry_exhausted() -> None:
    restore = _patch_fast_retry()
    calls = {"n": 0}
    try:
        def always_timeout():
            calls["n"] += 1
            raise FakeTimeout("timed out")

        try:
            ai_client.execute_ai_call(always_timeout, step="translate_text")
            raise AssertionError("expected timeout")
        except ai_client.AIError as err:
            assert err.error_code == "AI_TIMEOUT"
            assert err.retryable is True
            assert err.attempt == 3
        assert calls["n"] == 3
    finally:
        restore()


def test_cancellation_before_retry() -> None:
    restore = _patch_fast_retry()
    calls = {"n": 0}
    cancelled = {"value": False}
    try:
        def flaky():
            calls["n"] += 1
            cancelled["value"] = True
            raise FakeTimeout("timed out")

        with ai_client.ai_call_context(cancel_check=lambda: cancelled["value"]):
            try:
                ai_client.execute_ai_call(flaky, step="translate_text")
                raise AssertionError("expected cancellation")
            except ai_client.AICancelledError as err:
                assert err.error_code == "AI_CANCELLED"
        assert calls["n"] == 1
    finally:
        restore()


def test_invalid_response_classification() -> None:
    try:
        json.loads("{")
    except json.JSONDecodeError as exc:
        err = ai_client.classify_ai_exception(exc)
        assert err.error_code == "AI_INVALID_RESPONSE"
        assert err.retryable is False


def test_safe_logging() -> None:
    restore = _patch_fast_retry()
    secret = "sk-fake-secret-for-test"
    document_text = "CONFIDENTIAL DOCUMENT TEXT"
    buf = io.StringIO()
    try:
        def flaky():
            raise FakeTimeout(f"timed out while handling {secret} {document_text}")

        with redirect_stdout(buf):
            try:
                ai_client.execute_ai_call(flaky, step="translate_text")
            except ai_client.AIError:
                pass
        logs = buf.getvalue()
        assert secret not in logs
        assert document_text not in logs
        assert "AI_TIMEOUT" in logs
    finally:
        restore()


def main() -> None:
    test_retry_then_success(FakeTimeout("timed out"), "AI_TIMEOUT")
    test_retry_then_success(FakeHTTPError(429), "AI_RATE_LIMIT")
    test_retry_then_success(FakeHTTPError(503), "AI_SERVER_ERROR")
    test_retry_then_success(FakeConnectionError("connection reset"), "AI_NETWORK")
    test_no_retry(FakeHTTPError(401), "AI_AUTH")
    test_no_retry(FakeHTTPError(400), "AI_INVALID_REQUEST")
    test_retry_exhausted()
    test_cancellation_before_retry()
    test_invalid_response_classification()
    test_safe_logging()
    print("ai client tests passed")


if __name__ == "__main__":
    main()
