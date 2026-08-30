from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ai_client
import translation_core


class FakeHTTPError(Exception):
    def __init__(self, status_code: int, message: str = ""):
        super().__init__(message or f"HTTP {status_code}")
        self.status_code = status_code


class FakeTimeout(Exception):
    pass


class FakeResponse:
    def __init__(self, output_text: str):
        self.output_text = output_text


class CountingResponses:
    def __init__(self, plan):
        self.plan = list(plan)
        self.calls: list[str] = []

    def create(self, *, model: str, **_kwargs):
        self.calls.append(model)
        if not self.plan:
            return FakeResponse('{"items": [], "unrecorded_terms": []}')
        action = self.plan.pop(0)
        if isinstance(action, Exception):
            raise action
        return FakeResponse(str(action))


class FakeClient:
    def __init__(self, plan):
        self.responses = CountingResponses(plan)


def _patch_runtime():
    originals = {
        "attempts": ai_client.AI_RETRY_ATTEMPTS,
        "base": ai_client.AI_RETRY_BASE_SECONDS,
        "max": ai_client.AI_RETRY_MAX_SECONDS,
        "sleep": ai_client.time.sleep,
        "uniform": ai_client.random.uniform,
        "model": translation_core.ANTHROPIC_MODEL,
        "fallbacks": translation_core.OPENAI_FALLBACK_MODELS,
    }
    ai_client.AI_RETRY_ATTEMPTS = 2
    ai_client.AI_RETRY_BASE_SECONDS = 0.01
    ai_client.AI_RETRY_MAX_SECONDS = 0.01
    ai_client.time.sleep = lambda _seconds: None
    ai_client.random.uniform = lambda _a, _b: 0
    translation_core.ANTHROPIC_MODEL = "primary-model"
    translation_core.OPENAI_FALLBACK_MODELS = ["fallback-model"]

    def restore():
        ai_client.AI_RETRY_ATTEMPTS = originals["attempts"]
        ai_client.AI_RETRY_BASE_SECONDS = originals["base"]
        ai_client.AI_RETRY_MAX_SECONDS = originals["max"]
        ai_client.time.sleep = originals["sleep"]
        ai_client.random.uniform = originals["uniform"]
        translation_core.ANTHROPIC_MODEL = originals["model"]
        translation_core.OPENAI_FALLBACK_MODELS = originals["fallbacks"]

    return restore


def _call_low_level(client: FakeClient):
    return translation_core._create_anthropic_message(
        client,
        model="primary-model",
        messages=[{"role": "user", "content": "hello"}],
    )


def test_timeout_then_success_count() -> None:
    restore = _patch_runtime()
    try:
        client = FakeClient([
            FakeTimeout("timed out"),
            FakeTimeout("timed out"),
            '{"items": [], "unrecorded_terms": []}',
        ])
        _call_low_level(client)
        assert client.responses.calls == ["primary-model", "primary-model", "primary-model"]
    finally:
        restore()


def test_timeout_exhausted_count() -> None:
    restore = _patch_runtime()
    try:
        client = FakeClient([FakeTimeout("timed out")] * 5)
        try:
            _call_low_level(client)
            raise AssertionError("expected timeout")
        except ai_client.AIError as err:
            assert err.error_code == "AI_TIMEOUT"
        assert client.responses.calls == ["primary-model", "primary-model", "primary-model"]
    finally:
        restore()


def test_model_not_found_falls_back_without_transport_retry() -> None:
    restore = _patch_runtime()
    try:
        client = FakeClient([
            FakeHTTPError(404, "model not_found"),
            '{"items": [], "unrecorded_terms": []}',
        ])
        _call_low_level(client)
        assert client.responses.calls == ["primary-model", "fallback-model"]
    finally:
        restore()


def test_fallback_model_gets_full_retry_budget() -> None:
    restore = _patch_runtime()
    try:
        client = FakeClient([
            FakeHTTPError(404, "model not_found"),
            FakeTimeout("timed out"),
            FakeTimeout("timed out"),
            '{"items": [], "unrecorded_terms": []}',
        ])
        _call_low_level(client)
        assert client.responses.calls == [
            "primary-model",
            "fallback-model",
            "fallback-model",
            "fallback-model",
        ]
    finally:
        restore()


def test_error_classification_request_counts() -> None:
    restore = _patch_runtime()
    try:
        scenarios = [
            (FakeHTTPError(401), "AI_AUTH", 1),
            (FakeHTTPError(400), "AI_INVALID_REQUEST", 1),
            (FakeHTTPError(429), "AI_RATE_LIMIT", 3),
            (FakeHTTPError(500), "AI_SERVER_ERROR", 3),
            (FakeTimeout("timed out"), "AI_TIMEOUT", 3),
            (ConnectionError("connection reset"), "AI_NETWORK", 3),
        ]
        for exc, expected_code, expected_count in scenarios:
            client = FakeClient([exc] * 5)
            try:
                _call_low_level(client)
                raise AssertionError("expected AI error")
            except ai_client.AIError as err:
                assert err.error_code == expected_code
            assert len(client.responses.calls) == expected_count
    finally:
        restore()


def test_cancellation_prevents_second_transport_attempt() -> None:
    restore = _patch_runtime()
    cancelled = {"value": False}
    try:
        client = FakeClient([FakeTimeout("timed out"), '{"items": [], "unrecorded_terms": []}'])

        def cancel_check():
            return cancelled["value"]

        original_create = client.responses.create

        def create_and_cancel(**kwargs):
            cancelled["value"] = True
            return original_create(**kwargs)

        client.responses.create = create_and_cancel
        with ai_client.ai_call_context(cancel_check=cancel_check):
            try:
                _call_low_level(client)
                raise AssertionError("expected cancellation")
            except ai_client.AICancelledError:
                pass
        assert len(client.responses.calls) == 1
    finally:
        restore()


def test_pdf_batch_business_amplification_n4() -> None:
    restore = _patch_runtime()
    try:
        # Every batch call returns malformed JSON, so translate_batch_resilient
        # recursively splits 4 items down to singles, then each single uses
        # _force_translate once. _force_translate does not parse JSON.
        client = FakeClient(["not json"] * 20)
        mapping, terms = translation_core.translate_batch_resilient(
            client,
            ["alpha seam", "beta strap", "gamma cup", "delta lace"],
            {},
        )
        assert terms == set()
        assert set(mapping) == {"alpha seam", "beta strap", "gamma cup", "delta lace"}
        assert len(client.responses.calls) == 11
    finally:
        restore()


def main() -> None:
    test_timeout_then_success_count()
    test_timeout_exhausted_count()
    test_model_not_found_falls_back_without_transport_retry()
    test_fallback_model_gets_full_retry_budget()
    test_error_classification_request_counts()
    test_cancellation_prevents_second_transport_attempt()
    test_pdf_batch_business_amplification_n4()
    print("ai retry amplification tests passed")


if __name__ == "__main__":
    main()
