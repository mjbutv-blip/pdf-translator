from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, request


DEFAULT_RFQ_MANAGER_API_URL = "http://127.0.0.1:8000"
RFQ_MANAGER_API_URL_ENV = "RFQ_MANAGER_API_URL"


class UnifiedAuthError(PermissionError):
    """Unified RFQ authentication/authorization failed closed."""


def get_rfq_manager_api_url() -> str:
    return (os.getenv(RFQ_MANAGER_API_URL_ENV) or DEFAULT_RFQ_MANAGER_API_URL).rstrip("/")


def _require_token(access_token: str | None) -> str:
    token = str(access_token or "").strip()
    if not token:
        raise UnifiedAuthError("RFQ access token is required")
    return token


def _rfq_get_json(path: str, access_token: str, timeout: float = 10.0) -> Any:
    token = _require_token(access_token)
    url = f"{get_rfq_manager_api_url()}{path}"
    req = request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        },
        method="GET",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", 200)
            if status < 200 or status >= 300:
                raise UnifiedAuthError(f"RFQ auth failed with status {status}")
            raw = resp.read()
    except error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise UnifiedAuthError("RFQ token is invalid or unauthorized") from exc
        raise UnifiedAuthError(f"RFQ service error: HTTP {exc.code}") from exc
    except error.URLError as exc:
        raise UnifiedAuthError("RFQ service unavailable") from exc
    except TimeoutError as exc:
        raise UnifiedAuthError("RFQ service unavailable") from exc

    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UnifiedAuthError("RFQ response is malformed") from exc


def _normalize_context_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise UnifiedAuthError("RFQ /auth/context response is malformed")
    role = str(payload.get("role") or "").strip()
    username = str(payload.get("username") or "").strip()
    user_id = str(payload.get("user_id") or "").strip()
    is_active = payload.get("is_active")
    if not user_id or not username or not role:
        raise UnifiedAuthError("RFQ /auth/context response is missing required fields")
    if is_active is not True:
        raise UnifiedAuthError("RFQ user is inactive")
    if role not in {"company_admin", "group_leader", "staff", "viewer"}:
        raise UnifiedAuthError("RFQ role is not recognized")
    permissions = payload.get("permissions")
    return {
        "user_id": user_id,
        "username": username,
        "display_name": str(payload.get("display_name") or username),
        "role": role,
        "legacy_role": payload.get("legacy_role"),
        "group_id": payload.get("group_id"),
        "group_name": payload.get("group_name"),
        "is_active": True,
        "customer_access_mode": payload.get("customer_access_mode"),
        "permissions": permissions if isinstance(permissions, dict) else {},
    }


def _normalize_accessible_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        if isinstance(payload.get("customers"), list):
            rows = payload["customers"]
        elif isinstance(payload.get("data"), list):
            rows = payload["data"]
        else:
            raise UnifiedAuthError("RFQ /customers/accessible response is malformed")
    elif isinstance(payload, list):
        rows = payload
    else:
        raise UnifiedAuthError("RFQ /customers/accessible response is malformed")

    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise UnifiedAuthError("RFQ accessible customer row is malformed")
        normalized.append({
            "customer_id": row.get("customer_id"),
            "customer_code": row.get("customer_code"),
            "pdf_customer_id": row.get("pdf_customer_id"),
            "quote_profile_code": row.get("quote_profile_code"),
            "mapping_status": row.get("mapping_status"),
            "can_view": bool(row.get("can_view")),
            "can_use_for_quote": bool(row.get("can_use_for_quote")),
            "can_edit": bool(row.get("can_edit")),
        })
    return normalized


def get_unified_current_user(access_token: str) -> dict[str, Any]:
    """Validate a RFQ JWT through RFQ Manager and return the normalized principal."""
    return _normalize_context_payload(_rfq_get_json("/api/v1/auth/context", access_token))


def get_accessible_customers(access_token: str) -> list[dict[str, Any]]:
    """Return RFQ-authorized customers for the validated token."""
    return _normalize_accessible_payload(_rfq_get_json("/api/v1/customers/accessible", access_token))


def authorize_pdf_customer(
    access_token: str,
    pdf_customer_id: str,
    *,
    require_write: bool = False,
    require_quote_use: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    principal = get_unified_current_user(access_token)
    target = str(pdf_customer_id or "").strip()
    if not target:
        raise UnifiedAuthError("pdf_customer_id is required")
    if principal["role"] == "viewer" and (require_write or require_quote_use):
        raise UnifiedAuthError("permission denied: viewer has read-only access")

    matches = [
        row for row in get_accessible_customers(access_token)
        if str(row.get("pdf_customer_id") or "").strip() == target
    ]
    for row in matches:
        if row.get("mapping_status") != "confirmed":
            continue
        if not row.get("can_view"):
            continue
        if require_write and not row.get("can_edit"):
            continue
        if require_quote_use and not row.get("can_use_for_quote"):
            continue
        return principal, row

    if matches:
        raise UnifiedAuthError("permission denied: customer mapping is not confirmed or lacks required access")
    raise UnifiedAuthError("permission denied: pdf customer is not accessible")


def confirmed_pdf_customer_ids(access_token: str) -> tuple[dict[str, Any], list[str]]:
    principal = get_unified_current_user(access_token)
    ids: list[str] = []
    seen: set[str] = set()
    for row in get_accessible_customers(access_token):
        pdf_customer_id = str(row.get("pdf_customer_id") or "").strip()
        if (
            pdf_customer_id
            and row.get("mapping_status") == "confirmed"
            and row.get("can_view")
            and pdf_customer_id not in seen
        ):
            seen.add(pdf_customer_id)
            ids.append(pdf_customer_id)
    return principal, ids


__all__ = [
    "UnifiedAuthError",
    "get_rfq_manager_api_url",
    "get_unified_current_user",
    "get_accessible_customers",
    "authorize_pdf_customer",
    "confirmed_pdf_customer_ids",
]
