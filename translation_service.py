from __future__ import annotations

import io
import json
import traceback
from pathlib import Path
from typing import Any
import zipfile

import openpyxl
import pandas as pd
import fitz


APP_PATH = Path(__file__).with_name("app.py")
UI_MARKER = "# ── Streamlit page"


def _load_app_core() -> None:
    """Load the translation/core helpers from app.py without executing the UI."""
    source = APP_PATH.read_text(encoding="utf-8")
    marker_pos = source.find(UI_MARKER)
    if marker_pos == -1:
        raise RuntimeError(f"Cannot find UI marker in {APP_PATH.name}")
    core_source = source[:marker_pos]
    exec(compile(core_source, str(APP_PATH), "exec"), globals())


_load_app_core()


def _detect_file_type(filename: str, file_bytes: bytes, file_type: str | None = None) -> str:
    ext = (file_type or Path(filename or "").suffix.lstrip(".")).lower().strip()
    if ext in {"pdf", "xlsx", "xls"}:
        return ext
    if file_bytes[:4] == b"%PDF":
        return "pdf"
    if file_bytes[:8] == b"PK\x03\x04":
        return "xlsx"
    if file_bytes[:4] == b"\xd0\xcf\x11\xe0":
        return "xls"
    raise ValueError("无法识别文件类型，请提供 pdf / xlsx / xls 文件")


def _load_customer_glossary_bytes(customer_id: str) -> bytes:
    glossary_df = get_customer_glossary_df(customer_id)
    return glossary_df_to_xlsx_bytes(glossary_df)


def _convert_xls_to_xlsx_bytes(file_bytes: bytes, filename: str) -> tuple[bytes, dict[str, Any]]:
    sheet_rows, parse_error = _read_excel_sheet_rows(file_bytes, filename)
    if parse_error:
        raise ValueError(parse_error)

    wb = openpyxl.Workbook()
    first = True
    sheet_names: list[str] = []
    for sheet_name, rows in sheet_rows:
        sheet_names.append(sheet_name)
        if first:
            ws = wb.active
            ws.title = sheet_name[:31] or "Sheet1"
            first = False
        else:
            ws = wb.create_sheet(title=sheet_name[:31] or f"Sheet{len(sheet_names)}")
        for row in rows:
            ws.append([cell if cell is not None else "" for cell in row])
    out = io.BytesIO()
    wb.save(out)
    return out.getvalue(), {
        "excel_conversion": "xls_to_xlsx_values_only",
        "xls_sheet_names": sheet_names,
    }


def _build_pdf_report_bytes(
    source_filename: str,
    total_pages: int,
    selected_pages,
    scope_detection,
    scope_mode: str,
    glossary_report_bytes: bytes,
) -> tuple[bytes, dict[str, Any]]:
    scope_cfg = {
        "total_pages": total_pages,
        "detection": scope_detection or [],
    }
    scope_report = _pdf_scope_report_rows(
        source_filename,
        scope_cfg,
        selected_pages,
        scope_detection,
        scope_mode,
    )
    scope_report_bytes = build_scope_report_xlsx(scope_report) if scope_report else b""
    report_bytes = b""
    report_kind = ""
    if glossary_report_bytes and scope_report_bytes:
        report_zip = io.BytesIO()
        with zipfile.ZipFile(report_zip, "w") as zf:
            zf.writestr("unrecorded_terms.xlsx", glossary_report_bytes)
            zf.writestr("scope_report.xlsx", scope_report_bytes)
        report_bytes = report_zip.getvalue()
        report_kind = "zip"
    elif glossary_report_bytes:
        report_bytes = glossary_report_bytes
        report_kind = "unrecorded"
    elif scope_report_bytes:
        report_bytes = scope_report_bytes
        report_kind = "scope"

    meta = {
        "has_scope_report": bool(scope_report),
        "has_unrecorded_terms": bool(glossary_report_bytes),
        "report_kind": report_kind,
    }
    return report_bytes, meta


def _build_excel_report_bundle(
    report_rows: list[dict],
    img_report_rows: list[dict] | None = None,
) -> bytes:
    report_wb = openpyxl.Workbook()
    report_ws = report_wb.active
    report_ws.title = "翻译报告"
    headers = [
        "sheet_name", "cell_coordinate", "original_text", "translated_text",
        "status", "skip_reason", "is_merged_cell", "layout_warning",
        "scope_mode", "selected_sheets", "skipped_sheets",
        "detection_score", "detection_reason",
    ]
    report_ws.append(headers)
    for cell in report_ws[1]:
        cell.font = openpyxl.styles.Font(bold=True)
    for r in report_rows:
        report_ws.append([r.get(h, "") for h in headers])
    for col_letter, width in zip(
        "ABCDEFGHIJKLM",
        [12, 14, 35, 35, 12, 24, 14, 14, 18, 35, 35, 12, 70],
    ):
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


def translate_pdf_document(
    *,
    customer_id: str,
    filename: str,
    file_bytes: bytes,
    api_key: str,
    user: dict | None = None,
    font_path: str | None = None,
    selected_pages: list[int] | None = None,
    scope_mode: str = "all",
    scope_detection: list[dict] | None = None,
    source_file_name: str | None = None,
    created_by: str | None = None,
) -> dict[str, Any]:
    try:
        if user is not None and not can_use_customer_glossary(user, customer_id):
            raise PermissionError("当前用户无权使用该客户术语库")
        glossary_bytes = _load_customer_glossary_bytes(customer_id)
        resolved_font_path = font_path or (str(DEFAULT_FONT) if DEFAULT_FONT.exists() else "")
        if not resolved_font_path:
            raise FileNotFoundError("未找到默认字体，请提供 font_path")
        if not created_by:
            created_by = user.get("username") if isinstance(user, dict) and user.get("username") else "service"
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            total_pages = len(doc)
        pdf_out, xlsx_out, n_terms, review_summary = run_pdf_translation(
            pdf_bytes=file_bytes,
            glossary_bytes=glossary_bytes,
            font_path=resolved_font_path,
            api_key=api_key,
            on_page=lambda *_: None,
            on_block=lambda *_: None,
            on_progress=lambda *_: None,
            customer_id=customer_id,
            source_file_name=source_file_name or filename,
            created_by=created_by or (user.get("username") if isinstance(user, dict) else ""),
            selected_pages=selected_pages,
            scope_mode=scope_mode,
            scope_detection=scope_detection,
        )
        report_bytes, report_meta = _build_pdf_report_bytes(
            source_filename=source_file_name or filename,
            total_pages=total_pages,
            selected_pages=selected_pages,
            scope_detection=scope_detection,
            scope_mode=scope_mode,
            glossary_report_bytes=xlsx_out,
        )
        meta = {
            "n_terms": n_terms,
            "review_summary": review_summary,
            "total_pages": total_pages,
            **report_meta,
        }
        return {
            "status": "completed",
            "customer_id": customer_id,
            "source_filename": source_file_name or filename,
            "file_type": "pdf",
            "translated_file_bytes": pdf_out,
            "report_bytes": report_bytes,
            "meta": meta,
            "error": None,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "customer_id": customer_id,
            "source_filename": source_file_name or filename,
            "file_type": "pdf",
            "translated_file_bytes": None,
            "report_bytes": None,
            "meta": {"traceback": traceback.format_exc()},
            "error": str(exc),
        }


def translate_excel_document(
    *,
    customer_id: str,
    filename: str,
    file_bytes: bytes,
    api_key: str,
    user: dict | None = None,
    translate_images: bool = False,
    selected_sheets: list[str] | None = None,
    scope_mode: str = "all",
    scope_detection: list[dict] | None = None,
    source_file_name: str | None = None,
    created_by: str | None = None,
) -> dict[str, Any]:
    try:
        if user is not None and not can_use_customer_glossary(user, customer_id):
            raise PermissionError("当前用户无权使用该客户术语库")
        glossary_bytes = _load_customer_glossary_bytes(customer_id)
        glossary_dict = load_glossary(glossary_bytes)
        working_bytes = file_bytes
        conversion_meta: dict[str, Any] = {}
        if Path(filename).suffix.lower() == ".xls":
            working_bytes, conversion_meta = _convert_xls_to_xlsx_bytes(file_bytes, filename)
        if not created_by:
            created_by = user.get("username") if isinstance(user, dict) and user.get("username") else "service"
        excel_out, n_cells, n_images, report_rows, review_summary = run_excel_translation(
            xlsx_bytes=working_bytes,
            glossary_bytes=glossary_bytes,
            api_key=api_key,
            on_cell=lambda *_: None,
            on_progress=lambda *_: None,
            translate_images=False,
            customer_id=customer_id,
            source_file_name=source_file_name or filename,
            created_by=created_by or (user.get("username") if isinstance(user, dict) else ""),
            selected_sheets=selected_sheets,
            scope_mode=scope_mode,
            scope_detection=scope_detection,
        )
        img_report_rows: list[dict] = []
        if translate_images:
            client = OpenAI(api_key=api_key)
            excel_out, img_report_rows = add_translated_textboxes_to_excel(
                excel_out,
                client,
                glossary_dict,
                on_image=lambda *_: None,
                selected_sheets=selected_sheets,
            )
        report_bytes = _build_excel_report_bundle(report_rows, img_report_rows)
        meta = {
            "n_cells": n_cells,
            "n_images": n_images,
            "review_summary": review_summary,
            **conversion_meta,
        }
        return {
            "status": "completed",
            "customer_id": customer_id,
            "source_filename": source_file_name or filename,
            "file_type": "excel",
            "translated_file_bytes": excel_out,
            "report_bytes": report_bytes,
            "meta": meta,
            "error": None,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "customer_id": customer_id,
            "source_filename": source_file_name or filename,
            "file_type": "excel",
            "translated_file_bytes": None,
            "report_bytes": None,
            "meta": {"traceback": traceback.format_exc()},
            "error": str(exc),
        }


def translate_customer_document(
    *,
    customer_id: str,
    filename: str,
    file_bytes: bytes,
    api_key: str,
    user: dict | None = None,
    file_type: str | None = None,
    font_path: str | None = None,
    translate_images: bool = False,
    selected_pages: list[int] | None = None,
    selected_sheets: list[str] | None = None,
    scope_mode: str = "all",
    scope_detection: list[dict] | None = None,
    source_file_name: str | None = None,
    created_by: str | None = None,
) -> dict[str, Any]:
    kind = _detect_file_type(filename, file_bytes, file_type)
    if kind == "pdf":
        return translate_pdf_document(
            customer_id=customer_id,
            filename=filename,
            file_bytes=file_bytes,
            api_key=api_key,
            user=user,
            font_path=font_path,
            selected_pages=selected_pages,
            scope_mode=scope_mode,
            scope_detection=scope_detection,
            source_file_name=source_file_name,
            created_by=created_by,
        )
    if kind in {"xlsx", "xls"}:
        return translate_excel_document(
            customer_id=customer_id,
            filename=filename,
            file_bytes=file_bytes,
            api_key=api_key,
            user=user,
            translate_images=translate_images,
            selected_sheets=selected_sheets,
            scope_mode=scope_mode,
            scope_detection=scope_detection,
            source_file_name=source_file_name,
            created_by=created_by,
        )
    return {
        "status": "failed",
        "customer_id": customer_id,
        "source_filename": source_file_name or filename,
        "file_type": kind,
        "translated_file_bytes": None,
        "report_bytes": None,
        "meta": {},
        "error": f"不支持的文件类型：{kind}",
    }


__all__ = [
    "translate_customer_document",
    "translate_pdf_document",
    "translate_excel_document",
]
