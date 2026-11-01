from __future__ import annotations

import io
import sys
from pathlib import Path

import fitz
import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import translation_core


def _glossary_bytes() -> bytes:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["英文术语", "中文翻译"])
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def test_build_pdf_translation_units_joins_spans_and_wrapped_lines() -> None:
    blocks = [{
        "type": 0,
        "lines": [
            {"spans": [
                {"text": "2.5cm", "bbox": (20, 100, 55, 112), "size": 10},
                {"text": "strong", "bbox": (58, 100, 100, 112), "size": 10},
            ]},
            {"spans": [
                {"text": "knitted", "bbox": (20, 114, 65, 126), "size": 10},
            ]},
        ],
    }]

    units = translation_core._build_pdf_translation_units(blocks)

    assert len(units) == 1
    assert units[0]["text"] == "2.5cm strong knitted"
    assert units[0]["line_count"] == 2
    assert units[0]["span_count"] == 3
    assert units[0]["bbox"] == (20.0, 100.0, 100.0, 126.0)


def test_build_pdf_translation_units_keeps_distant_table_cells_separate() -> None:
    blocks = [{
        "type": 0,
        "lines": [{"spans": [
            {"text": "Season", "bbox": (20, 100, 55, 112), "size": 10},
            {"text": "Supplier", "bbox": (180, 100, 225, 112), "size": 10},
        ]}],
    }]

    units = translation_core._build_pdf_translation_units(blocks)

    assert [unit["text"] for unit in units] == ["Season", "Supplier"]


def test_header_margin_and_feedback_table_labels_are_preserved() -> None:
    assert translation_core._is_header_or_metadata_text(
        "arbitrary changing customer header",
        bbox=(20, 10, 240, 30),
        page_height=800,
    )
    assert translation_core._is_header_or_metadata_text(
        "Pantone colours",
        bbox=(20, 140, 150, 158),
        page_height=800,
    )
    assert not translation_core._is_header_or_metadata_text(
        "2.5cm strong knitted",
        bbox=(20, 300, 180, 325),
        page_height=800,
    )


def test_pdf_translation_sends_wrapped_phrase_as_one_ai_item(monkeypatch) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_textbox(
        fitz.Rect(72, 260, 260, 320),
        "2.5cm strong\nknitted",
        fontsize=11,
    )
    source = document.tobytes()
    document.close()
    calls: list[list[str]] = []

    def fake_batch(_client, texts, _glossary):
        calls.append(list(texts))
        return {text: "下摆宽度2.5cm的有支撑力针织结构" for text in texts}, set()

    monkeypatch.setattr(translation_core, "OpenAI", lambda **_kwargs: object())
    monkeypatch.setattr(translation_core, "translate_batch_resilient", fake_batch)

    translation_core.run_pdf_translation(
        source,
        _glossary_bytes(),
        str(translation_core.DEFAULT_FONT),
        "test-key",
        lambda *_args: None,
        lambda _text: None,
        lambda _progress: None,
    )

    assert calls == [["2.5cm strong knitted"]]


def test_color_code_and_name_group_never_reaches_ai(monkeypatch) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_textbox(
        fitz.Rect(72, 260, 260, 320),
        "18-1409 TCX\nPepper",
        fontsize=11,
    )
    source = document.tobytes()
    document.close()

    def fail_if_called(_client, texts, _glossary):
        raise AssertionError(f"color content reached AI: {texts}")

    monkeypatch.setattr(translation_core, "OpenAI", lambda **_kwargs: object())
    monkeypatch.setattr(translation_core, "translate_batch_resilient", fail_if_called)

    translation_core.run_pdf_translation(
        source,
        _glossary_bytes(),
        str(translation_core.DEFAULT_FONT),
        "test-key",
        lambda *_args: None,
        lambda _text: None,
        lambda _progress: None,
    )
