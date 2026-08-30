from __future__ import annotations

import ast
from pathlib import Path


APP_SOURCE = Path(__file__).resolve().parents[1] / "app.py"


def _source() -> str:
    return APP_SOURCE.read_text(encoding="utf-8")


def test_no_periodic_fragment_rerun() -> None:
    source = _source()
    assert "run_every" not in source
    assert "experimental_rerun" not in source


def test_uploaders_are_not_inside_fragment_functions() -> None:
    tree = ast.parse(_source())
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        is_fragment = any(
            isinstance(dec, ast.Attribute) and dec.attr == "fragment"
            or isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and dec.func.attr == "fragment"
            for dec in node.decorator_list
        )
        if not is_fragment:
            continue
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == "file_uploader"
            ):
                raise AssertionError(f"file_uploader found inside fragment {node.name}")


def test_uploader_keys_are_nonce_based() -> None:
    source = _source()
    assert 'key=f"pdf_uploader_{pdf_upload_nonce}"' in source
    assert 'key=f"pdf_font_uploader_{pdf_upload_nonce}"' in source
    assert 'key=f"excel_uploader_{excel_upload_nonce}"' in source
    assert "time.time()" not in source
    assert "uuid.uuid4()" not in source


def main() -> None:
    test_no_periodic_fragment_rerun()
    test_uploaders_are_not_inside_fragment_functions()
    test_uploader_keys_are_nonce_based()
    print("streamlit upload rerun boundary tests passed")


if __name__ == "__main__":
    main()
