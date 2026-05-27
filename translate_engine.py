import fitz
import openpyxl
import anthropic
import json
import os
import re
import sys
from pathlib import Path

BASE_DIR    = Path(__file__).parent
PDF_PATH    = BASE_DIR / "PASSION WINTER 25.pdf"
GLOSSARY    = BASE_DIR / "glossary.xlsx"
FONT_PATH   = str(BASE_DIR / "font.ttf")
OUTPUT_PDF  = BASE_DIR / "PASSION_WINTER_25_中文版.pdf"
OUTPUT_XLS  = BASE_DIR / "unrecorded_terms.xlsx"


def load_glossary() -> dict[str, str]:
    """Return {english_term: chinese_term} from glossary.xlsx."""
    wb = openpyxl.load_workbook(GLOSSARY)
    ws = wb.active
    headers = [str(c.value).strip() if c.value else "" for c in ws[1]]

    # locate columns – file has 中文 first, 英文 second
    chi_idx = eng_idx = None
    for i, h in enumerate(headers):
        if "中文" in h or h.lower() in ("chinese",):
            chi_idx = i
        if "英文" in h or h.lower() in ("english",):
            eng_idx = i

    if chi_idx is None or eng_idx is None:
        print(f"  [WARN] Could not identify columns; headers: {headers}")
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
    return {k: v for k, v in glossary.items() if k.lower() in tl}


def translate(client: anthropic.Anthropic, text: str, glossary: dict) -> dict:
    rel = relevant_glossary(text, glossary)

    gloss_block = ""
    if rel:
        lines = "\n".join(f"  {k} → {v}" for k, v in rel.items())
        gloss_block = f"强制术语对照（务必照搬，不得自行发挥）：\n{lines}\n\n"

    prompt = (
        "你是一名专业服装行业翻译，请将以下英文文本翻译成地道的中文服装术语。\n\n"
        f"{gloss_block}"
        f"待翻译文本：\n{text}\n\n"
        "规则：\n"
        "1. 凡出现上方术语对照中的词汇，必须使用对照表中的中文译法，禁止替换。\n"
        "2. 保留款号、数字、尺码、货号等编码不翻译。\n"
        "3. 识别文本中出现的、术语对照表里**未收录**的服装/纺织行业专业英文词汇，放入 unrecorded_terms。\n"
        "4. 只返回 JSON，不要有任何多余的文字或 markdown 代码块。\n\n"
        '返回格式：{"translated_text": "中文结果", "unrecorded_terms": ["term1", "term2"]}'
    )

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = msg.content[0].text.strip()
    # strip markdown fences if present
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    # extract outermost JSON object
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        raw = m.group()
    return json.loads(raw)


def insert_text(page, bbox, text: str, size: float):
    rect = fitz.Rect(bbox)
    for s in [size, size * 0.85, size * 0.7, 8.0, 7.0]:
        rc = page.insert_textbox(
            rect,
            text,
            fontname="myfont",
            fontfile=FONT_PATH,
            fontsize=s,
            color=(0, 0, 0),
            align=0,
        )
        if rc >= 0:
            return
    # last resort – clip without error
    page.insert_textbox(rect, text, fontname="myfont", fontfile=FONT_PATH,
                        fontsize=7.0, color=(0, 0, 0), align=0)


def main():
    for p in (PDF_PATH, GLOSSARY, FONT_PATH):
        if not Path(p).exists():
            sys.exit(f"[ERROR] 文件不存在: {p}")

    print("正在载入术语库 …")
    glossary = load_glossary()
    print(f"  共 {len(glossary)} 条术语")

    client = anthropic.Anthropic()

    print(f"正在打开 PDF: {PDF_PATH.name} …")
    doc = fitz.open(str(PDF_PATH))
    total = len(doc)
    print(f"  共 {total} 页")

    all_unrecorded: set[str] = set()

    for pn in range(total):
        page = doc[pn]
        print(f"\n── 第 {pn + 1}/{total} 页 ──")

        blocks = page.get_text("dict")["blocks"]

        text_blocks = []
        for blk in blocks:
            if blk["type"] != 0:
                continue
            parts, sizes = [], []
            for ln in blk.get("lines", []):
                for sp in ln.get("spans", []):
                    t = sp["text"]
                    if t.strip():
                        parts.append(t)
                        sizes.append(sp["size"])
                parts.append("\n")
            text = "".join(parts).strip()
            if not text:
                continue
            avg_size = sum(sizes) / len(sizes) if sizes else 10.0
            text_blocks.append({"bbox": blk["bbox"], "text": text, "size": avg_size})

        print(f"  找到 {len(text_blocks)} 个文本块")
        if not text_blocks:
            continue

        # translate
        results = []
        for i, blk in enumerate(text_blocks):
            preview = blk["text"][:60].replace("\n", " ")
            print(f"  [{i+1}/{len(text_blocks)}] 翻译: {preview} …")
            try:
                res = translate(client, blk["text"], glossary)
                translated = res.get("translated_text", blk["text"])
                terms = [t for t in res.get("unrecorded_terms", []) if t.strip()]
                all_unrecorded.update(terms)
                if terms:
                    print(f"    未收录术语: {terms}")
            except Exception as e:
                print(f"    [WARN] 翻译失败，保留原文: {e}")
                translated = blk["text"]

            results.append({**blk, "translated": translated})

        # redact originals
        for r in results:
            page.add_redact_annot(fitz.Rect(r["bbox"]), fill=(1, 1, 1))
        page.apply_redactions()

        # insert Chinese text
        for r in results:
            insert_text(page, r["bbox"], r["translated"], r["size"])

    print(f"\n正在保存 PDF → {OUTPUT_PDF.name} …")
    doc.save(str(OUTPUT_PDF), garbage=4, deflate=True)
    doc.close()
    print("  PDF 已保存！")

    if all_unrecorded:
        print(f"\n正在导出 {len(all_unrecorded)} 条未收录术语 → {OUTPUT_XLS.name} …")
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "未收录术语"
        ws.append(["未收录英文术语"])
        ws["A1"].font = openpyxl.styles.Font(bold=True)
        for term in sorted(all_unrecorded):
            ws.append([term])
        ws.column_dimensions["A"].width = 45
        wb.save(str(OUTPUT_XLS))
        print("  已保存！")
    else:
        print("\n无未收录术语。")

    print("\n========== 完成 ==========")
    print(f"中文 PDF : {OUTPUT_PDF}")
    print(f"未收录词汇: {OUTPUT_XLS if all_unrecorded else '无'}")


if __name__ == "__main__":
    main()
