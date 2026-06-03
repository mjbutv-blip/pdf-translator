import base64
import io
import json
import os
import re
import tempfile
import zipfile

import anthropic
import fitz
import openpyxl
import streamlit as st
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

DEFAULT_FONT     = Path(__file__).parent / "font.ttf"
DEFAULT_GLOSSARY = Path(__file__).parent / "glossary.xlsx"


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
    return {k: v for k, v in glossary.items() if k.lower() in tl}


# ── PDF translation ────────────────────────────────────────────────────────────

def translate_block(client: anthropic.Anthropic, text: str, glossary: dict) -> dict:
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
        system=(
            "你是专业内衣、泳衣、泳装及外贸成衣行业翻译。"
            "处理的文档均为服装工艺单、产品目录或设计说明书。"
            "遇到任何存在多义性的词汇，必须优先选取与内衣、泳衣、泳装、外贸成衣"
            "最贴切的中文行业专业术语，不得选用其他行业含义。"
            "例如：hipster → 平角内裤（而非潮人）；brief → 三角内裤（而非简短）；"
            "cup → 罩杯（而非杯子）；underwire → 钢圈（而非底线）。"
        ),
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        raw = m.group()
    return json.loads(raw)


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


def run_pdf_translation(
    pdf_bytes: bytes,
    glossary_bytes: bytes,
    font_path: str,
    api_key: str,
    on_page,
    on_block,
    on_progress,
) -> tuple[bytes, bytes, int]:
    """Returns (pdf_out, xlsx_out, n_unrecorded_terms)."""
    glossary = load_glossary(glossary_bytes)
    client = anthropic.Anthropic(api_key=api_key)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)
    all_unrecorded: set[str] = set()

    # 以 span 为最小单元预扫描：bbox 精确到每一段文字，数字 span 绝不擦除
    page_spans: list[list[dict]] = []
    for pn in range(total_pages):
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

    for pn, spans in enumerate(page_spans):
        page = doc[pn]
        on_page(pn, total_pages, len(spans))

        results = []
        for sp in spans:
            _t = sp["text"].strip()
            alpha     = len(re.findall(r'[a-zA-Z]', _t))
            has_digit = bool(re.search(r'\d', _t))

            # 纯数字/符号，无英文字母——绝不擦除，直接跳过
            if alpha == 0:
                done += 1; on_progress(done / total_spans); continue
            # 单字符或双字符：尺码字母(S/M/L)、连字符等
            if len(_t) <= 2:
                done += 1; on_progress(done / total_spans); continue
            # 含数字且英文字母极少(≤2)：32A、70B 等尺码代号
            if has_digit and alpha <= 2:
                done += 1; on_progress(done / total_spans); continue

            on_block(_t[:60])
            try:
                res = translate_block(client, sp["text"], glossary)
                translated = res.get("translated_text", sp["text"])
                for t in res.get("unrecorded_terms", []):
                    if t.strip():
                        all_unrecorded.add(t.strip())
            except Exception:
                translated = sp["text"]

            results.append({**sp, "translated": translated})
            done += 1
            on_progress(done / total_spans)

        # 直接叠加译文，不销毁英文原文
        for r in results:
            _insert_text(page, r["bbox"], r["translated"], r["size"], font_path)

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

    return pdf_buf.getvalue(), xlsx_buf.getvalue(), len(all_unrecorded)


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


def translate_cell_text(client: anthropic.Anthropic, text: str, glossary: dict) -> str:
    rel = relevant_glossary(text, glossary)
    gloss_block = ""
    if rel:
        lines = "\n".join(f"  {k} → {v}" for k, v in rel.items())
        gloss_block = f"强制术语对照（务必照搬）：\n{lines}\n\n"

    prompt = (
        "你是一名专业服装行业翻译，将下方英文翻译成中文服装术语。\n\n"
        f"{gloss_block}"
        f"待翻译文本：{text}\n\n"
        "规则：\n"
        "1. 对照表中的词汇必须使用对照表给出的中文译法。\n"
        "2. 款号、货号、数字、尺码保持原样，不翻译。\n"
        "3. 只返回翻译结果，不要任何解释或多余文字。"
    )
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=(
            "你是专业内衣、泳衣、泳装及外贸成衣行业翻译。"
            "处理的文档均为服装工艺单、产品目录或设计说明书。"
            "遇到任何存在多义性的词汇，必须优先选取与内衣、泳衣、泳装、外贸成衣"
            "最贴切的中文行业专业术语，不得选用其他行业含义。"
            "例如：hipster → 平角内裤；brief → 三角内裤；cup → 罩杯；underwire → 钢圈。"
        ),
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def run_excel_translation(
    xlsx_bytes: bytes,
    glossary_bytes: bytes,
    api_key: str,
    on_cell,
    on_progress,
) -> tuple[bytes, int, int]:
    """Translate text cells + embedded images across ALL sheets.
    Returns (xlsx_out, n_cells_translated, n_images_translated).
    """
    glossary = load_glossary(glossary_bytes)
    client = anthropic.Anthropic(api_key=api_key)

    # ── 阶段 1：翻译文字单元格 ──────────────────────────────────────────────────
    # data_only=True 读取公式计算结果而非公式字符串
    wb = openpyxl.load_workbook(io.BytesIO(xlsx_bytes), data_only=True)

    to_translate = [
        (ws.title, cell)
        for ws in wb.worksheets
        for row in ws.iter_rows()
        for cell in row
        if _is_translatable(cell.value)
    ]
    n_cells = len(to_translate)
    total_steps = max(n_cells, 1)

    for i, (sheet_name, cell) in enumerate(to_translate):
        on_cell(f"[{sheet_name}] {str(cell.value)[:50].replace(chr(10), ' ')}")
        try:
            cell.value = translate_cell_text(client, str(cell.value), glossary)
        except Exception:
            pass
        on_progress((i + 1) / total_steps * 0.6)   # 前 60% 进度给文字翻译

    buf = io.BytesIO()
    wb.save(buf)
    text_done_bytes = buf.getvalue()

    # ── 阶段 2：翻译嵌入图片 ──────────────────────────────────────────────────
    # 先数一下有多少图片
    with zipfile.ZipFile(io.BytesIO(text_done_bytes)) as zf:
        n_images = sum(
            1 for n in zf.namelist()
            if n.startswith("xl/media/") and n.rsplit(".", 1)[-1].lower() in _IMAGE_EXTS
        )

    if n_images == 0:
        on_progress(1.0)
        return text_done_bytes, n_cells, 0

    def on_image(idx, total, fname):
        on_cell(f"图片 {idx}/{total}：{fname}")
        on_progress(0.6 + idx / total * 0.4)   # 后 40% 进度给图片翻译

    final_bytes = translate_images_in_excel(text_done_bytes, client, glossary, on_image)
    on_progress(1.0)
    return final_bytes, n_cells, n_images


# ── Excel image translation ───────────────────────────────────────────────────

_IMAGE_EXTS = {"jpg", "jpeg", "png", "webp"}
_MIME = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}


def _translate_image_bytes(
    client: anthropic.Anthropic,
    img_bytes: bytes,
    ext: str,
    glossary: dict,
) -> bytes:
    """Send one image to Claude Vision, overlay Chinese translations, return new bytes."""
    media_type = _MIME.get(ext)
    if not media_type:
        return img_bytes

    b64 = base64.b64encode(img_bytes).decode("utf-8")

    gloss_hint = ""
    if glossary:
        sample = "\n".join(f"  {k} → {v}" for k, v in list(glossary.items())[:15])
        gloss_hint = f"\n参考术语对照（部分）：\n{sample}"

    try:
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=(
                "你是专业内衣、泳衣、泳装及外贸成衣行业翻译。"
                "遇到多义词优先选取与内衣、泳衣、外贸成衣最贴切的中文行业术语。"
            ),
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                    {"type": "text", "text": (
                        f"这是一张服装工艺/规格图，图中有英文标注。{gloss_hint}\n\n"
                        "请识别图中全部英文文字，翻译成中文，并给出每处文字在图片中的大致坐标。\n"
                        "坐标规则：左上角为(0,0)，右下角为(1,1)，用0~1之间的小数表示。\n"
                        "同时估计该处文字的字号（像素大小）。\n\n"
                        "只返回JSON，格式：\n"
                        '{"items": [{"en": "front strap", "zh": "前肩带", "x": 0.25, "y": 0.18, "size": 11}]}\n'
                        "如图中无英文则返回 {\"items\": []}。"
                    )}
                ],
            }],
        )
        raw = resp.content[0].text.strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            raw = m.group()
        items = json.loads(raw).get("items", [])
    except Exception:
        return img_bytes

    if not items:
        return img_bytes

    # Overlay Chinese text with PIL
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    draw = ImageDraw.Draw(img)
    iw, ih = img.size

    font_path = str(DEFAULT_FONT) if DEFAULT_FONT.exists() else None

    for item in items:
        zh = str(item.get("zh", "")).strip()
        if not zh:
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


# ── Streamlit page ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="服装行业翻译引擎",
    page_icon="🧵",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("🧵 服装行业翻译引擎")
st.caption("支持 PDF 与 Excel (.xlsx) 双格式 · 上传文件 + 术语库 · 调用 Claude 自动翻译为中文")
st.divider()

# ── 文件上传区 ─────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    source_file = st.file_uploader(
        "📄 上传待翻译文件（PDF 或 Excel）",
        type=["pdf", "xlsx"],
    )
with col2:
    glossary_hint = (
        f"可选，已内置默认术语库 `{DEFAULT_GLOSSARY.name}`"
        if DEFAULT_GLOSSARY.exists()
        else "请上传术语库 Excel"
    )
    glossary_file = st.file_uploader(
        f"📊 术语库 Excel（{glossary_hint}）",
        type=["xlsx", "xls"],
    )

# 仅 PDF 模式需要字体
source_is_pdf = source_file is not None and source_file.name.lower().endswith(".pdf")
source_is_xlsx = source_file is not None and source_file.name.lower().endswith(".xlsx")

if source_is_pdf:
    font_label = (
        f"🔤 上传中文字体 TTF（可选，已检测到默认字体 {DEFAULT_FONT.name}）"
        if DEFAULT_FONT.exists()
        else "🔤 上传中文字体 TTF（必填，未检测到默认字体）"
    )
    font_file = st.file_uploader(font_label, type=["ttf"])
else:
    font_file = None

st.divider()

# ── API Key ────────────────────────────────────────────────────────────────────
api_key = st.text_input(
    "🔑 Anthropic API Key",
    type="password",
    value=os.environ.get("ANTHROPIC_API_KEY", ""),
    placeholder="sk-ant-api03-...",
)
st.divider()

# ── 状态提示 & 开始按钮 ────────────────────────────────────────────────────────
font_ready     = bool(font_file or DEFAULT_FONT.exists())
glossary_ready = bool(glossary_file or DEFAULT_GLOSSARY.exists())

if source_is_pdf:
    can_start = bool(source_file and glossary_ready and api_key and font_ready)
elif source_is_xlsx:
    can_start = bool(source_file and glossary_ready and api_key)
else:
    can_start = False

if not can_start:
    missing = []
    if not source_file:
        missing.append("待翻译文件（PDF / Excel）")
    if not glossary_ready:
        missing.append("术语库 Excel")
    if not api_key:
        missing.append("Anthropic API Key")
    if source_is_pdf and not font_ready:
        missing.append("中文字体 TTF")
    st.info(f"请先提供：{'、'.join(missing)}" if missing else "")

start_btn = st.button(
    "🚀  开始翻译",
    disabled=not can_start,
    use_container_width=True,
    type="primary",
)

# ── 翻译流程 ───────────────────────────────────────────────────────────────────
if start_btn:
    for key in ("pdf_result", "xlsx_result", "excel_result", "n_terms", "n_cells", "mode"):
        st.session_state.pop(key, None)

    progress_bar = st.progress(0.0, text="初始化…")

    # ── PDF 分支 ──
    if source_is_pdf:
        with tempfile.TemporaryDirectory() as tmpdir:
            if font_file:
                fp = os.path.join(tmpdir, "font.ttf")
                with open(fp, "wb") as f:
                    f.write(font_file.read())
                font_path = fp
            else:
                font_path = str(DEFAULT_FONT)

            with st.status("翻译进行中，请稍候…", expanded=True) as status:
                block_ph = st.empty()

                def on_page(pn, total, n_blocks):
                    st.write(f"**── 第 {pn + 1} / {total} 页**　（{n_blocks} 个文本块）")

                def on_block(preview):
                    block_ph.caption(f"▶ 正在翻译：{preview}…")

                def on_progress(frac):
                    progress_bar.progress(frac, text=f"翻译进度 {frac:.0%}")

                try:
                    glossary_bytes = (
                        glossary_file.read() if glossary_file
                        else DEFAULT_GLOSSARY.read_bytes()
                    )
                    pdf_out, xlsx_out, n_terms = run_pdf_translation(
                        pdf_bytes=source_file.read(),
                        glossary_bytes=glossary_bytes,
                        font_path=font_path,
                        api_key=api_key,
                        on_page=on_page,
                        on_block=on_block,
                        on_progress=on_progress,
                    )
                    block_ph.empty()
                    status.update(label="✅ 翻译完成！", state="complete")
                    progress_bar.progress(1.0, text="完成 ✓")
                    st.session_state["mode"]        = "pdf"
                    st.session_state["pdf_result"]  = pdf_out
                    st.session_state["xlsx_result"] = xlsx_out
                    st.session_state["n_terms"]     = n_terms

                except Exception as e:
                    block_ph.empty()
                    status.update(label=f"❌ 出错：{e}", state="error")
                    st.error(str(e))

    # ── Excel 分支 ──
    elif source_is_xlsx:
        with st.status("翻译进行中，请稍候…", expanded=True) as status:
            cell_ph = st.empty()

            def on_cell(preview):
                cell_ph.caption(f"▶ 正在翻译：{preview}…")

            def on_progress(frac):
                progress_bar.progress(frac, text=f"翻译进度 {frac:.0%}")

            try:
                glossary_bytes = (
                    glossary_file.read() if glossary_file
                    else DEFAULT_GLOSSARY.read_bytes()
                )
                excel_out, n_cells, n_images = run_excel_translation(
                    xlsx_bytes=source_file.read(),
                    glossary_bytes=glossary_bytes,
                    api_key=api_key,
                    on_cell=on_cell,
                    on_progress=on_progress,
                )
                cell_ph.empty()
                status.update(label="✅ 翻译完成！", state="complete")
                progress_bar.progress(1.0, text="完成 ✓")
                st.session_state["mode"]         = "excel"
                st.session_state["excel_result"] = excel_out
                st.session_state["n_cells"]      = n_cells
                st.session_state["n_images"]     = n_images
                st.session_state["source_name"]  = source_file.name

            except Exception as e:
                cell_ph.empty()
                status.update(label=f"❌ 出错：{e}", state="error")
                st.error(str(e))

# ── 下载区 ─────────────────────────────────────────────────────────────────────
mode = st.session_state.get("mode")

if mode == "pdf" and st.session_state.get("pdf_result"):
    st.divider()
    n = st.session_state["n_terms"]
    st.success(f"翻译完成！共识别 **{n}** 条未收录术语。")

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            label="⬇️  下载中文 PDF",
            data=st.session_state["pdf_result"],
            file_name="translated_CN.pdf",
            mime="application/pdf",
            use_container_width=True,
            type="primary",
        )
    with dl2:
        if n > 0:
            st.download_button(
                label="⬇️  下载未收录术语 Excel",
                data=st.session_state["xlsx_result"],
                file_name="unrecorded_terms.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        else:
            st.info("本次翻译无未收录术语")

elif mode == "excel" and st.session_state.get("excel_result"):
    st.divider()
    n_c = st.session_state["n_cells"]
    n_i = st.session_state.get("n_images", 0)
    img_note = f"，图片 **{n_i}** 张" if n_i > 0 else ""
    st.success(f"翻译完成！文字单元格 **{n_c}** 个{img_note}。")

    src_name = st.session_state.get("source_name", "translated.xlsx")
    out_name = src_name.replace(".xlsx", "_中文版.xlsx")
    st.download_button(
        label="⬇️  下载中文 Excel",
        data=st.session_state["excel_result"],
        file_name=out_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        type="primary",
    )
