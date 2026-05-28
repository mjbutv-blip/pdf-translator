import io
import json
import os
import re
import tempfile

import anthropic
import fitz
import gspread
import openpyxl
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="服装行业 PDF 翻译引擎",
    page_icon="🧵",
    layout="wide",
)

DEFAULT_FONT     = Path(__file__).parent / "font.ttf"
SPREADSHEET_NAME = "Shared_Glossary"
PRESET_USERS     = ["User_A", "User_B", "User_C", "User_D", "User_E"]


# ══════════════════════════════════════════════════════════════════════════════
#  Google Sheets helpers
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def _get_gc():
    return gspread.service_account_from_dict(st.secrets["gcp_service_account"])

def get_or_create_worksheet(user: str) -> gspread.Worksheet:
    sh = _get_gc().open(SPREADSHEET_NAME)
    try:
        return sh.worksheet(user)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=user, rows=1000, cols=5)
        ws.append_row(["English", "Chinese"])
        return ws

def load_glossary(user: str) -> dict[str, str]:
    ws   = get_or_create_worksheet(user)
    rows = ws.get_all_records()
    return {
        str(r.get("English", "")).strip(): str(r.get("Chinese", "")).strip()
        for r in rows
        if r.get("English") and r.get("Chinese")
    }

def append_new_terms(user: str, pairs: list[tuple[str, str]]) -> None:
    ws = get_or_create_worksheet(user)
    for eng, chi in pairs:
        ws.append_row([eng, chi])


# ══════════════════════════════════════════════════════════════════════════════
#  Core translation logic
# ══════════════════════════════════════════════════════════════════════════════

def _relevant(text: str, glossary: dict) -> dict:
    tl = text.lower()
    return {k: v for k, v in glossary.items() if k.lower() in tl}

def _translate_block(client, text: str, glossary: dict) -> dict:
    rel = _relevant(text, glossary)
    gloss_block = ""
    if rel:
        lines = "\n".join(f"  {k} → {v}" for k, v in rel.items())
        gloss_block = f"强制术语对照（务必照搬）：\n{lines}\n\n"

    prompt = (
        "你是一名专业服装行业翻译，将下列英文翻译成地道的中文服装术语。\n\n"
        f"{gloss_block}"
        f"待翻译文本：\n{text}\n\n"
        "规则：\n"
        "1. 对照表中的词汇必须使用对照表给出的中文译法。\n"
        "2. 款号、货号、数字、尺码保持原样，不翻译。\n"
        "3. 将对照表里未收录的服装/纺织专业英文词汇放入 unrecorded_terms。\n"
        "4. 只返回 JSON，不得有多余文字或 markdown 代码块。\n\n"
        '格式：{"translated_text": "中文结果", "unrecorded_terms": ["term1"]}'
    )
    msg = client.messages.create(
        model="claude-3-5-sonnet-20241022", # 帮你修正了模型名称，"claude-sonnet-4-6" 并不是正式的API模型名
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$",       "", raw)
    m   = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        raw = m.group()
    return json.loads(raw)

def _insert_text(page, bbox, text: str, size: float, font_path: str) -> None:
    rect = fitz.Rect(bbox)
    for s in [size, size * 0.85, size * 0.7, 8.0, 7.0]:
        if page.insert_textbox(rect, text, fontname="myfont", fontfile=font_path,
                               fontsize=s, color=(0, 0, 0), align=0) >= 0:
            return
    page.insert_textbox(rect, text, fontname="myfont", fontfile=font_path,
                        fontsize=7.0, color=(0, 0, 0), align=0)

def run_translation(pdf_bytes, glossary, font_path, api_key,
                    on_page, on_block, on_progress):
    client      = anthropic.Anthropic(api_key=api_key)
    doc         = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)
    unrecorded  = set()

    page_blocks: list[list[dict]] = []
    for pn in range(total_pages):
        tb = []
        for blk in doc[pn].get_text("dict")["blocks"]:
            if blk["type"] != 0:
                continue
            parts, sizes = [], []
            for ln in blk.get("lines", []):
                for sp in ln.get("spans", []):
                    if sp["text"].strip():
                        parts.append(sp["text"])
                        sizes.append(sp["size"])
                parts.append("\n")
            text = "".join(parts).strip()
            if not text:
                continue
            tb.append({
                "bbox": blk["bbox"],
                "text": text,
                "size": sum(sizes) / len(sizes) if sizes else 10.0,
            })
        page_blocks.append(tb)

    total_blocks = max(sum(len(b) for b in page_blocks), 1)
    done = 0

    for pn, blocks in enumerate(page_blocks):
        page = doc[pn]
        on_page(pn, total_pages, len(blocks))

        results = []
        for blk in blocks:
            on_block(blk["text"][:60].replace("\n", " "))
            try:
                res        = _translate_block(client, blk["text"], glossary)
                translated = res.get("translated_text", blk["text"])
                for t in res.get("unrecorded_terms", []):
                    if t.strip():
                        unrecorded.add(t.strip())
            except Exception as exc:
                print(f"翻译接口报错: {exc}") # 在后台打印
                st.error(f"翻译中断！大模型报错: {exc}") # 在网页上弹出红字
                translated = blk["text"]
        results.append({**blk, "translated": translated})
                done += 1
            on_progress(done / total_blocks)

        for r in results:
            page.add_redact_annot(fitz.Rect(r["bbox"]), fill=(1, 1, 1))
        page.apply_redactions()
        for r in results:
            _insert_text(page, r["bbox"], r["translated"], r["size"], font_path)

    buf = io.BytesIO()
    doc.save(buf, garbage=4, deflate=True)
    doc.close()
    return buf.getvalue(), sorted(unrecorded)


# ══════════════════════════════════════════════════════════════════════════════
#  Streamlit UI
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("👤 用户选择")
    user = st.selectbox(
        "当前用户",
        PRESET_USERS,
        help="每位用户在 Google Sheets 中拥有独立的术语工作表",
    )
    st.caption(f"术语来源：**{SPREADSHEET_NAME}** › **{user}**")
    st.divider()
    st.info(
        "**使用说明**\n\n"
        "1. 选择用户\n"
        "2. 上传英文 PDF\n"
        "3. 填入 API Key\n"
        "4. 点击「开始翻译」\n"
        "5. 为生词填写释义，保存到 Google Sheets"
    )

    if st.session_state.get("_active_user") != user:
        for k in ("pdf_result", "unrecorded_terms", "terms_saved",
                  "saved_count", "translation_done"):
            st.session_state.pop(k, None)
        st.session_state["_active_user"] = user

st.title("🧵 服装行业 PDF 翻译引擎")
st.caption(
    f"多用户 · Google Sheets 云端术语库 · 在线学习新词条 ｜ 当前用户：**{user}**"
)
st.divider()

c1, c2 = st.columns(2)
with c1:
    pdf_file = st.file_uploader("📄 上传英文 PDF", type=["pdf"])
with c2:
    font_hint = (f"已检测到默认字体 `{DEFAULT_FONT.name}`，可跳过"
                 if DEFAULT_FONT.exists() else "未检测到默认字体，请上传")
    font_file = st.file_uploader(f"🔤 上传中文字体 TTF（{font_hint}）", type=["ttf"])

st.divider()

api_key = st.text_input(
    "🔑 Anthropic API Key",
    type="password",
    value=os.environ.get("ANTHROPIC_API_KEY", ""),
    placeholder="sk-ant-api03-...",
)

st.divider()

font_ok   = bool(font_file or DEFAULT_FONT.exists())
can_start = bool(pdf_file and api_key and font_ok)

if not can_start:
    parts = []
    if not pdf_file: parts.append("PDF 文件")
    if not api_key:  parts.append("API Key")
    if not font_ok:  parts.append("中文字体 TTF")
    st.info(f"还缺：{'、'.join(parts)}")

start = st.button(
    "🚀  开始翻译",
    disabled=not can_start,
    use_container_width=True,
    type="primary",
)

if start:
    for k in ("pdf_result", "unrecorded_terms", "terms_saved",
              "saved_count", "translation_done"):
        st.session_state.pop(k, None)

    with st.spinner(f"正在从 Google Sheets 加载「{user}」的术语库…"):
        try:
            glossary = load_glossary(user)
            st.toast(f"✅ 术语库就绪，共 {len(glossary)} 条", icon="📚")
        except Exception as exc:
            st.error(
                f"**无法连接 Google Sheets**\n\n{exc}\n\n"
                "请确认 Streamlit Secrets 中已配置 `gcp_service_account`，"
                f"且共享表格名称为 `{SPREADSHEET_NAME}`。"
            )
            st.stop()

    with tempfile.TemporaryDirectory() as tmp:
        if font_file:
            fp = os.path.join(tmp, "font.ttf")
            with open(fp, "wb") as f:
                f.write(font_file.read())
            font_path = fp
        else:
            font_path = str(DEFAULT_FONT)

        prog = st.progress(0.0, text="初始化…")

        with st.status("翻译进行中，请稍候…", expanded=True) as status:
            cur_block = st.empty()

            def on_page(pn, total, n):
                st.write(f"**── 第 {pn+1} / {total} 页**　（{n} 个文本块）")

            def on_block(preview):
                cur_block.caption(f"▶ 正在翻译：{preview}…")

            def on_progress(frac):
                prog.progress(frac, text=f"翻译进度 {frac:.0%}")

            try:
                pdf_out, unrecorded = run_translation(
                    pdf_bytes=pdf_file.read(),
                    glossary=glossary,
                    font_path=font_path,
                    api_key=api_key,
                    on_page=on_page,
                    on_block=on_block,
                    on_progress=on_progress,
                )
                cur_block.empty()
                status.update(label="✅ 翻译完成！", state="complete")
                prog.progress(1.0, text="完成 ✓")
                st.session_state["pdf_result"]       = pdf_out
                st.session_state["unrecorded_terms"] = unrecorded
                st.session_state["translation_done"] = True
                st.session_state["terms_saved"]      = False
            except Exception as exc:
                cur_block.empty()
                status.update(label=f"❌ 出错：{exc}", state="error")
                st.error(str(exc))

# ── Results ────────────────────────────────────────────────────────────────────
if st.session_state.get("translation_done"):
    st.divider()
    unrecorded: list[str] = st.session_state.get("unrecorded_terms", [])
    st.success(f"翻译完成！共识别 **{len(unrecorded)}** 条未收录术语。")

    st.download_button(
        label="⬇️   下载中文 PDF",
        data=st.session_state["pdf_result"],
        file_name="translated_CN.pdf",
        mime="application/pdf",
        use_container_width=True,
        type="primary",
    )

    if unrecorded and not st.session_state.get("terms_saved"):
        st.divider()
        st.subheader("📝 在线学习 — 为未收录术语填写中文释义")
        st.caption(
            f"填好后点击保存，新词条将追加到 **{SPREADSHEET_NAME} › {user}**。留空则跳过。"
        )

        with st.form("learn_form", clear_on_submit=False):
            term_vals: dict[str, str] = {}
            pairs = list(unrecorded)
            for i in range(0, len(pairs), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    if i + j < len(pairs):
                        term = pairs[i + j]
                        with col:
                            term_vals[term] = st.text_input(
                                label=term,
                                placeholder="输入中文释义…",
                                key=f"learn_{term}",
                            )

            save_btn = st.form_submit_button(
                "💾  保存到 Google Sheets 术语库",
                use_container_width=True,
                type="primary",
            )

        if save_btn:
            to_save = [(e, c) for e, c in term_vals.items() if c.strip()]
            with st.spinner("正在写入 Google Sheets…"):
                try:
                    if to_save:
                        append_new_terms(user, to_save)
                    st.session_state["terms_saved"] = True
                    st.session_state["saved_count"] = len(to_save)
                    st.rerun()
                except Exception as exc:
                    st.error(f"写入失败：{exc}")

    elif st.session_state.get("terms_saved"):
        n = st.session_state.get("saved_count", 0)
        if n > 0:
            st.success(
                f"✅ 已将 **{n}** 条新词条保存到 **{SPREADSHEET_NAME} › {user}**！"
                "下次翻译将自动使用这些术语。"
            )
        else:
            st.info("本次未填写新释义，未写入术语库。")
