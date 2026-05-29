import io
import json
import os
import re
import tempfile
import zipfile

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
        model="claude-sonnet-4-6",
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
    """Translate one PDF. Returns (pdf_bytes_out, sorted_unrecorded_terms)."""
    client      = anthropic.Anthropic(api_key=api_key)
    doc         = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)
    unrecorded  = set()

    # Pre-scan for accurate progress
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
                print(f"大模型调用报错: {exc}")
                st.error(f"⚠️ 翻译调用失败！API 报错信息：{exc}")
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
        "2. 上传一个或多个英文 PDF\n"
        "3. 填入 API Key\n"
        "4. 点击「开始翻译」\n"
        "5. 下载 ZIP 压缩包\n"
        "6. 为生词填写释义，保存到 Google Sheets"
    )

    if st.session_state.get("_active_user") != user:
        for k in ("zip_result", "zip_count", "unrecorded_terms",
                  "terms_saved", "saved_count", "translation_done"):
            st.session_state.pop(k, None)
        st.session_state["_active_user"] = user

st.title("🧵 服装行业 PDF 翻译引擎（批量版）")
st.caption(
    f"多用户 · Google Sheets 云端术语库 · 批量处理 · ZIP 打包下载 ｜ 当前用户：**{user}**"
)
st.divider()

# ── 文件上传 ───────────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    pdf_files = st.file_uploader(
        "📄 上传英文 PDF（可多选）",
        type=["pdf"],
        accept_multiple_files=True,
    )
with c2:
    font_hint = (f"已检测到默认字体 `{DEFAULT_FONT.name}`，可跳过"
                 if DEFAULT_FONT.exists() else "未检测到默认字体，请上传")
    font_file = st.file_uploader(f"🔤 上传中文字体 TTF（{font_hint}）", type=["ttf"])

if pdf_files:
    st.caption(f"已选择 **{len(pdf_files)}** 个文件：{', '.join(f.name for f in pdf_files)}")

st.divider()

api_key = st.text_input(
    "🔑 Anthropic API Key",
    type="password",
    value=os.environ.get("ANTHROPIC_API_KEY", ""),
    placeholder="sk-ant-api03-...",
)

st.divider()

# ── 就绪检查 & 开始按钮 ────────────────────────────────────────────────────────
font_ok   = bool(font_file or DEFAULT_FONT.exists())
can_start = bool(pdf_files and api_key and font_ok)

if not can_start:
    missing = []
    if not pdf_files: missing.append("PDF 文件（至少一个）")
    if not api_key:   missing.append("API Key")
    if not font_ok:   missing.append("中文字体 TTF")
    st.info(f"还缺：{'、'.join(missing)}")

start = st.button(
    f"🚀  开始翻译（共 {len(pdf_files)} 个文件）" if pdf_files else "🚀  开始翻译",
    disabled=not can_start,
    use_container_width=True,
    type="primary",
)

# ── 批量翻译主流程 ─────────────────────────────────────────────────────────────
if start:
    for k in ("zip_result", "zip_count", "unrecorded_terms",
              "terms_saved", "saved_count", "translation_done"):
        st.session_state.pop(k, None)

    # 加载术语库
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
        # 解析字体路径
        if font_file:
            fp = os.path.join(tmp, "font.ttf")
            with open(fp, "wb") as f:
                f.write(font_file.read())
            font_path = fp
        else:
            font_path = str(DEFAULT_FONT)

        n_files          = len(pdf_files)
        all_results      = []          # [(output_filename, pdf_bytes), ...]
        global_unrecorded: set[str] = set()

        # 双层进度条：文件级 + 文本块级
        file_prog  = st.progress(0.0, text="文件进度：0 / " + str(n_files))
        block_prog = st.progress(0.0, text="")

        with st.status(
            f"批量翻译进行中，共 {n_files} 个文件…", expanded=True
        ) as status:
            cur_file_ph  = st.empty()   # 当前文件提示（覆盖更新）
            cur_block_ph = st.empty()   # 当前文本块提示（覆盖更新）

            for i, pdf_file in enumerate(pdf_files):
                # 文件级状态
                cur_file_ph.info(
                    f"📄 正在处理第 **{i + 1} / {n_files}** 个文件：`{pdf_file.name}`"
                )
                file_prog.progress(
                    i / n_files,
                    text=f"文件进度：{i} / {n_files}",
                )
                block_prog.progress(0.0, text="")

                # 用 default 参数捕获当前循环变量，避免闭包陷阱
                def on_page(pn, total, n, _name=pdf_file.name):
                    st.write(
                        f"&nbsp;&nbsp;&nbsp;&nbsp;└─ **{_name}**"
                        f"　第 {pn + 1} / {total} 页（{n} 个文本块）"
                    )

                def on_block(preview):
                    cur_block_ph.caption(f"▶ 正在翻译：{preview}…")

                def on_progress(frac):
                    block_prog.progress(frac, text=f"当前文件翻译进度 {frac:.0%}")

                try:
                    pdf_out, file_unrecorded = run_translation(
                        pdf_bytes=pdf_file.read(),
                        glossary=glossary,
                        font_path=font_path,
                        api_key=api_key,
                        on_page=on_page,
                        on_block=on_block,
                        on_progress=on_progress,
                    )
                    stem = Path(pdf_file.name).stem
                    out_name = f"{stem}_CN.pdf"
                    all_results.append((out_name, pdf_out))
                    global_unrecorded.update(file_unrecorded)
                    st.write(
                        f"&nbsp;&nbsp;&nbsp;&nbsp;✅ **{pdf_file.name}** 翻译完成"
                        f"，发现 {len(file_unrecorded)} 条新生词"
                    )
                except Exception as exc:
                    print(f"处理 {pdf_file.name} 出错: {exc}")
                    st.error(f"❌ **{pdf_file.name}** 处理失败：{exc}")

            # 打包 ZIP（内存操作）
            cur_file_ph.info("📦 正在打包 ZIP 压缩包…")
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for fname, pdf_bytes in all_results:
                    zf.writestr(fname, pdf_bytes)

            # 收尾
            cur_file_ph.empty()
            cur_block_ph.empty()
            file_prog.progress(1.0, text=f"全部 {n_files} 个文件翻译完成 ✓")
            block_prog.empty()
            status.update(
                label=f"✅ 批量翻译完成！成功处理 {len(all_results)} / {n_files} 个文件",
                state="complete",
            )

            st.session_state["zip_result"]       = zip_buf.getvalue()
            st.session_state["zip_count"]        = len(all_results)
            st.session_state["unrecorded_terms"] = sorted(global_unrecorded)
            st.session_state["translation_done"] = True
            st.session_state["terms_saved"]      = False


# ── 结果区：下载 + 在线学习 ────────────────────────────────────────────────────
if st.session_state.get("translation_done"):
    st.divider()
    n_ok       = st.session_state.get("zip_count", 0)
    unrecorded = st.session_state.get("unrecorded_terms", [])
    st.success(
        f"批量翻译完成！成功 **{n_ok}** 个文件，"
        f"全局共识别 **{len(unrecorded)}** 条未收录术语。"
    )

    # ZIP 下载按钮
    st.download_button(
        label=f"⬇️  下载全部 {n_ok} 个中文 PDF（ZIP 压缩包）",
        data=st.session_state["zip_result"],
        file_name="translated_CN_batch.zip",
        mime="application/zip",
        use_container_width=True,
        type="primary",
    )

    # ── 在线学习表单 ────────────────────────────────────────────────────────────
    if unrecorded and not st.session_state.get("terms_saved"):
        st.divider()
        st.subheader("📝 在线学习 — 为未收录术语填写中文释义")
        st.caption(
            f"以下生词来自本批次所有文件，填好后点击保存，"
            f"新词条将追加到 **{SPREADSHEET_NAME} › {user}**。留空则跳过。"
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
