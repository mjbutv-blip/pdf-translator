import base64
import io
import json
import os
import re
import tempfile
import xml.etree.ElementTree as ET
import zipfile

import anthropic
import fitz
import openpyxl
import pandas as pd
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


# ── Glossary management (editable in-session glossary) ─────────────────────────

_GLOSSARY_EN_COL   = "英文术语"
_GLOSSARY_ZH_COL   = "中文翻译"
_GLOSSARY_NOTE_COL = "备注"
_GLOSSARY_CAT_COL  = "分类"
_GLOSSARY_EDIT_COLS = [_GLOSSARY_EN_COL, _GLOSSARY_ZH_COL, _GLOSSARY_NOTE_COL, _GLOSSARY_CAT_COL]


def _empty_glossary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_GLOSSARY_EDIT_COLS)


def parse_glossary_excel(data: bytes) -> tuple[pd.DataFrame | None, list[str]]:
    """Parse an uploaded glossary workbook into the standard 4-column shape.
    Returns (df, missing_columns); df is None if required columns are missing
    or the file can't be read at all — callers must check missing_columns
    instead of letting an exception propagate.
    """
    try:
        wb = openpyxl.load_workbook(io.BytesIO(data))
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
    except Exception:
        return None, ["无法解析该文件，请确认上传的是有效的 Excel (.xlsx) 文件"]

    if not rows:
        return None, ["文件内容为空"]

    headers = [str(h).strip() if h is not None else "" for h in rows[0]]
    en_idx = zh_idx = note_idx = cat_idx = None
    for i, h in enumerate(headers):
        hl = h.lower()
        if en_idx is None and ("英文" in h or "english" in hl):
            en_idx = i
        if zh_idx is None and ("中文" in h or "chinese" in hl):
            zh_idx = i
        if note_idx is None and ("备注" in h or "note" in hl or "remark" in hl):
            note_idx = i
        if cat_idx is None and ("分类" in h or "category" in hl or "来源" in h or "source" in hl):
            cat_idx = i

    missing = []
    if en_idx is None:
        missing.append("英文术语列（表头需包含「英文」或 English）")
    if zh_idx is None:
        missing.append("中文翻译列（表头需包含「中文」或 Chinese）")
    if missing:
        return None, missing

    def cell(row, idx):
        if idx is None or idx >= len(row) or row[idx] is None:
            return ""
        return str(row[idx])

    records = [
        {
            _GLOSSARY_EN_COL: cell(row, en_idx),
            _GLOSSARY_ZH_COL: cell(row, zh_idx),
            _GLOSSARY_NOTE_COL: cell(row, note_idx),
            _GLOSSARY_CAT_COL: cell(row, cat_idx),
        }
        for row in rows[1:]
    ]
    return pd.DataFrame(records, columns=_GLOSSARY_EDIT_COLS), []


def clean_glossary_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Trim whitespace, drop blank rows, de-dup English terms case-insensitively
    (last occurrence wins). Rows with an empty Chinese translation are kept
    (so the user can finish them later) but excluded from the active glossary
    by glossary_df_to_dict().
    Returns (cleaned_df, conflicts_df, n_incomplete).
    """
    conflicts_cols = ["英文术语", "旧翻译", "新翻译（采用）"]
    if df is None or df.empty:
        return _empty_glossary_df(), pd.DataFrame(columns=conflicts_cols), 0

    work = df.copy()
    for col in _GLOSSARY_EDIT_COLS:
        if col not in work.columns:
            work[col] = ""
        work.loc[:, col] = work[col].fillna("").astype(str).str.strip()

    work = work[work[_GLOSSARY_EN_COL] != ""]  # 去掉英文术语为空的行
    incomplete = work[work[_GLOSSARY_ZH_COL] == ""].copy()
    complete = work[work[_GLOSSARY_ZH_COL] != ""].copy()

    conflicts_rows = []
    if not complete.empty:
        complete = complete.copy()
        complete.loc[:, "_key"] = complete[_GLOSSARY_EN_COL].str.lower()
        deduped = []
        for _, grp in complete.groupby("_key", sort=False):
            if len(grp) > 1:
                zh_values = grp[_GLOSSARY_ZH_COL].tolist()
                if len(set(zh_values)) > 1:
                    conflicts_rows.append({
                        "英文术语": grp.iloc[0][_GLOSSARY_EN_COL],
                        "旧翻译": " / ".join(dict.fromkeys(zh_values[:-1])),
                        "新翻译（采用）": zh_values[-1],
                    })
            deduped.append(grp.iloc[-1])  # 同英文术语取最后一条（最新编辑生效）
        complete = pd.DataFrame(deduped, columns=complete.columns)[_GLOSSARY_EDIT_COLS].reset_index(drop=True)

    cleaned = pd.concat([complete, incomplete[_GLOSSARY_EDIT_COLS]], ignore_index=True)
    return cleaned, pd.DataFrame(conflicts_rows, columns=conflicts_cols), len(incomplete)


def glossary_df_to_dict(df: pd.DataFrame) -> dict[str, str]:
    """Active glossary used for translation — only rows with a non-empty
    Chinese translation are included."""
    if df is None or df.empty:
        return {}
    active = df[df[_GLOSSARY_ZH_COL].fillna("").astype(str).str.strip() != ""]
    return {
        str(r[_GLOSSARY_EN_COL]).strip(): str(r[_GLOSSARY_ZH_COL]).strip()
        for _, r in active.iterrows()
        if str(r[_GLOSSARY_EN_COL]).strip()
    }


def _df_to_simple_xlsx(df: pd.DataFrame, sheet_title: str) -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet_title[:31]
    ws.append(list(df.columns))
    for c in ws[1]:
        c.font = openpyxl.styles.Font(bold=True)
    for _, r in df.iterrows():
        ws.append([r[col] for col in df.columns])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def glossary_df_to_xlsx_bytes(df: pd.DataFrame) -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Glossary"
    ws.append(_GLOSSARY_EDIT_COLS)
    for c in ws[1]:
        c.font = openpyxl.styles.Font(bold=True)
    for _, r in df.iterrows():
        ws.append([r[col] for col in _GLOSSARY_EDIT_COLS])
    for col_letter, width in zip("ABCD", [28, 28, 24, 16]):
        ws.column_dimensions[col_letter].width = width
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def parse_unrecorded_excel(data: bytes) -> tuple[pd.DataFrame | None, list[str]]:
    """Parse a filled-in '未收录术语' file: needs an English column and a
    Chinese-translation column the user added by hand."""
    try:
        wb = openpyxl.load_workbook(io.BytesIO(data))
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
    except Exception:
        return None, ["无法解析该文件，请确认上传的是有效的 Excel (.xlsx) 文件"]

    if not rows:
        return None, ["文件内容为空"]

    headers = [str(h).strip() if h is not None else "" for h in rows[0]]
    en_idx = zh_idx = None
    for i, h in enumerate(headers):
        hl = h.lower()
        if en_idx is None and ("英文" in h or "english" in hl or "未收录" in h):
            en_idx = i
        if zh_idx is None and ("中文" in h or "chinese" in hl):
            zh_idx = i

    missing = []
    if en_idx is None:
        missing.append("英文术语列（表头需包含「英文」「未收录」或 English）")
    if zh_idx is None:
        missing.append("中文翻译列（请在表格中新增一列「中文翻译」并填好后再上传）")
    if missing:
        return None, missing

    records = []
    for row in rows[1:]:
        en = row[en_idx] if en_idx < len(row) else None
        zh = row[zh_idx] if zh_idx < len(row) else None
        en = "" if en is None else str(en).strip()
        zh = "" if zh is None else str(zh).strip()
        if en:
            records.append({_GLOSSARY_EN_COL: en, _GLOSSARY_ZH_COL: zh,
                            _GLOSSARY_NOTE_COL: "", _GLOSSARY_CAT_COL: ""})
    return pd.DataFrame(records, columns=_GLOSSARY_EDIT_COLS), []


def merge_unrecorded_into_glossary(
    glossary_df: pd.DataFrame,
    unrecorded_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int, int]:
    """Merge filled-in unrecorded terms into the glossary. New translations
    overwrite existing ones on conflict (per spec: default to newest wins).
    Returns (merged_df, conflicts_df, n_added, n_overwritten, n_skipped_blank).
    """
    base = glossary_df.copy(deep=True)
    for col in _GLOSSARY_EDIT_COLS:
        if col not in base.columns:
            base[col] = ""
    base.loc[:, "_key"] = base[_GLOSSARY_EN_COL].str.lower()
    lookup = {key: idx for idx, key in base["_key"].items()}

    filled = unrecorded_df[unrecorded_df[_GLOSSARY_ZH_COL].fillna("").astype(str).str.strip() != ""]
    n_skipped_blank = len(unrecorded_df) - len(filled)

    conflicts_rows = []
    new_rows = []
    n_added = n_overwritten = 0

    for _, r in filled.iterrows():
        en = str(r[_GLOSSARY_EN_COL]).strip()
        zh = str(r[_GLOSSARY_ZH_COL]).strip()
        key = en.lower()
        if key in lookup:
            idx = lookup[key]
            old_zh = str(base.loc[idx, _GLOSSARY_ZH_COL]).strip()
            if old_zh and old_zh != zh:
                conflicts_rows.append({"英文术语": en, "旧翻译": old_zh, "新翻译（采用）": zh})
            base.loc[idx, _GLOSSARY_ZH_COL] = zh
            n_overwritten += 1
        else:
            new_rows.append({_GLOSSARY_EN_COL: en, _GLOSSARY_ZH_COL: zh,
                             _GLOSSARY_NOTE_COL: "", _GLOSSARY_CAT_COL: ""})
            lookup[key] = None  # 防止同批次内重复英文术语都被当作新增
            n_added += 1

    base = base.drop(columns=["_key"])
    if new_rows:
        base = pd.concat([base, pd.DataFrame(new_rows)], ignore_index=True)

    conflicts_df = pd.DataFrame(conflicts_rows, columns=["英文术语", "旧翻译", "新翻译（采用）"])
    return base.reset_index(drop=True), conflicts_df, n_added, n_overwritten, n_skipped_blank


def _load_default_glossary_df() -> pd.DataFrame:
    if DEFAULT_GLOSSARY.exists():
        df, missing = parse_glossary_excel(DEFAULT_GLOSSARY.read_bytes())
        if df is not None:
            cleaned, _, _ = clean_glossary_df(df)
            return cleaned
    return _empty_glossary_df()


# ── PDF translation ────────────────────────────────────────────────────────────

_GARMENT_SYSTEM_PROMPT = (
    "你是一名资深内衣（含文胸/塑身贴身衣）、内裤、泳衣三大品类的外贸翻译顾问，"
    "同时具备版师/工艺师的实务经验。"
    "你处理的所有文本都来自这三类产品的工艺单（Tech Pack）、规格表或设计说明书，"
    "绝不会涉及童装、男装、户外服、运动服、牛仔裤、外套、鞋类、箱包等其他任何品类。\n\n"

    "【核心原则一：语境优先于字面，禁止逐词拼接】\n"
    "术语对照表给出的中文译法，是「这个英文词在服装语境下的行业标准称呼」，"
    "不是逐词替换表。当术语表中的多个词共同出现在同一个短语或句子里时，"
    "你必须先理解这句话在工艺单里描述的是哪个部位、哪种工艺、想达成什么效果，"
    "再给出一句通顺、符合行业表达习惯的完整翻译——而不是把每个词各自的译法生硬拼接在一起，"
    "导致中文读起来支离破碎、不像人话。\n\n"

    "【核心原则二：调用服装行业常识做合理推断】\n"
    "英文工艺描述经常很简略，或者同一个词在不同款式里实际工艺做法不同。"
    "遇到这种情况，你要结合上下文（款式类型、相邻的工艺描述、常见行业做法）"
    "推断客人真正想表达的工艺意图，用业内通行的专业说法给出延伸翻译，而不是停留在字面直译。\n"
    "例如：'crotch insert' 单独直译是「裆部插入物」，但结合内衣/泳装工艺单的上下文，"
    "这通常是指裆部内侧加一层裆衬/裆布（用于吸湿、防透），应翻译为「裆部内衬」或「裆衬」，"
    "而不是字面的「裆部插入件」。\n\n"

    "【核心原则三：行业语义优先于通用语义】\n"
    "遇到任何存在多义性的词汇，必须优先选取与内衣、泳衣、泳装、外贸成衣"
    "最贴切的中文行业专业术语，不得选用其他行业或日常含义。"
    "例如：hipster → 平角内裤（而非潮人）；brief → 三角内裤（而非简短）；"
    "cup → 罩杯（而非杯子）；underwire → 钢圈（而非底线）。\n\n"

    "【核心原则四：词汇范围严格限定——禁止输出内衣/内裤/泳衣以外品类的术语】\n"
    "你必须假设当前文本只可能描述内衣、内裤、泳衣这三类产品，不会是童装、男装、"
    "户外服、运动服、牛仔裤、外套、鞋类、箱包等其他任何品类，也不会是建筑、机械、"
    "电子、日用品等其他行业的文本。当一个英文词存在多种行业含义时，只允许选取"
    "内衣/内裤/泳衣语境下对应的中文术语，绝不允许使用其他服装品类或其他行业的常见译法，"
    "哪怕那个译法字面上更常见、更通用。\n"
    "示例（默认按这三类产品的语境理解，不做其他联想）：\n"
    "  'lining' → 里布/内衬（不是西装/外套语境下宽泛的「衬里」）\n"
    "  'strap' → 肩带（不是裤腰带、包带、鞋带）\n"
    "  'wire' → 钢圈（不是电线、钢丝）\n"
    "  'band' → 下扒/底围（不是乐队、橡皮筋的泛称）\n"
    "  'gusset' → 裆布/裆衬\n"
    "  'hook & eye' → 钩眼扣\n"
    "如果遇到术语库未收录、且不确定该词在这三类产品语境下准确译法的词，"
    "仍应给出内衣/内裤/泳衣行业最贴近的合理翻译，但不得套用其他品类或行业的常见说法。\n\n"

    "【参考示例（few-shot，请学习这种基于语境的翻译方式）】\n\n"
    "示例 1：\n"
    "输入：Crotch insert with gusset lining for moisture control\n"
    "✗ 错误的逐词拼接：裆部插入物带衬垫衬里用于湿度控制\n"
    "✓ 正确的语境翻译：裆部加裆衬里布，具备吸湿透气功能\n"
    "说明：'crotch insert' 和 'gusset lining' 在内裤/泳装工艺单里描述的是同一个部位、"
    "同一层结构——裆部内衬，应合并理解为一个完整工艺点，不能拆成两个词分别直译再堆叠。\n\n"
    "示例 2：\n"
    "输入：Underwire channel with cup seam binding\n"
    "✗ 错误的逐词拼接：钢圈通道带罩杯缝合捆绑\n"
    "✓ 正确的语境翻译：钢圈槽配罩杯接缝包边\n"
    "说明：'channel' 在文胸工艺单语境下专指钢圈槽（而非通道），'binding' 在缝份处理语境下"
    "是包边工艺（而非捆绑）；结合上下文给出符合行业习惯的完整短语，而非字面直译再堆叠。\n\n"

    "【核心原则五：强制全量翻译，禁止过度保护短词/指示句】\n"
    "除了纯数字（例如 '170 GSM' 中的 170）之外，文本里出现的所有英文字母、单词、缩写、短语、"
    "操作指示句，全部必须翻译成中文，不允许以'太短'、'像是代码'、'已经够清楚'、"
    "'是常见英文词不需要翻译'等理由跳过翻译、原样返回英文，或留空。"
    "哪怕只有一两个字母组成的词，只要它是一个独立单词而不是型号/尺码代号，也必须翻译。\n"
    "示例：'assortment' 必须译为「搭配/分类」；'pls see Artwork' 必须译为「请见图稿」；"
    "'no' 必须译为「否/无」；'incl. binding' 必须译为「含包边」。\n\n"

    "【核心原则六：正面指令——型号/编码原样输出，但不得连带跳过其前后的英文】\n"
    "所有英文字符均需翻译为中文。如果遇到必须保留的型号、货号、编码（例如 2118875、32A），"
    "请将该型号/编码本身原样输出，但绝不能以\"这是型号/编码\"为借口，"
    "连同它前后紧邻的英文单词、短语一起跳过不翻译——同一段文本里，"
    "编码以外的部分必须照常翻译成中文。\n\n"

    "【核心原则七：免翻译放行规则（Pass-through Rule）】\n"
    "当提取到的文本整体属于以下任一情况时，不需要翻译成中文，原样保留英文即可：\n"
    "  ① 单个英文字母（如表示罩杯/部位的 'D'、'C'）\n"
    "  ② 纯数字、纯标点符号，或数字与标点的组合（如 '2025)'）\n"
    "  ③ 简单型号/尺码代码（如 '32A'、'70B'）\n"
    "  ④ 行业专有缩写，即全大写字母组成的缩写词（如 'NOS'、'GSM'、'USA'）\n"
    "  ⑤ 数字+空格+大写单位的组合（如 '170 GSM'、'32 mm'）\n"
    "  ⑥ 无法确信中文含义的英文代号/色号/款式名（如 'Brick'——如果不确定请原样保留）\n"
    "这类内容本身不是需要翻译的语言文字，强行翻译反而会出错或不知所云。\n"
    "但\"不需要翻译\"绝不代表可以丢弃——你必须在输出的 JSON 中将其【原样】放入 "
    "\"translated\" 字段，与 \"original\" 完全一致，绝不能省略、留空，或跳过这一条不返回。\n"
    "示例：\n"
    '  {"original": "D", "translated": "D"}\n'
    '  {"original": "NOS", "translated": "NOS"}\n'
    '  {"original": "170 GSM", "translated": "170 GSM"}\n'
    '  {"original": "Brick", "translated": "Brick"}\n'
    '  {"original": "2025)", "translated": "2025)"}\n'
    "注意区分：放行规则只适用于上述整体属于代号/缩写/数字/标点的情况；"
    "只要文本里包含真正的英文单词或短语（哪怕很短，如 'no'、'ab'、'assortment'），"
    "仍必须依照核心原则五翻译成中文，不能套用本规则逃避翻译。"
)


_MAX_BATCH_ITEMS = 30


def _chunk(items: list, n: int) -> list[list]:
    return [items[i:i + n] for i in range(0, len(items), n)]


_RE_TABLE_ARTIFACT = re.compile(r'^\s*the following table\s*:?\s*', re.IGNORECASE)
_RE_REPEAT_PUNCT = re.compile(r'([,，.\-_]){3,}')


def _clean_extracted_text(text: str) -> str:
    """Strip known PyMuPDF extraction artifacts (stray 'The following table:'
    prefixes, runs of repeated commas/dots) before the text ever reaches the
    LLM, so this noise can't distract it from real content."""
    t = _RE_TABLE_ARTIFACT.sub('', text)
    t = _RE_REPEAT_PUNCT.sub(lambda m: m.group(1), t)
    return t.strip()


def translate_batch(client: anthropic.Anthropic, texts: list[str], glossary: dict) -> tuple[dict[str, str], set[str]]:
    """Translate many spans in a single LLM call. Returns
    ({original_text: translated_text}, unrecorded_terms)."""
    unique_texts = list(dict.fromkeys(t for t in texts if t.strip()))
    if not unique_texts:
        return {}, set()

    rel = relevant_glossary("\n".join(unique_texts), glossary)
    gloss_block = ""
    if rel:
        lines = "\n".join(f"  {k} → {v}" for k, v in rel.items())
        gloss_block = f"强制术语对照（务必照搬，不得自行发挥）：\n{lines}\n\n"

    items_xml = "\n".join(
        f"<item>{json.dumps(t, ensure_ascii=False)}</item>" for t in unique_texts
    )

    prompt = (
        f"{gloss_block}"
        "<text_to_translate>\n"
        f"{items_xml}\n"
        "</text_to_translate>\n\n"
        "你必须处理 <text_to_translate> 标签内的【每一个 <item>】，不允许遗漏任何一项。\n\n"
        "规则：\n"
        "1. 强制术语对照中词汇的中文译法必须出现在对应译文里，但要结合整条文本的语境自然融入，"
        "禁止逐词直译后生硬拼接。\n"
        "2. 所有英文字符均需翻译为中文。如果某个 item 里含有必须保留的型号/货号/编码"
        "（如 2118875、32A），请将该编码本身原样输出，但不得以此为借口跳过编码前后的英文单词——"
        "同一个 item 里编码以外的部分仍必须翻译。\n"
        "3. 当英文表述简略或存在多义性时，结合服装款式与工艺常识进行合理推断，"
        "给出业内真正想表达的工艺含义，不要停留在字面直译。\n"
        "4. 识别文本中术语对照表里**未收录**的服装/纺织行业专业英文词汇，统一放入返回对象的 "
        '"unrecorded_terms" 数组（去重）。\n'
        "5. 【免翻译放行】如果某个 item 整体仅是单个英文字母（如 'D'、'C'）、纯数字、"
        "纯标点符号，或简单的型号/尺码代码，不需要翻译——但必须原样保留，"
        '"translated" 与 "original" 完全一致地原样输出，绝不能省略或留空这一项。例如：\n'
        '   {"original": "D", "translated": "D"}\n'
        '   {"original": "2025)", "translated": "2025)"}\n'
        "除上述放行情形外，\"translated\" 不允许为空、不允许与 original 相同——"
        "只要 item 里包含真正的英文单词/短语（哪怕很短），必须翻译成中文，不能套用本条逃避翻译。\n"
        "6. 必须返回与 <item> 数量完全一致的 items 数组，每个元素的 \"original\" 必须与对应 "
        "<item> 内容逐字一致。\n"
        "7. 只返回 JSON，不要任何多余文字或 markdown 代码块。\n\n"
        '返回格式：{"items": [{"original": "原文1", "translated": "译文1"}, '
        '{"original": "原文2", "translated": "译文2"}], "unrecorded_terms": ["term1"]}'
    )

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        temperature=0,
        system=_GARMENT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        raw = m.group()
    data = json.loads(raw)

    mapping: dict[str, str] = {}
    for item in data.get("items", []):
        orig = str(item.get("original", "")).strip()
        trans = str(item.get("translated", "")).strip()
        if orig:
            mapping[orig] = trans

    unrecorded = {t.strip() for t in data.get("unrecorded_terms", []) if t.strip()}
    return mapping, unrecorded


_RE_SIZE_LABEL       = re.compile(r'^(?:X{0,3}[SML]|\d?XL|XXL|XXXL)$', re.IGNORECASE)
_RE_SINGLE_LETTER    = re.compile(r'^[A-Za-z]$')
_RE_PURE_DIGIT_PUNCT = re.compile(r'^[\d\W_]+$')           # digits/punctuation only, no letters
_RE_SIMPLE_CODE      = re.compile(r'^[A-Za-z0-9\-/]{1,8}$')  # short alnum model/size code
_RE_ALL_CAPS_ABBR    = re.compile(r'^[A-Z]{2,6}$')          # industry abbreviations: NOS, GSM, USA
_RE_NUM_UNIT         = re.compile(r'^\d+\s+[A-Z]{1,5}$')   # number + uppercase unit: 170 GSM


def _is_pure_size_code(t: str) -> bool:
    """True only for genuine size labels (S/M/L/XL/XXL/2XL...)."""
    return bool(_RE_SIZE_LABEL.match(t))


def _is_passthrough_eligible(t: str) -> bool:
    """True for content that is not translatable natural language: a lone
    letter, pure digits/punctuation, short alnum codes, all-caps industry
    abbreviations (NOS/GSM), or number+unit combos (170 GSM). These are
    allowed to come back unchanged from the LLM without triggering a
    force-retranslation retry. Ordinary short *words* like 'no'/'ab' are
    intentionally NOT eligible and must still be translated."""
    if _is_pure_size_code(t):
        return True
    if _RE_SINGLE_LETTER.match(t):
        return True
    if _RE_PURE_DIGIT_PUNCT.match(t):
        return True
    if _RE_SIMPLE_CODE.match(t) and bool(re.search(r'\d', t)):
        return True
    if _RE_ALL_CAPS_ABBR.match(t):
        return True
    if _RE_NUM_UNIT.match(t):
        return True
    return False


def _needs_retranslation(original: str, translated: str) -> bool:
    """Detect leftover untranslated English: identical to source, or still
    contains Latin letters with zero Chinese characters (excluding
    pass-through-eligible content, which is intentionally left as-is)."""
    o = original.strip()
    t = translated.strip()
    if not t:
        return True
    if o.lower() == t.lower():
        if _is_passthrough_eligible(o) or len(re.findall(r'[a-zA-Z]', o)) == 0:
            return False
        return True
    has_alpha = bool(re.search(r'[a-zA-Z]', t))
    has_cjk = bool(re.search(r'[一-鿿]', t))
    if has_alpha and not has_cjk:
        return True
    return False


def _force_translate(client: anthropic.Anthropic, text: str, glossary: dict) -> str:
    """Last-resort plain-text retry when translate_batch() returns an
    untranslated/empty/identical entry — used instead of silently keeping
    the English original."""
    rel = relevant_glossary(text, glossary)
    gloss_block = ""
    if rel:
        lines = "\n".join(f"  {k} → {v}" for k, v in rel.items())
        gloss_block = f"强制术语对照（务必照搬）：\n{lines}\n\n"
    prompt = (
        "上一次翻译时你返回了未翻译的英文原文，这是不允许的。\n"
        f"{gloss_block}"
        "<text_to_translate>\n"
        f"{text}\n"
        "</text_to_translate>\n\n"
        "你必须翻译 <text_to_translate> 标签内的所有内容，不允许遗漏任何一个单词。"
        "所有英文字符均需翻译为中文；如果遇到必须保留的型号/编码，请原样输出该编码本身，"
        "但不得以此为借口跳过其前后的英文单词，也必须原样保留文本中出现的所有数字。"
        "只返回中文翻译结果本身，不要输出 JSON、不要解释。"
    )
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=_GARMENT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


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

        to_translate = []
        for sp in spans:
            _t = _clean_extracted_text(sp["text"])
            alpha     = len(re.findall(r'[a-zA-Z]', _t))
            has_digit = bool(re.search(r'\d', _t))

            # 纯数字/符号，或清洗后已无内容（PDF 解析伪影）——绝不擦除，直接跳过
            if not _t or alpha == 0:
                done += 1; on_progress(done / total_spans); continue
            # 真正的尺码代号（S/M/L/XL/XXL...）——保持原样，不翻译
            if _is_pure_size_code(_t):
                done += 1; on_progress(done / total_spans); continue
            # 含数字且英文字母极少(≤2)：32A、70B 等尺码代号
            if has_digit and alpha <= 2:
                done += 1; on_progress(done / total_spans); continue

            to_translate.append({**sp, "clean_text": _t})

        results = []
        for batch in _chunk(to_translate, _MAX_BATCH_ITEMS):
            texts = [b["clean_text"] for b in batch]
            on_block(f"批量翻译 {len(texts)} 项…")
            try:
                mapping, unrecorded_batch = translate_batch(client, texts, glossary)
                all_unrecorded |= unrecorded_batch
            except Exception:
                mapping = {}

            for sp in batch:
                translated = mapping.get(sp["clean_text"], "")
                if _needs_retranslation(sp["clean_text"], translated):
                    try:
                        translated = _force_translate(client, sp["clean_text"], glossary)
                    except Exception:
                        pass

                # 渲染兜底：无论翻译是否成功，绝不能让该文本框因译文为空而被空白遮盖/丢失
                if not translated.strip():
                    translated = sp["text"]

                results.append({**sp, "translated": translated})
                done += 1
                on_progress(done / total_spans)

        # 直接叠加译文，不销毁英文原文
        # 原样放行项（translated == original）：跳过整个叠加步骤——不画背景色块也不重绘文字。
        # 底层英文原文保持裸露可见，避免中文字体无法渲染 Latin 字符导致色块遮挡。
        for r in results:
            if r["translated"].strip().lower() == r["clean_text"].strip().lower():
                continue
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
        "你是一名专业服装行业翻译，请翻译 <text_to_translate> 标签内的全部内容，"
        "不允许遗漏任何一个单词。\n\n"
        f"{gloss_block}"
        f"<text_to_translate>\n{text}\n</text_to_translate>\n\n"
        "规则：\n"
        "1. 对照表中词汇的中文译法必须出现在译文里，但要结合整句语境自然融入，"
        "禁止逐词直译后生硬拼接。\n"
        "2. 所有英文字符均需翻译为中文。如果遇到必须保留的型号/货号/编码，"
        "请将该编码本身原样输出，但不得以此为借口跳过编码前后的英文单词。\n"
        "3. 当英文表述简略或存在多义性时，结合服装款式与工艺常识进行合理推断，"
        "给出业内真正想表达的工艺含义，不要停留在字面直译。\n"
        "4. 除纯数字、型号、尺码代号外，所有英文单词/短语/指示句都必须翻译成中文，"
        "禁止因为词很短或是操作指示句就原样保留英文，不允许返回结果与原文相同。\n"
        "5. 只返回翻译结果，不要任何解释或多余文字。"
    )
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=_GARMENT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def _cell_merge_anchor(ws, cell) -> tuple[bool, bool]:
    """返回 (is_in_merged_range, is_anchor_cell)。
    非主单元格的 .value 在 openpyxl 里恒为 None，本来就不会被当作可翻译文本，
    这里只是为了在报告里明确标注，不改变跳过逻辑。"""
    for rng in ws.merged_cells.ranges:
        if cell.coordinate in rng:
            anchor = f"{rng.min_row}_{rng.min_col}"
            is_anchor = (cell.row == rng.min_row and cell.column == rng.min_col)
            return True, is_anchor
    return False, False


def _estimate_layout_warning(ws, cell, translated: str) -> bool:
    """粗略估算：译文字符数是否明显超出该列宽能容纳的字符数。
    中文字符按 1.8 倍英文字符宽度估算（粗略经验值，不追求精确）。"""
    col_letter = cell.column_letter
    dim = ws.column_dimensions.get(col_letter)
    width = dim.width if (dim and dim.width) else 8.43  # Excel 默认列宽
    capacity_chars = width / 1.2  # 默认字体下，约每字符占 1.2 个列宽单位
    cjk_weight = sum(1.8 if ord(ch) > 0x2E80 else 1.0 for ch in translated)
    return cjk_weight > capacity_chars * 1.1  # 留 10% 容差，避免假警报太多


def run_excel_translation(
    xlsx_bytes: bytes,
    glossary_bytes: bytes,
    api_key: str,
    on_cell,
    on_progress,
    translate_images: bool = False,
) -> tuple[bytes, int, int, list[dict]]:
    """Translate text cells (in place, formulas untouched) across ALL sheets.
    Embedded-image text translation is OFF by default (translate_images=False) —
    it rasterizes Chinese into image pixels, which is not editable and uses
    vision-model-estimated coordinates that can drift from the real layout.
    Returns (xlsx_out, n_cells_translated, n_images_translated, report_rows).
    """
    glossary = load_glossary(glossary_bytes)
    client = anthropic.Anthropic(api_key=api_key)

    # ── 阶段 1：翻译文字单元格 ──────────────────────────────────────────────────
    # 不用 data_only=True：公式必须保留原样（"="开头的字符串），否则保存时
    # 公式会被永久替换成当时的计算结果，且无法恢复。
    wb = openpyxl.load_workbook(io.BytesIO(xlsx_bytes))

    to_translate = [
        (ws.title, ws, cell)
        for ws in wb.worksheets
        for row in ws.iter_rows()
        for cell in row
        if _is_translatable(cell.value)
    ]
    n_cells = len(to_translate)
    total_steps = max(n_cells, 1)
    report_rows: list[dict] = []

    for i, (sheet_name, ws, cell) in enumerate(to_translate):
        on_cell(f"[{sheet_name}] {str(cell.value)[:50].replace(chr(10), ' ')}")
        original = str(cell.value)
        is_merged, is_anchor = _cell_merge_anchor(ws, cell)
        row_report = {
            "sheet_name": sheet_name,
            "cell_coordinate": cell.coordinate,
            "original_text": original,
            "translated_text": "",
            "status": "ok",
            "skip_reason": "",
            "is_merged_cell": is_merged,
            "layout_warning": False,
        }

        try:
            translated = translate_cell_text(client, original, glossary)
            if _needs_retranslation(original, translated):
                translated = _force_translate(client, original, glossary)
            cell.value = translated
            row_report["translated_text"] = translated

            # 译文过长：保留原对齐方式的其余设置，只追加开启自动换行；
            # 不动列宽，不写入相邻单元格。仍放不下则只记录 layout_warning。
            if _estimate_layout_warning(ws, cell, translated):
                row_report["layout_warning"] = True
                align = cell.alignment
                cell.alignment = openpyxl.styles.Alignment(
                    horizontal=align.horizontal, vertical=align.vertical,
                    wrap_text=True, text_rotation=align.text_rotation,
                    indent=align.indent,
                )
        except Exception as e:
            row_report["status"] = "error"
            row_report["skip_reason"] = str(e)

        report_rows.append(row_report)
        on_progress((i + 1) / total_steps * (0.6 if translate_images else 1.0))

    buf = io.BytesIO()
    wb.save(buf)
    text_done_bytes = buf.getvalue()

    # ── 阶段 2：翻译嵌入图片（默认关闭）────────────────────────────────────────
    # 关闭原因：会把中文像素烧录进图片本身（不可编辑），且依赖视觉模型估算的
    # 坐标（容易跟实际版式错位/串位）。保留函数实现，按需再开启。
    if not translate_images:
        on_progress(1.0)
        return text_done_bytes, n_cells, 0, report_rows

    with zipfile.ZipFile(io.BytesIO(text_done_bytes)) as zf:
        n_images = sum(
            1 for n in zf.namelist()
            if n.startswith("xl/media/") and n.rsplit(".", 1)[-1].lower() in _IMAGE_EXTS
        )

    if n_images == 0:
        on_progress(1.0)
        return text_done_bytes, n_cells, 0, report_rows

    def on_image(idx, total, fname):
        on_cell(f"图片 {idx}/{total}：{fname}")
        on_progress(0.6 + idx / total * 0.4)   # 后 40% 进度给图片翻译

    final_bytes = translate_images_in_excel(text_done_bytes, client, glossary, on_image)
    on_progress(1.0)
    return final_bytes, n_cells, n_images, report_rows


# ── Excel image translation ───────────────────────────────────────────────────

_IMAGE_EXTS = {"jpg", "jpeg", "png", "webp"}
_MIME = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}


def _vision_extract_items(
    client: anthropic.Anthropic,
    img_bytes: bytes,
    ext: str,
    glossary: dict,
) -> list[dict]:
    """让 Claude Vision 识别图片里的英文标注，返回
    [{"en":..., "zh":..., "x":0~1, "y":0~1, "size":int}, ...]（大致坐标，非精确测量）。
    失败或无文字时返回 []。"""
    media_type = _MIME.get(ext)
    if not media_type:
        return []

    b64 = base64.b64encode(img_bytes).decode("utf-8")

    gloss_hint = ""
    if glossary:
        sample = "\n".join(f"  {k} → {v}" for k, v in list(glossary.items())[:15])
        gloss_hint = f"\n参考术语对照（部分）：\n{sample}"

    try:
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=_GARMENT_SYSTEM_PROMPT,
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
        return json.loads(raw).get("items", [])
    except Exception:
        return []


def _translate_image_bytes(
    client: anthropic.Anthropic,
    img_bytes: bytes,
    ext: str,
    glossary: dict,
) -> bytes:
    """Send one image to Claude Vision, overlay Chinese translations, return new bytes."""
    items = _vision_extract_items(client, img_bytes, ext, glossary)
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


# ── Excel 图片译文：可编辑 TextBox 方案（替代 PIL 烧录像素） ──────────────────────
# 原图片像素完全不动；每条译文生成一个真正的 Excel TextBox（<xdr:sp>），
# 用户可以在 Excel/WPS 里直接点选、双击编辑、拖动——已用真实文件验证过可行。

_XDR_NS = "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing"
_A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
_R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_SML_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_EMU_PER_PT = 12700
_MDW = 7  # 默认字体(Calibri 11)下数字字符的像素宽度，列宽换算用，足够定位精度

ET.register_namespace("xdr", _XDR_NS)
ET.register_namespace("a", _A_NS)
ET.register_namespace("r", _R_NS)


def _drawing_to_sheet_name(all_data: dict) -> dict:
    """drawingN.xml 路径 -> 所在 worksheet 名称（要靠它的列宽/行高换算偏移量）。"""
    wb_xml = ET.fromstring(all_data["xl/workbook.xml"])
    sheets_el = wb_xml.find(f"{{{_SML_NS}}}sheets")
    rid_to_name = {
        sheet_el.get(f"{{{_R_NS}}}id"): sheet_el.get("name")
        for sheet_el in (sheets_el if sheets_el is not None else [])
    }

    wb_rels = ET.fromstring(all_data["xl/_rels/workbook.xml.rels"])
    rid_to_target = {rel.get("Id"): rel.get("Target") for rel in wb_rels}

    result = {}
    for rid, name in rid_to_name.items():
        target = rid_to_target.get(rid)
        if not target:
            continue
        if target.startswith("/"):
            sheet_path = target.lstrip("/")
        else:
            sheet_path = target if target.startswith("xl/") else f"xl/{target}"
        base_dir, fname = sheet_path.rsplit("/", 1)
        rels_data = all_data.get(f"{base_dir}/_rels/{fname}.rels")
        if not rels_data:
            continue
        for rel in ET.fromstring(rels_data):
            if rel.get("Type", "").endswith("/drawing"):
                result[_resolve_media_path(sheet_path, rel.get("Target"))] = name
    return result


def _col_width_emu(ws, col_idx_1based: int) -> int:
    from openpyxl.utils import get_column_letter
    from openpyxl.utils.units import pixels_to_EMU, DEFAULT_COLUMN_WIDTH
    dim = ws.column_dimensions.get(get_column_letter(col_idx_1based))
    width = dim.width if (dim and dim.width) else DEFAULT_COLUMN_WIDTH
    pixels = int(((256 * width + int(128 / _MDW)) / 256) * _MDW)
    return pixels_to_EMU(max(pixels, 1))


def _row_height_emu(ws, row_idx_1based: int) -> int:
    from openpyxl.utils.units import DEFAULT_ROW_HEIGHT
    dim = ws.row_dimensions.get(row_idx_1based)
    height_pt = dim.height if (dim and dim.height) else DEFAULT_ROW_HEIGHT
    return int(height_pt * _EMU_PER_PT)


def _carry_col_offset(ws, start_col_0based: int, offset_emu: int) -> tuple[int, int]:
    """把"从 start_col 起、偏移 offset_emu"换算成真正落在哪一列、列内偏移多少，
    避免单列塞进一个远超其宽度的 offset（很多阅读器会把这种情况渲染错位/堆叠）。"""
    col, off = start_col_0based, max(offset_emu, 0)
    for _ in range(500):
        w = _col_width_emu(ws, col + 1)
        if off < w:
            return col, off
        off -= w
        col += 1
    return col, off


def _carry_row_offset(ws, start_row_0based: int, offset_emu: int) -> tuple[int, int]:
    row, off = start_row_0based, max(offset_emu, 0)
    for _ in range(2000):
        h = _row_height_emu(ws, row + 1)
        if off < h:
            return row, off
        off -= h
        row += 1
    return row, off


def _drawing_rels_map(all_data: dict, drawing_path: str) -> dict:
    """drawingN.xml 里的 rId -> 关系目标路径（如 "../media/image1.png"）。"""
    base_dir, fname = drawing_path.rsplit("/", 1)
    rels_path = f"{base_dir}/_rels/{fname}.rels"
    data = all_data.get(rels_path)
    if not data:
        return {}
    root = ET.fromstring(data)
    return {
        rel.get("Id"): rel.get("Target")
        for rel in root
        if rel.get("Id") and rel.get("Target")
    }


def _resolve_media_path(drawing_path: str, target: str) -> str:
    """把 .rels 里的路径解析成 zip 内的绝对路径。target 可能是相对路径
    （如 "../media/image1.png"，原始 Excel/WPS 常见写法），也可能是包内绝对路径
    （如 "/xl/media/image1.png"，openpyxl 重新保存后常见写法）——两种都要处理。"""
    if target.startswith("/"):
        return target.lstrip("/")
    parts = drawing_path.rsplit("/", 1)[0].split("/")
    for seg in target.split("/"):
        if seg == "..":
            parts.pop()
        elif seg != ".":
            parts.append(seg)
    return "/".join(parts)


def _find_top_level_pics(drawing_root) -> list[tuple]:
    """返回 [(anchor_el, pic_el)]。只取直接挂在 anchor 下的 pic，
    不进 <xdr:grpSp> 分组（分组形状结构更复杂，本轮先跳过、在报告里标注）。"""
    results = []
    for anchor in list(drawing_root):
        tag = anchor.tag.rsplit("}", 1)[-1]
        if tag not in ("twoCellAnchor", "oneCellAnchor"):
            continue
        pic = anchor.find(f"{{{_XDR_NS}}}pic")
        if pic is not None:
            results.append((anchor, pic))
    return results


def _anchor_from(anchor_el) -> tuple[int, int, int, int]:
    frm = anchor_el.find(f"{{{_XDR_NS}}}from")
    return (
        int(frm.find(f"{{{_XDR_NS}}}col").text),
        int(frm.find(f"{{{_XDR_NS}}}colOff").text),
        int(frm.find(f"{{{_XDR_NS}}}row").text),
        int(frm.find(f"{{{_XDR_NS}}}rowOff").text),
    )


def _pic_extent(pic_el):
    """从 <xdr:pic>/<xdr:spPr>/<a:xfrm> 读取图片的局部偏移+宽高（EMU）。
    缺失则返回 None（无法据此换算坐标，调用方应跳过并记录原因）。"""
    xfrm = pic_el.find(f"{{{_XDR_NS}}}spPr/{{{_A_NS}}}xfrm")
    if xfrm is None:
        return None
    ext = xfrm.find(f"{{{_A_NS}}}ext")
    off = xfrm.find(f"{{{_A_NS}}}off")
    if ext is None:
        return None
    cx, cy = int(ext.get("cx")), int(ext.get("cy"))
    ox = int(off.get("x")) if off is not None else 0
    oy = int(off.get("y")) if off is not None else 0
    return ox, oy, cx, cy


def _pic_embed_rid(pic_el):
    blip = pic_el.find(f"{{{_XDR_NS}}}blipFill/{{{_A_NS}}}blip")
    return blip.get(f"{{{_R_NS}}}embed") if blip is not None else None


def _max_shape_id_in_drawing(root) -> int:
    max_id = 0
    for cNvPr in root.iter(f"{{{_XDR_NS}}}cNvPr"):
        try:
            max_id = max(max_id, int(cNvPr.get("id", 0)))
        except (TypeError, ValueError):
            pass
    return max_id


def _make_textbox_anchor(shape_id: int, col: int, colOff: int, row: int, rowOff: int,
                          cx: int, cy: int, zh_text: str, fontsize_pt: int,
                          fill_hex: str = "FFFF99", text_hex: str = "B40000"):
    """构造一个 <xdr:oneCellAnchor><xdr:sp>...黄底中文 TextBox。"""
    anchor = ET.Element(f"{{{_XDR_NS}}}oneCellAnchor")
    frm = ET.SubElement(anchor, f"{{{_XDR_NS}}}from")
    ET.SubElement(frm, f"{{{_XDR_NS}}}col").text = str(col)
    ET.SubElement(frm, f"{{{_XDR_NS}}}colOff").text = str(int(colOff))
    ET.SubElement(frm, f"{{{_XDR_NS}}}row").text = str(row)
    ET.SubElement(frm, f"{{{_XDR_NS}}}rowOff").text = str(int(rowOff))
    ext_el = ET.SubElement(anchor, f"{{{_XDR_NS}}}ext")
    ext_el.set("cx", str(int(cx)))
    ext_el.set("cy", str(int(cy)))

    sp = ET.SubElement(anchor, f"{{{_XDR_NS}}}sp")
    sp.set("macro", "")
    sp.set("textlink", "")
    nvSpPr = ET.SubElement(sp, f"{{{_XDR_NS}}}nvSpPr")
    cNvPr = ET.SubElement(nvSpPr, f"{{{_XDR_NS}}}cNvPr")
    cNvPr.set("id", str(shape_id))
    cNvPr.set("name", f"TextBox_CN_{shape_id}")
    cNvSpPr = ET.SubElement(nvSpPr, f"{{{_XDR_NS}}}cNvSpPr")
    cNvSpPr.set("txBox", "1")

    spPr = ET.SubElement(sp, f"{{{_XDR_NS}}}spPr")
    xfrm = ET.SubElement(spPr, f"{{{_A_NS}}}xfrm")
    off_el = ET.SubElement(xfrm, f"{{{_A_NS}}}off")
    off_el.set("x", "0")
    off_el.set("y", "0")
    ext2 = ET.SubElement(xfrm, f"{{{_A_NS}}}ext")
    ext2.set("cx", str(int(cx)))
    ext2.set("cy", str(int(cy)))
    prstGeom = ET.SubElement(spPr, f"{{{_A_NS}}}prstGeom")
    prstGeom.set("prst", "rect")
    ET.SubElement(prstGeom, f"{{{_A_NS}}}avLst")
    solidFill = ET.SubElement(spPr, f"{{{_A_NS}}}solidFill")
    ET.SubElement(solidFill, f"{{{_A_NS}}}srgbClr").set("val", fill_hex)
    ln = ET.SubElement(spPr, f"{{{_A_NS}}}ln")
    ET.SubElement(ln, f"{{{_A_NS}}}noFill")

    txBody = ET.SubElement(sp, f"{{{_XDR_NS}}}txBody")
    bodyPr = ET.SubElement(txBody, f"{{{_A_NS}}}bodyPr")
    bodyPr.set("wrap", "square")
    bodyPr.set("rtlCol", "0")
    ET.SubElement(bodyPr, f"{{{_A_NS}}}spAutoFit")
    ET.SubElement(txBody, f"{{{_A_NS}}}lstStyle")
    p_el = ET.SubElement(txBody, f"{{{_A_NS}}}p")
    r_el = ET.SubElement(p_el, f"{{{_A_NS}}}r")
    rPr = ET.SubElement(r_el, f"{{{_A_NS}}}rPr")
    rPr.set("lang", "zh-CN")
    rPr.set("sz", str(int(fontsize_pt * 100)))
    sf = ET.SubElement(rPr, f"{{{_A_NS}}}solidFill")
    ET.SubElement(sf, f"{{{_A_NS}}}srgbClr").set("val", text_hex)
    t_el = ET.SubElement(r_el, f"{{{_A_NS}}}t")
    t_el.text = zh_text

    ET.SubElement(anchor, f"{{{_XDR_NS}}}clientData")
    return anchor


def add_translated_textboxes_to_excel(
    xlsx_bytes: bytes,
    client: anthropic.Anthropic,
    glossary: dict,
    on_image,
) -> tuple[bytes, list[dict]]:
    """图片像素完全不动；每条识别出的英文标注生成一个独立、可编辑、可拖动的
    Excel TextBox（黄底+中文），定位在原图片范围内对应的相对坐标处。
    返回 (xlsx_out, report_rows)。"""
    with zipfile.ZipFile(io.BytesIO(xlsx_bytes)) as zf:
        infos = {info.filename: info for info in zf.infolist()}
        all_data = {name: zf.read(name) for name in zf.namelist()}

    # 列宽/行高换算偏移量需要 worksheet 的 column_dimensions/row_dimensions，
    # 这里只读不存，不会重新触发 openpyxl 的整书写出（那个交给 run_excel_translation）。
    dims_wb = openpyxl.load_workbook(io.BytesIO(xlsx_bytes))
    drawing_sheet_name = _drawing_to_sheet_name(all_data)

    drawing_paths = sorted(
        n for n in all_data if re.match(r"xl/drawings/drawing\d+\.xml$", n)
    )

    parsed = {}
    total_pics = 0
    for dp in drawing_paths:
        root = ET.fromstring(all_data[dp])
        pics = _find_top_level_pics(root)
        parsed[dp] = (root, pics)
        total_pics += len(pics)

    report_rows: list[dict] = []
    done = 0

    for dp, (root, pics) in parsed.items():
        if not pics:
            continue
        rels = _drawing_rels_map(all_data, dp)
        next_id = _max_shape_id_in_drawing(root) + 1

        for anchor_el, pic_el in pics:
            done += 1
            rid = _pic_embed_rid(pic_el)
            target = rels.get(rid) if rid else None
            media_path = _resolve_media_path(dp, target) if target else None
            fname = media_path.split("/")[-1] if media_path else f"(未知图片，{dp})"
            on_image(done, max(total_pics, 1), fname)

            if not media_path or media_path not in all_data:
                report_rows.append({"drawing": dp, "image": fname, "status": "skipped",
                                     "original_text": "", "translated_text": "",
                                     "skip_reason": "找不到图片关系映射"})
                continue

            extent = _pic_extent(pic_el)
            if extent is None:
                report_rows.append({"drawing": dp, "image": fname, "status": "skipped",
                                     "original_text": "", "translated_text": "",
                                     "skip_reason": "图片缺少 xfrm 定位信息"})
                continue

            ext = media_path.rsplit(".", 1)[-1].lower()
            items = _vision_extract_items(client, all_data[media_path], ext, glossary)
            if not items:
                report_rows.append({"drawing": dp, "image": fname, "status": "no_text",
                                     "original_text": "", "translated_text": "", "skip_reason": ""})
                continue

            from_col, from_colOff, from_row, from_rowOff = _anchor_from(anchor_el)
            _ox, _oy, cx, cy = extent  # ox/oy 是图片内部历史遗留的局部坐标，不可信，不用它

            sheet_name = drawing_sheet_name.get(dp)
            ws = dims_wb[sheet_name] if sheet_name in dims_wb.sheetnames else None
            if ws is None:
                report_rows.append({"drawing": dp, "image": fname, "status": "skipped",
                                     "original_text": "", "translated_text": "",
                                     "skip_reason": "找不到对应 worksheet，无法换算列宽/行高"})
                continue

            for item in items:
                zh = str(item.get("zh", "")).strip()
                en = str(item.get("en", "")).strip()
                if not zh:
                    continue
                x_norm = max(0.0, min(float(item.get("x", 0.5)), 0.95))
                y_norm = max(0.0, min(float(item.get("y", 0.5)), 0.95))
                fontsize_pt = max(8, min(int(item.get("size", 12)), 24))

                # 绝对偏移量按实际列宽/行高"进位"，换算成真正的目标列/行+列内偏移，
                # 不要把一个远超该列宽度的 offset 硬塞在同一列里（会被部分阅读器
                # 误判/堆叠在一起）。
                target_col, box_colOff = _carry_col_offset(
                    ws, from_col, from_colOff + int(x_norm * cx)
                )
                target_row, box_rowOff = _carry_row_offset(
                    ws, from_row, from_rowOff + int(y_norm * cy)
                )
                box_w = max(int(cx * 0.25), len(zh) * fontsize_pt * _EMU_PER_PT)
                box_h = int(fontsize_pt * 1.6 * _EMU_PER_PT)

                new_anchor = _make_textbox_anchor(
                    next_id, target_col, box_colOff, target_row, box_rowOff,
                    box_w, box_h, zh, fontsize_pt,
                )
                next_id += 1
                root.append(new_anchor)
                report_rows.append({"drawing": dp, "image": fname, "status": "ok",
                                     "original_text": en, "translated_text": zh, "skip_reason": ""})

        all_data[dp] = ET.tostring(root, encoding="utf-8", xml_declaration=True)

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zf_out:
        for name, data in all_data.items():
            zf_out.writestr(name, data, compress_type=infos[name].compress_type)
    return out.getvalue(), report_rows


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

# ── 会话内术语库初始化 ─────────────────────────────────────────────────────────
if "glossary_df" not in st.session_state:
    st.session_state["glossary_df"] = _load_default_glossary_df()
    st.session_state["glossary_source"] = (
        DEFAULT_GLOSSARY.name if DEFAULT_GLOSSARY.exists() else "（空，未找到默认术语库）"
    )
    st.session_state["glossary_conflicts"] = pd.DataFrame(columns=["英文术语", "旧翻译", "新翻译（采用）"])

# ── API Key（全局共用）───────────────────────────────────────────────────────
api_key = st.text_input(
    "🔑 Anthropic API Key",
    type="password",
    value=os.environ.get("ANTHROPIC_API_KEY", ""),
    placeholder="sk-ant-api03-...",
)
st.divider()

tab_pdf, tab_excel, tab_glossary = st.tabs(["📄 PDF 翻译（支持批量）", "📊 Excel 翻译", "📚 术语库管理"])

# ════════════════════════════════════════════════════════════════════════════
#  PDF Tab — batch upload, independent processing per file
# ════════════════════════════════════════════════════════════════════════════
with tab_pdf:
    pdf_files = st.file_uploader(
        "上传待翻译 PDF（可一次选择多个文件）",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
    )
    font_label = (
        f"🔤 上传中文字体 TTF（可选，已检测到默认字体 {DEFAULT_FONT.name}）"
        if DEFAULT_FONT.exists()
        else "🔤 上传中文字体 TTF（必填，未检测到默认字体）"
    )
    font_file = st.file_uploader(font_label, type=["ttf"], key="pdf_font_uploader")
    font_ready = bool(font_file or DEFAULT_FONT.exists())
    st.caption(f"📚 当前术语库：**{st.session_state['glossary_source']}** · {len(glossary_df_to_dict(st.session_state['glossary_df']))} 条术语（可在「术语库管理」标签页编辑）")

    can_start_pdf = bool(pdf_files and api_key and font_ready)
    if not can_start_pdf:
        missing = []
        if not pdf_files:
            missing.append("PDF 文件")
        if not api_key:
            missing.append("Anthropic API Key")
        if not font_ready:
            missing.append("中文字体 TTF")
        st.info(f"请先提供：{'、'.join(missing)}")

    start_pdf_btn = st.button(
        "🚀  开始翻译 PDF",
        disabled=not can_start_pdf,
        use_container_width=True,
        type="primary",
        key="start_pdf_btn",
    )

    if start_pdf_btn:
        st.session_state["pdf_batch_results"] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            if font_file:
                fp = os.path.join(tmpdir, "font.ttf")
                with open(fp, "wb") as f:
                    f.write(font_file.read())
                font_path = fp
            else:
                font_path = str(DEFAULT_FONT)

            glossary_bytes = glossary_df_to_xlsx_bytes(st.session_state["glossary_df"])

            overall = st.progress(0.0, text=f"准备处理 {len(pdf_files)} 个文件…")
            results = []

            for fi, pf in enumerate(pdf_files):
                with st.status(f"正在处理：{pf.name}", expanded=True) as status:
                    block_ph = st.empty()
                    file_prog = st.progress(0.0)

                    def on_page(pn, total, n_blocks):
                        st.write(f"**── 第 {pn + 1} / {total} 页**　（{n_blocks} 个文本块）")

                    def on_block(preview):
                        block_ph.caption(f"▶ 正在翻译：{preview}…")

                    def on_progress(frac):
                        file_prog.progress(frac, text=f"翻译进度 {frac:.0%}")

                    try:
                        pdf_out, xlsx_out, n_terms = run_pdf_translation(
                            pdf_bytes=pf.read(),
                            glossary_bytes=glossary_bytes,
                            font_path=font_path,
                            api_key=api_key,
                            on_page=on_page,
                            on_block=on_block,
                            on_progress=on_progress,
                        )
                        block_ph.empty()
                        status.update(label=f"✅ {pf.name} 翻译完成", state="complete")
                        results.append({
                            "name": pf.name,
                            "ok": True,
                            "pdf": pdf_out,
                            "xlsx": xlsx_out,
                            "n_terms": n_terms,
                        })
                    except Exception as e:
                        block_ph.empty()
                        status.update(label=f"❌ {pf.name} 出错：{e}", state="error")
                        results.append({
                            "name": pf.name,
                            "ok": False,
                            "error": str(e),
                        })

                overall.progress((fi + 1) / len(pdf_files), text=f"已完成 {fi + 1}/{len(pdf_files)}")

            st.session_state["pdf_batch_results"] = results

    batch_results = st.session_state.get("pdf_batch_results", [])
    if batch_results:
        st.divider()
        n_ok = sum(1 for r in batch_results if r["ok"])
        st.success(f"批量处理完成：成功 **{n_ok}** / 失败 **{len(batch_results) - n_ok}**（共 {len(batch_results)} 个文件）")

        # 全部成功结果打包为 ZIP
        ok_results = [r for r in batch_results if r["ok"]]
        if len(ok_results) > 1:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w") as zf:
                for r in ok_results:
                    base = r["name"].rsplit(".", 1)[0]
                    zf.writestr(f"{base}_translated.pdf", r["pdf"])
                    if r["n_terms"] > 0:
                        zf.writestr(f"{base}_unrecorded_terms.xlsx", r["xlsx"])
            st.download_button(
                label=f"⬇️  打包下载全部 {len(ok_results)} 个译文（ZIP）",
                data=zip_buf.getvalue(),
                file_name="translated_pdfs.zip",
                mime="application/zip",
                use_container_width=True,
                type="primary",
            )

        for i, r in enumerate(batch_results):
            with st.expander(f"{'✅' if r['ok'] else '❌'} {r['name']}", expanded=not r["ok"]):
                if r["ok"]:
                    base = r["name"].rsplit(".", 1)[0]
                    st.caption(f"未收录术语：{r['n_terms']} 条")
                    dl1, dl2 = st.columns(2)
                    with dl1:
                        st.download_button(
                            label="⬇️  下载中文 PDF",
                            data=r["pdf"],
                            file_name=f"{base}_translated.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            key=f"pdf_dl_{i}",
                        )
                    with dl2:
                        if r["n_terms"] > 0:
                            st.download_button(
                                label="⬇️  下载未收录术语",
                                data=r["xlsx"],
                                file_name=f"{base}_unrecorded_terms.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True,
                                key=f"xlsx_dl_{i}",
                            )
                else:
                    st.error(r["error"])

# ════════════════════════════════════════════════════════════════════════════
#  Excel Tab
# ════════════════════════════════════════════════════════════════════════════
def _build_excel_report_bytes(report_rows: list[dict], img_report_rows: list[dict] | None = None) -> bytes:
    report_wb = openpyxl.Workbook()
    report_ws = report_wb.active
    report_ws.title = "翻译报告"
    headers = ["sheet_name", "cell_coordinate", "original_text", "translated_text",
               "status", "skip_reason", "is_merged_cell", "layout_warning"]
    report_ws.append(headers)
    for cell in report_ws[1]:
        cell.font = openpyxl.styles.Font(bold=True)
    for r in report_rows:
        report_ws.append([r[h] for h in headers])
    for col_letter, width in zip("ABCDEFGH", [12, 14, 35, 35, 8, 20, 14, 14]):
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


with tab_excel:
    excel_files = st.file_uploader(
        "上传待翻译 Excel（可一次选择多个文件）",
        type=["xlsx"],
        accept_multiple_files=True,
        key="excel_uploader",
    )
    st.caption(f"📚 当前术语库：**{st.session_state['glossary_source']}** · {len(glossary_df_to_dict(st.session_state['glossary_df']))} 条术语（可在「术语库管理」标签页编辑）")

    can_start_excel = bool(excel_files and api_key)
    if not can_start_excel:
        missing = []
        if not excel_files:
            missing.append("Excel 文件")
        if not api_key:
            missing.append("Anthropic API Key")
        st.info(f"请先提供：{'、'.join(missing)}")

    translate_images_checked = st.checkbox(
        "🧪 同时翻译图片内文字（实验性：生成可编辑 TextBox，不改图片像素）",
        key="excel_translate_images_checkbox",
    )
    st.caption("勾选后，图片里识别到的英文标注会生成黄底中文 TextBox 叠在图片对应位置，"
               "可在 Excel/WPS 里直接双击编辑、拖动。坐标是视觉模型估算的，密集排版的图"
               "可能不够精确，请翻译后打开核对。不勾选则图片完全原样保留。")

    start_excel_btn = st.button(
        "🚀  开始翻译 Excel",
        disabled=not can_start_excel,
        use_container_width=True,
        type="primary",
        key="start_excel_btn",
    )

    if start_excel_btn:
        st.session_state["excel_batch_results"] = []

        glossary_bytes = glossary_df_to_xlsx_bytes(st.session_state["glossary_df"])
        glossary_dict = load_glossary(glossary_bytes)
        overall = st.progress(0.0, text=f"准备处理 {len(excel_files)} 个文件…")
        results = []

        for fi, ef in enumerate(excel_files):
            with st.status(f"正在处理：{ef.name}", expanded=True) as status:
                cell_ph = st.empty()
                file_prog = st.progress(0.0)

                def on_cell(preview):
                    cell_ph.caption(f"▶ 正在翻译：{preview}…")

                def on_progress(frac):
                    file_prog.progress(frac, text=f"翻译进度 {frac:.0%}")

                try:
                    excel_out, n_cells, n_images, report_rows = run_excel_translation(
                        xlsx_bytes=ef.read(),
                        glossary_bytes=glossary_bytes,
                        api_key=api_key,
                        on_cell=on_cell,
                        on_progress=on_progress,
                        translate_images=False,
                    )

                    img_report_rows = []
                    if translate_images_checked:
                        img_ph = st.empty()

                        def on_image(i, total, fname):
                            img_ph.caption(f"🖼️ 图片译文 {i}/{total}：{fname}")

                        client = anthropic.Anthropic(api_key=api_key)
                        excel_out, img_report_rows = add_translated_textboxes_to_excel(
                            excel_out, client, glossary_dict, on_image,
                        )
                        img_ph.empty()

                    cell_ph.empty()
                    status.update(label=f"✅ {ef.name} 翻译完成", state="complete")
                    results.append({
                        "name": ef.name,
                        "ok": True,
                        "excel": excel_out,
                        "report": report_rows,
                        "img_report": img_report_rows,
                        "n_cells": n_cells,
                    })
                except Exception as e:
                    cell_ph.empty()
                    status.update(label=f"❌ {ef.name} 出错：{e}", state="error")
                    results.append({"name": ef.name, "ok": False, "error": str(e)})

            overall.progress((fi + 1) / len(excel_files), text=f"已完成 {fi + 1}/{len(excel_files)}")

        st.session_state["excel_batch_results"] = results

    batch_results = st.session_state.get("excel_batch_results", [])
    if batch_results:
        st.divider()
        n_ok = sum(1 for r in batch_results if r["ok"])
        img_note = "图片已按勾选生成可编辑 TextBox 译文。" if translate_images_checked else "图片未处理，原样保留。"
        st.success(f"批量处理完成：成功 **{n_ok}** / 失败 **{len(batch_results) - n_ok}**（共 {len(batch_results)} 个文件）。{img_note}")

        ok_results = [r for r in batch_results if r["ok"]]
        if len(ok_results) > 1:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w") as zf:
                for r in ok_results:
                    base = r["name"].rsplit(".", 1)[0]
                    zf.writestr(f"{base}_中文版.xlsx", r["excel"])
                    zf.writestr(f"{base}_翻译报告.xlsx",
                                _build_excel_report_bytes(r["report"], r.get("img_report")))
            st.download_button(
                label=f"⬇️  打包下载全部 {len(ok_results)} 个译文（ZIP）",
                data=zip_buf.getvalue(),
                file_name="translated_excels.zip",
                mime="application/zip",
                use_container_width=True,
                type="primary",
            )

        for i, r in enumerate(batch_results):
            with st.expander(f"{'✅' if r['ok'] else '❌'} {r['name']}", expanded=not r["ok"]):
                if r["ok"]:
                    base = r["name"].rsplit(".", 1)[0]
                    report_rows = r["report"]
                    img_report_rows = r.get("img_report") or []
                    n_warn = sum(1 for row in report_rows if row["layout_warning"])
                    n_merged = sum(1 for row in report_rows if row["is_merged_cell"])
                    st.caption(
                        f"文字单元格 **{r['n_cells']}** 个"
                        f"（合并单元格内 **{n_merged}** 个，触发换行提醒 **{n_warn}** 个）"
                    )
                    if img_report_rows:
                        n_img_ok = sum(1 for row in img_report_rows if row["status"] == "ok")
                        n_img_skip = sum(1 for row in img_report_rows if row["status"] == "skipped")
                        st.caption(
                            f"图片译文 TextBox **{n_img_ok}** 条"
                            + (f"，跳过 **{n_img_skip}** 处（详见报告）" if n_img_skip else "")
                        )
                    dl1, dl2 = st.columns(2)
                    with dl1:
                        st.download_button(
                            label="⬇️  下载中文 Excel",
                            data=r["excel"],
                            file_name=f"{base}_中文版.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                            key=f"excel_dl_{i}",
                        )
                    with dl2:
                        st.download_button(
                            label="⬇️  下载翻译报告",
                            data=_build_excel_report_bytes(report_rows, img_report_rows),
                            file_name=f"{base}_翻译报告.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                            key=f"report_dl_{i}",
                        )
                else:
                    st.error(r["error"])

# ════════════════════════════════════════════════════════════════════════════
#  Glossary Management Tab
# ════════════════════════════════════════════════════════════════════════════
with tab_glossary:
    glossary_df = st.session_state["glossary_df"]
    n_complete = len(glossary_df_to_dict(glossary_df))
    n_incomplete = len(glossary_df) - n_complete
    st.caption(
        f"当前术语库来源：**{st.session_state['glossary_source']}** ｜ "
        f"完整术语 **{n_complete}** 条，待补充翻译 **{n_incomplete}** 条"
    )

    # ── ① 上传替换 ──────────────────────────────────────────────────────────
    st.markdown("#### ① 上传术语库（将替换当前术语库）")
    uploaded_glossary = st.file_uploader(
        "上传术语库 Excel（表头需包含「英文」「中文」列）",
        type=["xlsx", "xls"],
        key="glossary_replace_uploader",
    )
    if uploaded_glossary is not None:
        new_df, missing_cols = parse_glossary_excel(uploaded_glossary.read())
        if missing_cols:
            st.error("术语库格式不正确，缺少以下列：\n\n" + "\n".join(f"- {m}" for m in missing_cols))
        else:
            preview_cleaned, preview_conflicts, preview_incomplete = clean_glossary_df(new_df)
            st.info(
                f"解析成功：完整术语 **{len(glossary_df_to_dict(preview_cleaned))}** 条，"
                f"待补充翻译 **{preview_incomplete}** 条。点击下方按钮应用，将替换当前术语库。"
            )
            if st.button("✅ 应用此术语库", key="apply_uploaded_glossary"):
                st.session_state["glossary_df"] = preview_cleaned
                st.session_state["glossary_conflicts"] = preview_conflicts
                st.session_state["glossary_source"] = uploaded_glossary.name
                st.rerun()

    st.divider()

    # ── ② 查看 + 搜索 ───────────────────────────────────────────────────────
    st.markdown("#### ② 查看当前术语库")
    search_kw = st.text_input("🔍 搜索（英文 / 中文 / 备注 / 分类）", key="glossary_search")
    view_df = st.session_state["glossary_df"]
    if search_kw.strip():
        kw = search_kw.strip().lower()
        mask = view_df.apply(lambda r: kw in " ".join(str(v).lower() for v in r), axis=1)
        view_df = view_df[mask]
    st.dataframe(view_df, use_container_width=True, height=300)

    st.divider()

    # ── ③ 增 / 改 / 删 ──────────────────────────────────────────────────────
    st.markdown("#### ③ 新增 / 修改 / 删除术语")
    st.caption("直接编辑表格；选中整行后可删除；在底部空行新增。编辑完成后点击「保存」才会生效。")
    edited_df = st.data_editor(
        st.session_state["glossary_df"][_GLOSSARY_EDIT_COLS],
        num_rows="dynamic",
        use_container_width=True,
        key="glossary_editor",
    )
    if st.button("💾 保存术语库修改", use_container_width=True, type="primary", key="save_glossary_edits"):
        cleaned, conflicts, n_incomplete_after = clean_glossary_df(edited_df)
        st.session_state["glossary_df"] = cleaned
        st.session_state["glossary_conflicts"] = conflicts
        st.session_state["glossary_source"] = "手动编辑"
        if not conflicts.empty:
            st.warning(f"检测到 {len(conflicts)} 处英文术语重复且翻译不同，已自动采用最新一条翻译。")
        st.success(f"已保存：完整术语 {len(glossary_df_to_dict(cleaned))} 条，待补充翻译 {n_incomplete_after} 条。")
        st.rerun()

    st.divider()

    # ── ④ 未收录术语回填 ────────────────────────────────────────────────────
    st.markdown("#### ④ 未收录术语回填")
    st.caption("上传 PDF/Excel 翻译后下载的「未收录术语」文件，在其中新增「中文翻译」列填好后再上传，系统会自动合并进当前术语库。")
    unrecorded_upload = st.file_uploader(
        "上传补充翻译后的未收录术语 Excel",
        type=["xlsx", "xls"],
        key="unrecorded_uploader",
    )
    if unrecorded_upload is not None:
        unrecorded_df, missing_cols = parse_unrecorded_excel(unrecorded_upload.read())
        if missing_cols:
            st.error("文件格式不正确，缺少以下列：\n\n" + "\n".join(f"- {m}" for m in missing_cols))
        else:
            if st.button("🔀 合并到当前术语库", key="merge_unrecorded_btn"):
                merged, conflicts, n_added, n_overwritten, n_skipped = merge_unrecorded_into_glossary(
                    st.session_state["glossary_df"], unrecorded_df
                )
                cleaned, extra_conflicts, _ = clean_glossary_df(merged)
                all_conflicts = pd.concat([conflicts, extra_conflicts], ignore_index=True)
                st.session_state["glossary_df"] = cleaned
                st.session_state["glossary_conflicts"] = all_conflicts
                st.session_state["glossary_source"] = "术语库 + 未收录术语回填"
                msg = f"合并完成：新增 **{n_added}** 条，覆盖更新 **{n_overwritten}** 条"
                if n_skipped:
                    msg += f"，跳过 **{n_skipped}** 条未填写中文翻译的行"
                st.success(msg)
                st.rerun()

    st.divider()

    # ── ⑤ 下载 ──────────────────────────────────────────────────────────────
    st.markdown("#### ⑤ 下载")
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            label="⬇️  下载 updated_glossary.xlsx",
            data=glossary_df_to_xlsx_bytes(st.session_state["glossary_df"]),
            file_name="updated_glossary.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            type="primary",
        )
    with dl2:
        conflicts_now = st.session_state.get("glossary_conflicts")
        if conflicts_now is not None and not conflicts_now.empty:
            st.download_button(
                label=f"⬇️  下载 conflict_report.xlsx（{len(conflicts_now)} 条）",
                data=_df_to_simple_xlsx(conflicts_now, "Conflicts"),
                file_name="conflict_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        else:
            st.caption("暂无术语冲突")
