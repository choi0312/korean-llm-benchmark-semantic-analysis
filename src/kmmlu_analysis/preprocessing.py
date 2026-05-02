import re
import math
import unicodedata
import numpy as np
import pandas as pd

QUESTION_KEYS = ["question", "Question", "query", "Query", "problem", "Problem", "prompt", "Prompt", "stem", "Stem", "item", "Item"]
CHOICE_LIST_KEYS = ["choices", "Choices", "options", "Options", "choice", "Choice"]
CHOICE_COL_PATTERNS = [r"^[A-E]$", r"^[a-e]$", r"^option[_\-\s]?[1-9]$", r"^choice[_\-\s]?[1-9]$", r"^보기[_\-\s]?[1-9]$"]
EXCLUDE_TEXT_COLS = {"answer", "answers", "label", "labels", "gold", "target", "correct", "correct_answer", "correct_index", "score", "id", "idx", "index", "_dataset", "_split", "_config"}
DOMAIN_CANDIDATES = ["subject", "Subject", "category", "Category", "domain", "Domain", "profession", "Profession", "license", "License", "exam", "Exam", "field", "Field", "task", "Task", "source", "Source", "sub_category", "major"]


def stringify_value(x) -> str:
    if x is None:
        return ""
    try:
        if isinstance(x, float) and math.isnan(x):
            return ""
    except Exception:
        pass
    if isinstance(x, (list, tuple, np.ndarray)):
        return " ".join([stringify_value(v) for v in x if stringify_value(v)])
    if isinstance(x, dict):
        return " ".join([stringify_value(v) for v in x.values() if stringify_value(v)])
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def normalize_korean_text(text: str) -> str:
    text = stringify_value(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\ufeff", " ")
    text = re.sub(r"[\u200b\u200c\u200d]", " ", text)
    text = re.sub(r"[①②③④⑤⑥⑦⑧⑨⑩]", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_choice_column(col: str) -> bool:
    s = str(col).strip()
    low = s.lower()
    return any(re.match(p, s) or re.match(p, low) for p in CHOICE_COL_PATTERNS)


def build_item_text(row: pd.Series) -> str:
    parts = []
    cols = list(row.index)
    lower_to_col = {str(c).lower(): c for c in cols}
    for key in QUESTION_KEYS:
        if key.lower() in lower_to_col:
            v = stringify_value(row[lower_to_col[key.lower()]])
            if v:
                parts.append(v)
            break
    for key in CHOICE_LIST_KEYS:
        if key.lower() in lower_to_col:
            v = stringify_value(row[lower_to_col[key.lower()]])
            if v:
                parts.append(v)
    for c in sorted([c for c in cols if is_choice_column(c)], key=lambda x: str(x)):
        v = stringify_value(row[c])
        if v:
            parts.append(v)
    if not parts:
        for c in cols:
            if str(c).lower() in EXCLUDE_TEXT_COLS:
                continue
            v = stringify_value(row[c])
            if v and len(v) > 1:
                parts.append(v)
    return normalize_korean_text(" ".join(parts))


def infer_domain(row: pd.Series) -> str:
    for c in DOMAIN_CANDIDATES:
        if c in row.index:
            v = normalize_korean_text(row[c])
            if v:
                return v
    if "_config" in row.index:
        v = normalize_korean_text(row["_config"])
        if v:
            return v
    return "unknown"


def prepare_text_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df["_text"] = df.apply(build_item_text, axis=1)
    df["_domain"] = df.apply(infer_domain, axis=1)
    df = df[df["_text"].str.len() > 0].reset_index(drop=True)
    df = df.drop_duplicates(subset=["_dataset", "_text"]).reset_index(drop=True)
    return df
