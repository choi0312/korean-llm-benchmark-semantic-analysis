import re
import pandas as pd
from collections import Counter
from .preprocessing import normalize_korean_text


def count_sentences_simple(text: str) -> int:
    text = normalize_korean_text(text)
    if not text:
        return 0
    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    parts = [p.strip() for p in parts if p.strip()]
    return max(1, len(parts))


def add_text_statistics(df):
    df["char_count"] = df["_text"].apply(len)
    df["eojeol_count"] = df["_text"].apply(lambda x: len(str(x).split()))
    df["sentence_count"] = df["_text"].apply(count_sentences_simple)
    df["token_count"] = df["tokens"].apply(len)
    df["noun_count"] = df["nouns"].apply(len)
    df["unique_noun_count"] = df["nouns"].apply(lambda x: len(set(x)))
    return df


def compute_basic_statistics(df):
    rows = []
    for dataset_name, sub in df.groupby("_dataset"):
        all_tokens = [t for toks in sub["tokens"] for t in toks]
        all_nouns = [n for ns in sub["nouns"] for n in ns]
        rows.append({
            "dataset": dataset_name,
            "문항수": len(sub),
            "문장수": int(sub["sentence_count"].sum()),
            "어절수": int(sub["eojeol_count"].sum()),
            "전체토큰수": len(all_tokens),
            "명사수": len(all_nouns),
            "고유명사수": len(set(all_nouns)),
            "평균문항길이_어절": round(float(sub["eojeol_count"].mean()), 2),
            "평균토큰수": round(float(sub["token_count"].mean()), 2),
            "평균명사수": round(float(sub["noun_count"].mean()), 2),
        })
    return pd.DataFrame(rows)


def compute_top_nouns(df, top_n=50):
    rows = []
    for dataset_name, sub in df.groupby("_dataset"):
        counter = Counter([n for ns in sub["nouns"] for n in ns])
        total = sum(counter.values())
        for rank, (noun, freq) in enumerate(counter.most_common(top_n), start=1):
            rows.append({
                "dataset": dataset_name,
                "rank": rank,
                "noun": noun,
                "frequency": freq,
                "ratio": round(freq / total, 6) if total > 0 else 0,
            })
    return pd.DataFrame(rows)
