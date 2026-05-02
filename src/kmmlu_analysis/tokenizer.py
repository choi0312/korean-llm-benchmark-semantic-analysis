import re
import pickle
import hashlib
from pathlib import Path
from tqdm.auto import tqdm
from konlpy.tag import Okt
from .preprocessing import normalize_korean_text

STOPWORDS = {
    "다음", "중", "것", "있는", "없는", "대한", "대해", "관련", "설명", "옳은", "옳지", "않은",
    "가장", "적절한", "적절하지", "보기", "아래", "경우", "해당", "모두", "이를", "이는",
    "이며", "이다", "한다", "하여", "에서", "으로", "에게", "에는", "또는", "그리고", "그러나",
    "따라", "때", "수", "등", "및", "의", "를", "은", "는", "이", "가", "에", "와", "과",
    "로", "도", "만", "문항", "문제", "정답", "선택", "선지", "내용", "위한", "무엇",
    "어느", "몇", "각", "표", "그림", "있다", "없다", "된다", "하는", "하지", "되어",
    "따른", "관한", "하나", "하시오", "고르시오", "맞는", "틀린", "사항"
}
KEEP_ONE_CHAR = {"법", "세", "약", "병", "균", "암", "피", "뇌", "폐", "간", "눈", "코", "입", "뼈", "힘", "열", "값", "물", "불", "빛", "선", "점", "면", "각", "축", "파", "장", "항", "조", "식", "금", "은", "동", "철", "차", "집", "일", "권", "죄", "율", "률"}
KEEP_POS_FOR_TOKENS = {"Noun", "Alpha", "Verb", "Adjective"}


def clean_token(token: str) -> str:
    token = normalize_korean_text(token).lower()
    if not token:
        return ""
    if re.fullmatch(r"\d+([.,]\d+)?", token):
        return ""
    if re.fullmatch(r"[\W_]+", token):
        return ""
    if token in STOPWORDS:
        return ""
    if len(token) == 1 and token not in KEEP_ONE_CHAR:
        return ""
    return token


def tokenize_text(text: str, okt=None):
    okt = okt or Okt()
    try:
        pos = okt.pos(normalize_korean_text(text), norm=True, stem=True)
    except Exception:
        pos = []
    tokens, nouns = [], []
    for word, tag in pos:
        w = clean_token(word)
        if not w:
            continue
        if tag in KEEP_POS_FOR_TOKENS:
            tokens.append(w)
        if tag == "Noun":
            nouns.append(w)
    return tokens, nouns


def data_signature(texts) -> str:
    return hashlib.md5("\n".join(texts).encode("utf-8")).hexdigest()


def tokenize_dataframe(df, cache_dir=None, force=False):
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"tokenized_benchmark_{data_signature(df['_text'].tolist())}.pkl"
        if cache_path.exists() and not force:
            with open(cache_path, "rb") as f:
                obj = pickle.load(f)
            df["tokens"] = obj["tokens"]
            df["nouns"] = obj["nouns"]
            return df
    okt = Okt()
    all_tokens, all_nouns = [], []
    for text in tqdm(df["_text"].tolist(), desc="KoNLPy Okt benchmark tokenizing"):
        tokens, nouns = tokenize_text(text, okt)
        all_tokens.append(tokens)
        all_nouns.append(nouns)
    df["tokens"] = all_tokens
    df["nouns"] = all_nouns
    if cache_dir:
        with open(cache_path, "wb") as f:
            pickle.dump({"tokens": all_tokens, "nouns": all_nouns}, f)
    return df


def tokenize_corpus_lines(lines, max_lines=None, min_sentence_tokens=2):
    okt = Okt()
    tokenized = []
    for i, line in enumerate(tqdm(lines, desc="KoNLPy Okt corpus tokenizing")):
        if max_lines is not None and i >= max_lines:
            break
        line = normalize_korean_text(line)
        if not line:
            continue
        tokens, _ = tokenize_text(line, okt)
        if len(tokens) >= min_sentence_tokens:
            tokenized.append(tokens)
    return tokenized
