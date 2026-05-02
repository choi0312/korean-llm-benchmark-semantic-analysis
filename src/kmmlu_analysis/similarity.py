import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from .preprocessing import normalize_korean_text


def keywords_from_tokens(tokens, topn=8):
    if not tokens:
        return ""
    return ", ".join([w for w, _ in Counter(tokens).most_common(topn)])


def truncate_text(text, max_len=120):
    text = normalize_korean_text(text)
    return text[:max_len] + "..." if len(text) > max_len else text


def compute_cross_item_similarity(df, V, top_pair_n=50):
    redux_pos = np.where((df["_dataset"].values == "KMMLU-Redux") & (df["has_vector"].values))[0]
    pro_pos = np.where((df["_dataset"].values == "KMMLU-Pro") & (df["has_vector"].values))[0]
    if len(redux_pos) == 0 or len(pro_pos) == 0:
        return pd.DataFrame(), pd.DataFrame()
    S = cosine_similarity(V[redux_pos], V[pro_pos])
    flat = S.ravel()
    top_k = min(top_pair_n, flat.shape[0])
    idx = np.argpartition(flat, -top_k)[-top_k:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    public_rows, debug_rows = [], []
    for rank, flat_idx in enumerate(idx, start=1):
        i, j = np.unravel_index(flat_idx, S.shape)
        ri, pj = redux_pos[i], pro_pos[j]
        r, p = df.iloc[ri], df.iloc[pj]
        row = {
            "rank": rank,
            "similarity": round(float(S[i, j]), 4),
            "redux_domain": r["_domain"],
            "pro_domain": p["_domain"],
            "redux_keywords": keywords_from_tokens(r["tokens"]),
            "pro_keywords": keywords_from_tokens(p["tokens"]),
        }
        public_rows.append(row)
        debug_rows.append({**row, "redux_text_preview": truncate_text(r["_text"]), "pro_text_preview": truncate_text(p["_text"])})
    return pd.DataFrame(public_rows), pd.DataFrame(debug_rows)


def compute_domain_centroid_similarity(df, V, top_domains=12):
    def top_domain_names(dataset):
        return df[df["_dataset"] == dataset]["_domain"].value_counts().head(top_domains).index.tolist()
    def centroid(pos):
        vecs = V[pos]
        valid = np.linalg.norm(vecs, axis=1) > 0
        vecs = vecs[valid]
        return None if len(vecs) == 0 else vecs.mean(axis=0)
    r_domains, p_domains = top_domain_names("KMMLU-Redux"), top_domain_names("KMMLU-Pro")
    r_cen, r_valid, p_cen, p_valid = [], [], [], []
    for d in r_domains:
        pos = np.where((df["_dataset"].values == "KMMLU-Redux") & (df["_domain"].values == d) & (df["has_vector"].values))[0]
        c = centroid(pos)
        if c is not None:
            r_cen.append(c); r_valid.append(d)
    for d in p_domains:
        pos = np.where((df["_dataset"].values == "KMMLU-Pro") & (df["_domain"].values == d) & (df["has_vector"].values))[0]
        c = centroid(pos)
        if c is not None:
            p_cen.append(c); p_valid.append(d)
    if not r_cen or not p_cen:
        return pd.DataFrame()
    sim = cosine_similarity(np.vstack(r_cen), np.vstack(p_cen))
    return pd.DataFrame(sim, index=r_valid, columns=p_valid)
