import numpy as np
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer


def build_idf_dict(df):
    token_texts = [" ".join(tokens) for tokens in df["tokens"].tolist()]
    vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=1)
    vec.fit_transform(token_texts)
    return dict(zip(vec.get_feature_names_out(), vec.idf_))


def item_vector(tokens, model, idf_dict=None, vector_size=200):
    vecs, weights = [], []
    for tok in tokens:
        if tok in model.wv:
            vecs.append(model.wv[tok])
            weights.append(float(idf_dict.get(tok, 1.0)) if idf_dict else 1.0)
    if not vecs:
        return np.zeros(vector_size, dtype=np.float32)
    return np.average(np.vstack(vecs), axis=0, weights=np.array(weights, dtype=np.float32)).astype(np.float32)


def build_item_vectors(df, model, vector_size=200):
    idf_dict = build_idf_dict(df)
    V = np.vstack([
        item_vector(tokens, model, idf_dict, vector_size)
        for tokens in tqdm(df["tokens"].tolist(), desc="Building item vectors")
    ])
    df["has_vector"] = np.linalg.norm(V, axis=1) > 0
    return V, df
