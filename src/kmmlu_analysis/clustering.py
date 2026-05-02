import numpy as np
import pandas as pd
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


def cluster_terms(df, model, top_terms=180, n_clusters=7, seed=42):
    counter = Counter([t for toks in df["tokens"] for t in toks])
    terms = [w for w, c in counter.most_common(top_terms) if w in model.wv]
    if len(terms) < 20:
        raise RuntimeError("클러스터링할 단어 수가 부족합니다.")
    X = np.vstack([model.wv[w] for w in terms])
    n_clusters = min(n_clusters, max(3, len(terms) // 25))
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20)
    labels = km.fit_predict(X)
    detail = pd.DataFrame({"term": terms, "frequency": [counter[w] for w in terms], "cluster": labels})
    rows = []
    for cid, sub in detail.groupby("cluster"):
        sub = sub.sort_values("frequency", ascending=False)
        rows.append({"cluster": int(cid), "num_terms": len(sub), "representative_terms": ", ".join(sub["term"].head(12).tolist())})
    summary = pd.DataFrame(rows).sort_values("cluster")
    return detail, summary, X


def compute_tsne(detail, X, seed=42):
    n_terms = len(detail)
    perplexity = min(30, max(2, (n_terms - 1) // 3))
    tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity, init="pca", learning_rate="auto")
    coords = tsne.fit_transform(X)
    out = detail.copy()
    out["x"] = coords[:, 0]
    out["y"] = coords[:, 1]
    return out
