import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from .similarity import keywords_from_tokens, truncate_text


def textrank_items(df, V, dataset_name, topn=15, neighbor_k=10):
    pos = np.where((df["_dataset"].values == dataset_name) & (df["has_vector"].values))[0]
    if len(pos) == 0:
        return pd.DataFrame()
    Vd = V[pos]
    S = cosine_similarity(Vd)
    np.fill_diagonal(S, 0.0)
    G = nx.Graph()
    G.add_nodes_from(range(len(pos)))
    k = min(neighbor_k, len(pos) - 1)
    for i in range(len(pos)):
        nbr_idx = np.argpartition(S[i], -k)[-k:]
        for j in nbr_idx:
            w = float(S[i, j])
            if w > 0:
                G.add_edge(i, j, weight=w)
    scores = nx.pagerank(G, weight="weight", max_iter=300, tol=1e-6)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topn]
    rows = []
    for rank, (local_idx, score) in enumerate(ranked, start=1):
        row = df.iloc[pos[local_idx]]
        rows.append({
            "dataset": dataset_name,
            "rank": rank,
            "textrank_score": round(float(score), 6),
            "domain": row["_domain"],
            "keywords": keywords_from_tokens(row["tokens"], topn=10),
            "text_preview": truncate_text(row["_text"], max_len=120),
        })
    return pd.DataFrame(rows)
