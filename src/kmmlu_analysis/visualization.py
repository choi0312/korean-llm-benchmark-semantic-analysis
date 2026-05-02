import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_basic_statistics(summary_df, fig_dir):
    fig_dir = Path(fig_dir)
    cols = ["문항수", "어절수", "전체토큰수", "명사수", "고유명사수"]
    plot_df = summary_df.set_index("dataset")[cols].T
    plt.figure(figsize=(10, 6))
    x = np.arange(len(plot_df.index))
    width = 0.35
    datasets = list(plot_df.columns)
    for i, name in enumerate(datasets):
        offset = (i - (len(datasets) - 1) / 2) * width
        plt.bar(x + offset, plot_df[name].values, width=width, label=name)
    plt.xticks(x, plot_df.index, rotation=20, ha="right")
    plt.ylabel("Count")
    plt.title("KMMLU-Redux와 KMMLU-Pro의 기본 텍스트 통계")
    plt.legend()
    plt.tight_layout()
    path = fig_dir / "figure1_basic_statistics.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_term_clusters_tsne(tsne_df, fig_dir, label_top=80):
    fig_dir = Path(fig_dir)
    plt.figure(figsize=(11, 9))
    plt.scatter(tsne_df["x"], tsne_df["y"], c=tsne_df["cluster"], s=np.clip(tsne_df["frequency"] * 2, 20, 250), alpha=0.75)
    labels = set(tsne_df.sort_values("frequency", ascending=False).head(label_top)["term"].tolist())
    for _, row in tsne_df.iterrows():
        if row["term"] in labels:
            plt.text(row["x"], row["y"], row["term"], fontsize=8)
    plt.title("Large-corpus Word2Vec 기반 주요 어휘 군집")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    path = fig_dir / "figure3_term_clusters_tsne.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_domain_similarity_heatmap(domain_sim_df, fig_dir):
    fig_dir = Path(fig_dir)
    if domain_sim_df.empty:
        return None
    plt.figure(figsize=(12, 8))
    plt.imshow(domain_sim_df.values, aspect="auto")
    plt.colorbar(label="Cosine similarity")
    plt.xticks(np.arange(len(domain_sim_df.columns)), domain_sim_df.columns, rotation=45, ha="right")
    plt.yticks(np.arange(len(domain_sim_df.index)), domain_sim_df.index)
    plt.title("분야별 문항벡터 Centroid 유사도")
    plt.xlabel("KMMLU-Pro 분야")
    plt.ylabel("KMMLU-Redux 분야")
    plt.tight_layout()
    path = fig_dir / "figure4_domain_centroid_similarity_heatmap.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path
