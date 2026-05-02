import os
import sys
import argparse
from pathlib import Path
import yaml
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from kmmlu_analysis.io_utils import ensure_dirs, set_seed, set_korean_font, zip_outputs
from kmmlu_analysis.data_loader import load_benchmarks
from kmmlu_analysis.preprocessing import prepare_text_dataframe
from kmmlu_analysis.tokenizer import tokenize_dataframe, tokenize_corpus_lines
from kmmlu_analysis.corpus_loader import build_corpus_line_iterator
from kmmlu_analysis.statistics import add_text_statistics, compute_basic_statistics, compute_top_nouns
from kmmlu_analysis.embedding import train_word2vec_from_sentences, extract_similar_words
from kmmlu_analysis.vectorizer import build_item_vectors
from kmmlu_analysis.clustering import cluster_terms, compute_tsne
from kmmlu_analysis.similarity import compute_cross_item_similarity, compute_domain_centroid_similarity
from kmmlu_analysis.textrank import textrank_items
from kmmlu_analysis.visualization import plot_basic_statistics, plot_term_clusters_tsne, plot_domain_similarity_heatmap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_rows_per_dataset", type=str, default=None)
    parser.add_argument("--max_corpus_lines", type=str, default=None)
    parser.add_argument("--corpus_source", type=str, default=None)
    args = parser.parse_args()

    config_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.output_dir:
        cfg["project"]["output_dir"] = args.output_dir
    if args.max_rows_per_dataset is not None:
        cfg["datasets"]["max_rows_per_dataset"] = None if args.max_rows_per_dataset.lower() in ["none", "null"] else int(args.max_rows_per_dataset)
    if args.max_corpus_lines is not None:
        cfg["corpus"]["max_corpus_lines"] = None if args.max_corpus_lines.lower() in ["none", "null"] else int(args.max_corpus_lines)
    if args.corpus_source is not None:
        cfg["corpus"]["source"] = args.corpus_source

    output_dir = ROOT / cfg["project"]["output_dir"]
    paths = ensure_dirs(output_dir)
    set_seed(cfg["project"]["seed"])
    set_korean_font()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN 환경변수가 필요합니다.")

    repo_ids = {k: v for k, v in cfg["datasets"].items() if k not in ["max_rows_per_dataset"]}

    print("[1] Load KMMLU datasets")
    raw_df = load_benchmarks(
        repo_ids=repo_ids,
        token=hf_token,
        max_rows_per_dataset=cfg["datasets"].get("max_rows_per_dataset"),
        seed=cfg["project"]["seed"],
    )

    print("[2] Preprocess benchmark text")
    df = prepare_text_dataframe(raw_df)

    print("[3] Tokenize benchmark items")
    df = tokenize_dataframe(
        df,
        cache_dir=paths["cache"],
        force=cfg["tokenizer"].get("force_retokenize", False),
    )

    print("[4] Text statistics")
    df = add_text_statistics(df)
    summary_df = compute_basic_statistics(df)
    top_nouns_df = compute_top_nouns(df, top_n=cfg["analysis"].get("top_nouns", 50))
    summary_df.to_csv(paths["tables"] / "table1_dataset_basic_statistics.csv", index=False, encoding="utf-8-sig")
    top_nouns_df.to_csv(paths["tables"] / "table2_top_nouns.csv", index=False, encoding="utf-8-sig")
    plot_basic_statistics(summary_df, paths["figures"])

    print("[5] Build large-corpus Word2Vec training sentences")
    corpus_sentences = []
    source = cfg["corpus"].get("source", "benchmark_only")
    if source != "benchmark_only":
        line_iter = build_corpus_line_iterator(cfg["corpus"], token=hf_token)
        corpus_sentences = tokenize_corpus_lines(
            line_iter,
            max_lines=cfg["corpus"].get("max_corpus_lines"),
            min_sentence_tokens=cfg["tokenizer"].get("min_sentence_tokens", 2),
        )
    else:
        print("[WARN] corpus_source=benchmark_only: large corpus is not used.")

    benchmark_sentences = [tokens for tokens in df["tokens"].tolist() if len(tokens) >= cfg["tokenizer"].get("min_sentence_tokens", 2)]
    if cfg["corpus"].get("combine_benchmark_text", True):
        train_sentences = corpus_sentences + benchmark_sentences
    else:
        train_sentences = corpus_sentences

    print(f"Corpus tokenized sentences: {len(corpus_sentences):,}")
    print(f"Benchmark tokenized sentences: {len(benchmark_sentences):,}")
    print(f"Total Word2Vec training sentences: {len(train_sentences):,}")

    print("[6] Train Word2Vec")
    wv_cfg = cfg["word2vec"]
    model = train_word2vec_from_sentences(
        train_sentences,
        vector_size=wv_cfg["vector_size"],
        window=wv_cfg["window"],
        min_count=wv_cfg["min_count"],
        workers=wv_cfg.get("workers", max(1, os.cpu_count() or 2)),
        sg=wv_cfg["sg"],
        negative=wv_cfg["negative"],
        epochs=wv_cfg["epochs"],
        seed=cfg["project"]["seed"],
    )
    model.save(str(paths["models"] / "word2vec_large_corpus_kmmlu.model"))
    print("Word2Vec vocab size:", len(model.wv))

    similar_df = extract_similar_words(model, top_nouns_df, topn=cfg["analysis"].get("similar_topn", 8))
    similar_df.to_csv(paths["tables"] / "table3_word2vec_similar_words.csv", index=False, encoding="utf-8-sig")

    print("[7] Term clustering")
    c_cfg = cfg["clustering"]
    detail_df, cluster_summary_df, X_terms = cluster_terms(
        df,
        model,
        top_terms=c_cfg["top_terms"],
        n_clusters=c_cfg["n_clusters"],
        seed=cfg["project"]["seed"],
    )
    detail_df.to_csv(paths["tables"] / "table4_term_clusters_detail.csv", index=False, encoding="utf-8-sig")
    cluster_summary_df.to_csv(paths["tables"] / "table4_term_clusters_summary.csv", index=False, encoding="utf-8-sig")
    tsne_df = compute_tsne(detail_df, X_terms, seed=cfg["project"]["seed"])
    tsne_df.to_csv(paths["tables"] / "table4_term_clusters_tsne_coordinates.csv", index=False, encoding="utf-8-sig")
    plot_term_clusters_tsne(tsne_df, paths["figures"], label_top=c_cfg["tsne_top_labels"])

    print("[8] Item vectors and similarity")
    V, df = build_item_vectors(df, model, vector_size=wv_cfg["vector_size"])
    s_cfg = cfg["similarity"]
    cross_public_df, cross_debug_df = compute_cross_item_similarity(df, V, top_pair_n=s_cfg["top_pair_n"])
    cross_public_df.to_csv(paths["tables"] / "table5_cross_item_similarity_public.csv", index=False, encoding="utf-8-sig")
    cross_debug_df.to_csv(paths["tables"] / "table5_cross_item_similarity_debug_with_previews.csv", index=False, encoding="utf-8-sig")

    domain_sim_df = compute_domain_centroid_similarity(df, V, top_domains=s_cfg["top_domains"])
    if not domain_sim_df.empty:
        domain_sim_df.to_csv(paths["tables"] / "table6_domain_centroid_similarity.csv", encoding="utf-8-sig")
        plot_domain_similarity_heatmap(domain_sim_df, paths["figures"])

    print("[9] TextRank")
    t_cfg = cfg["textrank"]
    tr_list = []
    for dataset_name in df["_dataset"].unique():
        tr_list.append(textrank_items(df, V, dataset_name, topn=t_cfg["topn"], neighbor_k=t_cfg["neighbor_k"]))
    textrank_df = pd.concat(tr_list, ignore_index=True)
    textrank_df.to_csv(paths["tables"] / "table7_textrank_representative_items.csv", index=False, encoding="utf-8-sig")

    print("[10] Write interpretation draft")
    lines = ["KMMLU-Redux와 KMMLU-Pro 의미 구조 분석 결과 요약", ""]
    lines.append(f"Word2Vec training source: {source}")
    lines.append(f"Corpus tokenized sentences: {len(corpus_sentences):,}")
    lines.append(f"Benchmark tokenized sentences: {len(benchmark_sentences):,}")
    lines.append(f"Word2Vec vocab size: {len(model.wv):,}")
    lines.append("")
    for _, row in summary_df.iterrows():
        lines.append(f"- {row['dataset']}: 문항 {int(row['문항수'])}개, 어절 {int(row['어절수'])}개, 토큰 {int(row['전체토큰수'])}개, 명사 {int(row['명사수'])}개, 평균 어절 {row['평균문항길이_어절']}개")
    lines.append("")
    lines.append("대용량 말뭉치 기반 Word2Vec을 사용하면 분석 대상 문항만으로 학습했을 때보다 희소 전문용어와 일반 한국어 문맥의 의미 관계를 더 안정적으로 반영할 수 있다.")
    lines.append("다만 Wikipedia/KCC/WikiText의 도메인 분포와 전문시험 문항의 도메인 분포가 다르므로, 벤치마크 문항 추가 학습 또는 도메인별 말뭉치 보강이 필요하다.")
    with open(output_dir / "report_interpretation_draft.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    zip_path = zip_outputs(output_dir)
    print("[DONE] Outputs:", output_dir)
    print("[DONE] Zip:", zip_path)


if __name__ == "__main__":
    main()
