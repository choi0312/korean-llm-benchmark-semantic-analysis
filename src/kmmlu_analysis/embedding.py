import pandas as pd
from gensim.models import Word2Vec


def train_word2vec_from_sentences(sentences, vector_size=200, window=5, min_count=3, workers=4, sg=1, negative=10, epochs=10, seed=42):
    if len(sentences) < 10:
        raise RuntimeError("Word2Vec 학습 문장이 너무 적습니다.")
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        negative=negative,
        epochs=epochs,
        seed=seed,
    )
    return model


def extract_similar_words(model, top_nouns_df, topn=8):
    manual_terms = ["안전", "관리", "설비", "전기", "공정", "환경", "위험", "사고", "법", "법률", "계약", "의무", "책임", "환자", "진단", "치료", "세액", "소득", "비용", "자산", "회계", "세무", "기계", "의료"]
    auto_terms = top_nouns_df["noun"].head(100).tolist() if len(top_nouns_df) else []
    candidate_terms = []
    for t in manual_terms + auto_terms:
        if t not in candidate_terms and t in model.wv:
            candidate_terms.append(t)
    rows = []
    for term in candidate_terms:
        for rank, (word, score) in enumerate(model.wv.most_similar(term, topn=topn), start=1):
            rows.append({
                "query_term": term,
                "rank": rank,
                "similar_word": word,
                "cosine_similarity": round(float(score), 4),
            })
    return pd.DataFrame(rows)
