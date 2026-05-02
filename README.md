# Korean LLM Benchmark Semantic Analysis

> [Research] Korean LLM benchmark semantic structure analysis with large-corpus Word2Vec, Korean morphology, cosine similarity, and TextRank

**[2026년도 1학기 빅데이터최신기술 최종보고서]**

본 프로젝트는 한국어 LLM 벤치마크를 단순한 성능 점수의 집합이 아니라 **분석 가능한 전문지식 문항 텍스트**로 보고, KMMLU-Redux와 KMMLU-Pro의 형태소·어휘·문장 의미 구조를 비교하는 연구형 NLP 프로젝트입니다.

이번 버전에서는 기존의 "벤치마크 문항만으로 Word2Vec을 학습하는 방식"을 개선하여, **KCC, Wikipedia, WikiText와 같은 대용량 한국어 말뭉치 기반 Word2Vec 학습 구조**로 재설계했습니다. 대용량 말뭉치에서 일반 한국어 문맥을 먼저 학습하고, 벤치마크 문항을 함께 반영함으로써 희소 전문용어와 문항 어휘의 의미벡터 안정성을 높이는 것을 목표로 합니다.

## 1. 프로젝트 개요

본 연구는 LG AI Research가 Hugging Face를 통해 공개한 한국어 전문지식 LLM 벤치마크인 **KMMLU-Redux**와 **KMMLU-Pro**를 대상으로 다음 분석을 수행합니다.

- 한국어 형태소 분석 및 명사 추출
- 대용량 말뭉치 기반 Word2Vec 학습
- 고빈도 명사 기반 어휘 분포 비교
- 주요 어휘 유사도 계산
- KMeans 기반 term clustering
- TF-IDF 가중 Word2Vec 평균 기반 문항벡터 구성
- 문항벡터 cosine similarity 계산
- TextRank 기반 대표 문항 유형 추출

핵심 목표는 LLM의 정답률을 비교하는 것이 아니라, **벤치마크 문항 자체가 어떤 한국어 언어 구조와 전문지식 의미 구조를 가지는지 분석하는 것**입니다.

## 2. 설계 

본 프로젝트는 다음 구조로 재설계되었습니다.

~~~text
Large Korean Corpus
(KCC / Wikipedia / WikiText)
        +
KMMLU benchmark items
        ↓
KoNLPy morphological tokenization
        ↓
Large-corpus Word2Vec training
        ↓
Item vectorization and semantic analysis
~~~

이 구조는 실제 한국어 LLM 벤치마크 문항의 전문어휘 분석에도 적용할 수 있도록 설계되었습니다.

## 3. Dataset

| Dataset | Source | Description |
|---|---|---|
| KMMLU-Redux | LGAI-EXAONE/KMMLU-Redux | 한국 국가기술자격시험 기반 한국어 전문지식 벤치마크 |
| KMMLU-Pro | LGAI-EXAONE/KMMLU-Pro | 한국 국가전문자격시험 기반 고난도 전문직 지식 벤치마크 |
| Large Corpus | KCC / Wikipedia / WikiText | Word2Vec 사전학습 또는 보강 학습용 한국어 대용량 말뭉치 |

원본 데이터와 대용량 말뭉치는 저장소에 포함하지 않습니다. 실행 시 Hugging Face 또는 사용자가 지정한 텍스트 파일에서 직접 불러옵니다.

## 4. Method

분석 파이프라인은 다음과 같습니다.

~~~text
Hugging Face dataset loading
→ question + choices text construction
→ UTF-8 / Unicode normalization
→ KoNLPy Okt morphological analysis
→ large corpus tokenization
→ large-corpus Word2Vec training
→ similar word extraction
→ term clustering
→ TF-IDF weighted item vectorization
→ cosine similarity analysis
→ TextRank centrality analysis
→ tables and figures export
~~~

## 5. 주요 결과 해석 방향

실험 결과는 다음 관점에서 해석합니다.

- **KMMLU-Pro**: 보험, 계약, 등기, 회사, 채권 등 법률·계약·권리관계 중심 어휘가 두드러지는지 확인
- **KMMLU-Redux**: 검사, 시험, 온도, 시설, 압력 등 산업기술·절차·측정 중심 어휘가 강하게 나타나는지 확인
- 대용량 말뭉치 기반 Word2Vec이 희소 전문용어의 유사어휘를 더 안정적으로 제공하는지 확인
- 문항벡터 유사도 분석에서 규정 적용, 절차 판단, 제재 기준 선택과 같은 공통 의미 구조가 나타나는지 확인
- TextRank 분석을 통해 각 벤치마크 내부에서 중심성이 높은 대표 문항 영역을 추출

## 6. Repository Structure

~~~text
.
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml
├── src/
│   └── kmmlu_analysis/
│       ├── corpus_loader.py
│       ├── data_loader.py
│       ├── preprocessing.py
│       ├── tokenizer.py
│       ├── statistics.py
│       ├── embedding.py
│       ├── vectorizer.py
│       ├── clustering.py
│       ├── similarity.py
│       ├── textrank.py
│       ├── visualization.py
│       └── io_utils.py
├── scripts/
│   └── run_pipeline.py
├── reports/
│   └── figures/
└── outputs_sample/
    └── tables/
~~~

## 7. How to Run on Google Colab

~~~bash
git clone https://github.com/choi0312/korean-llm-benchmark-semantic-analysis.git
cd korean-llm-benchmark-semantic-analysis

pip install -r requirements.txt
apt-get -qq update
apt-get -qq install -y openjdk-17-jdk fonts-nanum

export HF_TOKEN=your_huggingface_token_here
python scripts/run_pipeline.py --config configs/default.yaml
~~~

빠른 테스트는 다음처럼 실행할 수 있습니다.

~~~bash
python scripts/run_pipeline.py --config configs/default.yaml   --output_dir outputs_validation   --max_rows_per_dataset 300   --max_corpus_lines 50000
~~~

벤치마크 문항만으로 빠르게 디버깅하려면 다음처럼 실행할 수 있습니다.

~~~bash
python scripts/run_pipeline.py --config configs/default.yaml   --output_dir outputs_debug   --max_rows_per_dataset 300   --corpus_source benchmark_only
~~~

## 8. Outputs

| Output | Description |
|---|---|
| table1_dataset_basic_statistics.csv | 문항 수, 문장 수, 어절 수, 토큰 수, 명사 수 |
| table2_top_nouns.csv | 데이터셋별 고빈도 명사 |
| table3_word2vec_similar_words.csv | Word2Vec 기반 유사어휘 |
| table4_term_clusters_summary.csv | 주요 어휘 군집 요약 |
| table5_cross_item_similarity_public.csv | Redux-Pro 간 문항벡터 유사도 |
| table7_textrank_representative_items.csv | TextRank 대표 문항 유형 |
| figure1_basic_statistics.png | 기본 텍스트 통계 비교 |
| figure3_term_clusters_tsne.png | 주요 어휘 군집 시각화 |
| figure4_domain_centroid_similarity_heatmap.png | 분야별 centroid 유사도 heatmap |

## 9. Limitations

대용량 말뭉치 기반 Word2Vec은 일반 한국어 문맥을 안정적으로 반영할 수 있지만, Wikipedia/KCC/WikiText의 도메인 분포가 국가기술자격시험 및 국가전문자격시험 문항과 완전히 일치하지는 않습니다. 따라서 향후에는 분야별 전문 말뭉치 추가, FastText subword 정보 활용, BERT 또는 Sentence-BERT 계열 문장 임베딩 비교를 통해 문항 전체의 문맥적 의미 유사도를 더 정교하게 분석할 필요가 있습니다.

## 10. Keywords

Korean NLP · LLM Benchmark · KMMLU · Morphological Analysis · Large-corpus Word2Vec · TextRank · Cosine Similarity · Semantic Structure Analysis
