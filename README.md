# HYU-lightRAG

AI 반도체 특허 100건에 대해 LightRAG, GraphRAG, Naive RAG를 비교하는 재현 실험 레포지토리.

한양대학교 산업데이터엔지니어링학과 KG/Ontology Lab 석사과정 연구의 일부로, Graph-based RAG가 특허 도메인에서 실제로 유효한지 정량 평가한다.

## 실험 개요

| 항목 | 값 |
|---|---|
| 문서 | AI 반도체 특허 100건 (AA/AB/AC/AD 각 25건, seed=20260609) |
| 시스템 | LightRAG (naive/local/global/hybrid), GraphRAG v3.1.0 (basic/local/global) |
| 생성 모델 | gpt-5.5 (codex-proxy) |
| 임베딩 | text-embedding-3-large (OpenRouter) |
| 평가 | Gemini 3.5 Flash (LLM-as-judge, 5 rubrics, pairwise A/B swap) |
| 쿼리 | 15개 고정 평가셋 (category-specific 8, cross-category 3, fact-check 2, comparison 1, exploratory 1) |

## 핵심 결과

| 비교 | Overall 승률 | 비고 |
|---|---|---|
| LightRAG hybrid vs naive | 53.3% vs 46.7% | 사실상 동등 — 그래프 부가가치 제한적 |
| LightRAG hybrid vs GraphRAG global | 13.3% vs 86.7% | GraphRAG global 우세, 단 답변 길이 2.43x 차이 (length bias confound) |
| LightRAG hybrid vs GraphRAG local | 93.3% vs 6.7% | LightRAG 압도 |
| GraphRAG global vs naive | 80.0% vs 20.0% | GraphRAG global 우세, length bias 동일 적용 |

**주요 발견**: GraphRAG global의 압도적 승률은 답변 길이 편향(평균 3218 vs 1325 chars)과 분리해서 해석해야 한다. LightRAG의 특허 특화 그래프 구축은 품질 지표상 성공(technical relation 94%, metadata relation 1%)했으나, hybrid가 naive 대비 뚜렷한 우위를 보이지 않았다.

## 레포 구조

```
HYU-lightRAG/
├── patent_lightrag/           # 실험 파이프라인 코드
│   ├── sample_patents.py      # 특허 샘플링 (seed 기반)
│   ├── build_patent_docs.py   # 특허 문서 변환
│   ├── index_patents.py       # LightRAG 인덱싱
│   ├── graphrag_repro.py      # GraphRAG 인덱싱/쿼리
│   ├── lightrag_query_batch.py # LightRAG 배치 쿼리
│   ├── rag_repro_judge.py     # Gemini judge 평가
│   ├── rag_repro_metrics.py   # 자동 메트릭 계산
│   ├── rag_repro_report.py    # HTML 리포트 생성
│   ├── graph_metrics.py       # 그래프 품질 지표
│   └── common.py              # 공통 유틸리티
├── LightRAG-main/prompts/entity_type/
│   └── patent_ai_semiconductor.yml  # 특허 특화 entity 추출 프롬프트
├── experiments/rag_repro_100_seed20260609/
│   ├── dataset/               # 샘플링 manifest
│   ├── queries/               # 15개 평가 쿼리
│   ├── lightrag_patent_prompt_100/  # LightRAG 인덱스/쿼리 결과
│   ├── graphrag_full_100_fresh/     # GraphRAG 인덱스/쿼리 결과
│   └── evaluation/            # Judge 결과 및 메트릭
└── reports/
    └── rag_repro_100_comparison.html  # 최종 비교 리포트
```

## 실행 방법

### 1. 환경 설정

```bash
# LightRAG 의존성
pip install -r LightRAG-main/requirements-offline.txt

# GraphRAG (external/ 에 v3.1.0 클론)
cd external/graphrag && pip install -e .
```

`.env` 파일에 API 키 설정:

```env
EMBEDDING_BINDING_API_KEY=<OPENROUTER_API_KEY>
```

### 2. Preflight 체크

```bash
python3 -m patent_lightrag.preflight
python3 -m patent_lightrag.preflight --check-openrouter
```

### 3. 데이터 준비 → 인덱싱 → 쿼리 → 평가 → 리포트

```bash
# 특허 100건 샘플링
python3 -m patent_lightrag.sample_patents

# LightRAG 인덱싱 + 쿼리
python3 -m patent_lightrag.index_patents
python3 -m patent_lightrag.lightrag_query_batch

# GraphRAG 인덱싱 + 쿼리
python3 -m patent_lightrag.graphrag_repro

# Judge 평가
python3 -m patent_lightrag.rag_repro_judge

# 리포트 생성
python3 -m patent_lightrag.rag_repro_report
```

## 그래프 품질 지표

| 지표 | 값 | 목표 |
|---|---|---|
| metadata relation ratio | 0.01 | ≤ 0.25 |
| technical relation ratio | 0.94 | ≥ 0.40 |
| excluded entity ratio | 0.004 | ≤ 0.10 |
| degree ≥ 50 hub count | 0 | 0 |
| nodes / edges | 1596 / 2203 | — |

## 제한사항

- LLM-as-judge의 verbosity bias가 주요 교란 요인 (길이 통제 실험 미실시)
- 쿼리 15개로 표본 수가 적어 통계적 유의성 검정 불가
- gpt-5.5로 생성과 인덱싱을 동시 수행해 self-enhancement bias 가능성
- 특허 도메인 한정 결과이므로 일반화에 주의
