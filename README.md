# GraphRAG PDF 챗봇

![Python](https://img.shields.io/badge/Python-3.11+-blue) ![LangGraph](https://img.shields.io/badge/LangGraph-0.*-green)

LangGraph 기반 하이브리드 GraphRAG 파이프라인 + Streamlit 챗봇. PDF를 인덱싱해 벡터검색(ChromaDB)과 지식 그래프(Neo4j)를 동시에 활용하며, 한국어 특화 온디바이스 모델로 추론한다.

---

## 목적

한국어 PDF 문서에 대한 정밀한 질의응답이 목표다. 단순 벡터 유사도 검색만으로는 엔티티 관계나 맥락 추론이 약하다는 한계를 극복하기 위해, 벡터검색과 Neo4j 지식 그래프 검색을 결합한 하이브리드 GraphRAG를 구성했다. 추론은 한국어 특화 소형 모델(Konan-LLM-OND, 4B)로 수행하며, 장기적으로 모바일 온디바이스 배포를 목표로 한다.

---

## 원리 / 동작 방식

두 개의 LangGraph `StateGraph`로 구성된다.

### 인제스천 파이프라인 (1회 실행)
```
load_pdf → split_chunks → embed_and_store → extract_entities → store_graph
```
- `PyMuPDF`로 PDF 텍스트 추출
- `RecursiveCharacterTextSplitter`로 청킹 (chunk_size=1000, overlap=200)
- `dragonkue/BGE-m3-ko` 임베딩 → ChromaDB 저장
- `konantech/Konan-LLM-OND`로 엔티티·관계 추출 → Neo4j 저장

### 질의응답 파이프라인 (실시간)
```
analyze_query → [vector_search ‖ graph_search] → generate_answer
```
- `analyze_query`: 쿼리에서 엔티티 추출 (LLM)
- `vector_search`: ChromaDB에서 상위 K개 청크 검색 (병렬)
- `graph_search`: 추출된 엔티티로 Neo4j 그래프 순회 (병렬)
- `generate_answer`: 벡터 컨텍스트 + 그래프 컨텍스트를 합쳐 LLM 최종 답변 생성

LLM/Embedding 모델은 싱글턴(`src/models.py`)으로 관리해 반복 로딩을 방지한다.

---

## 주요 기능

- **하이브리드 RAG**: 벡터 유사도 검색 + 엔티티 기반 그래프 검색 병렬 수행
- **한국어 특화 모델**: LLM `konantech/Konan-LLM-OND` (4B), 임베딩 `dragonkue/BGE-m3-ko` (0.6B)
- **Streamlit 챗봇 UI**: 대화형 질의응답 대시보드 (`app.py`)
- **CLI 인덱싱**: `ingest.py`로 폴더 또는 개별 PDF 일괄 처리
- **LangSmith 트레이싱**: 파이프라인 노드별 실행 추적
- **온디바이스 배포 전략**: PC에서 인덱싱·DB 생성 → 모바일(Galaxy S26 Ultra)에 DB + Q4 양자화 모델 이식, 모바일은 검색·추론만 수행

---

## 설치 & 사용법

```bash
# 1. 가상환경 설치
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

pip install -e .                   # 또는: poetry install
```

```bash
# 2. 환경변수 설정 (.env.example 복사 후 편집)
cp .env.example .env
```

```bash
# 3. Neo4j 실행 (Docker 필요)
docker compose up -d
```

```bash
# 4. PDF 인덱싱
python ingest.py data/             # data/ 폴더 내 모든 PDF
python ingest.py report.pdf        # 개별 파일
```

```bash
# 5. 챗봇 실행
streamlit run app.py
```

---

## 요구사항 / 의존성

| 항목 | 버전 / 비고 |
|------|-------------|
| Python | >=3.11, <3.13 |
| Docker | Neo4j 컨테이너 실행 필요 |
| HuggingFace 토큰 | `HUGGINGFACEHUB_API_TOKEN` (gated 모델 접근) |
| LangSmith API Key | 트레이싱 선택사항 |
| 주요 패키지 | `langgraph`, `langchain`, `langchain-huggingface`, `chromadb`, `neo4j`, `pymupdf`, `streamlit` |
| GPU | CUDA 권장 (CPU 추론 가능하나 느림) |

---

## 주요 변경 이력

| 날짜 | 내용 |
|------|------|
| 2026-03-06 | 초기 공개 — LangGraph GraphRAG 파이프라인 + Streamlit 챗봇 전체 구현 |
