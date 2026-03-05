# GraphRAG PDF 챗봇

LangGraph 기반 GraphRAG 파이프라인 + Streamlit 챗봇 대시보드.
PDF 문서를 인덱싱하고 벡터 검색 + 지식 그래프를 결합한 하이브리드 RAG로 질의응답하는 시스템.

## 기술 스택

| 구분 | 기술 |
|------|------|
| LLM | konantech/Konan-LLM-OND (4B, 한국어 특화) |
| Embedding | dragonkue/BGE-m3-ko (0.6B, 한국어) |
| 파이프라인 | LangGraph |
| 벡터 DB | ChromaDB |
| 그래프 DB | Neo4j (Docker) |
| UI | Streamlit |
| 추적 | LangSmith |
| Python | 3.11.9 |

## 프로젝트 구조

```
├── src/
│   ├── config.py          # 환경변수, 설정 상수
│   ├── models.py          # LLM/Embedding 초기화 (싱글턴)
│   ├── ingestion.py       # PDF → 청킹 → 벡터/그래프 저장
│   ├── retriever.py       # ChromaDB 벡터검색 + Neo4j 그래프검색
│   └── graph.py           # LangGraph 워크플로 (인제스천 + 질의응답)
├── app.py                 # Streamlit 대시보드 (채팅 전용)
├── ingest.py              # CLI 인덱싱 스크립트
├── data/                  # PDF 저장
├── docker-compose.yml     # Neo4j 컨테이너
└── pyproject.toml         # 의존성 정의
```

## LangGraph 파이프라인

### 인제스천 (1회, CLI)
```
load_pdf → split_chunks → embed_and_store → extract_entities → store_graph
```

### 질의응답 (실시간)
```
analyze_query → [vector_search, graph_search] → generate_answer
```

## 설치 & 실행

### 1. 환경 설정
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt  # 또는 pyproject.toml 기반 설치
```

### 2. 환경변수 (.env)
```
HUGGINGFACEHUB_API_TOKEN=your_token
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=RAG-proj
```

### 3. Neo4j 실행
```bash
docker compose up -d
```

### 4. PDF 인덱싱
```bash
python ingest.py data/          # data/ 폴더 내 모든 PDF
python ingest.py file1.pdf      # 개별 파일
```

### 5. 챗봇 실행
```bash
.venv\Scripts\streamlit run app.py
```

## 성능 벤치마크 (RTX 4080 SUPER)

### 인덱싱 (28페이지 PDF 기준)
| 단계 | 시간 |
|------|------|
| PDF 로딩 + 청킹 | < 0.1초 |
| 임베딩 + ChromaDB 저장 | 0.9초 |
| 엔티티 추출 (20청크) | ~4분 |

### 질의응답
| 단계 | 시간 |
|------|------|
| analyze_query | 0.59초 |
| vector_search + graph_search | ~0.1초 |
| generate_answer | ~44초 |

> generate_answer 병목 → llama.cpp 추론 엔진 교체로 최적화 예정

## 개발 타임라인

| 날짜 | 작업 |
|------|------|
| 2026-03-05 | 프로젝트 초기 설정 (pyproject.toml, .env, docker-compose.yml) |
| 2026-03-05 | HuggingFace 모델 선정 (Konan-LLM-OND, BGE-m3-ko) |
| 2026-03-05 | Neo4j Docker 실행, .env 설정 |
| 2026-03-05 | 백엔드 구현 (config, models, ingestion, retriever, graph) |
| 2026-03-05 | Streamlit 대시보드 구현 (Gemini 모델로 UI 디자인) |
| 2026-03-05 | PyTorch CPU → CUDA 전환 (RTX 4080 SUPER) |
| 2026-03-05 | 인제스천 파이프라인 버그 수정 (IngestionState, store_graph) |
| 2026-03-05 | CLI 인덱싱 스크립트 분리 (ingest.py) |
| 2026-03-05 | Streamlit → 채팅 전용 대시보드로 리팩토링 |
| 2026-03-05 | return_full_text=False 수정 → 답변 품질 정상화 |
| 2026-03-05 | CLI 테스트 5개 시나리오 통과 (할루시네이션 없음 확인) |
| 2026-03-05 | 100페이지 PDF 인덱싱 벤치마크 측정 |
| 2026-03-06 | LangSmith 트레이싱 연동 |
| 2026-03-06 | 성능 병목 분석 (generate_answer 44초) |

## 향후 계획

- [ ] 추론 엔진 교체 (HuggingFace Pipeline → llama.cpp GGUF)
- [ ] 모델 양자화 (Q4_K_M) → 추론 속도 3~5배 개선
- [ ] Galaxy S26 Ultra 온디바이스 배포 검증
- [ ] 멀티 PDF 인덱싱 지원 강화
- [ ] 대화 히스토리 기반 후속 질문 처리

## 온디바이스 배포 전략

```
[PC - RTX 4080]                    [Galaxy S26 Ultra]
PDF → 인덱싱 → DB 생성 → 전송 →   DB + Q4 모델 → 질문 → 답변
     (임베딩, 엔티티추출)              (검색 + 추론만)
```

- Konan-LLM-OND Q4: ~2.5GB → S26 VRAM 여유
- BGE-m3-ko Q4: ~0.4GB
- PC에서 인덱싱, 모바일은 추론만 → 배터리/발열 최적

## 라이선스

Private repository
