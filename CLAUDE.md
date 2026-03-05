# RAG-proj - LangGraph GraphRAG Chatbot

## 프로젝트 개요
LangGraph 기반 GraphRAG 파이프라인 + Streamlit 챗봇 대시보드

## 기술 스택 (확정)
- **LLM**: `konantech/Konan-LLM-OND` (4B, Qwen3-4B 기반 한국어 특화)
- **Embedding**: `dragonkue/BGE-m3-ko` (0.6B, 한국어 추가학습, 8192 토큰)
- **Graph DB**: Neo4j (Docker)
- **Vector Store**: ChromaDB
- **문서 소스**: PDF
- **파이프라인**: LangGraph
- **UI**: Streamlit
- **Python**: 3.11.9 (pyenv)


## 프로젝트 구조
```
D:\RAG-proj\
├── src/
│   ├── __init__.py
│   ├── config.py          # 환경변수, 설정 상수
│   ├── models.py          # LLM/Embedding 초기화 (싱글턴)
│   ├── ingestion.py       # PDF→청킹→벡터/그래프 저장
│   ├── retriever.py       # ChromaDB 벡터검색 + Neo4j 그래프검색
│   └── graph.py           # LangGraph 워크플로 (인제스천 + 질의응답)
├── app.py                 # Streamlit 대시보드 (진입점)
├── data/                  # 업로드 PDF 저장
├── chroma_db/             # ChromaDB 영속 저장소
├── .env                   # 환경변수 (HF 토큰, Neo4j)
├── docker-compose.yml     # Neo4j 컨테이너
└── pyproject.toml
```

## LangGraph 파이프라인
- **인제스천**: `load_pdf → split_chunks → embed_and_store → extract_entities → store_graph`
- **질의응답**: `analyze_query → [vector_search, graph_search] → generate_answer`

## 실행 방법
```bash
docker compose up -d          # Neo4j 실행
streamlit run app.py          # 대시보드 실행
```

## 설치된 주요 패키지
| 패키지 | 버전 |
|--------|------|
| langchain | 1.2.10 |
| langchain-core | 1.2.17 |
| langchain-community | 0.4.1 |
| langchain-huggingface | 1.2.1 |
| langchain-chroma | 1.1.0 |
| langgraph | 1.0.10 |
| chromadb | 1.5.2 |
| neo4j | 6.1.0 |
| pymupdf | 1.27.1 |
| streamlit | 1.55.0 |
| transformers | 5.3.0 |
| accelerate | 1.13.0 |

## 참고
- 이 프로젝트는 `C:\Users\hskim\Documents\langchain-kr` (TeddyNote LangChain 튜토리얼)에서 분리된 독립 프로젝트
- 가상환경: `D:\RAG-proj\.venv`
- Python 경로: `C:\Users\hskim\.pyenv\pyenv-win\versions\3.11.9`
