import time
import streamlit as st
import chromadb
from neo4j import GraphDatabase

from src.graph import run_query_pipeline
from src.config import CHROMA_DIR, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

# --- 페이지 설정 ---
st.set_page_config(layout="wide", page_title="GraphRAG PDF 챗봇", page_icon="🤖")


# --- DB 상태 조회 ---
@st.cache_resource
def get_chroma_count():
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_collection(name="pdf_chunks")
        return collection.count()
    except Exception:
        return 0


@st.cache_resource
def get_neo4j_count():
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        with driver.session() as session:
            result = session.run("MATCH (e:Entity) RETURN count(e) AS cnt")
            return result.single()["cnt"]
    except Exception:
        return 0


def stream_generator(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)


# --- 세션 상태 초기화 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 사이드바 ---
with st.sidebar:
    st.title("⚙️ 설정 및 상태")

    st.subheader("📊 데이터베이스 상태")
    chroma_count = get_chroma_count()
    neo4j_count = get_neo4j_count()

    col1, col2 = st.columns(2)
    col1.metric("벡터 문서 수", f"{chroma_count}개")
    col2.metric("그래프 엔티티", f"{neo4j_count}개")

    if chroma_count == 0 and neo4j_count == 0:
        st.warning("⚠️ 데이터베이스가 비어있습니다.\n`python ingest.py data/` 로 문서를 인덱싱해주세요.")

    st.divider()

    top_k = st.slider(
        "🔍 검색할 문서 수 (Top-K)",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
    )

    if st.button("🗑️ 대화 기록 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown(
        """
    **💡 사용 안내**
    1. 하단 입력창에 질문을 입력하세요.
    2. 벡터 DB와 지식 그래프를 동시에 검색하여 답변을 생성합니다.
    3. 답변 하단의 '참고 문서'를 클릭하면 출처를 확인할 수 있습니다.
    """
    )

# --- 메인 영역 ---
st.title("🤖 GraphRAG PDF 챗봇")
st.markdown("벡터 검색과 지식 그래프를 결합한 하이브리드 RAG 챗봇입니다. 질문을 입력하여 문서에 대해 알아보세요.")

# 기존 대화 이력 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 참고 문서"):
                for idx, source in enumerate(msg["sources"]):
                    content = source.get("page_content", str(source))
                    st.markdown(f"**[{idx + 1}]** {content}")
                    st.divider()

# 사용자 입력 처리
if prompt := st.chat_input("질문을 입력하세요..."):
    if chroma_count == 0 and neo4j_count == 0:
        st.error("데이터베이스가 구축되지 않았습니다. 인덱싱을 먼저 진행해주세요.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("답변을 생성 중입니다..."):
                try:
                    result = run_query_pipeline(query=prompt, top_k=top_k)
                    answer = result.get("answer", "답변을 생성하지 못했습니다.")
                    sources = result.get("sources", [])

                    st.write_stream(stream_generator(answer))

                    if sources:
                        with st.expander("📚 참고 문서"):
                            for idx, source in enumerate(sources):
                                content = source.get("page_content", str(source))
                                st.markdown(f"**[{idx + 1}]** {content}")
                                st.divider()

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })
                except Exception as e:
                    st.error(f"오류가 발생했습니다: {e}")
