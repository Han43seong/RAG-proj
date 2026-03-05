from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import PromptTemplate

from src.ingestion import load_pdf, split_chunks, store_vectors, extract_entities, store_graph
from src.retriever import vector_search, graph_search
from src.models import get_llm
from src.config import DATA_DIR


# --- State 정의 ---

class IngestionState(TypedDict):
    pdf_paths: list[str]
    pdf_text: str
    all_chunks: list[str]
    extraction_results: list[dict]
    num_chunks: int
    num_entities: int
    status: str


class QueryState(TypedDict):
    query: str
    top_k: int
    entities: list[str]
    vector_results: list[dict]
    vector_context: str
    graph_context: str
    answer: str
    sources: list[dict]


# --- 인제스천 노드 ---

def node_load_pdf(state: IngestionState) -> dict:
    all_text = ""
    for path in state["pdf_paths"]:
        all_text += load_pdf(path) + "\n\n"
    return {"all_chunks": [], "status": "loaded", "pdf_text": all_text}


def node_split_chunks(state: IngestionState) -> dict:
    pdf_text = state.get("pdf_text", "")
    chunks = split_chunks(pdf_text)
    return {"all_chunks": chunks, "num_chunks": len(chunks)}


def node_embed_and_store(state: IngestionState) -> dict:
    chunks = state["all_chunks"]
    source = ", ".join(state["pdf_paths"])
    store_vectors(chunks, source)
    return {}


def node_extract_entities(state: IngestionState) -> dict:
    chunks = state["all_chunks"]
    # 엔티티 추출은 청크 수가 많으면 시간이 오래 걸리므로 일부만 처리
    sample_chunks = chunks[:20] if len(chunks) > 20 else chunks
    results = extract_entities(sample_chunks)
    total_entities = sum(len(r.get("entities", [])) for r in results)
    return {"extraction_results": results, "num_entities": total_entities}


def node_store_graph(state: IngestionState) -> dict:
    results = state.get("extraction_results", [])
    if results:
        store_graph(results)
    return {"status": "success"}


# --- 질의응답 노드 ---

QUERY_ENTITY_PROMPT = PromptTemplate.from_template(
    """다음 질문에서 핵심 엔티티(인물, 조직, 장소, 개념 등)를 추출하세요.
쉼표로 구분된 엔티티 목록만 응답하세요. 없으면 '없음'이라고 응답하세요.

질문: {query}

엔티티:"""
)

QA_PROMPT = PromptTemplate.from_template(
    """다음 컨텍스트를 참고하여 질문에 답변해주세요.
컨텍스트에 없는 내용은 추측하지 말고, 정보가 부족하면 솔직히 말해주세요.

[문서 컨텍스트]
{vector_context}

[지식 그래프 컨텍스트]
{graph_context}

질문: {query}

답변:"""
)


def node_analyze_query(state: QueryState) -> dict:
    llm = get_llm()
    chain = QUERY_ENTITY_PROMPT | llm
    result = chain.invoke({"query": state["query"]})
    if isinstance(result, str):
        text = result
    else:
        text = result.content if hasattr(result, "content") else str(result)

    if "없음" in text:
        entities = []
    else:
        entities = [e.strip() for e in text.split(",") if e.strip()]
    return {"entities": entities}


def node_vector_search(state: QueryState) -> dict:
    results = vector_search(state["query"], state.get("top_k", 5))
    context = "\n\n".join(r["content"] for r in results)
    return {"vector_results": results, "vector_context": context}


def node_graph_search(state: QueryState) -> dict:
    entities = state.get("entities", [])
    context = graph_search(entities)
    return {"graph_context": context}


def node_generate_answer(state: QueryState) -> dict:
    llm = get_llm()
    chain = QA_PROMPT | llm
    result = chain.invoke({
        "vector_context": state.get("vector_context", ""),
        "graph_context": state.get("graph_context", "정보 없음"),
        "query": state["query"],
    })
    if isinstance(result, str):
        answer = result
    else:
        answer = result.content if hasattr(result, "content") else str(result)

    sources = [
        {"page_content": r["content"], "metadata": {"source": r["source"], "chunk_index": r["chunk_index"]}}
        for r in state.get("vector_results", [])
    ]
    return {"answer": answer, "sources": sources}


# --- 그래프 빌드 ---

def build_ingestion_graph():
    graph = StateGraph(IngestionState)
    graph.add_node("load_pdf", node_load_pdf)
    graph.add_node("split_chunks", node_split_chunks)
    graph.add_node("embed_and_store", node_embed_and_store)
    graph.add_node("extract_entities", node_extract_entities)
    graph.add_node("store_graph", node_store_graph)

    graph.add_edge(START, "load_pdf")
    graph.add_edge("load_pdf", "split_chunks")
    graph.add_edge("split_chunks", "embed_and_store")
    graph.add_edge("embed_and_store", "extract_entities")
    graph.add_edge("extract_entities", "store_graph")
    graph.add_edge("store_graph", END)

    return graph.compile()


def build_query_graph():
    graph = StateGraph(QueryState)
    graph.add_node("analyze_query", node_analyze_query)
    graph.add_node("vector_search", node_vector_search)
    graph.add_node("graph_search", node_graph_search)
    graph.add_node("generate_answer", node_generate_answer)

    graph.add_edge(START, "analyze_query")
    graph.add_edge("analyze_query", "vector_search")
    graph.add_edge("analyze_query", "graph_search")
    graph.add_edge("vector_search", "generate_answer")
    graph.add_edge("graph_search", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()


# --- 외부 인터페이스 ---

_ingestion_graph = None
_query_graph = None


def get_ingestion_graph():
    global _ingestion_graph
    if _ingestion_graph is None:
        _ingestion_graph = build_ingestion_graph()
    return _ingestion_graph


def get_query_graph():
    global _query_graph
    if _query_graph is None:
        _query_graph = build_query_graph()
    return _query_graph


def run_ingestion_pipeline(pdf_paths: list[str]) -> dict:
    graph = get_ingestion_graph()
    result = graph.invoke({"pdf_paths": pdf_paths, "pdf_text": "", "all_chunks": [], "extraction_results": [], "num_chunks": 0, "num_entities": 0, "status": ""})
    return {
        "status": result.get("status", "unknown"),
        "num_chunks": result.get("num_chunks", 0),
        "num_entities": result.get("num_entities", 0),
    }


def run_query_pipeline(query: str, top_k: int = 5) -> dict:
    graph = get_query_graph()
    result = graph.invoke({"query": query, "top_k": top_k, "entities": [], "vector_results": [], "vector_context": "", "graph_context": "", "answer": "", "sources": []})
    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "graph_context": result.get("graph_context", ""),
    }
