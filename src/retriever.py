from langchain_chroma import Chroma
from neo4j import GraphDatabase

from src.config import CHROMA_DIR, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from src.models import get_embeddings


def vector_search(query: str, top_k: int = 5) -> list[dict]:
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name="pdf_chunks",
    )
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    return [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", ""),
            "chunk_index": doc.metadata.get("chunk_index", 0),
            "score": float(score),
        }
        for doc, score in results
    ]


def graph_search(entities: list[str], max_depth: int = 2) -> str:
    if not entities:
        return ""

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    context_parts = []

    with driver.session() as session:
        for entity in entities:
            result = session.run(
                """
                MATCH (e:Entity {name: $name})-[r:RELATED*1..2]-(connected)
                RETURN e.name AS source,
                       [rel IN r | rel.type] AS relations,
                       connected.name AS target
                LIMIT 20
                """,
                name=entity,
            )
            for record in result:
                relations = " -> ".join(record["relations"])
                context_parts.append(
                    f"{record['source']} -[{relations}]-> {record['target']}"
                )

    driver.close()
    return "\n".join(context_parts) if context_parts else ""


def hybrid_search(query: str, entities: list[str], top_k: int = 5) -> dict:
    vector_results = vector_search(query, top_k)
    graph_context = graph_search(entities)

    vector_context = "\n\n".join(r["content"] for r in vector_results)

    return {
        "vector_results": vector_results,
        "vector_context": vector_context,
        "graph_context": graph_context,
    }
