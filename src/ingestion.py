import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from neo4j import GraphDatabase

from src.config import (
    CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_DIR,
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD,
)
from src.models import get_embeddings, get_llm


def load_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def split_chunks(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def store_vectors(chunks: list[str], source: str) -> Chroma:
    embeddings = get_embeddings()
    metadatas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=str(CHROMA_DIR),
        collection_name="pdf_chunks",
    )
    return vectorstore


ENTITY_EXTRACTION_PROMPT = PromptTemplate.from_template(
    """다음 텍스트에서 주요 엔티티(인물, 조직, 장소, 개념, 기술 등)와 엔티티 간의 관계를 추출해주세요.

텍스트:
{text}

반드시 아래 JSON 형식으로만 응답하세요:
{{"entities": ["엔티티1", "엔티티2"], "relations": [{{"source": "엔티티1", "relation": "관계", "target": "엔티티2"}}]}}"""
)


def extract_entities(chunks: list[str]) -> list[dict]:
    llm = get_llm()
    parser = JsonOutputParser()
    chain = ENTITY_EXTRACTION_PROMPT | llm | parser

    all_results = []
    for chunk in chunks:
        try:
            result = chain.invoke({"text": chunk})
            all_results.append(result)
        except Exception:
            all_results.append({"entities": [], "relations": []})
    return all_results


def store_graph(extraction_results: list[dict]):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    with driver.session() as session:
        for result in extraction_results:
            for entity in result.get("entities", []):
                session.run(
                    "MERGE (e:Entity {name: $name})",
                    name=entity,
                )
            for rel in result.get("relations", []):
                if not isinstance(rel, dict):
                    continue
                src = rel.get("source", rel.get("src", ""))
                tgt = rel.get("target", rel.get("tgt", ""))
                rtype = rel.get("relation", rel.get("type", rel.get("rel", "")))
                if not src or not tgt or not rtype:
                    continue
                session.run(
                    """
                    MERGE (s:Entity {name: $source})
                    MERGE (t:Entity {name: $target})
                    MERGE (s)-[r:RELATED {type: $relation}]->(t)
                    """,
                    source=src,
                    relation=rtype,
                    target=tgt,
                )
    driver.close()
