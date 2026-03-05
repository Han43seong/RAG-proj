import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# HuggingFace
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
LLM_MODEL_NAME = "konantech/Konan-LLM-OND"
EMBEDDING_MODEL_NAME = "dragonkue/BGE-m3-ko"

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval
DEFAULT_TOP_K = 5
