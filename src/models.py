from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

from src.config import LLM_MODEL_NAME, EMBEDDING_MODEL_NAME, HF_TOKEN

_llm = None
_embeddings = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_llm() -> HuggingFacePipeline:
    global _llm
    if _llm is None:
        tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL_NAME, token=HF_TOKEN, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            token=HF_TOKEN,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            return_full_text=False,
        )
        _llm = HuggingFacePipeline(pipeline=pipe)
    return _llm
