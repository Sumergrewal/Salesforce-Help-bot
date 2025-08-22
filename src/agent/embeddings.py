from typing import List
from src.agent.config import settings


from openai import OpenAI

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.OPENAI_API_KEY or None)
    return _client


def embed_query(text: str) -> List[float]:
    """
    Return a 1536-d (default) embedding for the query using the same model
    you used during ingestion (text-embedding-3-small by default).
    """
    if not text:
        text = " "
    client = _get_client()
    resp = client.embeddings.create(model=settings.OPENAI_EMBED_MODEL, input=[text])
    return resp.data[0].embedding


def vec_literal(v: list[float]) -> str:
    """
    Convert a Python list of floats to a pgvector literal: "[0.1,0.2,...]".
    """
    return "[" + ",".join(f"{float(x):.6f}" for x in v) + "]"
