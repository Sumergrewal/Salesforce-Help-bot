# src/agent/answer.py
from typing import List, Tuple, Optional
from openai import OpenAI

from src.agent.config import settings
from src.agent.prompts import build_messages, build_product_overview_messages
from src.models.schemas import Chunk, Source

_client: Optional[OpenAI] = None
def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.OPENAI_API_KEY or None)
    return _client

def _chunks_to_sources(chunks: List[Chunk]) -> List[Source]:
    return [
        Source(
            chunk_id=c.id,
            doc_id=c.doc_id,
            doc_title=c.doc_title,
            section_title=c.section_title,
            page_start=c.page_start,
            page_end=c.page_end,
            score=(c.hybrid_score or 0.0),
        )
        for c in chunks
    ]

def answer_with_citations(
    query: str,
    chunks: List[Chunk],
    memory_summary: str,
    *,
    mode: str = "default",
    product: Optional[str] = None,
) -> Tuple[str, List[Source]]:
    client = _get_client()
    k = getattr(settings, "TOPK_FINAL", 8)
    passages = chunks[:k]

    if mode == "overview" and product:
        messages = build_product_overview_messages(product, passages, memory_summary)
        temperature = 0.2
    else:
        messages = build_messages(query, passages, memory_summary)
        temperature = 0.1

    resp = client.chat.completions.create(
        model=getattr(settings, "OPENAI_MODEL", "gpt-4o-mini"),
        messages=messages,
        temperature=temperature,
    )
    answer = resp.choices[0].message.content.strip()
    sources = _chunks_to_sources(passages)
    return answer, sources
