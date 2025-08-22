from typing import List, Tuple, Optional
from openai import OpenAI

from src.agent.config import settings
from src.agent.prompts import build_messages
from src.models.schemas import Chunk, Source

_client: OpenAI | None = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.OPENAI_API_KEY or None)
    return _client

def _chunks_to_sources(chunks: List[Chunk]) -> List[Source]:
    out: List[Source] = []
    for c in chunks:
        out.append(
            Source(
                chunk_id=c.id,
                doc_id=c.doc_id,
                doc_title=c.doc_title,
                section_title=c.section_title,
                page_start=c.page_start,
                page_end=c.page_end,
                score=c.hybrid_score or c.vec_dist or 0.0,
            )
        )
    return out

def answer_with_citations(
    query: str,
    chunks: List[Chunk],
    memory_summary: Optional[str] = None,
) -> Tuple[str, List[Source]]:
    """
    Call the chat model with grounded context. Returns (answer_text, sources).
    """
    client = _get_client()
    # Trim to a safe number of passages (the caller should already have done this)
    passages = chunks[:8]
    messages = build_messages(query, passages, memory_summary)

    resp = client.chat.completions.create(
        model=getattr(settings, "OPENAI_MODEL", "gpt-4o-mini"),
        messages=messages,
        temperature=0.2,
    )
    answer = resp.choices[0].message.content.strip()
    sources = _chunks_to_sources(passages)
    return answer, sources
