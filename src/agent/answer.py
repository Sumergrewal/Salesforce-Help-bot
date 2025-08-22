from typing import List, Tuple
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
    """
    Build Source objects from the same chunks we provided to the LLM.
    This ensures convo memory can record used_doc_ids / used_chunk_ids reliably.
    """
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
                score=(c.hybrid_score or 0.0),
            )
        )
    return out

def answer_with_citations(query: str, chunks: List[Chunk], memory_summary: str) -> Tuple[str, List[Source]]:
    """
    Build messages via prompts.build_messages(...), call OpenAI, and return
    the model's text plus the Sources that mirror the context chunks.
    """
    client = _get_client()

    # Trim to a safe number of passages (caller usually does this already)
    k = getattr(settings, "TOPK_FINAL", 8)
    passages = chunks[:k]

    # messages = [ {"role":"system", SYSTEM_PROMPT}, {"role":"user", "...query + context..."} ]
    messages = build_messages(query, passages, memory_summary)

    resp = client.chat.completions.create(
        model=getattr(settings, "OPENAI_MODEL", "gpt-4o-mini"),
        messages=messages,
        temperature=0.2,
    )
    answer = resp.choices[0].message.content.strip()
    sources = _chunks_to_sources(passages)
    return answer, sources
