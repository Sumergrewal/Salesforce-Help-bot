from typing import Optional, List

from src.agent.config import settings
from src.agent.retrieval import hybrid_search
from src.agent.answer import answer_with_citations
from src.agent import memory as mem
from src.agent.guardrails import is_greeting, is_goodbye, is_low_info, make_welcome_msg, make_clarify_msg
from src.models.schemas import ChatAnswer

def _make_summary(last_user_msg: str, recent_topics: Optional[list[str]] = None) -> str:
    topics = f" Recent topics: {', '.join(recent_topics[:3])}." if recent_topics else ""
    return f"User is asking about: {last_user_msg[:180]}." + topics

def run_chat(session_id: str, user_text: str) -> ChatAnswer:
    mem.ensure_session(session_id)

    # --- Guardrails: greetings / low-info / goodbyes ---
    if is_greeting(user_text):
        answer = make_welcome_msg()
        mem.insert_turn(session_id, user_text, answer, [], [])
        mem.update_summary(session_id, _make_summary("greeting"))
        return ChatAnswer(session_id=session_id, message=user_text, answer=answer, sources=[])

    if is_goodbye(user_text):
        answer = "Goodbye! If you need anything else from Salesforce Help later, just ask."
        mem.insert_turn(session_id, user_text, answer, [], [])
        mem.update_summary(session_id, _make_summary("goodbye"))
        return ChatAnswer(session_id=session_id, message=user_text, answer=answer, sources=[])

    if is_low_info(user_text):
        answer = make_clarify_msg(user_text)
        mem.insert_turn(session_id, user_text, answer, [], [])
        mem.update_summary(session_id, _make_summary("clarification requested"))
        return ChatAnswer(session_id=session_id, message=user_text, answer=answer, sources=[])

    # --- Retrieval with light memory signals ---
    recent_docs = mem.get_recent_doc_ids(session_id)
    chunks = hybrid_search(
        query_text=user_text,
        recent_doc_ids=recent_docs,
        k_vec=settings.TOPK_VECTOR,
        k_fts=settings.TOPK_FTS,
        k_final=settings.TOPK_FINAL,
        alpha=settings.HYBRID_ALPHA,
    )

    if not chunks:
        # friendlier not-found with suggestions instead of a blunt "I don't know"
        answer = make_clarify_msg(user_text)
        mem.insert_turn(session_id, user_text, answer, [], [])
        mem.update_summary(session_id, _make_summary(user_text))
        return ChatAnswer(session_id=session_id, message=user_text, answer=answer, sources=[])

    # --- Answer + memory ---
    top_doc_titles = list({c.doc_title or c.doc_id for c in chunks})[:3]
    memory_summary = _make_summary(user_text, top_doc_titles)

    answer_text, sources = answer_with_citations(user_text, chunks, memory_summary)

    used_doc_ids = list({s.doc_id for s in sources if s.doc_id})
    used_chunk_ids = [s.chunk_id for s in sources if s.chunk_id]
    mem.insert_turn(session_id, user_text, answer_text, used_doc_ids, used_chunk_ids)
    mem.update_summary(session_id, memory_summary)

    return ChatAnswer(session_id=session_id, message=user_text, answer=answer_text, sources=sources)
