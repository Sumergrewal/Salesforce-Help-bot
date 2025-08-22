from typing import Optional, List
import re
import logging

from src.agent.config import settings
from src.agent.retrieval import hybrid_search
from src.agent.answer import answer_with_citations
from src.agent import memory as mem
from src.agent.guardrails import is_greeting, is_goodbye, is_low_info, make_welcome_msg, make_clarify_msg
from src.models.schemas import ChatAnswer, Chunk

log = logging.getLogger(__name__)
_SAME_PRODUCT_RE = re.compile(r"\b(same|this|that)\s+product\b", re.I)

def _make_summary(last_user_msg: str, recent_topics: Optional[list[str]] = None) -> str:
    topics = f" Recent topics: {', '.join(recent_topics[:3])}." if recent_topics else ""
    return f"User is asking about: {last_user_msg[:180]}." + topics

def _top_product(chunks: List[Chunk]) -> Optional[str]:
    counts = {}
    for c in chunks:
        p = getattr(c, "product", None)
        if p:
            counts[p] = counts.get(p, 0) + 1
    return max(counts, key=counts.get) if counts else None

def run_chat(session_id: str, user_text: str, product: Optional[str] = None) -> ChatAnswer:
    mem.ensure_session(session_id)

    # Greetings / goodbyes
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

    # Infer product EARLY and more aggressively:
    low = is_low_info(user_text)
    recent_docs = mem.get_recent_doc_ids(session_id)
    recent_product = mem.infer_recent_product(session_id)  # may be None
    chosen_product = product  # explicit from UI takes precedence

    # If no explicit product, use recent product when:
    # - user said "same/this/that product", OR
    # - the message is low-info (likely a follow-up), OR
    # - there is any recent grounded context at all.
    if chosen_product is None:
        if _SAME_PRODUCT_RE.search(user_text) or low or (recent_docs and recent_product):
            chosen_product = recent_product

    log.info(f"[chat] session={session_id} explicit_product={product!r} recent_product={recent_product!r} chosen_product={chosen_product!r} low={low} recent_docs_count={len(recent_docs)}")

    # LOW-INFO branch: try to help within chosen_product; if empty, fallback unfiltered
    if low:
        if chosen_product:
            fallback_query = mem.get_last_significant_user_query(session_id) or "overview"

            # 1) Try WITH product filter
            chunks = hybrid_search(
                query_text=fallback_query,
                recent_doc_ids=recent_docs,
                k_vec=settings.TOPK_VECTOR, k_fts=settings.TOPK_FTS,
                k_final=settings.TOPK_FINAL, alpha=settings.HYBRID_ALPHA,
                product=chosen_product,
            )
            # 2) If empty, try WITHOUT filter (safety net)
            if not chunks:
                chunks = hybrid_search(
                    query_text=fallback_query,
                    recent_doc_ids=recent_docs,
                    k_vec=settings.TOPK_VECTOR, k_fts=settings.TOPK_FTS,
                    k_final=settings.TOPK_FINAL, alpha=settings.HYBRID_ALPHA,
                    product=None,
                )

            if chunks:
                top_doc_titles = list({c.doc_title or c.doc_id for c in chunks})[:3]
                memory_summary = _make_summary(f"{user_text} (continued)", top_doc_titles)
                answer_text, sources = answer_with_citations(
                    "overview",  # query is ignored in overview mode
                    chunks,
                    memory_summary,
                    mode="overview",
                    product=chosen_product,
                )
                actually_filtered = chosen_product and any(getattr(c, "product", None) == chosen_product for c in chunks)
                prefix = f"Continuing with **{chosen_product}**:\n\n" if actually_filtered else \
                         f"Continuing (closest matches shown):\n\n"
                answer_text = prefix + answer_text

                used_doc_ids = list({s.doc_id for s in sources if s.doc_id})
                used_chunk_ids = [s.chunk_id for s in sources if s.chunk_id]
                mem.insert_turn(session_id, user_text, answer_text, used_doc_ids, used_chunk_ids)
                mem.update_summary(session_id, memory_summary)
                return ChatAnswer(session_id=session_id, message=user_text, answer=answer_text, sources=sources)

        # Still nothing (no product or no hits) -> clarify and nudge
        answer = make_clarify_msg(user_text)
        if recent_product:
            answer += f"\n\n(I can keep focusing on **{recent_product}**—try asking a specific task or feature.)"
        mem.insert_turn(session_id, user_text, answer, [], [])
        mem.update_summary(session_id, _make_summary("clarification requested"))
        return ChatAnswer(session_id=session_id, message=user_text, answer=answer, sources=[])

    # NORMAL retrieval (respect explicit or chosen product)
    chunks = hybrid_search(
        query_text=user_text,
        recent_doc_ids=recent_docs,
        k_vec=settings.TOPK_VECTOR, k_fts=settings.TOPK_FTS,
        k_final=settings.TOPK_FINAL, alpha=settings.HYBRID_ALPHA,
        product=chosen_product,
    )

    if not chunks:
        answer = make_clarify_msg(user_text)
        if chosen_product and product is None:
            answer += f"\n\n_Tip: Try selecting **{chosen_product}** in the product filter to narrow results._"
        mem.insert_turn(session_id, user_text, answer, [], [])
        mem.update_summary(session_id, _make_summary(user_text))
        return ChatAnswer(session_id=session_id, message=user_text, answer=answer, sources=[])

    top_doc_titles = list({c.doc_title or c.doc_id for c in chunks})[:3]
    memory_summary = _make_summary(user_text, top_doc_titles)
    answer_text, sources = answer_with_citations(user_text, chunks, memory_summary)

    if product is None:
        guessed = _top_product(chunks)
        if guessed:
            answer_text += f"\n\n_Tip: You can narrow future answers by selecting **{guessed}** in the sidebar._"

    used_doc_ids = list({s.doc_id for s in sources if s.doc_id})
    used_chunk_ids = [s.chunk_id for s in sources if s.chunk_id]
    mem.insert_turn(session_id, user_text, answer_text, used_doc_ids, used_chunk_ids)
    mem.update_summary(session_id, memory_summary)

    return ChatAnswer(session_id=session_id, message=user_text, answer=answer_text, sources=sources)
