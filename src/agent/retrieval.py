# src/agent/retrieval.py
from typing import List, Dict, Optional
from src.agent.config import settings
from src.agent.db import fetchall
from src.agent.embeddings import embed_query, vec_literal
from src.models.schemas import Chunk

def _minmax(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return 0.0, 1.0
    return min(vals), max(vals)

def _norm(value: Optional[float], lo: float, hi: float, invert: bool = False) -> float:
    if value is None:
        return 0.0
    if hi == lo:
        return 1.0 if invert else 0.0
    x = (value - lo) / (hi - lo)
    return (1.0 - x) if invert else x

def vector_search(qvec_literal: str, k: int, product: Optional[str] = None) -> List[Chunk]:
    """
    Returns top-k by embedding distance.
    IMPORTANT: we SELECT product via COALESCE(c.product, d.product) and join fsc_docs
               so it still works even if c.product wasn't backfilled yet.
    """
    if product:
        sql = """
          SELECT c.id, c.doc_id, c.doc_title, c.section_title, c.page_start, c.page_end,
                 c.content, (c.embedding <=> %s::vector) AS vec_dist,
                 COALESCE(c.product, d.product) AS product
          FROM fsc_chunks c
          LEFT JOIN fsc_docs d ON d.doc_id = c.doc_id
          WHERE COALESCE(c.product, d.product) = %s
          ORDER BY c.embedding <=> %s::vector
          LIMIT %s;
        """
        rows = fetchall(sql, (qvec_literal, product, qvec_literal, k))
    else:
        sql = """
          SELECT c.id, c.doc_id, c.doc_title, c.section_title, c.page_start, c.page_end,
                 c.content, (c.embedding <=> %s::vector) AS vec_dist,
                 COALESCE(c.product, d.product) AS product
          FROM fsc_chunks c
          LEFT JOIN fsc_docs d ON d.doc_id = c.doc_id
          ORDER BY c.embedding <=> %s::vector
          LIMIT %s;
        """
        rows = fetchall(sql, (qvec_literal, qvec_literal, k))
    return [Chunk(**row) for row in rows]

def fts_search(query_text: str, k: int, product: Optional[str] = None) -> List[Chunk]:
    """
    Keyword search over generated tsvector (section_title + content).
    Uses websearch_to_tsquery for natural language queries.
    """
    if product:
        sql = """
          SELECT c.id, c.doc_id, c.doc_title, c.section_title, c.page_start, c.page_end,
                 c.content,
                 ts_rank_cd(c.combined_tsv, websearch_to_tsquery('english', %s)) AS fts_rank,
                 COALESCE(c.product, d.product) AS product
          FROM fsc_chunks c
          LEFT JOIN fsc_docs d ON d.doc_id = c.doc_id
          WHERE COALESCE(c.product, d.product) = %s
            AND c.combined_tsv @@ websearch_to_tsquery('english', %s)
          ORDER BY fts_rank DESC
          LIMIT %s;
        """
        rows = fetchall(sql, (query_text, product, query_text, k))
    else:
        sql = """
          SELECT c.id, c.doc_id, c.doc_title, c.section_title, c.page_start, c.page_end,
                 c.content,
                 ts_rank_cd(c.combined_tsv, websearch_to_tsquery('english', %s)) AS fts_rank,
                 COALESCE(c.product, d.product) AS product
          FROM fsc_chunks c
          LEFT JOIN fsc_docs d ON d.doc_id = c.doc_id
          WHERE c.combined_tsv @@ websearch_to_tsquery('english', %s)
          ORDER BY fts_rank DESC
          LIMIT %s;
        """
        rows = fetchall(sql, (query_text, query_text, k))
    return [Chunk(**row) for row in rows]

def hybrid_search(
    query_text: str,
    recent_doc_ids: Optional[set[str]] = None,
    k_vec: Optional[int] = None,
    k_fts: Optional[int] = None,
    k_final: Optional[int] = None,
    alpha: Optional[float] = None,
    product: Optional[str] = None,
) -> List[Chunk]:
    """
    Merge vector + FTS; optional product filter applied to both.
    Returns top-k_final with hybrid_score.
    """
    k_vec = k_vec or settings.TOPK_VECTOR
    k_fts = k_fts or settings.TOPK_FTS
    k_final = k_final or settings.TOPK_FINAL
    alpha = settings.HYBRID_ALPHA if alpha is None else alpha

    # Embed query
    qvec = embed_query(query_text)
    qvec_lit = vec_literal(qvec)

    # Candidate pools
    vec_rows = vector_search(qvec_lit, k_vec, product=product)
    fts_rows = fts_search(query_text, k_fts, product=product)

    # Merge by id
    by_id: Dict[int, Chunk] = {}
    for r in vec_rows:
        by_id[r.id] = r
    for r in fts_rows:
        if r.id in by_id:
            base = by_id[r.id]
            base.fts_rank = r.fts_rank
        else:
            by_id[r.id] = r
    if not by_id:
        return []

    # Normalize scores and compute hybrid
    vec_vals = [c.vec_dist for c in by_id.values() if c.vec_dist is not None]
    fts_vals = [c.fts_rank for c in by_id.values() if c.fts_rank is not None]
    vec_lo, vec_hi = _minmax(vec_vals)
    fts_lo, fts_hi = _minmax(fts_vals)

    out: List[Chunk] = []
    for c in by_id.values():
        vec_component = _norm(c.vec_dist, vec_lo, vec_hi, invert=True)   # 1 = best
        fts_component = _norm(c.fts_rank, fts_lo, fts_hi, invert=False)  # 1 = best
        score = alpha * fts_component + (1.0 - alpha) * vec_component
        if recent_doc_ids and c.doc_id in recent_doc_ids:
            score += settings.MEMORY_DOC_BOOST
        c.hybrid_score = float(score)
        out.append(c)

    # Soft guardrail: keep if good vec OR any FTS
    kept = [c for c in out if (c.vec_dist is None or c.vec_dist <= settings.MIN_RELEVANCE) or (c.fts_rank and c.fts_rank > 0)]
    if len(kept) < k_final:
        kept = out

    kept.sort(key=lambda x: (x.hybrid_score or 0.0), reverse=True)
    return kept[:k_final]
