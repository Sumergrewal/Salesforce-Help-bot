from typing import List, Dict, Optional
from math import inf

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
        # missing -> worst
        return 0.0 if not invert else 0.0
    if hi == lo:
        return 1.0 if invert else 0.0
    x = (value - lo) / (hi - lo)
    return (1.0 - x) if invert else x


def vector_search(qvec_literal: str, k: int) -> List[Chunk]:
    sql = """
      SELECT id, doc_id, doc_title, section_title, page_start, page_end, content,
             (embedding <=> %s::vector) AS vec_dist
      FROM fsc_chunks
      ORDER BY embedding <=> %s::vector
      LIMIT %s;
    """
    rows = fetchall(sql, (qvec_literal, qvec_literal, k))
    return [Chunk(**row) for row in rows]


def fts_search(query_text: str, k: int) -> List[Chunk]:
    """
    Keyword search over combined_tsv (section_title + content).
    Use websearch_to_tsquery for natural language queries.
    """
    sql = """
      SELECT id, doc_id, doc_title, section_title, page_start, page_end, content,
             ts_rank_cd(combined_tsv, websearch_to_tsquery('english', %s)) AS fts_rank
      FROM fsc_chunks
      WHERE combined_tsv @@ websearch_to_tsquery('english', %s)
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
) -> List[Chunk]:
    """
    Hybrid retrieval:
      - embed query and run vector search (top K_vec)
      - FTS search over combined_tsv (top K_fts)
      - merge, normalize, score = alpha*fts + (1-alpha)*(1-vec)
      - small bonus for chunks from recently-cited doc_ids (memory)
      - return top K_final chunks
    """
    k_vec = k_vec or settings.TOPK_VECTOR
    k_fts = k_fts or settings.TOPK_FTS
    k_final = k_final or settings.TOPK_FINAL
    alpha = alpha if alpha is not None else settings.HYBRID_ALPHA

    # Embed query and get vector literal
    qvec = embed_query(query_text)
    qvec_lit = vec_literal(qvec)

    vec_rows = vector_search(qvec_lit, k_vec)
    fts_rows = fts_search(query_text, k_fts)

    # Merge on id
    by_id: Dict[int, Chunk] = {}
    for r in vec_rows:
        by_id[r.id] = r
    for r in fts_rows:
        if r.id in by_id:
            # merge fields
            base = by_id[r.id]
            base.fts_rank = r.fts_rank
        else:
            by_id[r.id] = r

    if not by_id:
        return []

    # Collect for normalization
    vec_vals = [c.vec_dist for c in by_id.values() if c.vec_dist is not None]
    fts_vals = [c.fts_rank for c in by_id.values() if c.fts_rank is not None]
    vec_lo, vec_hi = _minmax(vec_vals)  # lower is better
    fts_lo, fts_hi = _minmax(fts_vals)  # higher is better after normalization

    # Score
    out: List[Chunk] = []
    for c in by_id.values():
        vec_component = _norm(c.vec_dist, vec_lo, vec_hi, invert=True)  # 1 is best
        fts_component = _norm(c.fts_rank, fts_lo, fts_hi, invert=False)  # 1 is best
        score = alpha * fts_component + (1.0 - alpha) * vec_component

        # Optional memory bias
        if recent_doc_ids and c.doc_id in recent_doc_ids:
            score += settings.MEMORY_DOC_BOOST

        c.hybrid_score = float(score)
        out.append(c)

    # Thresholding by vector distance (guardrail)
    kept = [c for c in out if (c.vec_dist is None or c.vec_dist <= settings.MIN_RELEVANCE) or (c.fts_rank and c.fts_rank > 0)]

    # If too few after thresholding, fall back to best-scored overall (no threshold)
    if len(kept) < k_final:
        kept = out

    kept.sort(key=lambda x: (x.hybrid_score or 0.0), reverse=True)
    return kept[:k_final]
