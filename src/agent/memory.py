from typing import Optional, Set, List
from src.agent.db import fetchall, fetchone, execute

def ensure_session(session_id: str) -> None:
    execute(
        "INSERT INTO convo_session (session_id) VALUES (%s) ON CONFLICT (session_id) DO NOTHING;",
        (session_id,)
    )

def insert_turn(session_id: str, user_text: str, answer_text: str,
                used_doc_ids: List[str], used_chunk_ids: List[int]) -> None:
    execute(
        """
        INSERT INTO convo_turn (session_id, user_text, answer_text, used_doc_ids, used_chunk_ids)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (session_id, user_text, answer_text, used_doc_ids or None, used_chunk_ids or None)
    )

def update_summary(session_id: str, summary: str) -> None:
    execute(
        "UPDATE convo_session SET summary=%s WHERE session_id=%s",
        (summary, session_id)
    )

def get_recent_doc_ids(session_id: str, limit_turns: int = 20) -> Set[str]:
    rows = fetchall(
        """
        SELECT used_doc_ids
        FROM convo_turn
        WHERE session_id = %s AND used_doc_ids IS NOT NULL
        ORDER BY tstamp DESC
        LIMIT %s
        """,
        (session_id, limit_turns),
    )
    doc_ids: Set[str] = set()
    for r in rows:
        for d in (r["used_doc_ids"] or []):
            if d:
                doc_ids.add(d)
    return doc_ids

def infer_recent_product(session_id: str) -> Optional[str]:
    # (fixed CTE version you already applied)
    sql1 = """
    WITH recent AS (
      SELECT used_doc_ids
      FROM convo_turn
      WHERE session_id = %s
        AND used_doc_ids IS NOT NULL
      ORDER BY tstamp DESC
      LIMIT 50
    ),
    docs AS (
      SELECT DISTINCT unnest(used_doc_ids) AS doc_id
      FROM recent
    )
    SELECT d.product, COUNT(*) AS n
    FROM docs x
    LEFT JOIN fsc_docs d ON d.doc_id = x.doc_id
    WHERE d.product IS NOT NULL AND d.product <> ''
    GROUP BY d.product
    ORDER BY n DESC
    LIMIT 1;
    """
    row = fetchone(sql1, (session_id,))
    if row and row.get("product"):
        return row["product"]

    sql2 = """
    WITH recent AS (
      SELECT used_chunk_ids
      FROM convo_turn
      WHERE session_id = %s
        AND used_chunk_ids IS NOT NULL
      ORDER BY tstamp DESC
      LIMIT 200
    ),
    chunks AS (
      SELECT DISTINCT unnest(used_chunk_ids) AS cid
      FROM recent
    )
    SELECT COALESCE(c.product, d.product) AS product, COUNT(*) AS n
    FROM chunks ch
    LEFT JOIN fsc_chunks c ON c.id = ch.cid
    LEFT JOIN fsc_docs d ON d.doc_id = c.doc_id
    WHERE COALESCE(c.product, d.product) IS NOT NULL AND COALESCE(c.product, d.product) <> ''
    GROUP BY COALESCE(c.product, d.product)
    ORDER BY n DESC
    LIMIT 1;
    """
    row2 = fetchone(sql2, (session_id,))
    return row2["product"] if row2 and row2.get("product") else None

def get_last_significant_user_query(session_id: str) -> Optional[str]:
    row = fetchone(
        """
        SELECT user_text
        FROM convo_turn
        WHERE session_id = %s
          AND user_text IS NOT NULL
          AND length(trim(user_text)) >= 6
        ORDER BY
          CASE WHEN used_doc_ids IS NOT NULL AND array_length(used_doc_ids,1) > 0 THEN 0 ELSE 1 END,
          tstamp DESC
        LIMIT 1;
        """,
        (session_id,),
    )
    return (row["user_text"].strip() if row and row.get("user_text") else None)
