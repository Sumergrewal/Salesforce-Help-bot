from typing import Iterable, Optional, Set, List

from src.agent.db import execute, fetchall, fetchone


def ensure_session(session_id: str, initial_summary: str = "") -> None:
    sql = """
    INSERT INTO convo_session (session_id, summary)
    VALUES (%s, %s)
    ON CONFLICT (session_id) DO NOTHING;
    """
    execute(sql, (session_id, initial_summary))


def get_recent_turns(session_id: str, n: int = 5) -> List[dict]:
    sql = """
    SELECT id, tstamp, user_text, answer_text, used_doc_ids, used_chunk_ids
    FROM convo_turn
    WHERE session_id = %s
    ORDER BY tstamp DESC
    LIMIT %s;
    """
    return fetchall(sql, (session_id, n))


def get_recent_doc_ids(session_id: str) -> Set[str]:
    # Distinct set of doc_ids mentioned in recent turns
    sql = """
    SELECT DISTINCT unnest(used_doc_ids) AS doc_id
    FROM convo_turn
    WHERE session_id = %s
    """
    rows = fetchall(sql, (session_id,))
    return {r["doc_id"] for r in rows if r.get("doc_id")}


def update_summary(session_id: str, summary: str) -> None:
    sql = "UPDATE convo_session SET summary = %s WHERE session_id = %s;"
    execute(sql, (summary, session_id))


def insert_turn(
    session_id: str,
    user_text: str,
    answer_text: str,
    used_doc_ids: Optional[list[str]] = None,
    used_chunk_ids: Optional[list[int]] = None,
) -> None:
    sql = """
    INSERT INTO convo_turn (session_id, user_text, answer_text, used_doc_ids, used_chunk_ids)
    VALUES (%s, %s, %s, %s, %s);
    """
    execute(sql, (session_id, user_text, answer_text, used_doc_ids, used_chunk_ids))
