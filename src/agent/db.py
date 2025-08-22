from typing import Any, Iterable, Optional
import psycopg
from psycopg.rows import dict_row

from src.agent.config import settings


def get_conn():
    """
    Return a new psycopg connection with dict_row factory.
    Caller is responsible for closing/committing if using directly.
    """
    return psycopg.connect(settings.DATABASE_URL, row_factory=dict_row)


def fetchall(sql: str, params: Optional[Iterable[Any]] = None) -> list[dict]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params or [])
        return list(cur.fetchall())


def fetchone(sql: str, params: Optional[Iterable[Any]] = None) -> Optional[dict]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params or [])
        row = cur.fetchone()
        return dict(row) if row else None


def execute(sql: str, params: Optional[Iterable[Any]] = None) -> int:
    """
    Execute a write statement; returns rows affected.
    Commits automatically.
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params or [])
        conn.commit()
        return cur.rowcount
