from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: int
    doc_id: str
    doc_title: Optional[str] = None
    section_title: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    content: str

    # retrieval scores (optional, populated by retrieval)
    vec_dist: Optional[float] = None
    fts_rank: Optional[float] = None
    hybrid_score: Optional[float] = None


class Source(BaseModel):
    chunk_id: int
    doc_id: str
    doc_title: Optional[str] = None
    section_title: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    score: Optional[float] = None  # hybrid score or vec score used


class SearchRequest(BaseModel):
    query: str
    k_final: Optional[int] = Field(default=None, description="Override TOPK_FINAL")


class SearchResult(BaseModel):
    query: str
    chunks: List[Chunk]


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatAnswer(BaseModel):
    session_id: str
    message: str
    answer: str
    sources: List[Source] = []
