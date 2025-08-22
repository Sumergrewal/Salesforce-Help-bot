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
    # retrieval scores
    vec_dist: Optional[float] = None
    fts_rank: Optional[float] = None
    hybrid_score: Optional[float] = None
    # NEW: which product this chunk belongs to (from fsc_chunks or fsc_docs)
    product: Optional[str] = None

class Source(BaseModel):
    chunk_id: int
    doc_id: str
    doc_title: Optional[str] = None
    section_title: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    score: Optional[float] = None

class SearchRequest(BaseModel):
    query: str
    product: Optional[str] = Field(default=None, description="Restrict retrieval to a product")
    k_final: Optional[int] = Field(default=None, description="Override TOPK_FINAL")

class SearchResult(BaseModel):
    query: str
    chunks: List[Chunk]

class ChatRequest(BaseModel):
    session_id: str
    message: str
    product: Optional[str] = Field(default=None, description="Restrict retrieval to a product")

class ChatAnswer(BaseModel):
    session_id: str
    message: str
    answer: str
    sources: List[Source] = []

class ProductsResponse(BaseModel):
    products: List[str]
