from fastapi import APIRouter
from src.agent.retrieval import hybrid_search
from src.models.schemas import SearchRequest, SearchResult

router = APIRouter(prefix="/search", tags=["search"])

@router.post("", response_model=SearchResult)
def search(req: SearchRequest):
    chunks = hybrid_search(req.query)
    return SearchResult(query=req.query, chunks=chunks)
