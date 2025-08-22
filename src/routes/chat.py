from fastapi import APIRouter
from src.agent.orchestrator import run_chat
from src.models.schemas import ChatRequest, ChatAnswer

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("", response_model=ChatAnswer)
def chat(req: ChatRequest):
    return run_chat(req.session_id, req.message)
