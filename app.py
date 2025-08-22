from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.routes.chat import router as chat_router
from src.routes.search import router as search_router

app = FastAPI(
    title="SF Help Agent",
    version="0.1.0",
    description="Conversational agent over Salesforce Help content (hybrid retrieval + memory).",
)

# Broad CORS for local Streamlit and demos
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(search_router)

@app.get("/")
def root():
    return {"ok": True, "service": "SF Help Agent"}
