# src/agent/config.py
import os
from dataclasses import dataclass

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


@dataclass(frozen=True)
class _Settings:
    # --- DB ---
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://sumergrewal:sumergrewal@localhost:5433/sfhelp",
    )

    # --- OpenAI ---
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_EMBED_MODEL: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    VECTOR_DIM: int = int(os.getenv("VECTOR_DIM", "1536"))

    # --- Retrieval knobs ---
    TOPK_VECTOR: int = int(os.getenv("TOPK_VECTOR", "50"))     # vector candidate pool
    TOPK_FTS: int = int(os.getenv("TOPK_FTS", "50"))           # FTS candidate pool
    TOPK_FINAL: int = int(os.getenv("TOPK_FINAL", "8"))        # contexts to LLM
    HYBRID_ALPHA: float = float(os.getenv("HYBRID_ALPHA", "0.35"))   # weight for FTS in hybrid
    MIN_RELEVANCE: float = float(os.getenv("MIN_RELEVANCE", "0.25")) # vec_dist guardrail

    # --- Memory bias ---
    MEMORY_DOC_BOOST: float = float(os.getenv("MEMORY_DOC_BOOST", "0.03"))  # small boost if doc was cited recently


settings = _Settings()
