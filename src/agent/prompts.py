from typing import List, Optional
from textwrap import shorten
from src.models.schemas import Chunk

SYSTEM_PROMPT = """You are a helpful assistant specialized in Salesforce Help documentation.
Follow these rules STRICTLY:
- Only answer using the provided context passages.
- If the answer is not present in the context, say you don't know and suggest a narrower query.
- Be concise, step-by-step for procedural questions.
- Use Salesforce terminology accurately.
- Do NOT fabricate product names, settings, or steps.
- You do NOT need to include references inside the prose; the caller attaches sources separately.
"""

def _format_passage(i: int, c: Chunk, max_chars: int = 1000) -> str:
    label = f"[{i}] {c.doc_title or c.doc_id}"
    sect = f" • {c.section_title}" if c.section_title else ""
    pages = ""
    if c.page_start is not None and c.page_end is not None:
        pages = f" • p.{c.page_start}-{c.page_end}"
    header = f"{label}{sect}{pages}"
    body = shorten(c.content.replace("\n", " ").strip(), width=max_chars, placeholder=" …")
    return f"{header}\n{body}"

def build_messages(query: str, passages: List[Chunk], summary: Optional[str]) -> list[dict]:
    """
    Build OpenAI chat messages with a strict grounding context.
    """
    ctx_blocks = [_format_passage(i+1, p) for i, p in enumerate(passages)]
    ctx = "\n\n".join(ctx_blocks) if ctx_blocks else "(no context)"
    mem = f"\n\nConversation summary/hint: {summary}" if summary else ""
    user = f"User question: {query}\n\nContext passages:\n{ctx}{mem}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
