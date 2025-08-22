# src/agent/prompts.py
from typing import List, Optional
from src.models.schemas import Chunk

SYSTEM_PROMPT = """You are a helpful assistant specialized in Salesforce Help documentation.
- Base answers ONLY on the provided context passages.
- Keep answers precise and grounded; do not invent edition/limit info.
- If information is missing, say you can't find it in the provided docs.
"""

def _format_passages(passages: List[Chunk]) -> str:
    lines = []
    for i, c in enumerate(passages, 1):
        title = c.doc_title or c.doc_id
        sect = f" — {c.section_title}" if c.section_title else ""
        pages = ""
        if c.page_start is not None and c.page_end is not None:
            pages = f" (p.{c.page_start}-{c.page_end})"
        lines.append(f"[{i}] {title}{sect}{pages}\n{c.content}")
    return "\n\n".join(lines)

def build_messages(query: str, passages: List[Chunk], summary: Optional[str]) -> list[dict]:
    context = _format_passages(passages)
    mem = f"# Conversation note:\n{summary}\n\n" if summary else ""
    user = (
        f"{mem}"
        f"# Context passages:\n{context}\n\n"
        f"# User question:\n{query}\n\n"
        "Answer using ONLY the context. Keep the answer concise but complete, and include concrete steps or editions when stated."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

# NEW: richer overview mode for low-info + known product
def build_product_overview_messages(product: str, passages: List[Chunk], summary: Optional[str]) -> list[dict]:
    context = _format_passages(passages)
    mem = f"# Conversation note:\n{summary}\n\n" if summary else ""
    user = (
        f"{mem}"
        f"# Product: {product}\n"
        f"# Context passages:\n{context}\n\n"
        "Create a grounded PRODUCT OVERVIEW for the product above using ONLY the context. "
        "Prefer facts that appear multiple times or in summary/intro sections. "
        "Use this structured format (omit a section if not supported by the context):\n\n"
        "## What it is\n"
        "- 2-3 sentences.\n\n"
        "## Supported editions / availability\n"
        "- List exact editions, experiences (Lightning, mobile), and key availability notes.\n\n"
        "## Core capabilities\n"
        "- 4-8 bullet points of major features.\n\n"
        "## Setup highlights / prerequisites\n"
        "- 3-6 bullets of important setup steps/prereqs called out in docs.\n\n"
        "## Typical tasks\n"
        "- 3-6 bullets of common tasks users perform here.\n\n"
        "## Related features\n"
        "- Bullets of features frequently paired with this product.\n\n"
        "## Next questions you can ask\n"
        "- 3 concise, actionable follow-ups.\n\n"
        "Be precise; if editions or constraints are not explicitly stated in the context, say they aren't shown here."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
