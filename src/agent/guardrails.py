import re
from typing import List, Tuple
from src.agent.db import fetchall

# very small stopword set for "low-info" detection
_STOP = {
    "a","an","the","and","or","but","please","pls","about","on","of","for","to",
    "me","something","some","tell","say","explain","help","info","information"
}

_GREETING = re.compile(r"^\s*(hi|hello|hey|yo|hola|namaste|good\s*(morning|afternoon|evening))\b", re.I)
_GOODBYE  = re.compile(r"\b(bye|goodbye|see\s*you|see\s*ya|take\s*care)\b", re.I)

def _tokens(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", s.lower())

def is_greeting(text: str) -> bool:
    return bool(_GREETING.search(text or ""))

def is_goodbye(text: str) -> bool:
    return bool(_GOODBYE.search(text or ""))

def is_low_info(text: str, min_content_tokens: int = 3) -> bool:
    toks = _tokens(text or "")
    content = [t for t in toks if t not in _STOP]
    return len(content) < min_content_tokens

def top_products(limit: int = 10) -> List[str]:
    """
    Reads distinct products from your DB (filled by backfill script).
    Falls back to doc_title prefixes if product is NULL.
    """
    sql = """
      WITH prod AS (
        SELECT COALESCE(NULLIF(product,''), split_part(doc_title, '_', 1)) AS p, COUNT(*) AS n
        FROM fsc_chunks
        GROUP BY 1
      )
      SELECT p FROM prod WHERE p IS NOT NULL ORDER BY n DESC, p ASC LIMIT %s;
    """
    rows = fetchall(sql, (limit,))
    return [r["p"] for r in rows if r.get("p")]

def example_queries() -> List[str]:
    return [
        "How do I create a dashboard in CRM Analytics?",
        "What are supported editions for B2B Commerce?",
        "Enable managed checkout for D2C Commerce",
        "Set up DevOps Center in Salesforce",
        "What is Omnichannel Inventory?",
        "How to configure Sales Cloud Einstein features?"
    ]

def make_welcome_msg() -> str:
    prods = top_products(12)
    ex = example_queries()
    bullets = "\n".join(f"- {p}" for p in prods)
    samples = "\n".join(f"• {q}" for q in ex[:4])
    return (
        "Hi! I can answer questions from your **Salesforce Help** corpus.\n\n"
        "**Popular product areas I know:**\n" + bullets +
        "\n\n**Try asking:**\n" + samples
    )

def make_clarify_msg(user_text: str) -> str:
    prods = top_products(10)
    ex = example_queries()
    samples = "\n".join(f"• {q}" for q in ex[:3])
    return (
        "I can help with Salesforce Help content, but I need a bit more detail.\n"
        f"Your message was: _“{user_text.strip()}”_.\n\n"
        "Please specify a product/feature, like one of these product areas:\n"
        + ", ".join(prods) +
        "\n\nFor example:\n" + samples
    )
