import os, sys, textwrap
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
import psycopg
from psycopg.rows import dict_row

DB_URL = os.getenv("DATABASE_URL", "postgresql://sumergrewal:sumergrewal@localhost:5433/sfhelp")
MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY")

def vec_literal(v):
    return "[" + ",".join(f"{float(x):.6f}" for x in v) + "]"

def embed(q):
    client = OpenAI(api_key=API_KEY)
    e = client.embeddings.create(model=MODEL, input=[q])
    return e.data[0].embedding

def search(query, k=5):
    qvec = vec_literal(embed(query))
    sql = """
      SELECT doc_title, section_title, page_start, page_end,
             LEFT(content, 400) AS snippet,
             (embedding <=> %s::vector) AS distance
      FROM fsc_chunks
      ORDER BY embedding <=> %s::vector
      LIMIT %s;
    """
    with psycopg.connect(DB_URL, row_factory=dict_row) as conn, conn.cursor() as cur:
        cur.execute(sql, (qvec, qvec, k))
        return cur.fetchall()

if __name__ == "__main__":
    if not API_KEY:
        print("Set OPENAI_API_KEY in your .env"); sys.exit(1)
    q = " ".join(sys.argv[1:]) or "enable managed checkout for D2C"
    rows = search(q, k=5)
    print(f"\nQuery: {q}\n")
    for i, r in enumerate(rows, 1):
        where = f"{r['doc_title']} • {r['section_title']} • p.{r['page_start']}-{r['page_end']}"
        print(f"{i}. {where}  (dist={r['distance']:.4f})")
        print(textwrap.fill(r["snippet"].replace("\n", " "), width=100))
        print()
