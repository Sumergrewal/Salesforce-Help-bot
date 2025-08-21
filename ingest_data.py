"""
ingest_data.py  —  OpenAI embeddings -> Postgres (pgvector)
- Token-aware batching so each API call stays under ~8k tokens.
- Splits ultra-long chunks and averages their sub-embeddings.
- Skips duplicates via ON CONFLICT (doc_id, page_start, section_title, chunk_local_id) DO NOTHING

Setup:
  pip install -U openai psycopg[binary] tqdm python-dotenv tiktoken

.env (example):
  DATABASE_URL=postgresql://sumergrewal:sumergrewal@localhost:5433/sfhelp
  OPENAI_API_KEY=YOUR_OPENAI_KEY
  OPENAI_EMBED_MODEL=text-embedding-3-small
  VECTOR_DIM=1536
  DATA_DIR=/ABSOLUTE/PATH/TO/text files
  # Optional tuning:
  BATCH_TOKEN_BUDGET=7600
  PER_ITEM_TOKEN_LIMIT=8000
  MAX_ITEMS_PER_BATCH=64
  INGEST_LIMIT_FILES=0
  INGEST_LIMIT_CHUNKS=0
"""

import os, sys, time, json
from pathlib import Path
from typing import List, Dict, Any, Iterable, Tuple

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import psycopg
from psycopg.rows import dict_row
from tqdm import tqdm
from openai import OpenAI

# -------- Environment --------
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://sumergrewal:sumergrewal@localhost:5433/sfhelp")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY")
if not OPENAI_API_KEY:
    print("[ERR] Missing OPENAI_API_KEY in environment/.env")
    sys.exit(1)

OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")  # 1536 dims
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "1536"))
DATA_DIR = Path(os.getenv("DATA_DIR", "text files")).expanduser().resolve()

# Token-aware batching controls
BATCH_TOKEN_BUDGET = int(os.getenv("BATCH_TOKEN_BUDGET", "7600"))  # leave headroom under 8192
PER_ITEM_TOKEN_LIMIT = int(os.getenv("PER_ITEM_TOKEN_LIMIT", "8000"))
MAX_ITEMS_PER_BATCH = int(os.getenv("MAX_ITEMS_PER_BATCH", "64"))

# Ingestion limits (0 = no limit)
INGEST_LIMIT_FILES = int(os.getenv("INGEST_LIMIT_FILES", "0") or "0")
INGEST_LIMIT_CHUNKS = int(os.getenv("INGEST_LIMIT_CHUNKS", "0") or "0")

# -------- Tokenizer --------
try:
    import tiktoken
    try:
        _enc = tiktoken.encoding_for_model(OPENAI_EMBED_MODEL)
    except Exception:
        # embeddings-3 models use cl100k_base encoding
        _enc = tiktoken.get_encoding("cl100k_base")
except Exception:
    print("[WARN] tiktoken not installed; install it for token-aware batching: pip install tiktoken")
    _enc = None

def count_tokens(s: str) -> int:
    if _enc is None:
        # Fallback: rough approximation
        return max(1, len(s) // 4)
    return len(_enc.encode(s or ""))

def split_by_tokens(text: str, max_tokens: int) -> List[str]:
    """Split a long text into <= max_tokens-token pieces."""
    if _enc is None:
        # crude char-based split if tiktoken missing
        step = max_tokens * 4
        return [text[i:i+step] for i in range(0, len(text), step)]
    ids = _enc.encode(text or "")
    parts = []
    i = 0
    n = len(ids)
    while i < n:
        j = min(i + max_tokens, n)
        parts.append(_enc.decode(ids[i:j]))
        i = j
    return parts

# -------- OpenAI helpers --------
def embed_batch(client: OpenAI, texts: List[str], model: str, max_retries: int = 6) -> List[List[float]]:
    """Embed a batch of texts, retrying with backoff on transient errors."""
    attempt = 0
    while True:
        try:
            resp = client.embeddings.create(model=model, input=texts)
            return [d.embedding for d in resp.data]
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                raise
            sleep_s = min(20, 2 ** attempt) + 0.2 * attempt
            print(f"[WARN] embed batch failed ({e}); retrying in {sleep_s:.1f}s...")
            time.sleep(sleep_s)

def embed_long_text(client: OpenAI, text: str, model: str) -> List[float]:
    """
    For texts exceeding PER_ITEM_TOKEN_LIMIT, split into sub-parts,
    embed each, then mean-pool (vectors are normalized by API).
    """
    pieces = split_by_tokens(text, PER_ITEM_TOKEN_LIMIT - 50)  # small safety margin
    if len(pieces) == 1:
        return embed_batch(client, pieces, model)[0]
    vecs = embed_batch(client, pieces, model)
    # mean-pool
    dim = len(vecs[0])
    out = [0.0] * dim
    for v in vecs:
        for i, x in enumerate(v):
            out[i] += x
    out = [x / len(vecs) for x in out]
    return out

# -------- DB helpers --------
def to_vector_literal(vec: Iterable[float]) -> str:
    # pgvector accepts the "[x,y,...]" literal form
    return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"

def ensure_unique_index(conn: psycopg.Connection):
    """Make sure ON CONFLICT target exists."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS uq_fsc_doc_sectionpage_chunk
            ON fsc_chunks (doc_id, page_start, section_title, chunk_local_id);
        """)
        conn.commit()

# -------- File helpers --------
def find_jsonl_files(root: Path) -> List[Path]:
    if not root.exists():
        print(f"[ERR] DATA_DIR does not exist: {root}")
        sys.exit(1)
    return sorted(root.rglob("*.jsonl"))

# -------- Main ingest --------
def flush_batch(conn: psycopg.Connection, client: OpenAI, recs: List[Dict[str, Any]]) -> int:
    if not recs:
        return 0
    texts = [r["text"] for r in recs]
    vecs = embed_batch(client, texts, OPENAI_EMBED_MODEL)
    rows = []
    for r, v in zip(recs, vecs):
        rows.append({
            "doc_id": r.get("doc_id"),
            "doc_title": r.get("doc_title"),
            "section_title": r.get("section_title"),
            "section_level": r.get("section_level"),
            "page_start": r.get("page_start"),
            "page_end": r.get("page_end"),
            "chunk_local_id": r.get("chunk_local_id"),
            "content": r.get("text"),
            "embedding": to_vector_literal(v),
        })
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO fsc_chunks
            (doc_id, doc_title, section_title, section_level, page_start, page_end, chunk_local_id, content, embedding)
            VALUES (%(doc_id)s, %(doc_title)s, %(section_title)s, %(section_level)s, %(page_start)s, %(page_end)s, %(chunk_local_id)s, %(content)s, %(embedding)s::vector)
            ON CONFLICT (doc_id, page_start, section_title, chunk_local_id) DO NOTHING
            """,
            rows
        )
        conn.commit()
    return len(rows)

def main():
    print(f"[INFO] DB: {DATABASE_URL}")
    print(f"[INFO] Model: {OPENAI_EMBED_MODEL} (dim={VECTOR_DIM})")
    print(f"[INFO] Data dir: {DATA_DIR}")
    print(f"[INFO] Token budget per request: {BATCH_TOKEN_BUDGET} | Per-item limit: {PER_ITEM_TOKEN_LIMIT} | Max items: {MAX_ITEMS_PER_BATCH}")

    files = find_jsonl_files(DATA_DIR)
    if INGEST_LIMIT_FILES > 0:
        files = files[:INGEST_LIMIT_FILES]

    client = OpenAI(api_key=OPENAI_API_KEY)

    inserted_total = 0
    with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
        # Ensure ON CONFLICT target exists
        ensure_unique_index(conn)

        for f in files:
            # load records from one file
            recs: List[Dict[str, Any]] = []
            with open(f, "r", encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if INGEST_LIMIT_CHUNKS and i >= INGEST_LIMIT_CHUNKS:
                        break
                    r = json.loads(line)
                    txt = r.get("text") or ""
                    if not txt.strip():
                        continue
                    recs.append(r)

            if not recs:
                print(f"[SKIP] {f} (no text)"); continue

            print(f"[FILE] {f} | chunks={len(recs)}")
            pbar = tqdm(total=len(recs), unit="chunk")

            batch: List[Dict[str, Any]] = []
            token_sum = 0

            for r in recs:
                t = r["text"]
                n = count_tokens(t)

                # If a single item exceeds per-item limit: split+pool and insert immediately
                if n > PER_ITEM_TOKEN_LIMIT:
                    v = embed_long_text(client, t, OPENAI_EMBED_MODEL)
                    row = {
                        "doc_id": r.get("doc_id"),
                        "doc_title": r.get("doc_title"),
                        "section_title": r.get("section_title"),
                        "section_level": r.get("section_level"),
                        "page_start": r.get("page_start"),
                        "page_end": r.get("page_end"),
                        "chunk_local_id": r.get("chunk_local_id"),
                        "content": r.get("text"),
                        "embedding": to_vector_literal(v),
                    }
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO fsc_chunks
                            (doc_id, doc_title, section_title, section_level, page_start, page_end, chunk_local_id, content, embedding)
                            VALUES (%(doc_id)s, %(doc_title)s, %(section_title)s, %(section_level)s, %(page_start)s, %(page_end)s, %(chunk_local_id)s, %(content)s, %(embedding)s::vector)
                            ON CONFLICT (doc_id, page_start, section_title, chunk_local_id) DO NOTHING
                            """,
                            row
                        )
                        conn.commit()
                    inserted_total += 1
                    pbar.update(1)
                    continue

                # If adding this to the batch would exceed token budget or item cap: flush first
                if (token_sum + n > BATCH_TOKEN_BUDGET) or (len(batch) >= MAX_ITEMS_PER_BATCH):
                    inserted = flush_batch(conn, client, batch)
                    inserted_total += inserted
                    pbar.update(inserted)
                    batch, token_sum = [], 0

                batch.append(r)
                token_sum += n

            # flush tail
            if batch:
                inserted = flush_batch(conn, client, batch)
                inserted_total += inserted
                pbar.update(inserted)

            pbar.close()

    print(f"[DONE] Inserted rows: {inserted_total}")
    print("Tip: create ANN index now for fast search:")
    print("  CREATE INDEX IF NOT EXISTS fsc_chunks_embedding_idx ON fsc_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 200);")
    print("  VACUUM ANALYZE fsc_chunks;")

if __name__ == "__main__":
    main()
