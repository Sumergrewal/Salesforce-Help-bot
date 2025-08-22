import os, sys, json
from pathlib import Path
from typing import Dict, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import psycopg
from psycopg.rows import dict_row

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://sumergrewal:sumergrewal@localhost:5433/sfhelp")
ROOT = Path(os.getenv("DATA_DIR", "text files")).expanduser().resolve()

def scan_jsonl(root: Path) -> List[Dict]:
    docs = []
    for p in sorted(root.rglob("*.jsonl")):
        try:
            rel = p.relative_to(root)
        except Exception:
            rel = p.name
        product = p.parent.name                 # top-level folder under ROOT (e.g., analytics)
        filename = p.name
        relpath = str(rel)
        # read first non-empty line to get doc_id/doc_title
        with p.open("r", encoding="utf-8") as fh:
            first = ""
            for line in fh:
                if line.strip():
                    first = line
                    break
        if not first:
            continue
        rec = json.loads(first)
        doc_id = rec.get("doc_id")
        doc_title = rec.get("doc_title")
        if not doc_id:
            print(f"[WARN] {p} missing doc_id; skipping")
            continue
        docs.append({
            "doc_id": doc_id,
            "doc_title": doc_title,
            "product": product,
            "filename": filename,
            "relpath": relpath,
        })
    # dedupe by doc_id (last wins)
    uniq = {}
    for d in docs:
        uniq[d["doc_id"]] = d
    return list(uniq.values())

def upsert_docs(conn, docs: List[Dict]) -> None:
    sql = """
    INSERT INTO fsc_docs (doc_id, doc_title, product, filename, relpath)
    VALUES (%(doc_id)s, %(doc_title)s, %(product)s, %(filename)s, %(relpath)s)
    ON CONFLICT (doc_id) DO UPDATE
      SET doc_title = EXCLUDED.doc_title,
          product   = EXCLUDED.product,
          filename  = EXCLUDED.filename,
          relpath   = EXCLUDED.relpath;
    """
    with conn.cursor() as cur:
        cur.executemany(sql, docs)
    conn.commit()

def update_chunks_per_doc(conn, docs: List[Dict]) -> None:
    """
    Simple, dependency-free approach: one UPDATE per doc_id.
    With ~40k rows total this is fine, especially with fsc_chunks_doc_id_idx.
    """
    with conn.cursor() as cur:
        for d in docs:
            cur.execute(
                """
                UPDATE fsc_chunks
                SET product = %s, filename = %s
                WHERE doc_id = %s
                  AND (product IS DISTINCT FROM %s OR filename IS DISTINCT FROM %s)
                """,
                (d["product"], d["filename"], d["doc_id"], d["product"], d["filename"])
            )
    conn.commit()

def main():
    print(f"[INFO] DB: {DATABASE_URL}")
    print(f"[INFO] Scanning JSONLs under: {ROOT}")
    if not ROOT.exists():
        print(f"[ERR] DATA_DIR not found: {ROOT}"); sys.exit(1)

    docs = scan_jsonl(ROOT)
    print(f"[INFO] Found {len(docs)} documents")

    with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
        # ensure required DDL exists (safe if already applied)
        with conn.cursor() as cur:
            cur.execute("""
            ALTER TABLE fsc_chunks
              ADD COLUMN IF NOT EXISTS product  text,
              ADD COLUMN IF NOT EXISTS filename text;
            CREATE TABLE IF NOT EXISTS fsc_docs (
              doc_id    text PRIMARY KEY,
              doc_title text,
              product   text,
              filename  text,
              relpath   text
            );
            CREATE INDEX IF NOT EXISTS fsc_chunks_doc_id_idx ON fsc_chunks (doc_id);
            """)
            conn.commit()

        upsert_docs(conn, docs)
        update_chunks_per_doc(conn, docs)

    print("[DONE] Backfill complete.")
    print("Try: SELECT product, COUNT(*) FROM fsc_chunks GROUP BY product ORDER BY 2 DESC;")

if __name__ == "__main__":
    main()
