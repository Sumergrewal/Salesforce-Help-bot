"""
Salesforce Help PDF Extractor
-----------------------------
Walks an input folder like:
  pdfs/
    analytics/
    commerce/
    ...
and writes outputs to an output folder (e.g., "text files/") while mirroring the structure.

For each PDF:
  - Writes a raw text file with page markers (e.g., <<<PAGE 12>>>)
  - Writes a chunked JSONL (recommended for embeddings) with section-aware metadata
  - Updates a manifest.csv with file-level info

Usage:
  pip install pymupdf
  python3 extract_sf_pdfs.py --pdf-dir "pdfs" --out-dir "text files" --max-chunk-tokens 900 --overlap 180
"""
import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Iterable, Tuple

import fitz  # PyMuPDF


# ---------------------- helpers ----------------------
def sha1_of_path(p: Path) -> str:
    return hashlib.sha1(str(p.resolve()).encode("utf-8")).hexdigest()[:16]

def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def normalize_spaces(s: str) -> str:
    # collapse runs of spaces/tabs while preserving newlines
    s = re.sub(r"[\t ]+", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    return s

# Keep reading order: sort by top (y), then left (x)
def extract_page_text(page: fitz.Page) -> str:
    blocks = page.get_text("blocks")
    blocks = sorted(blocks, key=lambda b: (round(b[1], 1), round(b[0], 1)))
    parts = [b[4].strip() for b in blocks if isinstance(b, (list, tuple)) and len(b) >= 5 and b[4].strip()]
    txt = "\n\n".join(parts)
    return normalize_spaces(txt)

def iter_toc_sections(doc: fitz.Document) -> Iterable[Dict]:
    toc = doc.get_toc(simple=True) or []
    if not toc:
        # no TOC: yield a single section for the whole document
        yield {"level": 1, "title": "Document", "start": 0, "end": doc.page_count - 1}
        return
    for i, (lvl, title, page1) in enumerate(toc):
        start = max(0, (page1 or 1) - 1)
        end = (toc[i+1][2] - 2) if i + 1 < len(toc) else doc.page_count - 1
        end = min(end, doc.page_count - 1)
        yield {"level": int(lvl or 1), "title": (title or "").strip(), "start": start, "end": end}

def chunk_words(words: List[str], max_tokens: int, overlap: int) -> List[Tuple[int, int]]:
    # returns list of (start_idx, end_idx) windows over words
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    if overlap >= max_tokens:
        raise ValueError("overlap must be smaller than max_tokens")
    stride = max_tokens - overlap
    spans = []
    i = 0
    n = len(words)
    while i < n:
        j = min(i + max_tokens, n)
        spans.append((i, j))
        if j == n:
            break
        i += stride
    return spans

def write_txt_with_page_markers(doc: fitz.Document, out_txt: Path) -> int:
    ensure_parent(out_txt)
    pages_written = 0
    with out_txt.open("w", encoding="utf-8") as f:
        for p in range(doc.page_count):
            page_text = extract_page_text(doc.load_page(p))
            marker = f"\n\n<<<PAGE {p+1}>>>\n\n"
            if p == 0:
                f.write(f"<<<PAGE {p+1}>>>\n\n")
            else:
                f.write(marker)
            if page_text:
                f.write(page_text)
            pages_written += 1
    return pages_written

def write_chunked_jsonl(doc: fitz.Document, out_jsonl: Path, doc_id: str, doc_title: str,
                        max_tokens: int = 900, overlap: int = 180) -> int:
    ensure_parent(out_jsonl)
    written = 0
    with out_jsonl.open("w", encoding="utf-8") as out:
        for sec in iter_toc_sections(doc):
            # gather text across the section page span
            texts: List[Tuple[int, str]] = []
            for p in range(sec["start"], sec["end"] + 1):
                t = extract_page_text(doc.load_page(p))
                if t:
                    texts.append((p + 1, t))  # store human page number
            if not texts:
                continue
            # join while keeping lightweight page anchors to assist later mapping
            # we will inject a page marker token between pages to avoid accidental merges
            joined = ""
            for i, (pg, t) in enumerate(texts):
                if i > 0:
                    joined += f"\n\n[PAGE {pg}]\n\n"
                joined += t
            words = joined.split()
            spans = chunk_words(words, max_tokens, overlap)
            for cid, (a, b) in enumerate(spans):
                chunk_text = " ".join(words[a:b])
                rec = {
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "section_title": sec["title"],
                    "section_level": sec["level"],
                    "page_start": texts[0][0],
                    "page_end": texts[-1][0],
                    "chunk_local_id": cid,
                    "text": chunk_text
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
    return written

# ---------------------- main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", type=str, required=True, help="Root folder that contains product subfolders of PDFs")
    ap.add_argument("--out-dir", type=str, required=True, help="Output root folder (e.g., 'text files')")
    ap.add_argument("--max-chunk-tokens", type=int, default=900)
    ap.add_argument("--overlap", type=int, default=180)
    args = ap.parse_args()

    pdf_root = Path(args.pdf_dir).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    manifest_path = out_root / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # write or append manifest header
    write_header = not manifest_path.exists()
    mf = manifest_path.open("a", newline="", encoding="utf-8")
    mw = csv.writer(mf)
    if write_header:
        mw.writerow(["doc_id", "pdf_path", "txt_path", "jsonl_path", "pages", "bytes"])

    total_pdfs = 0
    total_chunks = 0

    for pdf in pdf_root.rglob("*.pdf"):
        total_pdfs += 1
        # mirror structure: replace pdf_root with out_root, and filenames .pdf -> .txt/.jsonl
        rel = pdf.relative_to(pdf_root)
        out_txt = out_root / rel.with_suffix(".txt")
        out_jsonl = out_root / rel.with_suffix(".jsonl")
        doc_id = sha1_of_path(pdf)
        try:
            doc = fitz.open(pdf)
        except Exception as e:
            print(f"[WARN] Failed to open {pdf}: {e}")
            continue

        pages = doc.page_count
        size_bytes = pdf.stat().st_size if pdf.exists() else 0

        # 1) raw txt with page markers
        pages_written = write_txt_with_page_markers(doc, out_txt)

        # 2) chunked jsonl (recommended for embeddings)
        chunks_written = write_chunked_jsonl(
            doc, out_jsonl, doc_id, pdf.stem,
            max_tokens=args.max_chunk_tokens, overlap=args.overlap
        )
        total_chunks += chunks_written

        mw.writerow([doc_id, str(pdf), str(out_txt), str(out_jsonl), pages, size_bytes])
        print(f"[OK] {pdf} -> {pages_written} pages, {chunks_written} chunks")
        doc.close()

    mf.close()
    print(f"Done. PDFs processed: {total_pdfs}. Total chunks: {total_chunks}. Manifest: {manifest_path}")

if __name__ == "__main__":
    main()
