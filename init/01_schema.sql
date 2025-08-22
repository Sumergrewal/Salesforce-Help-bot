-- init/01_schema.sql
-- Schema for Salesforce Help Agent
-- Idempotent: safe to run multiple times.

------------------------------------------------------------
-- 0) Extensions
------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS vector;

------------------------------------------------------------
-- 1) Content chunks (embeddings)
------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.fsc_chunks (
  id               bigserial PRIMARY KEY,
  doc_id           text NOT NULL,
  doc_title        text,
  section_title    text,
  section_level    smallint,
  page_start       int,
  page_end         int,
  chunk_local_id   int NOT NULL,
  content          text NOT NULL,
  embedding        vector(1536),   -- match your embedding dim
  -- New metadata (filled by backfill/ingest)
  product          text,
  filename         text
);

-- Uniqueness: one row per (doc + section/page + local chunk id)
CREATE UNIQUE INDEX IF NOT EXISTS uq_fsc_doc_sectionpage_chunk
  ON public.fsc_chunks (doc_id, page_start, section_title, chunk_local_id);

-- Handy lookups/updates
CREATE INDEX IF NOT EXISTS fsc_chunks_doc_id_idx   ON public.fsc_chunks (doc_id);
CREATE INDEX IF NOT EXISTS fsc_chunks_product_idx  ON public.fsc_chunks (product);
CREATE INDEX IF NOT EXISTS fsc_chunks_filename_idx ON public.fsc_chunks (filename);

-- Full-text search over section_title + content (generated column)
ALTER TABLE public.fsc_chunks
  ADD COLUMN IF NOT EXISTS combined_tsv tsvector
  GENERATED ALWAYS AS (
    to_tsvector('english', coalesce(section_title,'') || ' ' || content)
  ) STORED;

-- FTS index
CREATE INDEX IF NOT EXISTS fsc_chunks_combined_tsv_idx
  ON public.fsc_chunks USING GIN (combined_tsv);

-- (Optional, create later when table is large)
-- CREATE INDEX IF NOT EXISTS fsc_chunks_embedding_ivfflat
--   ON public.fsc_chunks USING ivfflat (embedding vector_cosine_ops)
--   WITH (lists = 200);
-- -- or HNSW (pgvector >= 0.5)
-- -- CREATE INDEX IF NOT EXISTS fsc_chunks_embedding_hnsw
-- --   ON public.fsc_chunks USING hnsw (embedding vector_cosine_ops)
-- --   WITH (m = 16, ef_construction = 200);

------------------------------------------------------------
-- 2) One-row-per-document metadata (normalized)
------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.fsc_docs (
  doc_id    text PRIMARY KEY,
  doc_title text,
  product   text,
  filename  text,
  relpath   text
);

-- Helpful filter/indexes on fsc_docs
CREATE INDEX IF NOT EXISTS fsc_docs_product_idx  ON public.fsc_docs (product);
CREATE INDEX IF NOT EXISTS fsc_docs_filename_idx ON public.fsc_docs (filename);

------------------------------------------------------------
-- 3) Conversation memory
------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.convo_session (
  session_id  text PRIMARY KEY,
  created_at  timestamptz DEFAULT now(),
  summary     text
);

CREATE TABLE IF NOT EXISTS public.convo_turn (
  id             bigserial PRIMARY KEY,
  session_id     text REFERENCES public.convo_session(session_id) ON DELETE CASCADE,
  tstamp         timestamptz DEFAULT now(),
  user_text      text,
  answer_text    text,
  used_doc_ids   text[],
  used_chunk_ids bigint[]
);

CREATE INDEX IF NOT EXISTS convo_turn_session_idx ON public.convo_turn (session_id);

