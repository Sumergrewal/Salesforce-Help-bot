CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS fsc_chunks (
  id              bigserial PRIMARY KEY,
  doc_id          text NOT NULL,
  doc_title       text,
  section_title   text,
  section_level   smallint,
  page_start      int,
  page_end        int,
  chunk_local_id  int NOT NULL,
  content         text NOT NULL,
  embedding       vector(1536)   -- set to 384 if you use MiniLM
);

-- avoid duplicate inserts for the same chunk
ALTER TABLE fsc_chunks
  ADD CONSTRAINT IF NOT EXISTS fsc_chunks_doc_chunk_unique
  UNIQUE (doc_id, chunk_local_id);

-- optional full text search column and index
ALTER TABLE fsc_chunks
  ADD COLUMN IF NOT EXISTS content_tsv tsvector
  GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX IF NOT EXISTS fsc_chunks_tsv_idx
  ON fsc_chunks USING GIN (content_tsv);

-- simple sanity check on pages
ALTER TABLE fsc_chunks
  ADD CONSTRAINT IF NOT EXISTS fsc_chunks_page_check
  CHECK (page_start >= 1 AND page_end >= page_start);

-- build ANN index after you finish bulk ingest (faster that way)
-- CREATE INDEX IF NOT EXISTS fsc_chunks_embedding_idx
--   ON fsc_chunks USING ivfflat (embedding vector_cosine_ops)
--   WITH (lists = 200);
