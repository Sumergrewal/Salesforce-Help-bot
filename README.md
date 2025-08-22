# Salesforce Help RAG Agent

Conversational AI agent over the **Salesforce Help** corpus. It uses **hybrid retrieval** (pgvector embeddings + Postgres Full-Text Search) with **session memory** so follow-ups like “tell me more about the same product” work. Includes a **FastAPI** backend and a **Streamlit** chat UI.

---

## ✨ Features

- **Hybrid RAG**: vector similarity (pgvector) + keyword search (Postgres FTS) blended with a tunable weight.
- **Product filter**: restrict retrieval to a selected product (e.g., *Commerce*, *Platform*, *Einstein*).
- **Citations**: every answer shows document title • section • page.
- **Session memory**: remembers recently used docs & inferred product for follow-ups (“same product”).
- **Seed-first startup**: optional SQL seed so others can run the app without re-ingesting.

> 🔕 Web search is **disabled** in this version (RAG-only), per project scope.

---

## 🗂️ Repository Structure

```
.
├── app.py                         # FastAPI entrypoint
├── streamlit_app.py               # Streamlit UI
├── docker-compose.yaml            # Postgres + pgvector
├── init/
│   ├── 01_schema.sql              # tables, indexes, triggers
│   └── 02_seed.sql[.gz]           # optional: preloaded data (docs & chunks & memory tables)
├── src/
│   ├── api/
│   │   └── app.py                 # FastAPI app + routers
│   ├── models/
│   │   └── schemas.py             # Pydantic models (ChatRequest/ChatAnswer/Chunk/Source)
│   ├── agent/
│   │   ├── config.py              # env knobs (TOPK/HYBRID_ALPHA/DB URL/openai model)
│   │   ├── db.py                  # DB helpers (fetchone/fetchall/execute/connect)
│   │   ├── retrieval.py           # vector + FTS + blending + boosts
│   │   ├── orchestrator.py        # conversation policy + memory + product inference
│   │   ├── answer.py              # calls OpenAI chat, returns answer + citations
│   │   ├── prompts.py             # default & overview prompts
│   │   ├── memory.py              # convo_session / convo_turn utils
│   │   └── guardrails.py          # greetings / goodbyes / low-info
│   └── routes/
│       ├── chat.py                # POST /chat
│       ├── search.py              # POST /search (debug)
│       └── products.py            # GET /products (for sidebar dropdown)
├── scripts/
│   └── fetch_seed.sh              # optional: download seed into init/
├── .streamlit/
│   └── config.toml                # dark theme
├── .env.example                   # sample environment variables
└── requirements.txt
```

---

## ✅ Prerequisites

- **Docker Desktop**
- **Python 3.10+** (for backend + Streamlit)
- **OpenAI API key** (env var `OPENAI_API_KEY`)
- (Optional) **pgAdmin** for DB inspection

---

## 🚀 Quickstart (with seed)
0. **Data files - download from link and place in root directory**  
   - link to pdfs: 
   - link to text/json files:https://drive.google.com/drive/folders/1B2UJqWeGHNZRoayYmbKZqIRojDAH2kYA?usp=drive_link

1. **Download the seed into `init/`**  
   - seed file link: https://drive.google.com/file/d/1Xumj25gDl1q3ZL1OekeRtgyvAngQDq_F/view?usp=drive_link

2. **Start database (first boot will load schema + seed)**
   ```bash
   docker compose down -v              # wipes only YOUR local volume; run on first boot
   docker compose up -d db
   ```
  **if db isnt populated:**
  run: 
  ```bash
   python3 extract_sf_pdfs.py
   python3 ingest_data.py
   ```

3. **Verify data + run backfill_doc_meta**
  ```bash
    python3 backfill_doc_meta.py
   ```

  make sure to keep the docker desktop open
  make sure venv is created and requirements are downloaded
  ```bash
  docker compose up --build (this will init and start)
  ```
  open new terminal as the above will not terminate after completion
  ```bash
    docker compose exec -T db psql -U sumergrewal -d sfhelp -c "SELECT COUNT(*) FROM fsc_chunks;" (for verifying if data is loaded)
  ```

4. **Create `.env`**
   ```bash
   cp .env.example .env
   # then edit values as needed
   ```

5. **Install deps & run API**
   ```bash
   pip install -r requirements.txt
   uvicorn app:app --reload --port 8000
   ```

6. **Run Streamlit UI (new terminal)**
   ```bash
   streamlit run streamlit_app.py
   ```

---

## 🔧 Configuration (.env)

```env
# DB
DATABASE_URL=postgresql://sumergrewal:sumergrewal@localhost:5433/sfhelp

# OpenAI
OPENAI_API_KEY=sk-***          # required
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small
VECTOR_DIM=1536

# Retrieval knobs
TOPK_VECTOR=50
TOPK_FTS=50
TOPK_FINAL=8
HYBRID_ALPHA=0.35
MIN_RELEVANCE=0.25

# Retrieval boosts
MEMORY_DOC_BOOST=0.05
MEMORY_PRODUCT_BOOST=0.02

# Where the Streamlit UI will call your API:
API_BASE_URL='http://localhost:8000'
```

**Knobs explained**  
- `TOPK_VECTOR` / `TOPK_FTS`: candidates per leg before blending.  
- `TOPK_FINAL`: chunks sent to the model.  
- `HYBRID_ALPHA`: 0.0–1.0 → weight of vector vs FTS (0.35 favors exact Salesforce terms).  
- `MEMORY_*_BOOST`: small nudges for documents seen recently / product match.

---

## 🧠 How it works (short)

1. **Streamlit** collects `session_id`, optional **product** filter, and the user’s message → `POST /chat`.
2. **Orchestrator**  
   - Applies guardrails (hi/bye/low-info), infers product from memory if not set.
   - Calls **hybrid_search** with/without product filter.
3. **Hybrid retrieval**  
   - **Vector leg**: pgvector cosine distance on `fsc_chunks.embedding`.  
   - **FTS leg**: GIN-indexed `combined_tsv` via `plainto_tsquery`, ranked by `ts_rank_cd`.  
   - Normalize & blend: `score = α*vector + (1-α)*fts`, apply small boosts, keep `TOPK_FINAL`.
4. **Answering**  
   - `answer_with_citations` prompts OpenAI using the selected chunks (default or overview prompt).  
   - Returns answer + structured **sources**.
5. **Memory**  
   - Store a **turn**: user_text, answer_text, `used_doc_ids[]`, `used_chunk_ids[]`.  
   - Follow-ups like “same product” use memory to keep context.

---

## 🛠️ API Endpoints

### `POST /chat`
**Request**
```json
{
  "session_id": "uuid-or-string",
  "message": "How do I enable Managed Checkout in a D2C store?",
  "product": "commerce"
}
```
**Response (abridged)**
```json
{
  "session_id": "…",
  "message": "…",
  "answer": "…",
  "sources": [
    {
      "source_type": "doc",
      "doc_title": "b2b_commerce_and_d2c_commerce_8-21-2025",
      "section_title": "Managed Checkout",
      "page_start": 120,
      "page_end": 121,
      "score": 0.62
    }
  ]
}
```

### `POST /search` (debug)
```bash
curl -s -X POST http://localhost:8000/search   -H "Content-Type: application/json"   -d '{"query":"managed checkout","product":"commerce"}' | jq .
```

### `GET /products`
Returns distinct product names for the UI filter.

---

## 🗃️ Database Schema (overview)

**`fsc_docs`**  
- `doc_id` (text, PK), `doc_title`, `product`, `filename`, …

**`fsc_chunks`**  
- `id` (PK), `doc_id` (FK), `product`, `section_title`, `section_level`,  
- `page_start`, `page_end`, `chunk_local_id`, `content` (text),  
- `embedding vector(1536)`,  
- `combined_tsv tsvector` (generated or maintained on insert).

**Indexes**  
- `fsc_chunks_combined_tsv_idx` (GIN on `combined_tsv`)  
- `uq_fsc_doc_sectionpage_chunk` (uniqueness across `doc_id`, `page_start`, `section_title`, `chunk_local_id`)  
- `fsc_chunks_pkey`

**Memory**  
- `convo_session(session_id text primary key, summary text, created_at timestamptz default now())`  
- `convo_turn(id bigserial pk, session_id text, tstamp timestamptz, user_text text, answer_text text, used_doc_ids text[], used_chunk_ids int[])`

---

## 🧪 Test Questions

**Straightforward**  
1. *Commerce* — What are the **supported editions for B2B Commerce**?  
2. *Commerce* — How do I **enable Managed Checkout** in a D2C store?  
3. *Platform/DevOps* — What are the **prerequisites to use DevOps Center**?  
4. *Einstein* — Overview of **Sales Cloud Einstein** capabilities.

**Complex**  
5. *Commerce* — **Compare Managed vs Custom Checkout** for D2C and when to choose each.  
6. *Commerce + Inventory* — Steps to use **Omnichannel Inventory** with a commerce store (include admin setup).

**Memory Showcase**  
7. “What are supported editions for B2B Commerce?” → “**Tell me more about the same product**.”  
8. “Switch to **Revenue Cloud**. What are the required editions?” → “**Continue on the same product—how is the product catalog shared?**”  
9. “**Sales Cloud Einstein**—what’s included?” → “**Still the same product—how do I enable it?**”

---

## 🧰 Troubleshooting

- **`role "X" does not exist`**  
  You changed `POSTGRES_USER` in `docker-compose.yaml` but kept an old volume. Recreate:
  ```bash
  docker compose down -v
  docker compose up -d db
  ```

- **Port conflict on 5433**  
  Edit `ports: "5433:5432"` → change the **left** side to a free port (e.g., `5434:5432`) and update `DATABASE_URL`.

- **Empty tables after restart**  
  If you launched without seed, DB is empty. Either import the seed:
  ```bash
  gunzip -c init/02_seed.sql.gz | psql "$DATABASE_URL"
  ```
  or run your ingestion pipeline (not required for demo).

- **OpenAI errors**  
  Ensure `OPENAI_API_KEY` is set; check model name; reduce prompt size if you hit context limits.

- **Streamlit “circular import”**  
  In `streamlit_app.py`, make sure you import **streamlit as st** (not the file itself), and avoid naming your file `streamlit.py`.

---

## 🎛️ Tuning

- If answers miss exact terms → raise `TOPK_FTS` or lower `HYBRID_ALPHA` (e.g., `0.30`).  
- If paraphrased questions underperform → raise `TOPK_VECTOR` or increase `HYBRID_ALPHA` (e.g., `0.45`).  
- If answers get rambling → reduce `TOPK_FINAL` (e.g., `6–8`).  
- Keep boosts small (0.02–0.05) so they guide, not dominate.

---

## 🔒 Security Notes

- The app is local by default. If you expose it, protect the API and redact PII in logs.  
- OpenAI key is read from `.env`; don’t commit secrets.

---

## 📝 License

Add your preferred license (MIT/Apache-2.0) here.

---

## 🙌 Acknowledgements

- PostgreSQL + pgvector  
- FastAPI, Streamlit  
- OpenAI models for embeddings and chat
