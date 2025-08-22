import os
import uuid
import requests
from dotenv import load_dotenv; load_dotenv()
import streamlit as st

st.set_page_config(page_title="SF Help Agent", page_icon="🔎", layout="wide")

# Where your FastAPI is running
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# --- Session bootstrap ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []  # [{role:"user"|"assistant", text:str, sources:list}]
if "selected_product" not in st.session_state:
    st.session_state.selected_product = None

@st.cache_data(ttl=3600)
def fetch_products(api_base: str) -> list[str]:
    try:
        r = requests.get(f"{api_base}/products", timeout=15)
        r.raise_for_status()
        data = r.json() or {}
        return data.get("products", [])
    except Exception:
        return []

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 🛠️ Settings")

    # Advanced (collapsed) section
    with st.expander("Advanced (API, session)", expanded=False):
        st.write(f"**API base:** `{API_BASE_URL}`")
        st.write(f"**Session:** `{st.session_state.session_id}`")

    # Main controls
    products = fetch_products(API_BASE_URL)
    sel = st.selectbox(
        "Filter by product",
        ["All products"] + products if products else ["All products"],
        index=0,
    )
    st.session_state.selected_product = None if sel == "All products" else sel

    if st.button("🔄  New session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.history.clear()
        (st.rerun if hasattr(st, "rerun") else st.experimental_rerun)()

    show_sources = st.checkbox("Show sources", value=True)

# --- Header ---
st.title("Salesforce Help Agent")
if not st.session_state.selected_product:
    st.info("Tip: you can narrow results by choosing a product in the sidebar.")

st.caption("Grounded answers from your Salesforce Help corpus.")

# --- Render chat history ---
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["text"])
        if show_sources and turn["role"] == "assistant" and turn.get("sources"):
            with st.expander("Sources"):
                for i, s in enumerate(turn["sources"], 1):
                    title = s.get("doc_title") or s.get("doc_id") or "Unknown document"
                    sect = f" • {s.get('section_title')}" if s.get("section_title") else ""
                    pages = ""
                    if s.get("page_start") is not None and s.get("page_end") is not None:
                        pages = f" • p.{s['page_start']}-{s['page_end']}"
                    score = s.get("score")
                    score_txt = f" (score: {score:.3f})" if isinstance(score, (int, float)) else ""
                    st.write(f"{i}. {title}{sect}{pages}{score_txt}")

# --- Input ---
prompt = st.chat_input("Ask a question (optionally pick a product in the sidebar)…")

if prompt:
    # Show user message
    st.session_state.history.append({"role": "user", "text": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare payload for /chat (include product if selected)
    payload = {
        "session_id": st.session_state.session_id,
        "message": prompt,
    }
    if st.session_state.selected_product:
        payload["product"] = st.session_state.selected_product

    # Call API
    try:
        resp = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json() or {}
        answer = (data.get("answer") or "").strip() or "_No answer returned._"
        sources = data.get("sources", []) or []
    except Exception as e:
        answer = f"⚠️ Error contacting API: `{e}`"
        sources = []

    # Render assistant reply
    st.session_state.history.append({"role": "assistant", "text": answer, "sources": sources})
    with st.chat_message("assistant"):
        st.markdown(answer)
        if show_sources and sources:
            with st.expander("Sources"):
                for i, s in enumerate(sources, 1):
                    title = s.get("doc_title") or s.get("doc_id") or "Unknown document"
                    sect = f" • {s.get('section_title')}" if s.get("section_title") else ""
                    pages = ""
                    if s.get("page_start") is not None and s.get("page_end") is not None:
                        pages = f" • p.{s['page_start']}-{s['page_end']}"
                    score = s.get("score")
                    score_txt = f" (score: {score:.3f})" if isinstance(score, (int, float)) else ""
                    st.write(f"{i}. {title}{sect}{pages}{score_txt}")

# ---------- style tweaks ----------
st.markdown(
    """
<style>
/* nicer chat bubble containers */
.stChatMessage > div {
    background: #262b36;
    border-radius: 10px;
    padding: 8px 14px;
}
/* tighten bullet spacing in answers */
ul, ol { margin-bottom: 0px; }
</style>
    """,
    unsafe_allow_html=True,
)