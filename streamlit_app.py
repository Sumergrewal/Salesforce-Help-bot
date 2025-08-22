import os
import uuid
import requests
import streamlit as st
st.set_page_config(page_title="SF Help Agent", page_icon="🔎", layout="wide")

# --- Config ---
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")  # where FastAPI is running

# --- Session bootstrap ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []  # list[{"role": "user"|"assistant", "text": str, "sources": list}]

# --- Sidebar ---
with st.sidebar:
    st.markdown("## Settings")
    st.write(f"**API**: {API_BASE_URL}")
    st.write(f"**Session**: `{st.session_state.session_id}`")

    if st.button("🔄 New session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()

    st.markdown("---")
    show_sources = st.checkbox("Show sources for assistant messages", value=True)
    st.caption("Sources are pulled directly from your indexed Salesforce Help content.")

st.title("Salesforce Help Agent")
st.caption("Ask about anything in your locally indexed Salesforce Help corpus. Hybrid retrieval + grounded answers.")

# --- Chat history render ---
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
prompt = st.chat_input("Type your question about Salesforce Help…")

if prompt:
    # Show the user message immediately
    st.session_state.history.append({"role": "user", "text": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call the API
    try:
        resp = requests.post(
            f"{API_BASE_URL}/chat",
            json={"session_id": st.session_state.session_id, "message": prompt},
            timeout=60,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"API error {resp.status_code}: {resp.text}")
        data = resp.json()
        answer = data.get("answer", "").strip() or "_No answer returned._"
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
