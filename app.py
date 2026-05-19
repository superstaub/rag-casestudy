"""
app.py — Streamlit UI for the Hilo Trustpilot RAG demo.
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
from datetime import datetime, time
from rag import query, get_index
import chromadb

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "trustpilot_reviews"
LANGUAGES = ["All", "de", "en", "it", "fr", "es", "ru"]
LANG_LABELS = {
    "All": "All languages",
    "de": "German (DE)",
    "en": "English (EN)",
    "it": "Italian (IT)",
    "fr": "French (FR)",
    "es": "Spanish (ES)",
    "ru": "Russian (RU)",
}

st.set_page_config(
    page_title="Hilo Review Assistant",
    page_icon="💓",
    layout="wide",
)

# ── Sidebar: index stats ─────────────────────────────────────────────────────
@st.cache_data
def load_stats():
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(COLLECTION_NAME)
    all_items = collection.get(include=["metadatas"])
    metadatas = all_items["metadatas"]
    total = len(metadatas)
    lang_counts = {}
    for m in metadatas:
        lang = m.get("language", "?")
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    return total, lang_counts

with st.sidebar:
    st.title("Index Stats")
    try:
        total, lang_counts = load_stats()
        st.metric("Total reviews", total)
        st.markdown("**By language:**")
        for lang in ["de", "en", "it", "fr", "es", "ru"]:
            count = lang_counts.get(lang, 0)
            if count:
                st.text(f"  {LANG_LABELS.get(lang, lang)}: {count}")
    except Exception as e:
        st.warning(f"Could not load stats: {e}")

    st.divider()
    st.markdown(
        "**Model:** GPT-4o-mini\n\n"
        "**Embeddings:** text-embedding-3-small\n\n"
        "**Vector store:** Chroma (cosine)"
    )

# ── Main UI ──────────────────────────────────────────────────────────────────
st.title("Hilo Customer Review Assistant")
st.caption("Ask questions about Hilo customer reviews. Answers cite source review IDs.")

question = st.text_input(
    "Your question",
    placeholder='e.g. "What do customers say about accuracy?" or "Were there skin irritation reports in the past two months?"',
    label_visibility="collapsed",
)

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    lang_choice = st.selectbox(
        "Language filter",
        LANGUAGES,
        format_func=lambda x: LANG_LABELS.get(x, x),
    )
with col2:
    date_from = st.date_input("From date (optional)", value=None)
with col3:
    date_to = st.date_input("To date (optional)", value=None)

st.caption(
    "Leave date fields empty to search all time, or let the query handle it naturally "
    '(e.g. "past two months" is detected automatically).'
)

search_clicked = st.button("Search", type="primary", use_container_width=False)

# ── Handle query ─────────────────────────────────────────────────────────────
if search_clicked and question.strip():
    lang_filter = None if lang_choice == "All" else lang_choice

    # Convert UI date pickers to Unix timestamps (explicit overrides auto-detection)
    since_ts = int(datetime.combine(date_from, time.min).timestamp()) if date_from else None
    until_ts = int(datetime.combine(date_to, time.max).timestamp()) if date_to else None

    with st.spinner("Searching reviews and generating answer..."):
        try:
            result = query(
                question.strip(),
                language_filter=lang_filter,
                since_ts=since_ts,
                until_ts=until_ts,
            )
        except Exception as e:
            st.error(f"Error during query: {e}")
            st.stop()

    # Show which filters were actually applied (including auto-extracted dates)
    applied = []
    if lang_filter:
        applied.append(f"Language: {LANG_LABELS.get(lang_filter, lang_filter)}")
    df_applied = result.get("date_filter", {})
    if df_applied.get("since_ts"):
        applied.append(f"From: {datetime.fromtimestamp(df_applied['since_ts']).strftime('%d %b %Y')}")
    if df_applied.get("until_ts"):
        applied.append(f"To: {datetime.fromtimestamp(df_applied['until_ts']).strftime('%d %b %Y')}")
    if applied:
        st.caption("Filters applied: " + " | ".join(applied))

    if result["refused"]:
        st.warning(result["answer"])
    else:
        st.markdown("### Answer")
        st.markdown(result["answer"])

        sources = result["sources"]
        if sources:
            st.markdown(f"### Sources ({len(sources)} reviews)")
            for src in sources:
                stars = int(src.get("stars", 0))
                star_str = "★" * stars + "☆" * (5 - stars)
                label = (
                    f"{star_str}  |  **{src.get('username', 'Unknown')}**  |  "
                    f"{LANG_LABELS.get(src.get('language', ''), src.get('language', ''))}  |  "
                    f"{src.get('date', '')}  |  `{src.get('review_id', '')}`"
                )
                with st.expander(label):
                    st.markdown(f"**{src.get('title', '')}**")
                    st.write(src.get("content", ""))
                    st.caption(
                        f"Webshop: {src.get('webshop', '')}  |  "
                        f"Review ID: `{src.get('review_id', '')}`"
                    )

elif search_clicked and not question.strip():
    st.info("Please enter a question.")

elif not question:
    st.markdown(
        """
        **Example queries:**
        - *What are the most common complaints?*
        - *Was sagen Kunden über die Genauigkeit der Messung?*
        - *Cosa pensano i clienti dell'app?*
        - *Quels problèmes les clients ont-ils signalés ?*
        """
    )
