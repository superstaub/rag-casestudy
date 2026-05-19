"""
rag.py — Query function: retrieve relevant reviews + generate a cited answer.
Usage (standalone test):
    python rag.py "Was ist der häufigste Beschwerdegrund?"
    python rag.py "Were there skin irritation reports in the past two months?"
"""
import os
import sys
import json
from datetime import datetime, date
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.vector_stores import (
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
    FilterCondition,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import chromadb
from sentence_transformers import CrossEncoder

load_dotenv()

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "trustpilot_reviews"

# Embedding-stage fast-fail: if the best cosine score across ALL candidates
# is below this, refuse immediately without calling the cross-encoder.
# Kept loose (0.20) because the cross-encoder does the real relevance filtering.
EMBED_SCORE_THRESHOLD = 0.20

# Cross-encoder reranker: multilingual MS MARCO MiniLM, covers DE/EN/IT/FR/ES/RU.
# Scores are raw logits — higher = more relevant.
# Rough scale: >3 strong match, 0–3 relevant, <0 weak, <-5 irrelevant.
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# Pull this many candidates from Chroma before reranking, then keep only top_k.
# Higher = better recall, slightly more reranker latency (runs locally, ~50ms/batch).
RETRIEVAL_MULTIPLIER = 4

# Words that suggest the query contains a date constraint.
# Used to avoid an unnecessary LLM call on queries with no date intent.
_DATE_KEYWORDS = {
    "month", "year", "week", "day", "recent", "last", "past", "since",
    "before", "after", "monat", "jahr", "woche", "mois", "an", "semaine",
    "mese", "anno", "settimana", "mes", "ano", "semana", "heute", "gestern",
    "aujourd", "ieri", "ayer", "recently", "latest", "newest",
}

SYSTEM_PROMPT = """You are a customer-insights analyst for Hilo by Aktiia.
You answer questions about customer reviews by citing the source reviews inline.

Rules:
1. Always cite review IDs inline using the format [review_id] immediately after the claim they support.
2. Answer in the same language as the user's question.
3. If the provided reviews are NOT relevant to the question, output ONLY the token [NO_RELEVANT_REVIEWS]
   followed by a one-sentence polite explanation. Do not invent information.
4. Be concise and factual. Do not pad your answer.
5. When summarizing sentiment, note the star ratings of the cited reviews.
"""


def _load_index():
    """Load the persisted Chroma index. Call once and reuse."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=api_key)
    llm = OpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.1)
    Settings.embed_model = embed_model
    Settings.llm = llm

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )
    return index


# Module-level caches so Streamlit doesn't reload on every query.
# The cross-encoder downloads ~120 MB on first use, then caches locally.
_index_cache = None
_reranker_cache = None

def get_index():
    global _index_cache
    if _index_cache is None:
        _index_cache = _load_index()
    return _index_cache

def get_reranker():
    global _reranker_cache
    if _reranker_cache is None:
        _reranker_cache = CrossEncoder(RERANKER_MODEL)
    return _reranker_cache


def extract_date_filter(question: str) -> tuple[int | None, int | None]:
    """
    Use the LLM to detect and parse any date range constraint in the question.
    Returns (since_ts, until_ts) as Unix timestamps, or (None, None) if none found.

    Only called when the question contains date-related keywords, so the extra
    LLM call is avoided for the majority of queries.
    """
    today = date.today().isoformat()
    llm = Settings.llm
    messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                f"Today is {today}. Extract any date range constraint from the user's question.\n"
                "Reply with ONLY a JSON object with keys 'since' and 'until' "
                "(ISO date strings YYYY-MM-DD, or null if not specified).\n"
                "Examples:\n"
                '  "past two months"  -> {"since": "COMPUTE_FROM_TODAY", "until": null}\n'
                '  "last year"        -> {"since": "YEAR_START", "until": "YEAR_END"}\n'
                '  "before March"     -> {"since": null, "until": "YEAR-02-28"}\n'
                '  no date constraint -> {"since": null, "until": null}'
            ),
        ),
        ChatMessage(role=MessageRole.USER, content=question),
    ]
    try:
        response = llm.chat(messages)
        raw = str(response.message.content).strip()
        # Strip markdown code fences if the model adds them
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        data = json.loads(raw)

        since_ts = (
            int(datetime.strptime(data["since"], "%Y-%m-%d").timestamp())
            if data.get("since") else None
        )
        until_ts = (
            int(datetime.strptime(data["until"], "%Y-%m-%d").timestamp())
            if data.get("until") else None
        )
        return since_ts, until_ts
    except Exception:
        return None, None


def query(
    question: str,
    language_filter: str | None = None,
    top_k: int = 8,
    since_ts: int | None = None,
    until_ts: int | None = None,
) -> dict:
    """
    Run a RAG query against the review index.

    Args:
        question:        Natural-language question from the user.
        language_filter: Optional ISO language code ('de', 'en', 'it', 'fr', 'es', 'ru').
        top_k:           Number of reviews to return after reranking (default 8).
        since_ts:        Optional start of date range as Unix timestamp.
        until_ts:        Optional end of date range as Unix timestamp.
                         If both are None and the question contains date keywords,
                         the date range is extracted automatically from the question.

    Returns:
        {
            "answer":      str  — generated answer (or refusal message),
            "sources":     list — list of metadata dicts for cited reviews,
            "refused":     bool — True if no relevant reviews were found,
            "date_filter": dict — {"since_ts": int|None, "until_ts": int|None}
                                  (reflects the filter actually applied, including
                                   any auto-extracted range)
        }
    """
    get_index()  # ensures LLM + embed_model are initialised

    # Auto-extract date range from natural language if not given explicitly.
    # The keyword check avoids an extra LLM call on queries with no date intent.
    if since_ts is None and until_ts is None:
        words = set(question.lower().split())
        if words & _DATE_KEYWORDS:
            since_ts, until_ts = extract_date_filter(question)

    # ── Build metadata filters ────────────────────────────────────────────────
    filter_list = []
    if language_filter and language_filter.lower() != "all":
        filter_list.append(
            MetadataFilter(key="language", value=language_filter.lower(),
                           operator=FilterOperator.EQ)
        )
    if since_ts is not None:
        filter_list.append(
            MetadataFilter(key="date_ts", value=since_ts,
                           operator=FilterOperator.GTE)
        )
    if until_ts is not None:
        filter_list.append(
            MetadataFilter(key="date_ts", value=until_ts,
                           operator=FilterOperator.LTE)
        )
    filters = (
        MetadataFilters(filters=filter_list, condition=FilterCondition.AND)
        if filter_list else None
    )

    # ── Stage 1: broad embedding retrieval ───────────────────────────────────
    index = get_index()
    retriever = index.as_retriever(
        similarity_top_k=top_k * RETRIEVAL_MULTIPLIER,
        filters=filters,
    )
    candidates = retriever.retrieve(question)

    # Fast-fail: if even the best embedding score is below the floor threshold
    # the query is completely off-topic — skip the cross-encoder and refuse.
    best_embed_score = max(
        (n.score for n in candidates if n.score is not None), default=0.0
    )
    if best_embed_score < EMBED_SCORE_THRESHOLD:
        return {
            "answer": (
                "I could not find any reviews relevant to your question. "
                "Please try rephrasing or broadening your query."
            ),
            "sources": [],
            "refused": True,
            "date_filter": {"since_ts": since_ts, "until_ts": until_ts},
        }

    # ── Stage 2: cross-encoder reranking ─────────────────────────────────────
    reranker = get_reranker()
    pairs = [(question, n.get_content()) for n in candidates]
    ce_scores = reranker.predict(pairs)

    ranked = sorted(zip(ce_scores, candidates), key=lambda x: x[0], reverse=True)
    relevant_nodes = [node for _, node in ranked[:top_k]]

    # ── Build LLM context ────────────────────────────────────────────────────
    context_parts = []
    for node in relevant_nodes:
        meta = node.metadata
        context_parts.append(
            f"Review ID: {meta.get('review_id', 'N/A')}\n"
            f"Stars: {meta.get('stars', '?')}/5 | Language: {meta.get('language', '?')} | "
            f"Date: {meta.get('date', '?')} | User: {meta.get('username', '?')}\n"
            f"Title: {meta.get('title', '')}\n"
            f"Content: {meta.get('content', node.get_content())}\n"
        )
    context = "\n---\n".join(context_parts)

    prompt = (
        f"Based only on the following customer reviews, answer the question.\n\n"
        f"REVIEWS:\n{context}\n\n"
        f"QUESTION: {question}"
    )

    llm = Settings.llm
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
        ChatMessage(role=MessageRole.USER, content=prompt),
    ]
    response = llm.chat(messages)
    answer = str(response.message.content)

    # LLM signals refusal with a structured token (reliable vs. substring matching)
    llm_refused = answer.strip().startswith("[NO_RELEVANT_REVIEWS]")
    if llm_refused:
        answer = answer.replace("[NO_RELEVANT_REVIEWS]", "").strip()

    sources = [n.metadata for n in relevant_nodes]
    return {
        "answer": answer,
        "sources": [] if llm_refused else sources,
        "refused": llm_refused,
        "date_filter": {"since_ts": since_ts, "until_ts": until_ts},
    }


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "What do customers say about accuracy?"
    print(f"Query: {question}\n")
    result = query(question)
    print("Answer:\n", result["answer"])
    df = result["date_filter"]
    if df["since_ts"] or df["until_ts"]:
        since = datetime.fromtimestamp(df["since_ts"]).date() if df["since_ts"] else "—"
        until = datetime.fromtimestamp(df["until_ts"]).date() if df["until_ts"] else "—"
        print(f"\nDate filter applied: from {since} to {until}")
    print(f"\nSources ({len(result['sources'])}):")
    for s in result["sources"]:
        print(f"  [{s.get('review_id')}] {s.get('stars')}* ({s.get('language')}) "
              f"{s.get('date')} — {s.get('username')}")
