"""
build_index.py — Load Trustpilot CSV, embed reviews, store in persistent Chroma.
Run once (or re-run to rebuild): python build_index.py
Idempotent: deletes and recreates the collection on each run.
"""
import os
import sys
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

load_dotenv()

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "trustpilot_reviews"
CSV_PATH = "trustpilot.csv"

# ── 1. Load CSV ──────────────────────────────────────────────────────────────
print("Loading CSV...")
df = pd.read_csv(CSV_PATH, sep="\t", encoding="utf-16")
print(f"  Loaded {len(df)} rows")

# Fix the one Lithuanian review that is actually English
lt_count = (df["Review Language"] == "lt").sum()
if lt_count:
    df.loc[df["Review Language"] == "lt", "Review Language"] = "en"
    print(f"  Relabeled {lt_count} 'lt' row(s) to 'en'")

print("\nLanguage breakdown:")
for lang, cnt in df["Review Language"].value_counts().items():
    print(f"  {lang}: {cnt}")

# ── 2. Build LlamaIndex Documents ────────────────────────────────────────────
print(f"\nBuilding {len(df)} documents...")
documents = []
for _, row in df.iterrows():
    title = str(row["Review Title"]).strip()
    content = str(row["Review Content"]).strip()
    # Include star rating in text so embedding captures sentiment context
    text = f"[{row['Review Stars']} stars] {title}\n{content}"
    date_str = str(row["Review Created"]).strip()
    try:
        date_ts = int(datetime.strptime(date_str, "%Y-%m-%d %H:%M").timestamp())
    except ValueError:
        date_ts = 0

    metadata = {
        "review_id": str(row["Review Id"]).strip(),
        "language":  str(row["Review Language"]).strip(),
        "stars":     int(row["Review Stars"]),
        "date":      date_str,
        "date_ts":   date_ts,   # Unix timestamp — enables range filtering in Chroma
        "username":  str(row["Review Username"]).strip(),
        "webshop":   str(row["Webshop Name"]).strip(),
        # Store original text in metadata for citation cards
        "title":     title,
        "content":   content,
    }
    documents.append(Document(
        text=text,
        metadata=metadata,
        # content/title are already in the text; exclude from metadata prefix
        # to avoid exceeding default chunk size during embedding
        excluded_embed_metadata_keys=["content", "title", "date_ts"],
        excluded_llm_metadata_keys=["content", "date_ts"],
    ))

print(f"  Created {len(documents)} documents")

# ── 3. Set up Chroma (delete + recreate = idempotent) ────────────────────────
print(f"\nSetting up Chroma at '{CHROMA_PATH}'...")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

try:
    chroma_client.delete_collection(COLLECTION_NAME)
    print(f"  Deleted existing collection '{COLLECTION_NAME}'")
except Exception:
    pass

collection = chroma_client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)
print(f"  Created collection '{COLLECTION_NAME}'")

# ── 4. Embed and index ───────────────────────────────────────────────────────
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY not set in .env")
    sys.exit(1)

embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=api_key)
Settings.embed_model = embed_model
Settings.llm = None  # no LLM needed during indexing

vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Estimated cost: ~150 tokens/review × 3548 × $0.02/1M ≈ $0.011
print("\nEmbedding and indexing (estimated cost: ~$0.01)...")
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True,
)

print(f"\nDone. Index stored at '{CHROMA_PATH}'.")
print(f"Total reviews indexed: {len(documents)}")
