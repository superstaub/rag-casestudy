# Hilo Trustpilot RAG — Case Study Prototype

A multilingual RAG (Retrieval-Augmented Generation) system over 3,548 Hilo Trustpilot reviews.  
Supports DE / EN / IT / FR / ES / RU queries. Every answer cites source review IDs.

---

## Stack

| Component | Technology |
|---|---|
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | OpenAI `gpt-4o-mini` |
| Vector store | Chroma (persistent, cosine similarity) |
| RAG framework | LlamaIndex |
| UI | Streamlit |

---

## Setup

### 1. Prerequisites
- Python 3.11+
- An OpenAI API key

### 2. Obtain the dataset
The CSV file is not included in the repository. Place `trustpilot.csv` (UTF-16 LE, tab-separated) in the project root before running `build_index.py`.

### 3. Create virtual environment
```bash
cd "Case Study"
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 4. Install dependencies
```bash
pip install llama-index llama-index-embeddings-openai llama-index-llms-openai \
            llama-index-vector-stores-chroma chromadb pandas streamlit python-dotenv \
            sentence-transformers
```

### 5. Set your API key
Create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-...
```

### 6. Build the index (one-time, ~$0.01)
```bash
python build_index.py
```
This embeds all 3,548 reviews and stores them in `./chroma_db`.  
Re-running deletes and recreates the collection (idempotent).

---

## Running

### Streamlit UI
```bash
streamlit run app.py
```
Opens at http://localhost:8501

### CLI query (quick test)
```bash
python rag.py "What do customers say about accuracy?"
python rag.py "Was sagen Kunden über den Kundenservice?"
```

### Evaluation suite
```bash
python eval.py
```
Runs 18 test queries (4× EN, DE, IT, FR + 2 refusal cases) and prints pass/fail results.

---

## Project structure

```
.
├── build_index.py   # Load CSV → embed → persist in Chroma
├── rag.py           # query() function: retrieve + generate with citations
├── app.py           # Streamlit UI
├── eval.py          # Evaluation harness (18 test queries)
├── cost_model.md    # Token cost breakdown + monthly projection
├── trustpilot.csv   # Source data — not in repo, must be provided separately
├── chroma_db/       # Persistent vector store — not in repo, created by build_index.py
├── .env             # OPENAI_API_KEY — not in repo
└── venv/            # Python virtual environment — not in repo
```

---

## Key design decisions

**Two-stage retrieval with cross-encoder reranking**  
The embedding model (`text-embedding-3-small`) retrieves 32 candidates (4× top-k) as a broad first pass. A local multilingual cross-encoder (`mmarco-mMiniLMv2-L12-H384-v1`) then re-scores each (query, review) pair by meaning rather than vector distance, and the best 8 are kept. This eliminates language-clustering bias: a French query can surface the most relevant German review without penalising it for being in a different language.

**Embedding fast-fail threshold (0.20)**  
If the best cosine similarity score across all 32 candidates is below 0.20, the query is completely off-topic and the pipeline refuses immediately without calling the cross-encoder or LLM. A second refusal layer exists at LLM level via a structured `[NO_RELEVANT_REVIEWS]` token.

**Metadata exclusion from embedding prefix**  
`content`, `title`, and `date_ts` fields are excluded from the LlamaIndex metadata prefix that gets prepended to document text during embedding. They are stored in metadata for citation cards and filtering but not double-counted in the vector representation.

**Date range filtering**  
Reviews are indexed with a `date_ts` Unix timestamp field, enabling Chroma pre-filters (`>=`, `<=`) before retrieval. Date ranges can be set via the UI or expressed in natural language (e.g. "past two months") — the pipeline detects date keywords and makes a small LLM call to parse the range automatically.

**Language filter**  
Uses LlamaIndex `MetadataFilter` on the `language` metadata field. Selecting a language restricts the Chroma search space to that language's reviews before retrieval.

**Answer language**  
The system prompt instructs GPT-4o-mini to answer in the user's query language. No translation step needed — the model handles this natively.

---

## Cost

See [cost_model.md](cost_model.md) for full breakdown.

| Scenario | Monthly cost |
|---|---|
| 20 employees × 5 queries/day × 22 days | **~$0.73** |

---

## Documented failure modes

See the bottom of `eval.py` for documented failure modes and mitigations.
