# Cost Model — Hilo RAG Prototype

## Pricing (as of May 2026)

| Component | Model | Price |
|---|---|---|
| Embeddings (query) | text-embedding-3-small | $0.02 / 1M tokens |
| Generation | gpt-4o-mini (input) | $0.15 / 1M tokens |
| Generation | gpt-4o-mini (output) | $0.60 / 1M tokens |

---

## One-time indexing cost (already paid)

| Item | Tokens | Cost |
|---|---|---|
| 3,548 reviews × ~150 tokens avg | ~532,000 | **~$0.011** |

---

## Per-query cost breakdown

### Standard query (no date filter)

**Step 1 — Query embedding**
- Query text: ~15 tokens → **$0.000000030** (negligible)

**Step 2 — LLM generation**

| Token type | Tokens | Cost |
|---|---|---|
| Input (system prompt ~120 + 8 reviews ~1,200 + question ~15) | 1,335 | $0.000200 |
| Output (average answer) | 200 | $0.000120 |
| **Total** | | **~$0.000320** |

**Rounded: ~$0.0003 per query**

---

### Query with natural language date filter

When the question contains date keywords (e.g. "past two months"), an additional
LLM call extracts the date range before retrieval.

| Step | Tokens | Cost |
|---|---|---|
| Date extraction — input (~220 tokens system + ~15 query) | 235 | $0.000035 |
| Date extraction — output (JSON, ~20 tokens) | 20 | $0.000012 |
| Standard query (as above) | — | $0.000320 |
| **Total** | | **~$0.000367** |

**Rounded: ~$0.00037 per date-filtered query** (~15% more than a standard query)

Assuming ~20% of queries contain date intent:
- Blended cost = (0.8 × $0.000320) + (0.2 × $0.000367) = **~$0.000330 per query**

---

## Monthly projection

**Usage assumptions:**
- 20 employees
- 5 queries/day per employee
- 22 working days/month

| Metric | Value |
|---|---|
| Queries/month | 20 × 5 × 22 = **2,200** |
| Blended cost/query | $0.000330 |
| **Monthly total** | **~$0.73** |

**Annual estimate: ~$8.70**

---

## Sensitivity analysis

| Scenario | Queries/month | Monthly cost |
|---|---|---|
| Conservative (10 users, 3q/day) | 660 | $0.22 |
| Base case (20 users, 5q/day) | 2,200 | **$0.73** |
| Heavy use (50 users, 10q/day) | 11,000 | $3.63 |

---

## Notes

1. **Re-indexing is negligible** — the $0.011 one-time cost only recurs if the review dataset is rebuilt from scratch.
2. **Context window scales with top-k** — reducing `top_k` from 8 to 4 roughly halves LLM input tokens and saves ~$0.0001/query.
3. **GPT-4o (full)** would cost ~15× more on input tokens ($2.50/1M vs $0.15/1M). For a prototype, gpt-4o-mini is the right choice.
4. **Token counts are estimates** — actual usage varies with review length (German reviews tend to be longer, averaging ~200 tokens).
5. **Cross-encoder reranker adds $0.00 API cost** — the model (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, ~120 MB) runs locally on CPU. One-time download only; no per-query fee. The embedding retrieval now fetches 32 candidates instead of 8, but at $0.02/1M tokens that delta is ~$0.000000006/query — immeasurable. In production at scale, replacing the local model with a managed reranking API (e.g. Cohere, Voyage AI) would cost ~$1–2/1M pairs scored — roughly **$0.07–0.14/month** at base-case usage (2,200 queries × 32 candidates).
