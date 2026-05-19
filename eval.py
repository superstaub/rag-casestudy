"""
eval.py — Lightweight evaluation of the RAG pipeline.
Runs ~20 test queries and prints a results table.
Usage: python eval.py
"""
import time
from rag import query

# ── Eval set ─────────────────────────────────────────────────────────────────
# Format:
#   query       : the natural-language question
#   language    : optional filter ('de', 'en', etc.) or None
#   expected_ids: list of review IDs that SHOULD appear in sources (partial match OK)
#                 Empty list = just check refused=False (no ground-truth IDs)
#   expect_refuse: True if we expect the pipeline to refuse (no relevant reviews)
#
# Ground-truth IDs were manually spot-checked from the CSV.

EVAL_SET = [
    # ── English (4 queries) ──────────────────────────────────────────────────
    {
        "id": "EN-1",
        "query": "What do customers say about the accuracy of measurements?",
        "language": "en",
        "expected_ids": ["67155af05323691734ab3ccc", "671645a2b7490bff3d78833d"],
        "expect_refuse": False,
    },
    {
        "id": "EN-2",
        "query": "Are there complaints about connectivity or Bluetooth issues?",
        "language": "en",
        "expected_ids": [],
        "expect_refuse": False,
    },
    {
        "id": "EN-3",
        "query": "What do customers say about customer support response time?",
        "language": "en",
        "expected_ids": [],
        "expect_refuse": False,
    },
    {
        "id": "EN-4",
        "query": "Which reviews mention problems with the wrist strap or cuff?",
        "language": "en",
        "expected_ids": [],
        "expect_refuse": False,
    },
    # ── German (4 queries) ───────────────────────────────────────────────────
    {
        "id": "DE-1",
        "query": "Was sind die häufigsten Beschwerden der deutschen Kunden?",
        "language": "de",
        "expected_ids": [],
        "expect_refuse": False,
    },
    {
        "id": "DE-2",
        "query": "Welche Kunden loben den Kundenservice besonders?",
        "language": "de",
        "expected_ids": ["697341a358d28d9b3d9d7d93", "68b82f73166e4d89d4dfa02f"],
        "expect_refuse": False,
    },
    {
        "id": "DE-3",
        "query": "Gibt es Probleme mit der Lieferung oder dem Versand?",
        "language": "de",
        "expected_ids": [],
        "expect_refuse": False,
    },
    {
        "id": "DE-4",
        "query": "Wie bewerten Kunden die App-Anbindung des Geräts?",
        "language": "de",
        "expected_ids": [],
        "expect_refuse": False,
    },
    # ── Italian (4 queries) ──────────────────────────────────────────────────
    {
        "id": "IT-1",
        "query": "Cosa pensano i clienti italiani della precisione del dispositivo?",
        "language": "it",
        "expected_ids": [],
        "expect_refuse": False,
    },
    {
        "id": "IT-2",
        "query": "Ci sono recensioni negative sull'assistenza clienti in italiano?",
        "language": "it",
        "expected_ids": [],
        "expect_refuse": False,
    },
    {
        "id": "IT-3",
        "query": "I clienti italiani menzionano problemi con la spedizione?",
        "language": "it",
        "expected_ids": [],
        "expect_refuse": False,
    },
    {
        "id": "IT-4",
        "query": "Come valutano i clienti italiani l'applicazione mobile?",
        "language": "it",
        "expected_ids": [],
        "expect_refuse": False,
    },
    # ── French (4 queries) ───────────────────────────────────────────────────
    {
        "id": "FR-1",
        "query": "Quels problèmes les clients français ont-ils signalés avec l'appareil?",
        "language": "fr",
        "expected_ids": [],
        "expect_refuse": False,
    },
    {
        "id": "FR-2",
        "query": "Les clients français sont-ils satisfaits du service client?",
        "language": "fr",
        "expected_ids": [],
        "expect_refuse": False,
    },
    {
        "id": "FR-3",
        "query": "Y a-t-il des avis positifs sur la précision des mesures en français?",
        "language": "fr",
        "expected_ids": [],
        "expect_refuse": False,
    },
    {
        "id": "FR-4",
        "query": "Que disent les clients francophones de la livraison?",
        "language": "fr",
        "expected_ids": [],
        "expect_refuse": False,
    },
    # ── Refusal cases (2 queries) ────────────────────────────────────────────
    {
        "id": "REF-1",
        "query": "What is the weather forecast for Berlin next week?",
        "language": None,
        "expected_ids": [],
        "expect_refuse": True,
    },
    {
        "id": "REF-2",
        "query": "Empfehlen Sie mir ein gutes Restaurant in Wien?",
        "language": None,
        "expected_ids": [],
        "expect_refuse": True,
    },
]


def run_eval():
    print("Running evaluation...\n")
    results = []

    for case in EVAL_SET:
        t0 = time.time()
        try:
            result = query(case["query"], language_filter=case["language"])
        except Exception as e:
            results.append({**case, "error": str(e), "latency": 0, "pass": False})
            continue
        latency = time.time() - t0

        refused = result["refused"]
        source_ids = {s.get("review_id", "") for s in result["sources"]}

        # Evaluate
        if case["expect_refuse"]:
            passed = refused
            hit_rate = "N/A (refusal)"
        else:
            if refused:
                passed = False
                hit_rate = "0/? (wrongly refused)"
            else:
                expected = case["expected_ids"]
                if not expected:
                    # No ground truth: pass if we got at least 1 source
                    passed = len(source_ids) > 0
                    hit_rate = f"? ({len(source_ids)} sources)"
                else:
                    hits = sum(1 for eid in expected if eid in source_ids)
                    hit_rate = f"{hits}/{len(expected)}"
                    passed = hits == len(expected)

        results.append({
            "id": case["id"],
            "query": case["query"][:55] + ("…" if len(case["query"]) > 55 else ""),
            "lang_filter": case["language"] or "—",
            "refused": refused,
            "expect_refuse": case["expect_refuse"],
            "hit_rate": hit_rate,
            "sources": len(source_ids),
            "latency_s": round(latency, 1),
            "pass": passed,
        })

        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {case['id']}: {hit_rate} | {latency:.1f}s | refused={refused}")

    # Summary
    n_pass = sum(1 for r in results if r["pass"])
    n_total = len(results)
    print(f"\n{'='*60}")
    print(f"Results: {n_pass}/{n_total} passed ({100*n_pass//n_total}%)")
    print(f"{'='*60}")

    # Failure analysis
    failures = [r for r in results if not r["pass"]]
    if failures:
        print("\nFailed cases:")
        for f in failures:
            print(f"  {f['id']}: hit_rate={f['hit_rate']}, refused={f['refused']}, expect_refuse={f['expect_refuse']}")

    # Documented failure modes
    print("""
Failure modes (documented):
1. CROSS-LANGUAGE LEAKAGE — if language_filter is None, a German query may
   retrieve EN reviews and vice versa, lowering precision for language-specific
   questions. Mitigation: instruct users to set the language filter.

2. SHORT REVIEWS / LOW SIGNAL — 1–2 sentence reviews embed poorly; cosine
   similarity can miss them even when they are topically relevant.

3. THRESHOLD TOO LOW — MIN_SCORE_THRESHOLD=0.25 is conservative. A slightly
   off-topic review may slip through. Raise to 0.35 to tighten precision
   (at the cost of more refusals on niche queries).

4. PARAPHRASING MISMATCH — queries using different vocabulary than the review
   text (e.g., "precision" vs "accuracy") may miss relevant reviews.

5. REFUSAL COVERAGE — the refusal check only triggers when ALL top-k nodes
   fall below the threshold. A vague query about a tangentially related topic
   will NOT be refused even if the answer is unhelpful.
""")


if __name__ == "__main__":
    run_eval()
