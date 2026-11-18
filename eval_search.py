"""Evaluate semantic search retrieval quality using Hit@K (Recall@K) and MRR.

Usage:
    python eval_search.py
"""

from typing import Callable

import search

# (query, expected job id) pairs, two natural-language queries per job in sample_jobs.csv.
BENCHMARK_QUERIES: list[tuple[str, int]] = [
    ("build python rest apis with fastapi and postgres", 1),
    ("backend internship working on a payments platform in bengaluru", 1),
    ("android app development with kotlin and jetpack compose", 2),
    ("remote mobile developer intern using firebase", 2),
    ("analyse sales numbers using sql and python dashboards", 3),
    ("business intelligence and data analytics role in gurugram", 3),
    ("train computer vision deep learning models in tensorflow", 4),
    ("machine learning model evaluation and accuracy reports", 4),
    ("react and typescript UI components for an admin dashboard", 5),
    ("remote frontend intern working on responsive web layouts", 5),
    ("help customers fix problems using server logs and sql", 6),
    ("linux troubleshooting and incident support engineer in bengaluru", 6),
    ("nlp chatbot and text classification research internship", 7),
    ("evaluate language model output quality remotely", 7),
    ("full stack node.js backend and react frontend intern", 8),
    ("software intern writing integration tests and git code reviews", 8),
]


def hit_at_k(ranked_ids: list[int], expected_id: int, k: int) -> int:
    return int(expected_id in ranked_ids[:k])


def reciprocal_rank(ranked_ids: list[int], expected_id: int) -> float:
    if expected_id not in ranked_ids:
        return 0.0
    return 1.0 / (ranked_ids.index(expected_id) + 1)


def evaluate(
    queries: list[tuple[str, int]] = BENCHMARK_QUERIES,
    search_fn: Callable[[str, int], list[dict]] = search.search,
    k: int = 5,
) -> dict[str, float]:
    hits_1 = hits_3 = hits_k = 0
    rr_sum = 0.0
    for query, expected_id in queries:
        ranked_ids = [int(r["id"]) for r in search_fn(query, k)]
        hits_1 += hit_at_k(ranked_ids, expected_id, 1)
        hits_3 += hit_at_k(ranked_ids, expected_id, 3)
        hits_k += hit_at_k(ranked_ids, expected_id, k)
        rr_sum += reciprocal_rank(ranked_ids, expected_id)
    n = len(queries)
    return {
        "num_queries": n,
        "hit@1": round(hits_1 / n, 3),
        "hit@3": round(hits_3 / n, 3),
        f"hit@{k}": round(hits_k / n, 3),
        "mrr": round(rr_sum / n, 3),
    }


def random_baseline(corpus_size: int, k: int) -> float:
    """Expected Hit@k if results were ranked at random (gives the metrics some context)."""
    return min(k, corpus_size) / corpus_size if corpus_size else 0.0


if __name__ == "__main__":
    print(f"Running retrieval evaluation on {len(BENCHMARK_QUERIES)} queries...\n")
    for q, expected in BENCHMARK_QUERIES:
        top = search.search(q, k=5)
        ids = [int(r["id"]) for r in top]
        if expected in ids:
            pos = ids.index(expected) + 1
            status = "OK  #1" if pos == 1 else f"#{pos}   "
        else:
            status = "MISS "
        top_title = top[0]["title"] if top else "no results"
        print(f"[{status}] expected {expected} | top: {top_title} | {q}")
    metrics = evaluate()
    corpus = len(search.load_jobs())
    print("\n--- Summary Metrics ---")
    print(f"Corpus size       : {corpus} jobs")
    print(f"Queries Evaluated : {metrics['num_queries']}")
    print(
        f"Hit@1 (Top-1 Acc) : {metrics['hit@1'] * 100:.1f}%   (random baseline {random_baseline(corpus, 1) * 100:.1f}%)"
    )
    print(
        f"Hit@3 (Recall@3)  : {metrics['hit@3'] * 100:.1f}%   (random baseline {random_baseline(corpus, 3) * 100:.1f}%)"
    )
    print(
        f"Hit@5 (Recall@5)  : {metrics['hit@5'] * 100:.1f}%   (random baseline {random_baseline(corpus, 5) * 100:.1f}%)"
    )
    print(f"MRR               : {metrics['mrr']:.3f}")
