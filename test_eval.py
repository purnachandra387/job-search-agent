import eval_search
def test_hit_at_k_and_reciprocal_rank():
    ranked = [4, 2, 6, 1, 8]
    assert eval_search.hit_at_k(ranked, 4, k=1) == 1
    assert eval_search.hit_at_k(ranked, 2, k=1) == 0
    assert eval_search.hit_at_k(ranked, 2, k=3) == 1
    assert eval_search.reciprocal_rank(ranked, 4) == 1.0
    assert eval_search.reciprocal_rank(ranked, 2) == 0.5
    assert eval_search.reciprocal_rank(ranked, 99) == 0.0
def test_evaluate_aggregates_metrics():
    sample_queries = [("q1", 10), ("q2", 20)]
    def fake_search(q, k=5):
        if q == "q1":
            return [{"id": 10}, {"id": 30}]
        return [{"id": 30}, {"id": 20}]
    metrics = eval_search.evaluate(queries=sample_queries, search_fn=fake_search, k=5)
    assert metrics["num_queries"] == 2
    assert metrics["hit@1"] == 0.5
    assert metrics["hit@5"] == 1.0
    assert metrics["mrr"] == 0.75
