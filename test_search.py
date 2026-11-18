import numpy as np

import search


def fake_embed(texts):
    """Tiny bag-of-words embedder so tests run fast, offline and without the real model."""
    vocab = ["python", "kotlin", "react", "sql", "android", "backend", "frontend"]
    vecs = np.array([[t.lower().count(w) for w in vocab] for t in texts], dtype=np.float32)
    return search.normalise(vecs)


def test_rank_returns_best_match_first():
    vectors = fake_embed(["python backend api", "kotlin android app", "react frontend"])
    query = fake_embed(["android kotlin"])[0]
    ranked = search.rank(query, vectors, k=3)
    assert ranked[0][0] == 1
    assert ranked[0][1] > ranked[1][1]


def test_rank_respects_k():
    vectors = fake_embed(["python", "sql", "react", "kotlin"])
    assert len(search.rank(fake_embed(["python"])[0], vectors, k=2)) == 2
