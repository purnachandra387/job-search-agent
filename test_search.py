import numpy as np
import pytest

import load_jobs
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


@pytest.fixture
def indexed_db(tmp_path, monkeypatch):
    csv_path = tmp_path / "jobs.csv"
    csv_path.write_text(
        "id,title,company,location,description\n"
        "1,Backend Intern,Acme,Bengaluru,Python backend APIs\n"
        "2,Android Intern,PixelNest,Remote,Kotlin Android screens\n"
        "3,Frontend Intern,Webly,Remote,React frontend UI\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(load_jobs, "DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.setattr(search, "DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.setattr(search, "INDEX_PATH", str(tmp_path / "index.npz"))
    monkeypatch.setattr(search, "embed", fake_embed)
    load_jobs.load_csv(str(csv_path))
    return tmp_path


def test_build_and_search_end_to_end(indexed_db):
    assert search.build_index() == 3
    results = search.search("kotlin android", k=2)
    assert len(results) == 2
    assert results[0]["id"] == 2
    assert results[0]["score"] >= results[1]["score"]


def test_search_without_index_gives_actionable_error(indexed_db):
    with pytest.raises(FileNotFoundError, match="search.py build"):
        search.search("anything")


def test_load_is_idempotent(indexed_db, tmp_path):
    csv_path = tmp_path / "jobs.csv"
    load_jobs.load_csv(str(csv_path))
    assert len(search.load_jobs()) == 3
