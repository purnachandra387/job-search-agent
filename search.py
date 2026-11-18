"""Semantic search over the jobs table using local embeddings (fastembed).

Usage:
    python search.py build                      # embed all jobs and save the index
    python search.py "python backend in bengaluru"   # search
"""
import sqlite3
import sys

import numpy as np

DB_PATH = "jobs.db"
INDEX_PATH = "index.npz"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

_model = None


def embed(texts: list[str]) -> np.ndarray:
    """Turn texts into unit-length vectors. The model downloads once (about 130 MB)."""
    global _model
    if _model is None:
        from fastembed import TextEmbedding

        _model = TextEmbedding(MODEL_NAME)
    vectors = np.array(list(_model.embed(texts)), dtype=np.float32)
    return normalise(vectors)


def normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def load_jobs() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT id, title, company, location, description FROM jobs ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def job_text(job: dict) -> str:
    return f"{job['title']} at {job['company']} in {job['location']}. {job['description']}"


def build_index() -> int:
    jobs = load_jobs()
    vectors = embed([job_text(j) for j in jobs])
    np.savez(INDEX_PATH, ids=np.array([j["id"] for j in jobs]), vectors=vectors)
    return len(jobs)


def rank(query_vector: np.ndarray, vectors: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Return (row_index, cosine_score) for the top k rows, best first."""
    scores = vectors @ query_vector  # vectors are unit length, so dot product = cosine similarity
    top = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in top]


def search(query: str, k: int = 5) -> list[dict]:
    data = np.load(INDEX_PATH)
    jobs = {j["id"]: j for j in load_jobs()}
    results = []
    for idx, score in rank(embed([query])[0], data["vectors"], k):
        job = dict(jobs[int(data["ids"][idx])])
        job["score"] = round(score, 3)
        results.append(job)
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python search.py build | python search.py "your query"')
    if sys.argv[1] == "build":
        print(f"Indexed {build_index()} jobs into {INDEX_PATH}")
    else:
        for r in search(" ".join(sys.argv[1:])):
            print(f"{r['score']:.3f}  {r['title']} - {r['company']} ({r['location']})")
