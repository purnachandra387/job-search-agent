"""Semantic search over the jobs table using local embeddings (fastembed).

Usage:
    python search.py build                           # embed all jobs and save the index
    python search.py "python backend in bengaluru"   # search
"""

import os
import sqlite3
import sys

import numpy as np

DB_PATH = os.getenv("JOBS_DB_PATH", "jobs.db")
INDEX_PATH = os.getenv("JOBS_INDEX_PATH", "index.npz")
MODEL_NAME = "BAAI/bge-small-en-v1.5"

_model = None
_index_cache: dict = {}


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


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def load_jobs() -> list[dict]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id, title, company, location, description FROM jobs ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def fetch_jobs(ids: list[int]) -> dict[int, dict]:
    """Fetch only the requested jobs (used at query time instead of loading the whole table)."""
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    conn = _connect()
    try:
        rows = conn.execute(
            f"SELECT id, title, company, location, description FROM jobs WHERE id IN ({placeholders})",
            [int(i) for i in ids],
        ).fetchall()
    finally:
        conn.close()
    return {r["id"]: dict(r) for r in rows}


def job_text(job: dict) -> str:
    return f"{job['title']} at {job['company']} in {job['location']}. {job['description']}"


def build_index() -> int:
    jobs = load_jobs()
    if not jobs:
        raise ValueError(f"No jobs found in {DB_PATH}. Run `python load_jobs.py sample_jobs.csv` first.")
    vectors = embed([job_text(j) for j in jobs])
    np.savez(INDEX_PATH, ids=np.array([j["id"] for j in jobs]), vectors=vectors)
    return len(jobs)


def _load_index() -> tuple[np.ndarray, np.ndarray]:
    """Load the index once and reload it only when index.npz changes on disk."""
    try:
        mtime = os.path.getmtime(INDEX_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Search index '{INDEX_PATH}' not found. Run `python search.py build` first."
        ) from None
    key = (INDEX_PATH, mtime)
    if _index_cache.get("key") != key:
        data = np.load(INDEX_PATH)
        _index_cache.update(key=key, ids=data["ids"], vectors=data["vectors"])
    return _index_cache["ids"], _index_cache["vectors"]


def rank(query_vector: np.ndarray, vectors: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Return (row_index, cosine_score) for the top k rows, best first."""
    scores = vectors @ query_vector  # vectors are unit length, so dot product = cosine similarity
    top = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in top]


def search(query: str, k: int = 5) -> list[dict]:
    k = max(1, int(k))
    ids, vectors = _load_index()
    ranked = rank(embed([query])[0], vectors, k)
    jobs = fetch_jobs([int(ids[idx]) for idx, _ in ranked])
    results = []
    for idx, score in ranked:
        job = jobs.get(int(ids[idx]))
        if job is None:  # row deleted since the index was built; rebuild the index to fix
            continue
        results.append({**job, "score": round(score, 3)})
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python search.py build | python search.py "your query"')
    if sys.argv[1] == "build":
        print(f"Indexed {build_index()} jobs into {INDEX_PATH}")
    else:
        for r in search(" ".join(sys.argv[1:])):
            print(f"{r['score']:.3f}  {r['title']} - {r['company']} ({r['location']})")
