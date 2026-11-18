# AI Job Search Agent

An LLM-powered career assistant and semantic job search engine built with Python, FastAPI, SQLite, FastEmbed (`BAAI/bge-small-en-v1.5`) and Pydantic.

## How it works

```
CSV / public job APIs ──► SQLite (jobs.db) ──► embeddings (index.npz, 384-d, unit-normalised)
                                                      │
                          GET /search ◄───────────────┤  cosine similarity (dot product)
                          POST /agent ──► LLM ◄─tools─┘  search_jobs / get_job_details
```

## Features

- **Idempotent job loader** (`load_jobs.py`): ingests postings into SQLite with `INSERT OR REPLACE`, so re-running never duplicates rows.
- **Dense vector semantic search** (`search.py`): embeds postings locally and ranks them by cosine similarity. The index is cached in memory and reloaded only when `index.npz` changes; only the top-k rows are read from SQLite per query.
- **Tool-using LLM career agent** (`agent.py`):
  - calls `search_jobs` and `get_job_details` in a bounded tool loop;
  - keeps multi-turn memory per `session_id` (LRU-capped sessions, trimmed history, one in-flight request per session);
  - validates the final answer with Pydantic (`AgentResponse`), retries once if the model returns malformed JSON, and **drops any recommended `job_id` that no tool actually returned** (hallucination guardrail);
  - tool failures are fed back to the model instead of crashing the loop.
- **FastAPI backend** (`app.py`): `GET /health`, `GET /search`, `GET /jobs/{job_id}`, `POST /agent`, with OpenAPI docs at `/docs`. Failures map to meaningful codes: `503` for a missing API key or search index, `502` for LLM provider errors.
- **Retrieval evaluation** (`eval_search.py`): benchmarks search on 16 natural-language queries with Hit@1, Hit@3, Hit@5 (Recall@5) and MRR, and prints the random-ranking baseline next to each metric for context.
- **Real data** (`fetch_remotive_jobs.py`): optionally pulls hundreds of live postings from the Remotive and Arbeitnow APIs. Ids are derived from company/title/location, so re-runs update rather than duplicate.

## Quickstart

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python load_jobs.py sample_jobs.csv
python search.py build
python -m pytest -q
python eval_search.py
python -m uvicorn app:app --reload
```

macOS / Linux: use `source .venv/bin/activate` instead of the `Activate.ps1` line.

The first `search.py build` downloads the embedding model (about 130 MB). Tests use a tiny fake embedder and run offline.

Try it: open <http://127.0.0.1:8000/docs>, or

```bash
curl "http://127.0.0.1:8000/search?q=python%20backend%20internship%20in%20bengaluru&k=3"
```

## Using the LLM agent

```powershell
$env:GEMINI_API_KEY="your-api-key"      # or: $env:GROQ_API_KEY="your-api-key"
python agent.py "Find me a Python backend internship in Bengaluru"
```

or over HTTP:

```bash
curl -X POST http://127.0.0.1:8000/agent -H "Content-Type: application/json" \
  -d '{"message": "Find me a Python backend internship in Bengaluru", "session_id": "me"}'
```

| Variable | Purpose | Default |
| --- | --- | --- |
| `GROQ_API_KEY` | Use Groq (takes priority if both keys are set) | – |
| `GEMINI_API_KEY` | Use Google Gemini | – |
| `LLM_MODEL` | Override the model name | `llama-3.3-70b-versatile` (Groq) / `gemini-2.5-flash` (Gemini) |
| `JOBS_DB_PATH` | SQLite file location | `jobs.db` |
| `JOBS_INDEX_PATH` | Vector index location | `index.npz` |

## Evaluation

```powershell
python eval_search.py
```

The sample corpus has only 8 jobs, so a random ranking already scores 62.5% Hit@5; read the metrics next to the printed baseline. To stress-test retrieval on a larger corpus:

```powershell
python fetch_remotive_jobs.py
python search.py build
python eval_search.py
```

## Docker

```bash
docker build -t job-search-agent .
docker run -p 8000:8000 -e GEMINI_API_KEY="your-api-key" job-search-agent
```

The image bakes in the database, the embedding model and the index, runs as a non-root user, and exposes a `/health` healthcheck.

## Project layout

| File | Role |
| --- | --- |
| `load_jobs.py` | CSV → SQLite loader |
| `fetch_remotive_jobs.py` | Pull live postings from public APIs |
| `search.py` | Embedding, index build, semantic search |
| `agent.py` | Tool-calling LLM agent, session memory, output validation |
| `app.py` | FastAPI service |
| `eval_search.py` | Retrieval metrics |
| `test_*.py` | Offline unit tests (fake embedder and fake LLM) |

## Known limitations

- Sessions are held in process memory: they reset on restart and are not shared across workers. Use Redis or a database for multi-worker deployments.
- Search is exact brute-force cosine similarity, which is fine up to tens of thousands of postings; use an ANN index (FAISS, sqlite-vec, pgvector) beyond that.
- The guardrail removes invented job ids from `recommended_jobs`, but the free-text `answer` is still model-written.
