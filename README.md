# AI Job Search Agent
An LLM-powered career assistant and semantic job search engine built with Python, FastAPI, SQLite, FastEmbed (BAAI/bge-small-en-v1.5), and Pydantic.
## Features
- Idempotent SQLite Job Loader (load_jobs.py): Ingests job postings (id, title, company, location, description) into SQLite using INSERT OR REPLACE so repeated loads never duplicate rows.
- Dense Vector Semantic Search (search.py): Encodes job postings into 384-dimensional unit-normalised embeddings using BAAI/bge-small-en-v1.5 (fastembed) and ranks candidates by cosine similarity.
- Tool-Using LLM Career Agent (agent.py): Calls search_jobs and get_job_details, preserves multi-turn conversation context per session_id, and validates structured JSON outputs with Pydantic (AgentResponse). Supports both Google Gemini (GEMINI_API_KEY) and Groq (GROQ_API_KEY).
- FastAPI Backend (app.py): Exposes GET /health, GET /search, GET /jobs/{job_id}, and POST /agent with automatic OpenAPI documentation.
- Retrieval Evaluation Suite (eval_search.py): Benchmarks semantic search across 16 natural-language queries using Hit@1, Hit@3, Hit@5 (Recall@5), and MRR (Mean Reciprocal Rank).
## Quickstart
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    python load_jobs.py sample_jobs.csv
    python search.py build
    python -m pytest -q
    python eval_search.py
    python -m uvicorn app:app --reload
## Using the LLM Agent
    $env:GEMINI_API_KEY="your-api-key"
    python agent.py "Find me a Python backend internship in Bengaluru"
## Docker
    docker build -t job-search-agent .
    docker run -p 8000:8000 -e GEMINI_API_KEY="your-api-key" job-search-agent
