"""FastAPI endpoints for semantic job search, job details, and the LLM career agent.

Usage:
    uvicorn app:app --reload
"""

import sqlite3

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import agent
import search

app = FastAPI(title="Job Search Agent API", version="0.3.0")


class JobSummary(BaseModel):
    id: int
    title: str
    company: str
    location: str | None = None
    description: str
    score: float = Field(..., description="Cosine similarity score")


class JobDetail(BaseModel):
    id: int
    title: str
    company: str
    location: str | None = None
    description: str


class AgentRequest(BaseModel):
    message: str = Field(
        ..., min_length=1, max_length=2000, description="Candidate question or job preference"
    )
    session_id: str = Field(
        default="default", min_length=1, max_length=64, description="Session ID for conversation memory"
    )


def get_job_by_id(job_id: int) -> dict | None:
    conn = sqlite3.connect(search.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT id, title, company, location, description FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


# --- Error mapping: turn infrastructure failures into meaningful HTTP codes, not bare 500s ---
@app.exception_handler(agent.AgentConfigError)
def config_error_handler(request: Request, exc: agent.AgentConfigError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(agent.AgentError)
def agent_error_handler(request: Request, exc: agent.AgentError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(FileNotFoundError)
def missing_index_handler(request: Request, exc: FileNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/search", response_model=list[JobSummary])
def search_jobs(
    q: str = Query(..., min_length=1, description="Natural-language search query"),
    k: int = Query(5, ge=1, le=20, description="Maximum number of results to return"),
) -> list[dict]:
    return search.search(q, k=k)


@app.get("/jobs/{job_id}", response_model=JobDetail)
def get_job_details(job_id: int) -> dict:
    job = get_job_by_id(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


@app.post("/agent", response_model=agent.AgentResponse)
def chat_with_agent(req: AgentRequest) -> agent.AgentResponse:
    return agent.run_agent(req.message, session_id=req.session_id)
