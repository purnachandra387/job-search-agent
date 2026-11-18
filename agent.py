"""Tool-using LLM job search agent with conversation memory and Pydantic structured output.
Supports Google Gemini (GEMINI_API_KEY) or Groq (GROQ_API_KEY) over httpx.
Usage:
    python agent.py "Find me a backend internship in Bengaluru"
"""
import json
import os
import sqlite3
import sys
from typing import Any, Callable
import httpx
from pydantic import BaseModel, Field
import search
class RecommendedJob(BaseModel):
    job_id: int
    title: str
    company: str
    location: str | None = None
    match_reason: str = Field(..., description="Why this job matches the user's request")
class AgentResponse(BaseModel):
    answer: str = Field(..., description="Helpful natural-language reply to the candidate")
    recommended_jobs: list[RecommendedJob] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
SYSTEM_PROMPT = """You are a helpful AI career agent that helps candidates find matching jobs.
You have two tools:
1. search_jobs(query, k): Semantic search over the jobs database. Always call this first when a user asks for jobs.
2. get_job_details(job_id): Fetch full details for a specific job ID when the user asks about a specific role or needs deeper inspection.
When you have enough information from the tools to answer the user, respond ONLY with a valid JSON object matching this exact schema:
{
  "answer": "Clear summary advice for the user",
  "recommended_jobs": [
    {
      "job_id": 1,
      "title": "Job Title",
      "company": "Company Name",
      "location": "Location",
      "match_reason": "Specific reason from the job description"
    }
  ]
}
Only recommend jobs that were returned by your tools. Never invent job IDs or companies."""
TOOLS_OPENAI_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_jobs",
            "description": "Run semantic search over the jobs database to find relevant postings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query describing skills, role, or location"},
                    "k": {"type": "integer", "description": "Number of top matches to return (default 5)", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_job_details",
            "description": "Retrieve full job posting details for a single job_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "integer", "description": "Numeric ID of the job"},
                },
                "required": ["job_id"],
            },
        },
    },
]
SESSIONS: dict[str, list[dict[str, Any]]] = {}
def get_job_details(job_id: int) -> dict[str, Any]:
    conn = sqlite3.connect(search.DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT id, title, company, location, description FROM jobs WHERE id = ?",
        (int(job_id),),
    ).fetchone()
    conn.close()
    if row is None:
        return {"error": f"Job {job_id} not found"}
    return dict(row)
def execute_tool(name: str, args: dict[str, Any]) -> Any:
    if name == "search_jobs":
        query = str(args.get("query", ""))
        k = int(args.get("k", 5))
        return search.search(query, k=k)
    if name == "get_job_details":
        return get_job_details(int(args["job_id"]))
    raise ValueError(f"Unknown tool: {name}")
def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = [line for line in lines if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end >= start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)
def call_llm(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Call Groq or Gemini using their OpenAI-compatible chat completions endpoint."""
    groq_key = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    if groq_key:
        url = "https://api.groq.com/openai/v1/chat/completions"
        api_key = groq_key
        model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    elif gemini_key:
        url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        api_key = gemini_key
        model = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    else:
        raise RuntimeError("Set either GEMINI_API_KEY or GROQ_API_KEY in your environment.")
    payload = {
        "model": model,
        "messages": messages,
        "tools": TOOLS_OPENAI_SCHEMA,
        "tool_choice": "auto",
        "temperature": 0.1,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = httpx.post(url, headers=headers, json=payload, timeout=30.0)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]
def run_agent(
    user_message: str,
    session_id: str = "default",
    llm: Callable[[list[dict[str, Any]]], dict[str, Any]] = call_llm,
    max_steps: int = 4,
) -> AgentResponse:
    """Run the tool-calling loop, update session memory, and validate output with Pydantic."""
    history = SESSIONS.setdefault(
        session_id, [{"role": "system", "content": SYSTEM_PROMPT}]
    )
    history.append({"role": "user", "content": user_message})
    tools_used: list[str] = []
    for _ in range(max_steps):
        assistant_msg = llm(history)
        tool_calls = assistant_msg.get("tool_calls") or []
        if not tool_calls:
            raw_content = assistant_msg.get("content") or "{}"
            history.append({"role": "assistant", "content": raw_content})
            parsed = _extract_json(raw_content)
            parsed["tools_used"] = tools_used
            return AgentResponse.model_validate(parsed)
        history.append(
            {
                "role": "assistant",
                "content": assistant_msg.get("content") or "",
                "tool_calls": tool_calls,
            }
        )
        for call in tool_calls:
            fn = call["function"]
            name = fn["name"]
            args = json.loads(fn.get("arguments") or "{}")
            tools_used.append(name)
            result = execute_tool(name, args)
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": name,
                    "content": json.dumps(result),
                }
            )
    raise RuntimeError("Agent exceeded max_steps without returning a final answer")
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python agent.py "your question"')
    reply = run_agent(" ".join(sys.argv[1:]))
    print(json.dumps(reply.model_dump(), indent=2))
