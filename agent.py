"""Tool-using LLM job search agent with conversation memory and Pydantic structured output.

Supports Google Gemini (GEMINI_API_KEY) or Groq (GROQ_API_KEY) over httpx.

Usage:
    python agent.py "Find me a backend internship in Bengaluru"
"""

import json
import os
import sqlite3
import sys
import threading
from collections import OrderedDict
from typing import Any, Callable

import httpx
from pydantic import BaseModel, Field, ValidationError

import search

MAX_SESSIONS = 200  # least-recently-used sessions are evicted beyond this
MAX_HISTORY_MESSAGES = 40  # per session, older turns are dropped
MAX_TOOL_K = 20
MAX_TOOL_DESCRIPTION_CHARS = 600  # keeps tool results (and token usage) small


class AgentError(RuntimeError):
    """Base class for agent failures."""


class AgentConfigError(AgentError):
    """Missing or invalid configuration (e.g. no API key)."""


class AgentUpstreamError(AgentError):
    """The LLM provider failed or returned something unusable."""


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
                    "query": {
                        "type": "string",
                        "description": "Search query describing skills, role, or location",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of top matches to return (default 5)",
                        "default": 5,
                    },
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

# --- Session memory (in-process, bounded, thread-safe) ---------------------------------------
SESSIONS: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict()
_session_locks: dict[str, threading.Lock] = {}
_table_lock = threading.Lock()


def _get_session(session_id: str) -> tuple[list[dict[str, Any]], threading.Lock]:
    with _table_lock:
        history = SESSIONS.get(session_id)
        if history is None:
            history = [{"role": "system", "content": SYSTEM_PROMPT}]
            SESSIONS[session_id] = history
        SESSIONS.move_to_end(session_id)
        lock = _session_locks.setdefault(session_id, threading.Lock())
        while len(SESSIONS) > MAX_SESSIONS:
            evicted, _ = SESSIONS.popitem(last=False)
            _session_locks.pop(evicted, None)
        return history, lock


def _trim_history(history: list[dict[str, Any]]) -> None:
    """Drop the oldest turns, always cutting at a user message so tool-call pairs stay intact."""
    if len(history) <= MAX_HISTORY_MESSAGES:
        return
    cut = len(history) - (MAX_HISTORY_MESSAGES - 1)
    while cut < len(history) and history[cut]["role"] != "user":
        cut += 1
    if cut < len(history):
        del history[1:cut]


# --- Tools -----------------------------------------------------------------------------------
def get_job_details(job_id: int) -> dict[str, Any]:
    jobs = search.fetch_jobs([int(job_id)])
    return jobs.get(int(job_id)) or {"error": f"Job {job_id} not found"}


def execute_tool(name: str, args: dict[str, Any]) -> Any:
    """Run a tool. Errors are returned to the model as {"error": ...} so it can recover."""
    try:
        if name == "search_jobs":
            query = str(args.get("query", "")).strip()
            if not query:
                return {"error": "query must not be empty"}
            k = min(max(int(args.get("k", 5)), 1), MAX_TOOL_K)
            results = search.search(query, k=k)
            return [
                {**r, "description": str(r.get("description", ""))[:MAX_TOOL_DESCRIPTION_CHARS]}
                for r in results
            ]
        if name == "get_job_details":
            return get_job_details(int(args["job_id"]))
        return {"error": f"Unknown tool: {name}"}
    except (KeyError, TypeError, ValueError, FileNotFoundError, sqlite3.Error) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _seen_job_ids(history: list[dict[str, Any]]) -> set[int]:
    """Every job id that a tool actually returned in this session."""
    seen: set[int] = set()
    for msg in history:
        if msg.get("role") != "tool":
            continue
        try:
            payload = json.loads(msg.get("content") or "null")
        except ValueError:
            continue
        for item in payload if isinstance(payload, list) else [payload]:
            if isinstance(item, dict) and "error" not in item and "id" in item:
                try:
                    seen.add(int(item["id"]))
                except (TypeError, ValueError):
                    pass
    return seen


# --- LLM plumbing ----------------------------------------------------------------------------
def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = [line for line in cleaned.splitlines() if not line.strip().startswith("```")]
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
        raise AgentConfigError("Set either GEMINI_API_KEY or GROQ_API_KEY in your environment.")
    payload = {
        "model": model,
        "messages": messages,
        "tools": TOOLS_OPENAI_SCHEMA,
        "tool_choice": "auto",
        "temperature": 0.1,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        resp = httpx.post(url, headers=headers, json=payload, timeout=30.0)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]
    except httpx.HTTPStatusError as exc:
        raise AgentUpstreamError(f"LLM provider returned HTTP {exc.response.status_code}") from exc
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
        raise AgentUpstreamError(f"LLM request failed: {type(exc).__name__}") from exc


def _finalise(raw: str, tools_used: list[str], history: list[dict[str, Any]]) -> AgentResponse:
    parsed = _extract_json(raw)
    if not isinstance(parsed, dict):
        raise ValueError("expected a JSON object")
    parsed["tools_used"] = list(tools_used)
    reply = AgentResponse.model_validate(parsed)
    # Guardrail: drop any recommendation whose job_id no tool ever returned (hallucination check).
    seen = _seen_job_ids(history)
    reply.recommended_jobs = [j for j in reply.recommended_jobs if j.job_id in seen]
    return reply


def run_agent(
    user_message: str,
    session_id: str = "default",
    llm: Callable[[list[dict[str, Any]]], dict[str, Any]] = call_llm,
    max_steps: int = 5,
) -> AgentResponse:
    """Run the tool-calling loop, update session memory, and validate output with Pydantic."""
    history, lock = _get_session(session_id)
    with lock:  # one in-flight request per session keeps the message history consistent
        _trim_history(history)
        start_len = len(history)
        history.append({"role": "user", "content": user_message})
        try:
            return _run_loop(history, llm, max_steps)
        except Exception:
            del history[start_len:]  # roll back so a failed turn doesn't poison the session
            raise


def _run_loop(
    history: list[dict[str, Any]],
    llm: Callable[[list[dict[str, Any]]], dict[str, Any]],
    max_steps: int,
) -> AgentResponse:
    tools_used: list[str] = []
    repaired = False
    for _ in range(max_steps):
        assistant_msg = llm(history)
        tool_calls = assistant_msg.get("tool_calls") or []
        if not tool_calls:
            raw = assistant_msg.get("content") or ""
            history.append({"role": "assistant", "content": raw})
            try:
                return _finalise(raw, tools_used, history)
            except (ValueError, ValidationError) as exc:  # JSONDecodeError is a ValueError
                if repaired:
                    raise AgentUpstreamError("Model did not return valid structured JSON") from exc
                repaired = True
                history.append(
                    {
                        "role": "user",
                        "content": "Your last reply was not valid JSON matching the schema. "
                        "Reply again with ONLY the JSON object.",
                    }
                )
                continue
        history.append(
            {"role": "assistant", "content": assistant_msg.get("content") or "", "tool_calls": tool_calls}
        )
        for call in tool_calls:
            fn = call["function"]
            name = fn["name"]
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {}
            tools_used.append(name)
            result = execute_tool(name, args if isinstance(args, dict) else {})
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": name,
                    "content": json.dumps(result),
                }
            )
    raise AgentUpstreamError("Agent exceeded max_steps without returning a final answer")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python agent.py "your question"')
    try:
        reply = run_agent(" ".join(sys.argv[1:]))
    except AgentError as exc:
        sys.exit(f"Agent error: {exc}")
    print(json.dumps(reply.model_dump(), indent=2))
