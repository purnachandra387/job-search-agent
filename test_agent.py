import json

import pytest

import agent


def test_agent_executes_tool_and_returns_validated_response(monkeypatch):
    agent.SESSIONS.clear()
    fake_jobs = [
        {
            "id": 1,
            "title": "Backend Engineer Intern",
            "company": "Acme Pay",
            "location": "Bengaluru",
            "description": "Build Python REST APIs with FastAPI and PostgreSQL.",
            "score": 0.812,
        }
    ]
    monkeypatch.setattr(agent.search, "search", lambda query, k=5: fake_jobs)
    responses = [
        {
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "search_jobs",
                        "arguments": json.dumps({"query": "python backend bengaluru", "k": 3}),
                    },
                }
            ],
        },
        {
            "content": json.dumps(
                {
                    "answer": "Acme Pay has a Backend Engineer Intern role in Bengaluru using FastAPI.",
                    "recommended_jobs": [
                        {
                            "job_id": 1,
                            "title": "Backend Engineer Intern",
                            "company": "Acme Pay",
                            "location": "Bengaluru",
                            "match_reason": "Matches Python, FastAPI, and Bengaluru location.",
                        }
                    ],
                }
            )
        },
    ]

    def fake_llm(messages):
        return responses.pop(0)

    result = agent.run_agent(
        "Find me a Python backend internship in Bengaluru",
        session_id="test_session",
        llm=fake_llm,
    )
    assert result.tools_used == ["search_jobs"]
    assert len(result.recommended_jobs) == 1
    assert result.recommended_jobs[0].company == "Acme Pay"


def test_agent_remembers_context_across_turns():
    agent.SESSIONS.clear()
    captured_lengths = []

    def fake_llm(messages):
        captured_lengths.append(len(messages))
        return {
            "content": json.dumps(
                {
                    "answer": "Follow-up response using session history.",
                    "recommended_jobs": [],
                }
            )
        }

    agent.run_agent("First message", session_id="mem_test", llm=fake_llm)
    agent.run_agent("Second message referring to the first", session_id="mem_test", llm=fake_llm)
    assert captured_lengths == [2, 4]


def _search_call():
    return {
        "content": "",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "search_jobs", "arguments": json.dumps({"query": "python"})},
            }
        ],
    }


def _answer(job_id):
    return {
        "content": json.dumps(
            {
                "answer": "Here you go.",
                "recommended_jobs": [
                    {"job_id": job_id, "title": "T", "company": "C", "location": "L", "match_reason": "fits"}
                ],
            }
        )
    }


def test_agent_drops_hallucinated_job_ids(monkeypatch):
    agent.SESSIONS.clear()
    monkeypatch.setattr(
        agent.search, "search", lambda q, k=5: [{"id": 1, "title": "T", "company": "C", "description": "d"}]
    )
    replies = [
        _search_call(),
        {
            "content": json.dumps(
                {
                    "answer": "Two picks.",
                    "recommended_jobs": [
                        {"job_id": 1, "title": "T", "company": "C", "match_reason": "real"},
                        {"job_id": 999, "title": "Ghost", "company": "Nope", "match_reason": "invented"},
                    ],
                }
            )
        },
    ]
    result = agent.run_agent("python jobs", session_id="halluc", llm=lambda m: replies.pop(0))
    assert [j.job_id for j in result.recommended_jobs] == [1]


def test_agent_retries_once_on_invalid_json():
    agent.SESSIONS.clear()
    replies = [{"content": "Sure! here are some jobs"}, {"content": json.dumps({"answer": "ok"})}]
    result = agent.run_agent("hi", session_id="repair", llm=lambda m: replies.pop(0))
    assert result.answer == "ok"


def test_agent_raises_clean_error_and_rolls_back_history():
    agent.SESSIONS.clear()

    with pytest.raises(agent.AgentUpstreamError):
        agent.run_agent("hi", session_id="bad", llm=lambda m: {"content": "not json"})
    assert len(agent.SESSIONS["bad"]) == 1  # only the system prompt remains


def test_tool_errors_are_returned_to_the_model_not_raised():
    assert "error" in agent.execute_tool("nope", {})
    assert "error" in agent.execute_tool("get_job_details", {})  # missing job_id
    assert "error" in agent.execute_tool("search_jobs", {"query": "  "})


def test_sessions_are_bounded(monkeypatch):
    agent.SESSIONS.clear()
    monkeypatch.setattr(agent, "MAX_SESSIONS", 3)
    for i in range(6):
        agent.run_agent("hi", session_id=f"s{i}", llm=lambda m: {"content": json.dumps({"answer": "ok"})})
    assert list(agent.SESSIONS) == ["s3", "s4", "s5"]


def test_history_is_trimmed_at_turn_boundaries(monkeypatch):
    agent.SESSIONS.clear()
    monkeypatch.setattr(agent, "MAX_HISTORY_MESSAGES", 6)
    llm = lambda m: {"content": json.dumps({"answer": "ok"})}
    for i in range(10):
        agent.run_agent(f"msg {i}", session_id="long", llm=llm)
    history = agent.SESSIONS["long"]
    assert history[0]["role"] == "system"
    assert history[1]["role"] == "user"
    assert len(history) <= 8
