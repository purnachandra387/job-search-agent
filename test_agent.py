import json
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
