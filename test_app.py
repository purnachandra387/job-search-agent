from fastapi.testclient import TestClient

import app as api_app


def test_health_endpoint():
    client = TestClient(api_app.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_search_endpoint_returns_ranked_jobs(monkeypatch):
    fake_results = [
        {
            "id": 6,
            "title": "Customer Support Engineer",
            "company": "CloudDesk",
            "location": "Bengaluru",
            "description": "Troubleshoot customer issues using logs and SQL.",
            "score": 0.699,
        }
    ]
    monkeypatch.setattr(api_app.search, "search", lambda q, k=5: fake_results[:k])
    client = TestClient(api_app.app)
    resp = client.get("/search", params={"q": "help customers fix problems using logs", "k": 1})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["id"] == 6
    assert data[0]["title"] == "Customer Support Engineer"


def test_get_job_details_found_and_not_found(monkeypatch):
    sample_job = {
        "id": 2,
        "title": "Android Developer Intern",
        "company": "PixelNest",
        "location": "Remote",
        "description": "Build Kotlin and Jetpack Compose screens.",
    }
    monkeypatch.setattr(
        api_app,
        "get_job_by_id",
        lambda jid: sample_job if jid == 2 else None,
    )
    client = TestClient(api_app.app)
    ok_resp = client.get("/jobs/2")
    assert ok_resp.status_code == 200
    assert ok_resp.json()["company"] == "PixelNest"
    missing_resp = client.get("/jobs/999")
    assert missing_resp.status_code == 404


def test_agent_endpoint_success(monkeypatch):
    reply = api_app.agent.AgentResponse(answer="Try Acme Pay.", tools_used=["search_jobs"])
    monkeypatch.setattr(api_app.agent, "run_agent", lambda msg, session_id="default": reply)
    resp = TestClient(api_app.app).post("/agent", json={"message": "python jobs"})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "Try Acme Pay."


def test_agent_errors_map_to_http_status(monkeypatch):
    def no_key(msg, session_id="default"):
        raise api_app.agent.AgentConfigError("Set either GEMINI_API_KEY or GROQ_API_KEY")

    def upstream(msg, session_id="default"):
        raise api_app.agent.AgentUpstreamError("LLM provider returned HTTP 429")

    client = TestClient(api_app.app)
    monkeypatch.setattr(api_app.agent, "run_agent", no_key)
    assert client.post("/agent", json={"message": "hi"}).status_code == 503
    monkeypatch.setattr(api_app.agent, "run_agent", upstream)
    assert client.post("/agent", json={"message": "hi"}).status_code == 502


def test_agent_rejects_empty_message():
    assert TestClient(api_app.app).post("/agent", json={"message": ""}).status_code == 422


def test_search_returns_503_when_index_missing(monkeypatch):
    def boom(q, k=5):
        raise FileNotFoundError("Search index 'index.npz' not found. Run `python search.py build` first.")

    monkeypatch.setattr(api_app.search, "search", boom)
    resp = TestClient(api_app.app).get("/search", params={"q": "python"})
    assert resp.status_code == 503
    assert "search.py build" in resp.json()["detail"]
