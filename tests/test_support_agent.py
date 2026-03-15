"""Integration smoke tests for upgraded support agent + API."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.conversation.support_agent import SupportAgent


@pytest.fixture(autouse=True)
def disable_llm(monkeypatch):
    monkeypatch.setenv("DISABLE_LLM", "true")


def test_agent_chat_and_resume_smoke(isolated_business_db):
    agent = SupportAgent(enable_memory=True, enable_sentiment=True)
    response = agent.chat(user_id="it_user", message="请帮我创建工单")
    assert response.thread_id is not None
    assert response.run_status in {"completed", "interrupted", "error"}

    resumed = agent.resume(thread_id="unknown-thread", decisions=[{"type": "approve"}])
    assert resumed.run_status == "error"


def test_fastapi_core_flow(isolated_business_db):
    client = TestClient(app)
    chat_resp = client.post(
        "/chat",
        json={"user_id": "api_it_user", "content": "我的账户有问题，请帮我查一下"},
    )
    assert chat_resp.status_code == 200
    data = chat_resp.json()
    assert data["thread_id"]

    resume_resp = client.post(
        f"/runs/{data['thread_id']}/resume",
        json={"decisions": [{"type": "approve"}]},
    )
    assert resume_resp.status_code == 200

    reindex_resp = client.post("/knowledge/reindex", json={"clear_existing": False})
    assert reindex_resp.status_code == 200
