"""Unit tests for upgraded FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, manager


@pytest.fixture(autouse=True)
def disable_llm(monkeypatch):
    monkeypatch.setenv("DISABLE_LLM", "true")


@pytest.fixture
def client():
    return TestClient(app)


def test_root(client):
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert "endpoints" in data
    assert "resume" in data["endpoints"]


def test_chat_contract_default_response_is_clean(client):
    response = client.post(
        "/chat",
        json={"user_id": "api_user", "content": "如何重置密码？"},
    )

    assert response.status_code == 200
    data = response.json()
    for key in (
        "message",
        "thread_id",
        "run_status",
        "active_agent",
        "intent",
        "sentiment",
        "result",
        "next_action",
        "timestamp",
    ):
        assert key in data

    assert "debug" not in data
    assert "approval" not in data
    assert isinstance(data["sentiment"], dict)
    assert isinstance(data["result"], dict)


def test_chat_contract_with_debug(client):
    response = client.post(
        "/chat",
        json={"user_id": "api_user", "content": "如何重置密码？", "debug": True},
    )

    assert response.status_code == 200
    data = response.json()
    assert "debug" in data
    assert data["debug"]["route_path"]
    assert isinstance(data["debug"]["validation_notes"], list)
    assert isinstance(data["debug"]["trace_preview"], list)
    assert data["debug"]["decision_summary"]


def test_resume_endpoint(client):
    response = client.post(
        "/runs/non-existent-thread/resume",
        json={"decisions": [{"type": "approve"}], "debug": True},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["run_status"] == "error"
    assert data["next_action"] == "请确认使用的是上一次 /chat 返回的 thread_id。"
    assert data["debug"]["route_path"] == ["resume"]
    assert data["debug"]["decision_summary"] == "恢复失败：未找到待审批线程。"


def test_sse_stream(client):
    response = client.get("/chat/stream", params={"user_id": "sse_user", "content": "你好"})

    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")


def test_knowledge_reindex(client):
    response = client.post("/knowledge/reindex", json={"clear_existing": False})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert "components" in data


def test_tickets_and_history(client):
    tickets = client.get("/users/user_001/tickets")
    assert tickets.status_code == 200

    history = client.get("/users/user_001/history")
    assert history.status_code == 200


def test_session_manager_helpers():
    assert manager.get_session_info("not-exist") is None
    assert manager.get_user_sessions("nobody") == []
