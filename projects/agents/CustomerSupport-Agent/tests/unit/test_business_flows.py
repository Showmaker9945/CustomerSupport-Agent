"""Regression tests for small-scale business support flows."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.conversation.support_agent import SupportAgent
from src.conversation.support_agent import service as support_service
from src.tools.support_tools import (
    create_ticket,
    explain_invoice_charge,
    get_latest_invoice,
    get_subscription_status,
)


@pytest.fixture(autouse=True)
def isolate_runtime(monkeypatch, isolated_business_db):
    """Keep global agent/ticket state deterministic across tests."""
    monkeypatch.setenv("DISABLE_LLM", "true")

    if support_service._support_agent is not None:
        support_service._support_agent.close()
    support_service._support_agent = None
    yield

    if support_service._support_agent is not None:
        support_service._support_agent.close()
        support_service._support_agent = None


@pytest.fixture
def agent():
    return SupportAgent(enable_memory=True, enable_sentiment=True)


@pytest.fixture
def client():
    return TestClient(app)


def test_subscription_tool_returns_business_fields():
    result = get_subscription_status.invoke({"user_id": "user_001"})

    assert "当前套餐：Pro 团队版" in result
    assert "下次续费：2026-04-01" in result
    assert "自动续费：开启" in result
    assert "来源：Subscription Ledger" in result


def test_latest_invoice_tool_returns_breakdown():
    result = get_latest_invoice.invoke({"user_id": "user_001"})

    assert "账单号：INV-202603-0001" in result
    assert "总金额：¥199.00" in result
    assert "超额存储包 100GB" in result
    assert "来源：Billing Ledger" in result


def test_invoice_explanation_tool_returns_reasoning():
    result = explain_invoice_charge.invoke({"invoice_id": "INV-202603-0001"})

    assert "账单扣费说明" in result
    assert "Pro 团队版月费：¥159.00" in result
    assert "超额存储" in result
    assert "来源：Billing Explanation Engine" in result


def test_billing_ticket_creation_uses_category():
    result = create_ticket.invoke(
        {
            "user_id": "user_001",
            "subject": "账单金额异常",
            "description": "本月账单看起来不对，请帮我核查。",
            "priority": "high",
            "category": "billing",
        }
    )

    assert "工单创建成功" in result
    assert "分类：billing" in result
    assert "来源：Support Ticket Store" in result


def test_agent_routes_subscription_question_to_action(agent):
    response = agent.chat(user_id="user_001", message="我当前是什么套餐？下次什么时候续费？")

    assert response.route_path == ["analyze", "action", "validate", "respond"]
    assert response.active_agent == "action"
    assert "套餐" in response.message
    assert response.citations


def test_agent_routes_invoice_question_to_action(agent):
    response = agent.chat(user_id="user_001", message="为什么这笔账单扣了 199？")

    assert response.route_path == ["analyze", "action", "validate", "respond"]
    assert "¥199.00" in response.message
    assert any("Billing" in citation for citation in response.citations)


def test_agent_routes_billing_ticket_request_to_action(agent):
    response = agent.chat(user_id="user_001", message="请帮我创建一个账单异常工单")

    assert response.route_path == ["analyze", "action", "validate", "respond"]
    assert response.ticket_created is not None
    assert "分类：billing" in response.message


def test_agent_routes_subscription_cancel_question_to_knowledge(agent):
    response = agent.chat(user_id="user_001", message="如何取消套餐？")

    assert response.route_path == ["analyze", "knowledge", "validate", "respond"]
    assert response.active_agent == "knowledge"
    assert "取消" in response.message
    assert any("FAQ::billing" in citation for citation in response.citations)


def test_subscription_endpoint(client):
    response = client.get("/users/user_001/subscription")

    assert response.status_code == 200
    data = response.json()
    assert data["subscription"]["plan_name"] == "Pro 团队版"
    assert data["subscription"]["renewal_date"] == "2026-04-01"


def test_latest_invoice_endpoint(client):
    response = client.get("/users/user_001/invoices/latest")

    assert response.status_code == 200
    data = response.json()
    assert data["invoice"]["invoice_id"] == "INV-202603-0001"
    assert data["invoice"]["total_amount"] == 199.0


def test_missing_latest_invoice_endpoint_returns_404(client):
    response = client.get("/users/user_002/invoices/latest")

    assert response.status_code == 404
    assert response.json()["error"] == "未找到该用户的最新账单"
