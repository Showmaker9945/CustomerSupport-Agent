"""Unit tests for the upgraded multi-agent support core."""

import os

import pytest
from langchain_core.messages import AIMessage

from src.conversation.support_agent import SupportAgent, SupportResponse


@pytest.fixture
def agent(monkeypatch):
    """Use deterministic non-LLM mode for stable tests."""
    monkeypatch.setenv("DISABLE_LLM", "true")
    return SupportAgent(enable_memory=True, enable_sentiment=True)


def test_chat_returns_extended_contract(agent):
    response = agent.chat(user_id="user_a", message="我想知道如何重置密码")
    assert isinstance(response, SupportResponse)
    assert response.thread_id is not None
    assert response.run_status in {"completed", "interrupted", "error"}
    assert response.active_agent in {"supervisor", "knowledge", "action", "escalation"}
    assert response.trace_id is not None
    assert response.message


def test_route_to_knowledge_agent(agent):
    response = agent.chat(user_id="user_q", message="请问如何邀请团队成员？")
    assert response.active_agent == "knowledge"
    assert response.intent in {"question", "other"}
    assert any(source in {"Hybrid RAG", "FAQ Knowledge Base", "Fallback Policy"} for source in response.sources)


def test_route_to_action_agent(agent):
    response = agent.chat(user_id="user_r", message="请帮我创建工单，我的账单有问题")
    assert response.active_agent == "action"
    assert response.intent in {"request", "other"}


def test_route_to_escalation(agent):
    response = agent.chat(user_id="user_e", message="我要投诉并要求人工经理现在处理")
    assert response.active_agent == "escalation"
    assert response.escalated is True


def test_history_and_reset(agent):
    user_id = "history_user"
    agent.chat(user_id=user_id, message="你好")
    agent.chat(user_id=user_id, message="我需要帮助")
    history = agent.get_conversation_history(user_id=user_id, limit=20)
    assert len(history) >= 2
    agent.reset_conversation(user_id)
    assert len(agent.get_conversation_history(user_id=user_id, limit=20)) == 0


def test_stream_chat_events(agent):
    events = list(agent.stream_chat(user_id="stream_user", message="请问如何修改密码"))
    assert events
    assert events[0]["type"] == "node"
    assert events[-1]["type"] == "done"


def test_resume_without_pending(agent):
    response = agent.resume(thread_id="missing-thread", decisions=[{"type": "approve"}])
    assert response.run_status == "error"
    assert "无法恢复" in response.message


class _FakeRoleAgent:
    def __init__(self, text: str):
        self.text = text

    def invoke(self, _payload, config=None, context=None):
        return {"messages": [AIMessage(content=self.text)]}


def test_resume_approve_does_not_false_escalate(agent):
    agent.llm_enabled = True
    agent.role_agents = {
        "action": _FakeRoleAgent("工单创建成功。\n工单号：TKT-20260307-0001"),
        "responder": _FakeRoleAgent(
            "已为你创建工单，当前无需人工升级。\n来源：工具执行结果。"
        ),
    }

    thread_id = "thread-approve"
    agent._thread_user[thread_id] = "resume_user"
    agent._trace_by_thread[thread_id] = "trace-approve"
    agent._pending_role[thread_id] = "action"
    agent._pending_state[thread_id] = {
        "user_id": "resume_user",
        "thread_id": thread_id,
        "current_message": "请帮我创建工单",
        "intent": "request",
        "risk": "medium",
        "selected_agent": "action",
        "active_agent": "action",
        "run_status": "interrupted",
        "interrupts": [{"id": "interrupt-1", "value": {"tool": "create_ticket"}}],
        "citations": [],
    }

    response = agent.resume(thread_id=thread_id, decisions=[{"type": "approve"}])

    assert response.run_status == "completed"
    assert response.active_agent == "action"
    assert response.escalated is False
    assert response.ticket_created == "TKT-20260307-0001"
