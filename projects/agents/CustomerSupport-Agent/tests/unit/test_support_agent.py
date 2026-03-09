"""Unit tests for the upgraded multi-agent support core."""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from src.conversation.support_agent import SupportAgent, SupportResponse
from src.conversation.support_agent.graph import build_trace_event


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
    assert response.route_path
    assert isinstance(response.validation_notes, list)
    assert isinstance(response.trace_preview, list)
    assert response.decision_summary
    assert response.next_action
    assert response.message


def test_route_to_knowledge_agent(agent):
    response = agent.chat(user_id="user_q", message="请问如何邀请团队成员？")

    assert response.active_agent == "knowledge"
    assert response.intent in {"question", "other"}
    assert response.route_path == ["analyze", "knowledge", "validate", "respond"]
    assert any(source in {"Hybrid RAG", "FAQ Knowledge Base", "Fallback Policy"} for source in response.sources)


def test_route_to_action_agent(agent):
    response = agent.chat(user_id="user_r", message="请帮我创建工单，我的账单有问题")

    assert response.active_agent == "action"
    assert response.intent in {"request", "other"}
    assert response.route_path == ["analyze", "action", "validate", "respond"]


def test_route_to_escalation(agent):
    response = agent.chat(user_id="user_e", message="我要投诉并要求人工经理现在处理")

    assert response.active_agent == "escalation"
    assert response.escalated is True
    assert response.route_path == ["analyze", "escalation", "validate", "respond"]


def test_route_to_knowledge_then_action(agent):
    response = agent.chat(user_id="user_qa", message="怎么更新账单地址并帮我更新账户信息")

    assert response.route_path == ["analyze", "knowledge", "action", "validate", "respond"]
    assert response.active_agent == "knowledge"
    assert "执行路径=knowledge -> action" in response.decision_summary


def test_extract_interrupts_flattens_hitl_actions(agent):
    interrupt = SimpleNamespace(
        id="hitl-1",
        value={
            "action_requests": [
                {"name": "create_ticket", "args": {"subject": "账单异常"}, "description": "请审批是否创建工单"},
                {"name": "escalate_to_human", "args": {"reason": "用户投诉"}},
            ],
            "review_configs": [
                {"action_name": "create_ticket", "allowed_decisions": ["approve", "reject"]},
                {"action_name": "escalate_to_human", "allowed_decisions": ["approve", "edit", "reject"]},
            ],
        },
    )

    interrupts = agent._extract_interrupts({"__interrupt__": [interrupt]})

    assert len(interrupts) == 2
    assert interrupts[0]["tool"] == "create_ticket"
    assert interrupts[0]["tool_label"] == "创建工单"
    assert interrupts[0]["reason"] == "请审批是否创建工单"
    assert interrupts[1]["tool"] == "escalate_to_human"
    assert interrupts[1]["tool_label"] == "升级人工客服"

    approval = agent._build_approval_payload(interrupts)
    assert approval is not None
    assert approval["count"] == 2
    assert approval["required_decisions"] == 2
    assert "创建工单" in approval["message"]


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
    assert events[-1]["payload"]["debug"]["route_path"]
    assert "decision_summary" in events[-1]["payload"]["debug"]


def test_structured_memory_writes_preferences_and_profile(agent):
    response = agent.chat(
        user_id="memory_user",
        message="我叫小王，请以后都用中文回复，并通过邮件通知我。",
    )

    debug_payload = response.to_dict(include_debug=True)["debug"]
    assert "memory" in debug_payload
    write_ids = {item["memory_id"] for item in debug_payload["memory"]["writes"]}
    assert "profile:name" in write_ids
    assert "preference:language" in write_ids
    assert "preference:contact_channel" in write_ids

    memory_hits = agent._search_memory("memory_user", "中文 邮件 小王", limit=5)
    hit_ids = {item["memory_id"] for item in memory_hits}
    assert "profile:name" in hit_ids
    assert "preference:language" in hit_ids
    assert "preference:contact_channel" in hit_ids


def test_open_issue_memory_can_be_closed(agent):
    user_id = "memory_issue_user"
    agent.chat(user_id=user_id, message="这个账单问题还没解决，请帮我继续跟进")
    stored_before = {item["memory_id"] for item in agent._list_structured_memory(user_id)}
    assert "open_issue:billing_anomaly" in stored_before

    resolved = agent.chat(user_id=user_id, message="谢谢，这个账单问题已经解决了")
    stored_after = {item["memory_id"] for item in agent._list_structured_memory(user_id)}
    assert "open_issue:billing_anomaly" not in stored_after
    assert "resolved_issue:billing_anomaly" in stored_after
    assert "memory" in resolved.to_dict(include_debug=True)["debug"]


def test_resume_without_pending(agent):
    response = agent.resume(thread_id="missing-thread", decisions=[{"type": "approve"}])

    assert response.run_status == "error"
    assert "无法恢复" in response.message
    assert response.route_path == ["resume"]
    assert response.decision_summary == "恢复失败：未找到待审批线程。"
    assert response.next_action == "请确认使用的是上一次 /chat 返回的 thread_id。"


class _FakeRoleAgent:
    def __init__(self, text: str):
        self.text = text

    def invoke(self, _payload, config=None, context=None):
        return {"messages": [AIMessage(content=self.text)]}


def test_resume_approve_does_not_false_escalate(agent):
    agent.llm_enabled = True
    agent.role_agents = {
        "action": _FakeRoleAgent("工单创建成功。\n工单号：TKT-20260307-0001"),
        "responder": _FakeRoleAgent("已为你创建工单，当前无需人工升级。\n来源：工具执行结果。"),
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
        "route_path": ["analyze", "action"],
        "trace_events": [
            build_trace_event(node="analyze", agent="supervisor", summary="已完成路由决策。"),
            build_trace_event(node="action", agent="action", summary="动作节点因审批中断。", status="interrupted"),
        ],
        "decision_summary": "意图=request，风险=medium，首选代理=action，执行路径=action，原因=test。",
        "run_status": "interrupted",
        "interrupts": [
            {
                "id": "interrupt-1",
                "tool": "create_ticket",
                "tool_label": "创建工单",
                "reason": "创建工单属于高风险写操作，需要人工审批。",
                "args_preview": {"subject": "账单异常"},
                "allowed_decisions": ["approve", "edit", "reject"],
            }
        ],
        "citations": [],
    }

    response = agent.resume(thread_id=thread_id, decisions=[{"type": "approve"}])

    assert response.run_status == "completed"
    assert response.active_agent == "action"
    assert response.escalated is False
    assert response.ticket_created == "TKT-20260307-0001"
    assert response.route_path == ["analyze", "action", "validate", "respond"]
    assert response.decision_summary.startswith("意图=request")
    assert any(event["node"] == "respond" for event in response.trace_preview)
