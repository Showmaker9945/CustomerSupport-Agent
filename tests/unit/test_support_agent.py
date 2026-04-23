"""Unit tests for the upgraded multi-agent support core."""

import src.conversation.support_agent.graph as support_graph_module
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from src.conversation.support_agent import SupportAgent, SupportResponse
from src.conversation.support_agent.graph import build_trace_event
from src.db.repositories import get_conversation_thread, list_ticket_records, save_pending_conversation_state


@pytest.fixture
def agent(monkeypatch, isolated_business_db):
    """Use deterministic non-LLM mode for stable tests."""
    monkeypatch.setenv("DISABLE_LLM", "true")
    instance = SupportAgent(enable_memory=True, enable_sentiment=True)
    yield instance
    instance.close()


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
    assert isinstance(response.node_timings, list)
    assert response.node_timings
    assert response.total_duration_ms is not None
    assert response.decision_summary
    assert response.next_action
    assert response.message


def test_orchestrator_build_wires_main_graph_persistence(monkeypatch):
    checkpointer = object()
    store = object()
    captured = {}
    compiled_graph = object()

    def fake_compile(self, *args, **kwargs):
        captured.update(kwargs)
        return compiled_graph

    monkeypatch.setattr(support_graph_module.StateGraph, "compile", fake_compile)
    owner = SimpleNamespace(persistence=SimpleNamespace(checkpointer=checkpointer, store=store))

    orchestrator = support_graph_module.SupportAgentOrchestrator(owner)
    result = orchestrator.build(checkpointer=checkpointer, store=store)

    assert result is compiled_graph
    assert captured["checkpointer"] is checkpointer
    assert captured["store"] is store


def test_chat_uses_stable_graph_thread_id(agent):
    thread_id = "thread-graph-persist"
    fake_graph = _FakeOrchestrationGraph(
        {
            "intent": "question",
            "risk": "low",
            "selected_agent": "knowledge",
            "active_agent": "knowledge",
            "route_path": ["analyze", "knowledge", "validate", "respond"],
            "trace_events": [build_trace_event(node="respond", agent="responder", summary="完成回复。")],
            "node_timings": [
                {"node": "analyze", "agent": "supervisor", "duration_ms": 1.0, "status": "completed"}
            ],
            "decision_summary": "主图已使用稳定 checkpoint thread id。",
            "validation_notes": [],
            "run_status": "completed",
            "final_message": "这是一次测试回复。",
            "citations": [],
            "evidence_items": [],
            "retrieval_text": "",
            "tool_text": "",
            "tool_source": "",
            "escalated": False,
        }
    )
    agent.orchestration_graph = fake_graph

    response = agent.chat(user_id="graph_user", message="如何重置密码？", thread_id=thread_id)
    invocation = fake_graph.calls[-1]
    expected_graph_thread_id = agent._graph_thread_id(thread_id)

    assert invocation["state"]["graph_thread_id"] == expected_graph_thread_id
    assert invocation["config"]["configurable"]["thread_id"] == expected_graph_thread_id
    assert response.thread_id == thread_id


def test_route_to_knowledge_agent(agent):
    response = agent.chat(user_id="user_q", message="请问如何邀请团队成员？")

    assert response.active_agent == "knowledge"
    assert response.intent in {"question", "other"}
    assert response.route_path == ["analyze", "knowledge", "validate", "respond"]
    assert any(source in {"Hybrid RAG", "Help Center Knowledge Base", "Fallback Policy"} for source in response.sources)


def test_route_to_action_agent(agent):
    response = agent.chat(user_id="user_r", message="请帮我创建工单，我的账单有问题")

    assert response.active_agent == "action"
    assert response.intent in {"request", "other"}
    assert response.route_path == ["analyze", "action", "validate", "respond"]


def test_knowledge_citations_are_derived_from_structured_evidence(agent):
    response = agent.chat(user_id="citation_user", message="如何重置密码？")

    assert response.run_status == "completed"
    assert response.citations
    assert all(citation.startswith("帮助中心::") for citation in response.citations)
    assert "来源：" not in response.message


def test_hallucinated_source_line_is_stripped_from_responder_output(agent):
    agent.llm_enabled = True
    agent.basic_model = None
    agent.role_agents = {
        "responder": _FakeRoleAgent("你可以通过找回流程重置密码。\n来源：官方《不存在的文档》。"),
    }

    response = agent.chat(user_id="fake_source_user", message="如何重置密码？")

    assert response.run_status == "completed"
    assert response.citations
    assert "不存在的文档" not in response.message
    assert all("不存在的文档" not in citation for citation in response.citations)


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
    assert all("thread_id" in item for item in history)

    agent.reset_conversation(user_id)
    assert len(agent.get_conversation_history(user_id=user_id, limit=20)) == 0


def test_short_term_history_survives_new_agent_instance(monkeypatch, isolated_business_db):
    monkeypatch.setenv("DISABLE_LLM", "true")
    first_agent = SupportAgent(enable_memory=True, enable_sentiment=True)
    try:
        first_response = first_agent.chat(user_id="persist_user", message="你好，我想查一下账单")
        thread_id = first_response.thread_id
    finally:
        first_agent.close()

    second_agent = SupportAgent(enable_memory=True, enable_sentiment=True)
    try:
        history = second_agent.get_conversation_history(user_id="persist_user", limit=20)
        context = second_agent._load_recent_history_context(thread_id)
        assert any(item["content"] == "你好，我想查一下账单" for item in history)
        assert "你好，我想查一下账单" in context["text"]
        assert context["messages"]
    finally:
        second_agent.close()


def test_load_pending_context_backfills_graph_thread_id(agent):
    pending_state = {
        "user_id": "pending_graph_user",
        "thread_id": "thread-pending-graph",
        "intent": "request",
        "risk": "medium",
        "selected_agent": "action",
        "active_agent": "action",
        "route_path": ["analyze", "action"],
        "run_status": "interrupted",
        "interrupts": [{"id": "interrupt-1", "tool": "create_ticket"}],
    }
    save_pending_conversation_state(
        thread_id="thread-pending-graph",
        user_id="pending_graph_user",
        pending_role="action",
        pending_state=pending_state,
        trace_id="trace-pending-graph",
        status="interrupted",
        last_active_agent="action",
    )

    _, _, role, loaded_state = agent._load_pending_context("thread-pending-graph")

    assert role == "action"
    assert loaded_state["graph_thread_id"] == agent._graph_thread_id("thread-pending-graph")


def test_resume_can_recover_pending_state_from_database(monkeypatch, isolated_business_db):
    monkeypatch.setenv("DISABLE_LLM", "true")
    agent = SupportAgent(enable_memory=True, enable_sentiment=True)
    agent.llm_enabled = True
    agent.role_agents = {
        "action": _FakeRoleAgent("工单创建成功。\n工单号：TKT-20260315-0001"),
        "responder": _FakeRoleAgent("已为你创建工单，后续将继续跟进。\n来源：工具执行结果。"),
    }

    pending_state = {
        "user_id": "db_resume_user",
        "thread_id": "thread-db-resume",
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
        "decision_summary": "意图=request，风险=medium，首选代理=action，执行路径=action，原因=test-db。",
        "run_status": "interrupted",
        "interrupts": [
            {
                "id": "interrupt-db-1",
                "tool": "create_ticket",
                "tool_label": "创建工单",
                "reason": "创建工单属于高风险写操作，需要人工审批。",
                "args_preview": {"subject": "账单异常"},
                "allowed_decisions": ["approve", "edit", "reject"],
            }
        ],
        "citations": [],
    }
    save_pending_conversation_state(
        thread_id="thread-db-resume",
        user_id="db_resume_user",
        pending_role="action",
        pending_state=pending_state,
        trace_id="trace-db-resume",
        status="interrupted",
        last_active_agent="action",
    )

    response = agent.resume(thread_id="thread-db-resume", decisions=[{"type": "approve"}])
    thread_record = get_conversation_thread("thread-db-resume")

    assert response.run_status == "completed"
    assert response.trace_id == "trace-db-resume"
    assert response.ticket_created == "TKT-20260315-0001"
    assert thread_record is not None
    assert not thread_record["pending_role"]
    assert thread_record["pending_state"] == {}
    agent.close()


def test_long_thread_generates_rolling_summary(agent):
    first = agent.chat(user_id="summary_user", message="你好，我先想确认一下当前订阅套餐。")
    thread_id = first.thread_id
    for idx in range(1, 8):
        agent.chat(
            user_id="summary_user",
            thread_id=thread_id,
            message=f"补充问题 {idx}：我还想继续了解账单、套餐和工单处理进度。",
        )

    thread_record = get_conversation_thread(thread_id)
    context = agent._load_recent_history_context(thread_id)

    assert thread_record is not None
    assert thread_record["message_count"] >= 16
    assert thread_record["rolling_summary"]
    assert "用户近期诉求：" in thread_record["rolling_summary"]
    assert "待跟进事项：" in thread_record["rolling_summary"]
    assert context["rolling_summary"] == thread_record["rolling_summary"]
    assert "更早对话摘要：" in context["text"]
    assert len(context["messages"]) <= 12


def test_heuristic_rolling_summary_is_structured(agent):
    summary = agent._heuristic_rolling_summary(
        [
            {"role": "user", "content": "我想确认套餐续费和最近账单。"},
            {"role": "assistant", "content": "已帮你查到当前是 Pro 套餐，账单正在核对。"},
            {"role": "interrupt", "content": "检测到高风险动作，等待人工审批后再继续处理。"},
        ],
        previous_summary="之前已经确认过用户需要中文回复。",
    )

    assert "历史摘要：" in summary
    assert "用户近期诉求：" in summary
    assert "已处理/已确认：" in summary
    assert "待跟进事项：" in summary


def test_stream_chat_events(agent):
    events = list(agent.stream_chat(user_id="stream_user", message="请问如何修改密码"))

    assert events
    assert events[0]["type"] == "node"
    assert events[-1]["type"] == "done"
    assert events[-1]["payload"]["debug"]["route_path"]
    assert "node_timings" in events[-1]["payload"]["debug"]
    assert "decision_summary" not in events[-1]["payload"]["debug"]


def test_structured_memory_writes_preferences_and_profile(agent):
    response = agent.chat(
        user_id="memory_user",
        message="我叫小王，请以后都用中文回复，并通过邮件通知我。",
    )

    debug_payload = response.to_dict(include_debug=True)["debug"]
    assert "memory" not in debug_payload

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
    assert "memory" not in resolved.to_dict(include_debug=True)["debug"]


def test_resume_without_pending(agent):
    response = agent.resume(thread_id="missing-thread", decisions=[{"type": "approve"}])

    assert response.run_status == "error"
    assert "无法恢复" in response.message
    assert response.route_path == ["resume"]
    assert response.decision_summary == "恢复失败：未找到待审批线程。"
    assert response.next_action == "请确认使用的是上一次 /chat 返回的 thread_id。"
    assert response.total_duration_ms is not None


def test_billing_ticket_request_requires_hitl_when_llm_enabled(monkeypatch, isolated_business_db):
    monkeypatch.setenv("DISABLE_LLM", "true")
    agent = SupportAgent(enable_memory=True, enable_sentiment=True)
    agent.llm_enabled = True
    agent.basic_model = None
    agent.role_agents = {"action": _FakeRoleAgent("unused"), "responder": _FakeRoleAgent("unused")}

    response = agent.chat(user_id="user_001", message="帮我创建一个账单异常工单")

    assert response.run_status == "interrupted"
    assert response.active_agent == "action"
    assert response.approval is not None
    assert response.approval["tools"][0]["tool"] == "create_ticket"
    assert response.approval["tools"][0]["tool_label"] == "创建工单"
    assert response.approval["tools"][0]["args_preview"]["subject"] == "账单异常核查"
    assert any(item["node"] == "action" for item in response.node_timings)
    agent.close()


def test_billing_ticket_process_question_routes_to_knowledge(agent):
    response = agent.chat(user_id="billing_process_user", message="账单异常工单创建后会发生什么？")

    assert response.run_status == "completed"
    assert response.active_agent == "knowledge"
    assert response.route_path == ["analyze", "knowledge", "validate", "respond"]
    assert response.citations


class _FakeRoleAgent:
    def __init__(self, text: str):
        self.text = text

    def invoke(self, _payload, config=None, context=None):
        return {"messages": [AIMessage(content=self.text)]}


class _FakeOrchestrationGraph:
    def __init__(self, result):
        self.result = dict(result)
        self.calls = []

    def invoke(self, state, config=None):
        self.calls.append({"state": dict(state), "config": dict(config or {})})
        merged = dict(state)
        merged.update(self.result)
        return merged


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


def test_deterministic_resume_approve_executes_ticket(agent):
    agent.llm_enabled = True
    agent.basic_model = None
    agent.role_agents = {
        "action": _FakeRoleAgent("unused"),
        "responder": _FakeRoleAgent("已为你创建账单异常工单，后续会继续跟进。\n来源：工具执行结果。"),
    }

    interrupted = agent.chat(user_id="resume_ticket_user", message="帮我创建一个账单异常工单")
    resumed = agent.resume(thread_id=interrupted.thread_id, decisions=[{"type": "approve"}])
    thread_record = get_conversation_thread(interrupted.thread_id)
    tickets = list_ticket_records("resume_ticket_user")

    assert interrupted.run_status == "interrupted"
    assert resumed.run_status == "completed"
    assert resumed.ticket_created is not None
    assert resumed.ticket_created.startswith("TKT-")
    assert resumed.route_path == ["analyze", "action", "validate", "respond"]
    assert any(item["node"] == "action" and item["status"] == "completed" for item in resumed.node_timings)
    assert thread_record is not None
    assert thread_record["pending_state"] == {}
    assert len(tickets) == 1
    assert tickets[0]["ticket_id"] == resumed.ticket_created


def test_deterministic_resume_edit_overrides_tool_args(agent):
    agent.llm_enabled = True
    agent.basic_model = None
    agent.role_agents = {
        "action": _FakeRoleAgent("unused"),
        "responder": _FakeRoleAgent("已按修改后的内容创建工单。\n来源：工具执行结果。"),
    }

    interrupted = agent.chat(user_id="resume_edit_user", message="帮我创建一个账单异常工单")
    resumed = agent.resume(
        thread_id=interrupted.thread_id,
        decisions=[
            {
                "type": "edit",
                "edited_action": {
                    "args": {
                        "subject": "人工审批后的账单工单",
                        "priority": "urgent",
                    }
                },
            }
        ],
    )
    tickets = list_ticket_records("resume_edit_user")

    assert resumed.run_status == "completed"
    assert resumed.ticket_created is not None
    assert len(tickets) == 1
    assert tickets[0]["subject"] == "人工审批后的账单工单"
    assert tickets[0]["priority"] == "urgent"


def test_deterministic_resume_reject_skips_tool_execution(agent):
    agent.llm_enabled = True
    agent.basic_model = None
    agent.role_agents = {
        "action": _FakeRoleAgent("unused"),
        "responder": _FakeRoleAgent("本次未执行创建工单操作，如需继续处理请重新提交。\n来源：审批决策。"),
    }

    interrupted = agent.chat(user_id="resume_reject_user", message="帮我创建一个账单异常工单")
    resumed = agent.resume(thread_id=interrupted.thread_id, decisions=[{"type": "reject"}])

    assert interrupted.run_status == "interrupted"
    assert resumed.run_status == "completed"
    assert resumed.ticket_created is None
    assert "取消" in resumed.message or "未执行" in resumed.message
    assert "Approval Decision" in resumed.sources
    assert list_ticket_records("resume_reject_user") == []


def test_resume_merges_existing_knowledge_evidence_with_new_tool_evidence(agent):
    agent.llm_enabled = True
    agent.basic_model = None
    agent.role_agents = {
        "action": _FakeRoleAgent("unused"),
        "responder": _FakeRoleAgent("已为你创建账单异常工单，后续会继续跟进。"),
    }

    thread_id = "thread-resume-evidence"
    agent._thread_user[thread_id] = "resume_evidence_user"
    agent._trace_by_thread[thread_id] = "trace-resume-evidence"
    agent._pending_role[thread_id] = "action"
    agent._pending_state[thread_id] = {
        "user_id": "resume_evidence_user",
        "thread_id": thread_id,
        "current_message": "帮我创建一个账单异常工单",
        "intent": "request",
        "risk": "medium",
        "selected_agent": "action",
        "active_agent": "action",
        "route_path": ["analyze", "knowledge", "action"],
        "trace_events": [
            build_trace_event(node="analyze", agent="supervisor", summary="已完成路由决策。"),
            build_trace_event(node="knowledge", agent="knowledge", summary="已完成知识检索。"),
            build_trace_event(node="action", agent="action", summary="动作节点因审批中断。", status="interrupted"),
        ],
        "decision_summary": "意图=request，风险=medium，首选代理=action，执行路径=knowledge -> action，原因=test。",
        "run_status": "interrupted",
        "interrupts": [
            {
                "id": "interrupt-evidence-1",
                "tool": "create_ticket",
                "tool_label": "创建工单",
                "reason": "创建工单属于高风险写操作，需要人工审批。",
                "args_preview": {"subject": "账单异常核查"},
                "allowed_decisions": ["approve", "edit", "reject"],
            }
        ],
        "pending_approval_plan": {
            "mode": "deterministic_tool_call",
            "tool": "create_ticket",
            "tool_label": "创建工单",
            "args": {
                "user_id": "resume_evidence_user",
                "subject": "账单异常核查",
                "description": "账单存在重复扣费，请创建工单",
                "priority": "high",
                "category": "billing",
            },
            "reject_message": "已取消创建工单。",
        },
        "retrieval_text": "账单异常工单会先进入财务复核，再由客服同步处理结果。",
        "evidence_items": [
            {
                "evidence_id": "knowledge:billing-ticket-flow",
                "kind": "knowledge",
                "source_type": "help_center",
                "source_label": "帮助中心::Customer Support Help Center > 工单与服务 > 账单异常工单审核流程",
                "document_title": "Customer Support Help Center",
                "section_path": "Customer Support Help Center > 工单与服务 > 账单异常工单审核流程",
                "source_path": "knowledge/help-center.md",
                "snippet": "账单异常工单会先进入财务复核，再由客服同步处理结果。",
                "confidence": 0.93,
                "metadata": {"parent_chunk_id": "parent-flow-1"},
            }
        ],
        "citations": [],
    }

    response = agent.resume(thread_id=thread_id, decisions=[{"type": "approve"}])

    assert response.run_status == "completed"
    assert "帮助中心::Customer Support Help Center > 工单与服务 > 账单异常工单审核流程" in response.citations
    assert "Support Tools::创建工单" in response.citations
    assert response.ticket_created is not None


def test_tool_result_ticket_id_ignores_failure_payload(agent):
    failure_text = "创建工单失败：数据库写入失败，但日志里出现了 TKT-20260315-0001。"

    assert agent._tool_result_ticket_id("create_ticket", failure_text) is None
    assert agent._tool_result_ticket_id("escalate_to_human", failure_text) is None
