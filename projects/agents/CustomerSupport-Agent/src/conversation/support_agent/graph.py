"""Core graph, state, prompts and helpers for the support agent."""

from __future__ import annotations

import json
import re
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langgraph.graph import END, StateGraph

from ...config import settings
from ...sentiment.analyzer import SentimentResult

QUESTION_HINTS = (
    "怎么",
    "如何",
    "为什么",
    "what",
    "how",
    "why",
    "?",
    "？",
    "说明",
    "教程",
    "帮助文档",
)
REQUEST_HINTS = (
    "创建",
    "新建",
    "工单",
    "ticket",
    "状态",
    "进度",
    "查询",
    "更新",
    "账户",
    "账号",
    "账单",
)
ESCALATION_HINTS = (
    "人工",
    "投诉",
    "经理",
    "马上处理",
    "退款",
    "起诉",
    "furious",
    "unacceptable",
    "sue",
)
ACCOUNT_HINTS = ("账户", "账号", "账单", "member", "plan", "invoice", "account")
TICKET_HINTS = ("工单", "ticket", "状态", "进度", "升级", "催单")


class ConversationState(TypedDict):
    """Backward-compatible conversation state exposed by the API layer."""

    user_id: str
    thread_id: str
    messages: List[Dict[str, str]]
    current_message: str
    intent: Optional[str]
    risk: Optional[str]
    response: Optional[str]
    active_agent: Optional[str]
    citations: Optional[List[str]]
    run_status: Optional[str]


class OrchestrationState(TypedDict, total=False):
    """Shared LangGraph orchestration state."""

    user_id: str
    thread_id: str
    current_message: str
    intent: str
    risk: str
    sentiment_label: str
    frustration_score: float
    route_reason: str
    selected_agent: str
    active_agent: str
    execution_steps: List[str]
    needs_knowledge: bool
    needs_action: bool
    needs_action_after_knowledge: bool
    needs_escalation: bool
    retrieval_text: str
    tool_text: str
    validation_notes: List[str]
    validation_passed: bool
    final_message: str
    citations: List[str]
    run_status: Literal["completed", "interrupted", "error"]
    interrupts: List[Dict[str, Any]]
    ticket_id: Optional[str]
    escalated: bool


@dataclass
class AgentRuntimeContext:
    """Runtime context injected into LangChain middleware."""

    user_id: str
    thread_id: str
    active_agent: str
    intent: str = "other"
    risk: str = "low"
    locale: str = settings.default_response_language


@dataclass
class SupportResponse:
    """Normalized support response returned by API and tests."""

    message: str
    intent: str
    sentiment: SentimentResult
    sources: List[str] = field(default_factory=list)
    escalated: bool = False
    ticket_created: Optional[str] = None
    thread_id: Optional[str] = None
    run_status: Literal["completed", "interrupted", "error"] = "completed"
    interrupts: List[Dict[str, Any]] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    active_agent: str = "supervisor"
    trace_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the dataclass into an API-friendly payload."""
        return {
            "message": self.message,
            "intent": self.intent,
            "sentiment": {
                "label": self.sentiment.label,
                "polarity": self.sentiment.polarity,
                "frustration_score": self.sentiment.frustration_score,
            },
            "sources": self.sources,
            "escalated": self.escalated,
            "ticket_created": self.ticket_created,
            "thread_id": self.thread_id,
            "run_status": self.run_status,
            "interrupts": self.interrupts,
            "citations": self.citations,
            "active_agent": self.active_agent,
            "trace_id": self.trace_id,
        }


def infer_intent(message: str) -> str:
    lowered = message.lower().strip()
    if not lowered:
        return "other"
    if any(token in lowered for token in ESCALATION_HINTS):
        return "complaint"
    if any(token in lowered for token in REQUEST_HINTS):
        return "request"
    if any(token in lowered for token in QUESTION_HINTS):
        return "question"
    if any(token in lowered for token in ("hello", "hi", "你好", "感谢", "谢谢", "thanks")):
        return "greeting"
    return "other"


def infer_risk(message: str, sentiment: Optional[SentimentResult]) -> str:
    lowered = message.lower()
    if any(token in lowered for token in ESCALATION_HINTS):
        return "high"
    if sentiment and sentiment.frustration_score >= 0.75:
        return "high"
    if sentiment and sentiment.frustration_score >= 0.45:
        return "medium"
    return "low"


def normalize_intent(value: Any) -> str:
    valid = {"question", "complaint", "request", "feedback", "greeting", "other"}
    text = str(value or "").strip().lower()
    return text if text in valid else "other"


def normalize_risk(value: Any) -> str:
    valid = {"low", "medium", "high"}
    text = str(value or "").strip().lower()
    return text if text in valid else "low"


def normalize_sentiment_label(value: Any) -> str:
    valid = {"positive", "neutral", "negative"}
    text = str(value or "").strip().lower()
    return text if text in valid else "neutral"


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return False


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return max(0.0, min(1.0, result))
    except Exception:
        return default


def extract_json_payload(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    candidates: List[str] = [text.strip()]
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    candidates.extend(fenced)
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        candidates.append(brace_match.group(0))

    for candidate in candidates:
        with suppress(Exception):
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload
    return None


def merge_unique(left: List[str], right: List[str]) -> List[str]:
    merged = list(left)
    for item in right:
        cleaned = str(item).strip()
        if cleaned and cleaned not in merged:
            merged.append(cleaned)
    return merged


def ticket_id_from_text(text: str) -> Optional[str]:
    match = re.search(r"TKT-\d{8,14}-\d{3,6}", text, re.IGNORECASE)
    return match.group(0).upper() if match else None


def extract_citations(text: str) -> List[str]:
    citations = re.findall(r"来源：([^\n]+)", text)
    unique: List[str] = []
    for cite in citations:
        cleaned = cite.strip()
        if cleaned and cleaned not in unique:
            unique.append(cleaned)
    return unique


def is_positive_escalation_text(text: str) -> bool:
    if not text:
        return False
    positive_patterns = (
        r"已升级到人工",
        r"已转人工",
        r"转交人工",
        r"人工客服已接手",
        r"已提交人工",
    )
    return any(re.search(pattern, text) for pattern in positive_patterns)


def contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def normalize_execution_steps(value: Any) -> List[str]:
    valid = {"knowledge", "action", "escalation"}
    if not isinstance(value, list):
        return []
    steps: List[str] = []
    for item in value:
        text = str(item or "").strip().lower()
        if text in valid and text not in steps:
            steps.append(text)
    if "escalation" in steps:
        return ["escalation"]
    return steps


def build_execution_steps(
    *,
    selected_agent: str,
    needs_knowledge: bool,
    needs_action: bool,
    needs_action_after_knowledge: bool,
    needs_escalation: bool,
) -> List[str]:
    if selected_agent == "escalation" or needs_escalation:
        return ["escalation"]

    steps: List[str] = []
    if selected_agent == "knowledge" or needs_knowledge:
        steps.append("knowledge")

    if needs_action_after_knowledge:
        if "knowledge" not in steps:
            steps.append("knowledge")
        steps.append("action")
    elif selected_agent == "action" or needs_action:
        if "action" not in steps:
            steps.append("action")

    return steps


def build_neutral_sentiment() -> SentimentResult:
    return SentimentResult(
        polarity=0.0,
        subjectivity=0.0,
        label="neutral",
        frustration_score=0.0,
        keywords=[],
    )


def build_role_system_prompt(role: str, memory_items: List[Dict[str, Any]]) -> str:
    role_rules = {
        "supervisor": "你是客服总调度，负责判断问题类型并给出决策建议。",
        "knowledge": "你是知识检索专家，优先调用 search_faq，严格基于证据回复。",
        "action": "你是客服执行专家，负责工单与账户类动作，必要时调用工具。",
        "escalation": "你是升级专员，负责人工介入、风险沟通与交接摘要。",
        "responder": "你是最终回答代理，负责融合证据与工具结果，生成最终答复。",
    }
    memory_text = "\n".join(
        f"- {item.get('fact', item.get('message', ''))}"
        for item in memory_items
        if item
    )
    memory_block = f"\n用户长期记忆：\n{memory_text}" if memory_text else "\n用户长期记忆：暂无"
    return (
        f"{role_rules.get(role, '你是客服助手。')}\n"
        "输出必须为中文，语气专业、简洁、可执行。\n"
        "若信息不足，明确说明并提出下一步收集项。\n"
        "回答尽量包含来源与依据，不编造事实。"
        f"{memory_block}"
    )


def compose_knowledge_prompt(state: OrchestrationState) -> str:
    return (
        f"用户问题：{state.get('current_message', '')}\n"
        "任务：调用知识检索工具，输出关键结论，并附上来源行（格式：来源：xxx）。"
    )


def compose_action_prompt(state: OrchestrationState) -> str:
    retrieval = state.get("retrieval_text", "")
    retrieval_block = f"\n可用检索证据：\n{retrieval}\n" if retrieval else ""
    return (
        f"用户请求：{state.get('current_message', '')}\n"
        f"{retrieval_block}"
        "任务：判断是否需要调用工单/账户类工具，必要时执行，输出执行结果。"
    )


def compose_escalation_prompt(state: OrchestrationState) -> str:
    retrieval = state.get("retrieval_text", "")
    retrieval_block = f"\n可用检索证据：\n{retrieval}\n" if retrieval else ""
    return (
        f"用户消息：{state.get('current_message', '')}\n"
        f"{retrieval_block}"
        "任务：按升级策略处理，高风险场景优先人工升级并给出交接摘要。"
    )


def compose_responder_prompt(state: OrchestrationState) -> str:
    notes = state.get("validation_notes", [])
    note_block = "\n".join(f"- {note}" for note in notes) if notes else "无"
    retrieval_text = state.get("retrieval_text") or "无"
    tool_text = state.get("tool_text") or "无"
    return (
        "请基于以下结构化上下文生成最终客服答复：\n"
        f"- 用户问题：{state.get('current_message', '')}\n"
        f"- 识别意图：{state.get('intent', 'other')}\n"
        f"- 风险等级：{state.get('risk', 'low')}\n"
        f"- 情绪标签：{state.get('sentiment_label', 'neutral')}\n"
        f"- 挫败分：{state.get('frustration_score', 0.0):.2f}\n"
        f"- 知识证据：\n{retrieval_text}\n"
        f"- 工具执行结果：\n{tool_text}\n"
        f"- 是否人工升级：{'是' if state.get('escalated') else '否'}\n"
        f"- 工单号：{state.get('ticket_id') or '无'}\n"
        f"- 校验修正要求：\n{note_block}\n"
        "要求：中文输出；先结论后步骤；若有来源请保留“来源：xxx”；如果证据不足，要明确告知并给出下一步建议。"
    )


class SupportAgentOrchestrator:
    """Owns graph construction and node implementations."""

    def __init__(self, owner: Any) -> None:
        self.owner = owner

    def build(self) -> Any:
        workflow = StateGraph(OrchestrationState)
        workflow.add_node("analyze", self.node_analyze)
        workflow.add_node("knowledge", self.node_knowledge)
        workflow.add_node("action", self.node_action)
        workflow.add_node("escalation", self.node_escalation)
        workflow.add_node("validate", self.node_validate)
        workflow.add_node("respond", self.node_respond)

        workflow.set_entry_point("analyze")
        workflow.add_conditional_edges(
            "analyze",
            self._route_after_analyze,
            {
                "knowledge": "knowledge",
                "action": "action",
                "escalation": "escalation",
                "validate": "validate",
            },
        )
        workflow.add_conditional_edges(
            "knowledge",
            self._route_after_knowledge,
            {
                "action": "action",
                "validate": "validate",
            },
        )
        workflow.add_conditional_edges(
            "action",
            self._route_after_execution,
            {
                "validate": "validate",
                "end": END,
            },
        )
        workflow.add_conditional_edges(
            "escalation",
            self._route_after_execution,
            {
                "validate": "validate",
                "end": END,
            },
        )
        workflow.add_edge("validate", "respond")
        workflow.add_edge("respond", END)
        return workflow.compile()

    def finalize_after_execution(self, state: OrchestrationState) -> OrchestrationState:
        merged = dict(state)
        for node in (self.node_validate, self.node_respond):
            merged.update(node(merged))
        return merged

    def _route_after_analyze(self, state: OrchestrationState) -> str:
        selected = state.get("selected_agent", "supervisor")
        steps = state.get("execution_steps", [])
        if selected == "escalation" or "escalation" in steps:
            return "escalation"
        if "knowledge" in steps:
            return "knowledge"
        if selected == "action" or "action" in steps:
            return "action"
        return "validate"

    def _route_after_knowledge(self, state: OrchestrationState) -> str:
        if state.get("run_status") == "interrupted":
            return "validate"
        if "action" in state.get("execution_steps", []) and not state.get("tool_text"):
            return "action"
        return "validate"

    def _route_after_execution(self, state: OrchestrationState) -> str:
        if state.get("run_status") == "interrupted":
            return "end"
        return "validate"

    def node_analyze(self, state: OrchestrationState) -> Dict[str, Any]:
        message = state.get("current_message", "")
        baseline_sentiment = SentimentResult(
            polarity=0.0,
            subjectivity=0.0,
            label=self.owner._normalize_sentiment_label(state.get("sentiment_label")),
            frustration_score=self.owner._safe_float(state.get("frustration_score"), 0.0),
            keywords=[],
        )
        baseline_intent = self.owner._infer_intent(message)
        baseline_risk = self.owner._infer_risk(message, baseline_sentiment)

        route = self.owner._analyze_and_route(
            message=message,
            baseline_intent=baseline_intent,
            baseline_risk=baseline_risk,
            baseline_sentiment=baseline_sentiment,
        )

        self.owner._save_memory_item(
            user_id=state.get("user_id", "unknown_user"),
            payload={
                "kind": "routing_decision",
                "thread_id": state.get("thread_id", "unknown_thread"),
                "message": message,
                "intent": route["intent"],
                "risk": route["risk"],
                "selected_agent": route["selected_agent"],
                "execution_steps": route["execution_steps"],
                "needs_knowledge": route["needs_knowledge"],
                "needs_action": route["needs_action"],
                "needs_action_after_knowledge": route["needs_action_after_knowledge"],
                "needs_escalation": route["needs_escalation"],
                "reason": route["route_reason"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        return {
            "intent": route["intent"],
            "risk": route["risk"],
            "sentiment_label": route["sentiment_label"],
            "frustration_score": route["frustration_score"],
            "route_reason": route["route_reason"],
            "selected_agent": route["selected_agent"],
            "active_agent": route["selected_agent"],
            "execution_steps": route["execution_steps"],
            "needs_knowledge": route["needs_knowledge"],
            "needs_action": route["needs_action"],
            "needs_action_after_knowledge": route["needs_action_after_knowledge"],
            "needs_escalation": route["needs_escalation"],
            "run_status": "completed",
            "interrupts": [],
        }

    def node_knowledge(self, state: OrchestrationState) -> Dict[str, Any]:
        user_id = state.get("user_id", "unknown_user")
        thread_id = state.get("thread_id", "unknown_thread")
        intent = state.get("intent", "question")
        risk = state.get("risk", "low")

        if self.owner.llm_enabled:
            result = self.owner._call_role_agent(
                role="knowledge",
                user_id=user_id,
                thread_id=thread_id,
                intent=intent,
                risk=risk,
                message=compose_knowledge_prompt(state),
            )
            retrieval_text = self.owner._extract_ai_text(result)
        else:
            retrieval_text = self.owner._fallback_response("knowledge", user_id, state.get("current_message", ""), None)

        citations = merge_unique(state.get("citations", []), extract_citations(retrieval_text))
        return {
            "active_agent": "knowledge",
            "retrieval_text": retrieval_text,
            "citations": citations,
            "run_status": "completed",
        }

    def node_action(self, state: OrchestrationState) -> Dict[str, Any]:
        user_id = state.get("user_id", "unknown_user")
        thread_id = state.get("thread_id", "unknown_thread")
        intent = state.get("intent", "request")
        risk = state.get("risk", "medium")

        if self.owner.llm_enabled:
            result = self.owner._call_role_agent(
                role="action",
                user_id=user_id,
                thread_id=thread_id,
                intent=intent,
                risk=risk,
                message=compose_action_prompt(state),
            )
            interrupts = self.owner._extract_interrupts(result)
            if interrupts:
                return {
                    "active_agent": "action",
                    "run_status": "interrupted",
                    "interrupts": interrupts,
                    "citations": state.get("citations", []),
                }
            tool_text = self.owner._extract_ai_text(result)
        else:
            tool_text = self.owner._fallback_response("action", user_id, state.get("current_message", ""), None)

        ticket_id = ticket_id_from_text(tool_text) or state.get("ticket_id")
        citations = merge_unique(state.get("citations", []), extract_citations(tool_text))
        escalated = bool(state.get("escalated")) or is_positive_escalation_text(tool_text)
        return {
            "active_agent": "action",
            "tool_text": tool_text,
            "ticket_id": ticket_id,
            "escalated": escalated,
            "run_status": "completed",
            "interrupts": [],
            "citations": citations,
        }

    def node_escalation(self, state: OrchestrationState) -> Dict[str, Any]:
        user_id = state.get("user_id", "unknown_user")
        thread_id = state.get("thread_id", "unknown_thread")
        intent = state.get("intent", "complaint")

        if self.owner.llm_enabled:
            result = self.owner._call_role_agent(
                role="escalation",
                user_id=user_id,
                thread_id=thread_id,
                intent=intent,
                risk="high",
                message=compose_escalation_prompt(state),
            )
            interrupts = self.owner._extract_interrupts(result)
            if interrupts:
                return {
                    "active_agent": "escalation",
                    "run_status": "interrupted",
                    "interrupts": interrupts,
                    "escalated": True,
                    "citations": state.get("citations", []),
                }
            tool_text = self.owner._extract_ai_text(result)
        else:
            tool_text = self.owner._fallback_response("escalation", user_id, state.get("current_message", ""), None)

        ticket_id = ticket_id_from_text(tool_text) or state.get("ticket_id")
        citations = merge_unique(state.get("citations", []), extract_citations(tool_text))
        return {
            "active_agent": "escalation",
            "tool_text": tool_text,
            "ticket_id": ticket_id,
            "escalated": True,
            "run_status": "completed",
            "interrupts": [],
            "citations": citations,
        }

    def node_validate(self, state: OrchestrationState) -> Dict[str, Any]:
        notes: List[str] = []
        retrieval_text = (state.get("retrieval_text") or "").strip()
        tool_text = (state.get("tool_text") or "").strip()
        combined_text = "\n".join(part for part in (retrieval_text, tool_text) if part).strip()
        has_evidence = bool(state.get("retrieval_text") or state.get("tool_text"))

        if not combined_text:
            notes.append("当前没有有效证据或执行结果，需要明确告知用户信息不足并给出下一步建议。")
        elif not contains_chinese(combined_text):
            notes.append("最终回复必须保持中文输出。")

        if has_evidence and not state.get("citations"):
            notes.append("若引用了知识库或工具结果，需要在最终回复中保留来源。")
        if state.get("ticket_id") and state["ticket_id"] not in combined_text:
            notes.append("如果已经生成工单号，需要在最终回复中明确展示。")
        if state.get("escalated") and "人工" not in combined_text and "升级" not in combined_text:
            notes.append("如果已经升级人工，需要明确告知用户后续人工跟进。")
        if not has_evidence and state.get("intent") in {"question", "request", "complaint"}:
            notes.append("如果证据不足，需要主动说明信息缺口并引导补充信息。")

        return {
            "validation_notes": notes,
            "validation_passed": not notes,
        }

    def node_respond(self, state: OrchestrationState) -> Dict[str, Any]:
        if state.get("run_status") == "interrupted":
            return {}

        if self.owner.llm_enabled:
            result = self.owner._call_role_agent(
                role="responder",
                user_id=state.get("user_id", "unknown_user"),
                thread_id=state.get("thread_id", "unknown_thread"),
                intent=state.get("intent", "other"),
                risk=state.get("risk", "low"),
                message=compose_responder_prompt(state),
            )
            final_message = self.owner._extract_ai_text(result)
        else:
            final_message = self.owner._fallback_response(
                role="responder",
                user_id=state.get("user_id", "unknown_user"),
                message=state.get("current_message", ""),
                sentiment=None,
                retrieval_text=state.get("retrieval_text", ""),
                tool_text=state.get("tool_text", ""),
            )

        citations = merge_unique(state.get("citations", []), extract_citations(final_message))
        ticket_id = state.get("ticket_id") or ticket_id_from_text(final_message)
        escalated = bool(state.get("escalated") or state.get("selected_agent") == "escalation")
        return {
            "final_message": final_message,
            "ticket_id": ticket_id,
            "escalated": escalated,
            "citations": citations,
            "run_status": "completed",
        }
