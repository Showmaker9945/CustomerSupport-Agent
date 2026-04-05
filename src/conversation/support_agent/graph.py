"""Core graph, state, prompts and helpers for the support agent."""

from __future__ import annotations

import json
import re
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
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
    "发票",
    "订阅",
    "套餐",
    "续费",
    "扣费",
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
ACCOUNT_HINTS = (
    "账户",
    "账号",
    "账单",
    "发票",
    "订阅",
    "套餐",
    "续费",
    "扣费",
    "member",
    "plan",
    "invoice",
    "billing",
    "subscription",
    "renewal",
    "charge",
    "account",
)
TICKET_HINTS = ("工单", "ticket", "状态", "进度", "升级", "催单", "账单异常")


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


class EvidenceItem(TypedDict, total=False):
    """Structured evidence item captured during retrieval or tool execution."""

    evidence_id: str
    kind: Literal["knowledge", "tool"]
    source_type: str
    source_label: str
    document_title: str
    section_path: str
    source_path: str
    snippet: str
    confidence: float
    tool_name: str
    tool_label: str
    metadata: Dict[str, Any]


class OrchestrationState(TypedDict, total=False):
    """Shared LangGraph orchestration state."""

    user_id: str
    thread_id: str
    trace_id: str
    current_message: str
    recent_history_text: str
    rolling_summary: str
    intent: str
    risk: str
    sentiment_label: str
    frustration_score: float
    route_reason: str
    selected_agent: str
    active_agent: str
    execution_steps: List[str]
    route_path: List[str]
    trace_events: List[Dict[str, Any]]
    node_timings: List[Dict[str, Any]]
    decision_summary: str
    needs_knowledge: bool
    needs_action: bool
    needs_action_after_knowledge: bool
    needs_escalation: bool
    retrieval_text: str
    tool_text: str
    evidence_items: List[EvidenceItem]
    validation_notes: List[str]
    validation_passed: bool
    final_message: str
    citations: List[str]
    tool_source: str
    pending_approval_plan: Dict[str, Any]
    langsmith_parent_headers: Dict[str, str]
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
    route_path: List[str] = field(default_factory=list)
    validation_notes: List[str] = field(default_factory=list)
    trace_preview: List[Dict[str, Any]] = field(default_factory=list)
    node_timings: List[Dict[str, Any]] = field(default_factory=list)
    decision_summary: str = ""
    approval: Optional[Dict[str, Any]] = None
    memory_debug: Dict[str, Any] = field(default_factory=dict)
    langsmith: Dict[str, Any] = field(default_factory=dict)
    next_action: str = ""
    total_duration_ms: Optional[float] = None

    def to_dict(self, include_debug: bool = True) -> Dict[str, Any]:
        """Convert the dataclass into an API-friendly payload."""
        payload: Dict[str, Any] = {
            "message": self.message,
            "thread_id": self.thread_id,
            "run_status": self.run_status,
            "active_agent": self.active_agent,
            "intent": self.intent,
            "sentiment": {
                "label": self.sentiment.label,
                "polarity": self.sentiment.polarity,
                "frustration_score": self.sentiment.frustration_score,
            },
            "result": {
                "escalated": self.escalated,
                "ticket_created": self.ticket_created,
                "sources": self.sources,
                "citations": self.citations,
            },
            "next_action": self.next_action,
        }
        if self.approval:
            payload["approval"] = self.approval
        if include_debug:
            payload["debug"] = {
                "trace_id": self.trace_id,
                "route_path": self.route_path,
                "node_timings": self.node_timings,
                "total_duration_ms": self.total_duration_ms,
            }
            if self.langsmith:
                payload["debug"]["langsmith"] = self.langsmith
        return payload


def build_node_timing(
    *,
    node: str,
    agent: str,
    duration_ms: float,
    status: str = "completed",
) -> Dict[str, Any]:
    """Build a lightweight node timing record for debug output."""
    return {
        "node": node,
        "agent": agent,
        "duration_ms": round(float(duration_ms), 2),
        "status": status,
    }


def append_node_timing(
    state: OrchestrationState,
    *,
    node: str,
    agent: str,
    duration_ms: float,
    status: str = "completed",
) -> Dict[str, Any]:
    """Append a node timing entry to orchestration state."""
    return {
        "node_timings": [
            *state.get("node_timings", []),
            build_node_timing(
                node=node,
                agent=agent,
                duration_ms=duration_ms,
                status=status,
            ),
        ]
    }


def build_trace_event(
    *,
    node: str,
    agent: str,
    summary: str,
    status: str = "completed",
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a lightweight trace event for graph debugging and demos."""
    event: Dict[str, Any] = {
        "node": node,
        "agent": agent,
        "status": status,
        "summary": summary,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    cleaned_details = {key: value for key, value in (details or {}).items() if value is not None}
    if cleaned_details:
        event["details"] = cleaned_details
    return event


def extend_trace(
    state: OrchestrationState,
    *,
    node: str,
    agent: str,
    summary: str,
    status: str = "completed",
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Append route path and trace event updates for a graph node."""
    return {
        "route_path": [*state.get("route_path", []), node],
        "trace_events": [
            *state.get("trace_events", []),
            build_trace_event(
                node=node,
                agent=agent,
                summary=summary,
                status=status,
                details=details,
            ),
        ],
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


def _clean_evidence_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _evidence_identity(item: EvidenceItem) -> str:
    metadata = item.get("metadata") or {}
    parts = [
        _clean_evidence_text(item.get("evidence_id")),
        _clean_evidence_text(item.get("kind")),
        _clean_evidence_text(item.get("source_label")),
        _clean_evidence_text(item.get("section_path")),
        _clean_evidence_text(item.get("tool_name")),
        _clean_evidence_text(item.get("snippet")),
        _clean_evidence_text(metadata.get("parent_chunk_id")),
        _clean_evidence_text(metadata.get("tool_call_id")),
    ]
    joined = "|".join(part for part in parts if part)
    return joined or json.dumps(item, ensure_ascii=False, sort_keys=True)


def merge_evidence_items(
    left: List[EvidenceItem],
    right: List[EvidenceItem],
) -> List[EvidenceItem]:
    merged: List[EvidenceItem] = []
    seen: set[str] = set()
    for candidate in [*(left or []), *(right or [])]:
        if not isinstance(candidate, dict):
            continue
        item: EvidenceItem = {
            key: value
            for key, value in candidate.items()
            if key
            in {
                "evidence_id",
                "kind",
                "source_type",
                "source_label",
                "document_title",
                "section_path",
                "source_path",
                "snippet",
                "confidence",
                "tool_name",
                "tool_label",
                "metadata",
            }
        }
        if isinstance(item.get("confidence"), (int, float)):
            item["confidence"] = round(float(item["confidence"]), 4)
        if item.get("metadata") and not isinstance(item["metadata"], dict):
            item["metadata"] = {"raw": item["metadata"]}
        identity = _evidence_identity(item)
        if identity in seen:
            continue
        seen.add(identity)
        merged.append(item)
    return merged


def evidence_source_label(item: EvidenceItem) -> str:
    source_label = _clean_evidence_text(item.get("source_label"))
    if source_label:
        return source_label
    section_path = _clean_evidence_text(item.get("section_path"))
    if section_path:
        return f"帮助中心::{section_path}"
    document_title = _clean_evidence_text(item.get("document_title"))
    if document_title:
        return f"帮助中心::{document_title}"
    tool_label = _clean_evidence_text(item.get("tool_label"))
    if tool_label:
        return tool_label
    tool_name = _clean_evidence_text(item.get("tool_name"))
    if tool_name:
        return f"工具::{tool_name}"
    return ""


def build_citations_from_evidence_items(items: List[EvidenceItem]) -> List[str]:
    citations: List[str] = []
    for item in items or []:
        label = evidence_source_label(item)
        if label and label not in citations:
            citations.append(label)
    return citations


def ticket_id_from_text(text: str) -> Optional[str]:
    match = re.search(r"TKT-\d{8,14}-\d{3,6}", text, re.IGNORECASE)
    return match.group(0).upper() if match else None


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


def strip_source_annotations(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"(?mi)^\s*(来源|出处|source)\s*[:：].*$", "", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


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
        "knowledge": "你是知识检索专家，优先调用 search_knowledge_base，负责回答帮助中心文档、订阅规则、操作说明与常见问题。",
        "action": "你是客服执行专家，负责订阅状态、账单解释、工单与账户类动作，必要时调用工具。",
        "escalation": "你是升级专员，负责人工介入、风险沟通与交接摘要。",
        "responder": "你是最终回答代理，负责融合证据与工具结果，生成最终答复。",
    }
    memory_lines: List[str] = []
    for item in memory_items:
        if not item:
            continue
        memory_type = str(item.get("memory_type", item.get("kind", "memory"))).strip().lower()
        content = (
            item.get("content")
            or item.get("summary")
            or item.get("fact")
            or item.get("message")
            or item.get("response")
            or ""
        )
        if not content:
            continue
        label_map = {
            "profile": "用户画像",
            "preference": "用户偏好",
            "open_issue": "待跟进问题",
            "resolved_issue": "已解决问题",
        }
        memory_lines.append(f"- [{label_map.get(memory_type, '记忆')}] {content}")

    memory_text = "\n".join(memory_lines)
    memory_block = f"\n用户长期记忆：\n{memory_text}" if memory_text else "\n用户长期记忆：暂无"
    return (
        f"{role_rules.get(role, '你是客服助手。')}\n"
        "输出必须为中文，语气专业、简洁、可执行。\n"
        "若信息不足，明确说明并提出下一步收集项。\n"
        "回答必须基于已给出的依据，不编造事实。"
        f"{memory_block}"
    )


def compose_action_prompt(state: OrchestrationState) -> str:
    history_text = state.get("recent_history_text", "")
    history_block = f"\n短期对话上下文：\n{history_text}\n" if history_text else ""
    retrieval = state.get("retrieval_text", "")
    retrieval_block = f"\n可用检索证据：\n{retrieval}\n" if retrieval else ""
    return (
        f"用户请求：{state.get('current_message', '')}\n"
        f"{history_block}"
        f"{retrieval_block}"
        "任务：判断是否需要调用订阅/账单/工单/账户类工具，必要时执行，输出执行结果。"
    )


def compose_escalation_prompt(state: OrchestrationState) -> str:
    history_text = state.get("recent_history_text", "")
    history_block = f"\n短期对话上下文：\n{history_text}\n" if history_text else ""
    retrieval = state.get("retrieval_text", "")
    retrieval_block = f"\n可用检索证据：\n{retrieval}\n" if retrieval else ""
    return (
        f"用户消息：{state.get('current_message', '')}\n"
        f"{history_block}"
        f"{retrieval_block}"
        "任务：按升级策略处理，高风险场景优先人工升级并给出交接摘要。"
    )


def compose_responder_prompt(state: OrchestrationState) -> str:
    notes = state.get("validation_notes", [])
    note_block = "\n".join(f"- {note}" for note in notes) if notes else "无"
    history_text = state.get("recent_history_text") or "无"
    retrieval_text = strip_source_annotations(state.get("retrieval_text") or "") or "无"
    tool_text = strip_source_annotations(state.get("tool_text") or "") or "无"
    decision_summary = state.get("decision_summary") or "无"
    evidence_count = len(state.get("evidence_items", []))
    return (
        "请基于以下结构化上下文生成最终客服答复：\n"
        f"- 用户问题：{state.get('current_message', '')}\n"
        f"- 短期对话上下文：\n{history_text}\n"
        f"- 识别意图：{state.get('intent', 'other')}\n"
        f"- 风险等级：{state.get('risk', 'low')}\n"
        f"- 情绪标签：{state.get('sentiment_label', 'neutral')}\n"
        f"- 挫败分：{state.get('frustration_score', 0.0):.2f}\n"
        f"- 图决策摘要：{decision_summary}\n"
        f"- 知识证据：\n{retrieval_text}\n"
        f"- 工具执行结果：\n{tool_text}\n"
        f"- 可用结构化证据数：{evidence_count}\n"
        f"- 是否人工升级：{'是' if state.get('escalated') else '否'}\n"
        f"- 工单号：{state.get('ticket_id') or '无'}\n"
        f"- 校验修正要求：\n{note_block}\n"
        "要求：中文输出；先结论后步骤；只输出答复正文；不要输出“来源：”“出处：”或自造文档标题；如果证据不足，要明确告知并给出下一步建议。"
    )


class SupportAgentOrchestrator:
    """Owns graph construction and node implementations."""

    def __init__(self, owner: Any) -> None:
        self.owner = owner

    def _run_timed_node(
        self,
        state: OrchestrationState,
        *,
        node: str,
        agent: str,
        handler: Any,
    ) -> Dict[str, Any]:
        started = perf_counter()
        result = dict(handler(state) or {})
        elapsed_ms = (perf_counter() - started) * 1000

        status = "completed"
        if result.get("run_status") == "interrupted":
            status = "interrupted"
        elif result.get("run_status") == "error":
            status = "error"
        elif node == "validate" and result.get("validation_passed") is False:
            status = "needs_revision"

        result.update(
            append_node_timing(
                state,
                node=node,
                agent=agent,
                duration_ms=elapsed_ms,
                status=status,
            )
        )
        return result

    def build(self) -> Any:
        workflow = StateGraph(OrchestrationState)
        workflow.add_node(
            "analyze",
            lambda state: self._run_timed_node(
                state,
                node="analyze",
                agent="supervisor",
                handler=self.node_analyze,
            ),
        )
        workflow.add_node(
            "knowledge",
            lambda state: self._run_timed_node(
                state,
                node="knowledge",
                agent="knowledge",
                handler=self.node_knowledge,
            ),
        )
        workflow.add_node(
            "action",
            lambda state: self._run_timed_node(
                state,
                node="action",
                agent="action",
                handler=self.node_action,
            ),
        )
        workflow.add_node(
            "escalation",
            lambda state: self._run_timed_node(
                state,
                node="escalation",
                agent="escalation",
                handler=self.node_escalation,
            ),
        )
        workflow.add_node(
            "validate",
            lambda state: self._run_timed_node(
                state,
                node="validate",
                agent="validator",
                handler=self.node_validate,
            ),
        )
        workflow.add_node(
            "respond",
            lambda state: self._run_timed_node(
                state,
                node="respond",
                agent="responder",
                handler=self.node_respond,
            ),
        )

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
        for node_name, agent_name, handler in (
            ("validate", "validator", self.node_validate),
            ("respond", "responder", self.node_respond),
        ):
            merged.update(
                self._run_timed_node(
                    merged,
                    node=node_name,
                    agent=agent_name,
                    handler=handler,
                )
            )
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
                "decision_summary": route["decision_summary"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        trace_update = extend_trace(
            state,
            node="analyze",
            agent="supervisor",
            summary=route["decision_summary"],
            details={
                "selected_agent": route["selected_agent"],
                "execution_steps": route["execution_steps"],
                "intent": route["intent"],
                "risk": route["risk"],
            },
        )
        return {
            "intent": route["intent"],
            "risk": route["risk"],
            "sentiment_label": route["sentiment_label"],
            "frustration_score": route["frustration_score"],
            "route_reason": route["route_reason"],
            "decision_summary": route["decision_summary"],
            "selected_agent": route["selected_agent"],
            "active_agent": route["selected_agent"],
            "execution_steps": route["execution_steps"],
            "needs_knowledge": route["needs_knowledge"],
            "needs_action": route["needs_action"],
            "needs_action_after_knowledge": route["needs_action_after_knowledge"],
            "needs_escalation": route["needs_escalation"],
            "run_status": "completed",
            "interrupts": [],
            **trace_update,
        }

    def node_knowledge(self, state: OrchestrationState) -> Dict[str, Any]:
        bundle = self.owner._run_knowledge_lookup_bundle(state.get("current_message", ""))
        retrieval_text = strip_source_annotations(bundle.get("text", ""))
        evidence_items = merge_evidence_items(
            state.get("evidence_items", []),
            bundle.get("evidence_items", []),
        )
        trace_update = extend_trace(
            state,
            node="knowledge",
            agent="knowledge",
            summary="知识检索已完成。",
            details={
                "has_retrieval": bool(retrieval_text),
                "evidence_count": len(evidence_items),
                "text_length": len(retrieval_text),
            },
        )
        return {
            "active_agent": "knowledge",
            "retrieval_text": retrieval_text,
            "evidence_items": evidence_items,
            "run_status": "completed",
            **trace_update,
        }

    def node_action(self, state: OrchestrationState) -> Dict[str, Any]:
        user_id = state.get("user_id", "unknown_user")
        thread_id = state.get("thread_id", "unknown_thread")
        intent = state.get("intent", "request")
        risk = state.get("risk", "medium")

        if self.owner.llm_enabled and self.owner._is_billing_ticket_request(state.get("current_message", "")):
            interrupt_payload = self.owner._build_billing_ticket_interrupt(
                user_id=user_id,
                message=state.get("current_message", ""),
            )
            interrupts = interrupt_payload.get("interrupts", [])
            interrupt_tools = [item.get("tool_label") or item.get("tool") for item in interrupts]
            trace_update = extend_trace(
                state,
                node="action",
                agent="action",
                summary=f"识别到显式写操作，请先审批：{'、'.join(interrupt_tools)}。",
                status="interrupted",
                details={"interrupt_count": len(interrupts), "tools": interrupt_tools},
            )
            return {
                "active_agent": "action",
                "run_status": "interrupted",
                "interrupts": interrupts,
                "evidence_items": state.get("evidence_items", []),
                "pending_approval_plan": interrupt_payload.get("pending_approval_plan", {}),
                **trace_update,
            }

        structured_action = self.owner._run_structured_business_action(
            user_id=user_id,
            message=state.get("current_message", ""),
        )
        evidence_items = state.get("evidence_items", [])
        tool_source = state.get("tool_source", "")
        if structured_action is not None:
            tool_text = structured_action.get("text", "")
            evidence_items = merge_evidence_items(
                evidence_items,
                structured_action.get("evidence_items", []),
            )
            tool_source = structured_action.get("tool_source", "") or tool_source
        elif self.owner.llm_enabled:
            result = self.owner._call_role_agent(
                role="action",
                user_id=user_id,
                thread_id=thread_id,
                trace_id=state.get("trace_id", ""),
                intent=intent,
                risk=risk,
                message=compose_action_prompt(state),
            )
            interrupts = self.owner._extract_interrupts(result)
            if interrupts:
                interrupt_tools = [item.get("tool_label") or item.get("tool") for item in interrupts]
                trace_update = extend_trace(
                    state,
                    node="action",
                    agent="action",
                    summary=f"动作执行命中 HITL，待审批动作：{'、'.join(interrupt_tools)}。",
                    status="interrupted",
                    details={"interrupt_count": len(interrupts), "tools": interrupt_tools},
                )
                return {
                    "active_agent": "action",
                    "run_status": "interrupted",
                    "interrupts": interrupts,
                    "evidence_items": state.get("evidence_items", []),
                    **trace_update,
                }
            tool_text = self.owner._extract_ai_text(result)
            evidence_items = merge_evidence_items(
                evidence_items,
                self.owner._extract_tool_evidence_items(result),
            )
            tool_source = tool_source or "Support Tools"
        else:
            tool_text = self.owner._fallback_response("action", user_id, state.get("current_message", ""), None)

        tool_text = strip_source_annotations(tool_text)
        ticket_id = (
            (structured_action or {}).get("ticket_id")
            if isinstance(structured_action, dict)
            else None
        ) or ticket_id_from_text(tool_text) or state.get("ticket_id")
        escalated = (
            bool(state.get("escalated"))
            or bool((structured_action or {}).get("escalated"))
            or is_positive_escalation_text(tool_text)
        )
        trace_update = extend_trace(
            state,
            node="action",
            agent="action",
            summary="动作执行已完成。",
            details={
                "ticket_id": ticket_id,
                "escalated": escalated,
                "evidence_count": len(evidence_items),
            },
        )
        return {
            "active_agent": "action",
            "tool_text": tool_text,
            "tool_source": tool_source or "Support Tools",
            "ticket_id": ticket_id,
            "escalated": escalated,
            "run_status": "completed",
            "interrupts": [],
            "evidence_items": evidence_items,
            **trace_update,
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
                trace_id=state.get("trace_id", ""),
                intent=intent,
                risk="high",
                message=compose_escalation_prompt(state),
            )
            interrupts = self.owner._extract_interrupts(result)
            if interrupts:
                interrupt_tools = [item.get("tool_label") or item.get("tool") for item in interrupts]
                trace_update = extend_trace(
                    state,
                    node="escalation",
                    agent="escalation",
                    summary=f"人工升级动作待审批：{'、'.join(interrupt_tools)}。",
                    status="interrupted",
                    details={"interrupt_count": len(interrupts), "tools": interrupt_tools},
                )
                return {
                    "active_agent": "escalation",
                    "run_status": "interrupted",
                    "interrupts": interrupts,
                    "escalated": True,
                    "evidence_items": state.get("evidence_items", []),
                    **trace_update,
                }
            tool_text = self.owner._extract_ai_text(result)
            evidence_items = merge_evidence_items(
                state.get("evidence_items", []),
                self.owner._extract_tool_evidence_items(result),
            )
        else:
            tool_text = self.owner._fallback_response("escalation", user_id, state.get("current_message", ""), None)
            evidence_items = state.get("evidence_items", [])

        tool_text = strip_source_annotations(tool_text)
        ticket_id = ticket_id_from_text(tool_text) or state.get("ticket_id")
        trace_update = extend_trace(
            state,
            node="escalation",
            agent="escalation",
            summary="人工升级流程已完成。",
            details={
                "ticket_id": ticket_id,
                "evidence_count": len(evidence_items),
            },
        )
        return {
            "active_agent": "escalation",
            "tool_text": tool_text,
            "tool_source": "Support Tools",
            "ticket_id": ticket_id,
            "escalated": True,
            "run_status": "completed",
            "interrupts": [],
            "evidence_items": evidence_items,
            **trace_update,
        }

    def node_validate(self, state: OrchestrationState) -> Dict[str, Any]:
        notes: List[str] = []
        retrieval_text = (state.get("retrieval_text") or "").strip()
        tool_text = (state.get("tool_text") or "").strip()
        evidence_items = state.get("evidence_items", [])
        current_message = state.get("current_message", "")
        combined_text = "\n".join(part for part in (retrieval_text, tool_text) if part).strip()
        has_evidence = bool(combined_text or evidence_items)
        selected_agent = state.get("selected_agent")
        is_subscription_query = any(token in current_message for token in ("套餐", "订阅", "续费"))
        is_billing_query = any(token in current_message for token in ("账单", "发票", "扣费", "扣款", "金额"))
        needs_next_step = (
            not combined_text
            or selected_agent in {"action", "escalation"}
            or state.get("intent") == "complaint"
        )
        has_next_step_signal = any(
            token in combined_text
            for token in ("下一步", "接下来", "稍后", "分钟", "小时", "工作日", "请", "建议")
        )

        if not combined_text:
            notes.append("当前没有有效证据或执行结果，需要明确告知用户信息不足并给出下一步建议。")
        elif not contains_chinese(combined_text):
            notes.append("最终回复必须保持中文输出。")

        if (retrieval_text or tool_text) and not evidence_items:
            notes.append("知识检索或工具执行已产出结果，但缺少结构化证据，无法稳定生成引用。")
        if state.get("ticket_id") and state["ticket_id"] not in combined_text:
            notes.append("如果已经生成工单号，需要在最终回复中明确展示。")
        if state.get("escalated") and "人工" not in combined_text and "升级" not in combined_text:
            notes.append("如果已经升级人工，需要明确告知用户后续人工跟进。")
        if state.get("selected_agent") in {"action", "escalation"} and not tool_text and state.get("run_status") == "completed":
            notes.append("如果本轮未真正执行动作，需要解释原因并给出下一步处理方式。")
        if is_subscription_query and not any(token in combined_text for token in ("套餐", "订阅", "续费", "状态")):
            notes.append("订阅类问题的最终回复需要明确写出套餐、订阅状态或续费信息。")
        if is_billing_query and not any(token in combined_text for token in ("账单", "发票", "金额", "扣费")):
            notes.append("账单类问题的最终回复需要明确写出账单、金额或扣费说明。")
        if needs_next_step and not has_next_step_signal:
            notes.append("最终回复需要先直接回答用户核心问题，再给出明确的下一步建议或处理时效。")
        if not has_evidence and state.get("intent") in {"question", "request", "complaint"}:
            notes.append("如果证据不足，需要主动说明信息缺口并引导补充信息。")

        trace_update = extend_trace(
            state,
            node="validate",
            agent="validator",
            summary="校验通过。" if not notes else f"校验发现 {len(notes)} 项修正要求。",
            status="completed" if not notes else "needs_revision",
            details={"validation_passed": not notes, "note_count": len(notes)},
        )
        return {
            "validation_notes": notes,
            "validation_passed": not notes,
            **trace_update,
        }

    def node_respond(self, state: OrchestrationState) -> Dict[str, Any]:
        if state.get("run_status") == "interrupted":
            return {}

        if self.owner.llm_enabled:
            result = self.owner._call_role_agent(
                role="responder",
                user_id=state.get("user_id", "unknown_user"),
                thread_id=state.get("thread_id", "unknown_thread"),
                trace_id=state.get("trace_id", ""),
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

        final_message = strip_source_annotations(final_message)
        ticket_id = state.get("ticket_id") or ticket_id_from_text(final_message)
        escalated = bool(state.get("escalated") or state.get("selected_agent") == "escalation")
        trace_update = extend_trace(
            state,
            node="respond",
            agent="responder",
            summary="最终回复已生成。",
            details={
                "message_length": len(final_message),
                "evidence_count": len(state.get("evidence_items", [])),
                "validation_passed": state.get("validation_passed", True),
            },
        )
        return {
            "final_message": final_message,
            "ticket_id": ticket_id,
            "escalated": escalated,
            "run_status": "completed",
            **trace_update,
        }
