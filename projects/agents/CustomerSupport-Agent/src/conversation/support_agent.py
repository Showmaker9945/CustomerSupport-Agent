"""
基于 LangGraph 条件边编排的多 Agent 客服核心。

能力说明：
- 显式条件边流程：分析 -> 检索/执行/升级 -> 回答
- middleware（before_model / dynamic_prompt / wrap_model_call / wrap_tool_call / after_model）
- 高风险工具 HITL 中断与恢复
- 线程级持久化（checkpointer）+ 用户级长期记忆（store）
- 中文优先输出
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    after_model,
    before_model,
    dynamic_prompt,
    wrap_model_call,
    wrap_tool_call,
)
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, StateGraph
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import PostgresStore
from langgraph.types import Command

from ..config import settings
from ..sentiment.analyzer import SentimentResult, get_sentiment_analyzer
from ..tools.support_tools import (
    create_ticket,
    escalate_to_human,
    get_ticket_status,
    get_user_tickets,
    lookup_account,
    reindex_knowledge_base,
    search_faq,
    update_ticket,
)

logger = logging.getLogger(__name__)


class ConversationState(TypedDict):
    """对外兼容的会话状态定义。"""

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
    """LangGraph 编排状态。"""

    user_id: str
    thread_id: str
    current_message: str
    intent: str
    risk: str
    sentiment_label: str
    frustration_score: float
    selected_agent: str
    active_agent: str
    needs_knowledge: bool
    needs_action: bool
    needs_action_after_knowledge: bool
    needs_escalation: bool
    retrieval_text: str
    tool_text: str
    final_message: str
    citations: List[str]
    run_status: Literal["completed", "interrupted", "error"]
    interrupts: List[Dict[str, Any]]
    ticket_id: Optional[str]
    escalated: bool


@dataclass
class AgentRuntimeContext:
    """LangGraph runtime context（注入 middleware）。"""

    user_id: str
    thread_id: str
    active_agent: str
    intent: str = "other"
    risk: str = "low"
    locale: str = settings.default_response_language


@dataclass
class SupportResponse:
    """统一客服返回结构（兼容旧字段 + 新字段）。"""

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
        """转换为 API 友好字典。"""
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


class LangGraphPersistence:
    """封装 checkpointer/store 初始化与资源释放。"""

    def __init__(self) -> None:
        self.checkpointer: Any = None
        self.store: Any = None
        self._checkpointer_cm: Any = None
        self._store_cm: Any = None
        self.backend = "memory"
        self._init_backend()

    def _init_backend(self) -> None:
        """优先使用 Postgres，不可用时回退内存实现。"""
        if settings.use_postgres_langgraph:
            try:
                self._store_cm = PostgresStore.from_conn_string(settings.postgres_uri)
                self._checkpointer_cm = PostgresSaver.from_conn_string(settings.postgres_uri)
                self.store = self._store_cm.__enter__()
                self.checkpointer = self._checkpointer_cm.__enter__()

                if settings.auto_setup_postgres:
                    with suppress(Exception):
                        self.store.setup()
                    with suppress(Exception):
                        self.checkpointer.setup()

                self.backend = "postgres"
                logger.info("LangGraph persistence backend: postgres")
                return
            except Exception as error:
                logger.warning(f"Postgres backend unavailable, fallback to memory: {error}")
                with suppress(Exception):
                    if self._store_cm:
                        self._store_cm.__exit__(None, None, None)
                with suppress(Exception):
                    if self._checkpointer_cm:
                        self._checkpointer_cm.__exit__(None, None, None)

        self.checkpointer = InMemorySaver()
        self.store = InMemoryStore()
        self.backend = "memory"
        logger.info("LangGraph persistence backend: memory")

    def close(self) -> None:
        """释放资源。"""
        with suppress(Exception):
            if self._store_cm:
                self._store_cm.__exit__(None, None, None)
        with suppress(Exception):
            if self._checkpointer_cm:
                self._checkpointer_cm.__exit__(None, None, None)


class SupportAgent:
    """
    多 Agent 客服调度器（LangGraph 条件边编排）。

    流程：
    - analyze: LLM/回退规则做意图、风险、情绪与路由规划
    - knowledge/action/escalation: 执行子代理任务
    - respond: 回答代理统一收敛输出
    """

    QUESTION_HINTS = (
        "怎么", "如何", "为什么", "what", "how", "why", "?", "？", "说明", "教程", "帮助文档"
    )
    REQUEST_HINTS = (
        "创建", "新建", "工单", "ticket", "状态", "进度", "查询", "更新", "账户", "账号", "账单"
    )
    ESCALATION_HINTS = (
        "人工", "投诉", "经理", "马上处理", "退款", "起诉", "furious", "unacceptable", "sue"
    )
    ACCOUNT_HINTS = ("账户", "账号", "账单", "member", "plan", "invoice", "account")
    TICKET_HINTS = ("工单", "ticket", "状态", "进度", "升级", "催单")

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        enable_memory: bool = True,
        enable_sentiment: bool = True,
    ) -> None:
        self.model_name = model_name or settings.llm_model
        self.temperature = settings.llm_temperature if temperature is None else temperature
        self.enable_memory = enable_memory
        self.enable_sentiment = enable_sentiment
        disable_llm = os.getenv("DISABLE_LLM", "").strip().lower() in {"1", "true", "yes", "on"}
        self.llm_enabled = settings.has_valid_llm_api_key and not disable_llm
        self.persistence = LangGraphPersistence()
        self.sentiment_analyzer = get_sentiment_analyzer() if enable_sentiment else None

        self._lock = threading.Lock()
        self._history: Dict[str, List[Dict[str, str]]] = {}
        self._thread_user: Dict[str, str] = {}
        self._pending_role: Dict[str, str] = {}
        self._pending_state: Dict[str, OrchestrationState] = {}
        self._trace_by_thread: Dict[str, str] = {}

        self.basic_model: Optional[ChatOpenAI] = None
        self.advanced_model: Optional[ChatOpenAI] = None
        self.role_agents: Dict[str, Any] = {}

        if self.llm_enabled:
            self.basic_model = self._create_model(self.model_name, self.temperature)
            self.advanced_model = self._create_model(settings.llm_high_quality_model, 0.2)
            self._build_role_agents()
        else:
            logger.warning("LLM 未可用：将使用规则路由与模板回复。")

        self.orchestration_graph = self._build_orchestration_graph()

    def close(self) -> None:
        """释放底层资源。"""
        self.persistence.close()

    def _create_model(self, model_name: str, temperature: float) -> ChatOpenAI:
        kwargs: Dict[str, Any] = {
            "model": model_name,
            "temperature": temperature,
            "api_key": settings.resolved_llm_api_key,
            "request_timeout": 45.0,
        }
        if settings.llm_base_url:
            kwargs["base_url"] = settings.llm_base_url
        return ChatOpenAI(**kwargs)

    def _memory_namespace(self, user_id: str) -> Tuple[str, str]:
        return (settings.long_term_memory_namespace, user_id)

    def _save_memory_item(self, user_id: str, payload: Dict[str, Any]) -> None:
        if not self.enable_memory or self.persistence.store is None:
            return
        digest = hashlib.md5(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        self.persistence.store.put(self._memory_namespace(user_id), digest, payload)

    def _search_memory(self, user_id: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        if not self.enable_memory or self.persistence.store is None:
            return []
        try:
            items = self.persistence.store.search(
                self._memory_namespace(user_id),
                query=query,
                limit=limit,
            )
            return [item.value for item in items]
        except Exception as error:
            logger.warning(f"Memory search failed: {error}")
            return []

    def _build_role_agents(self) -> None:
        """创建角色 Agent。"""
        self.role_agents = {
            "supervisor": self._create_role_agent(
                role="supervisor",
                tools=[search_faq, lookup_account, get_ticket_status],
                enable_hitl=False,
            ),
            "knowledge": self._create_role_agent(
                role="knowledge",
                tools=[search_faq],
                enable_hitl=False,
            ),
            "action": self._create_role_agent(
                role="action",
                tools=[
                    search_faq,
                    create_ticket,
                    update_ticket,
                    get_ticket_status,
                    get_user_tickets,
                    lookup_account,
                    escalate_to_human,
                ],
                enable_hitl=True,
            ),
            "escalation": self._create_role_agent(
                role="escalation",
                tools=[escalate_to_human, create_ticket, get_ticket_status],
                enable_hitl=True,
            ),
            "responder": self._create_role_agent(
                role="responder",
                tools=[],
                enable_hitl=False,
            ),
        }

    def _create_role_agent(self, role: str, tools: List[Any], enable_hitl: bool) -> Any:
        """按角色创建 create_agent 实例。"""

        @dynamic_prompt
        def role_prompt(request: ModelRequest) -> str:
            context = getattr(request.runtime, "context", None)
            user_id = getattr(context, "user_id", "unknown_user")
            latest_user = ""
            for message in reversed(request.state.get("messages", [])):
                if isinstance(message, HumanMessage):
                    latest_user = str(message.content)
                    break

            memory_items = self._search_memory(user_id=user_id, query=latest_user, limit=4)
            memory_text = "\n".join(
                f"- {item.get('fact', item.get('message', ''))}"
                for item in memory_items
                if item
            )
            memory_block = f"\n用户长期记忆：\n{memory_text}" if memory_text else "\n用户长期记忆：暂无"

            role_rules = {
                "supervisor": "你是客服总调度，负责判断问题类型并给出决策建议。",
                "knowledge": "你是知识检索专家，优先调用 search_faq，严格基于证据回复。",
                "action": "你是客服执行专家，负责工单与账户类动作，必要时调用工具。",
                "escalation": "你是升级专员，负责人工介入、风险沟通与交接摘要。",
                "responder": "你是最终回答代理，负责融合证据与工具结果，生成最终答复。",
            }
            return (
                f"{role_rules.get(role, '你是客服助手。')}\n"
                "输出必须为中文，语气专业、简洁、可执行。\n"
                "若信息不足，明确说明并提出下一步收集项。\n"
                "回答尽量包含来源与依据，不编造事实。"
                f"{memory_block}"
            )

        @before_model
        def trim_history(state: Dict[str, Any], _runtime: Any) -> Dict[str, Any] | None:
            messages = state.get("messages", [])
            max_keep = max(8, settings.max_conversation_history)
            if len(messages) <= max_keep:
                return None
            removals = [
                RemoveMessage(id=msg.id) for msg in messages[:-max_keep] if getattr(msg, "id", None)
            ]
            if removals:
                return {"messages": removals}
            return None

        @wrap_model_call
        def dynamic_model_selector(
            request: ModelRequest, handler: Any
        ) -> ModelResponse:
            if self.advanced_model is None:
                return handler(request)
            message_count = len(request.state.get("messages", []))
            context = getattr(request.runtime, "context", None)
            risk = getattr(context, "risk", "low")
            if message_count > 12 or risk == "high":
                return handler(request.override(model=self.advanced_model))
            return handler(request)

        @wrap_tool_call
        def safe_tool_wrapper(request: ToolCallRequest, handler: Any) -> Any:
            tool_name = request.tool_call.get("name", "unknown_tool")
            args = request.tool_call.get("args", {})
            logger.info(f"[tool_call] role={role} tool={tool_name} args={args}")

            context = getattr(request.runtime, "context", None)
            user_id = getattr(context, "user_id", "unknown_user")
            thread_id = getattr(context, "thread_id", "unknown_thread")
            self._save_memory_item(
                user_id=user_id,
                payload={
                    "kind": "tool_call",
                    "role": role,
                    "thread_id": thread_id,
                    "tool_name": tool_name,
                    "args": args,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            try:
                return handler(request)
            except Exception as error:
                logger.error(f"Tool execution failed: {tool_name}, error={error}")
                return ToolMessage(
                    content=f"工具 `{tool_name}` 执行失败：{str(error)}",
                    tool_call_id=request.tool_call.get("id", "unknown_id"),
                    name=tool_name,
                    status="error",
                )

        @after_model
        def output_guard(state: Dict[str, Any], _runtime: Any) -> Dict[str, Any] | None:
            messages = state.get("messages", [])
            last_ai: Optional[AIMessage] = next(
                (msg for msg in reversed(messages) if isinstance(msg, AIMessage)),
                None,
            )
            if last_ai is None:
                return None

            raw_content = str(last_ai.content or "")
            guarded = re.sub(r"sk-[A-Za-z0-9_\-]{8,}", "[REDACTED]", raw_content)
            if not re.search(r"[\u4e00-\u9fff]", guarded):
                guarded = "以下为中文回复：\n" + guarded

            if guarded == raw_content:
                return None
            if getattr(last_ai, "id", None):
                return {"messages": [RemoveMessage(id=last_ai.id), AIMessage(content=guarded)]}
            return None

        middleware: List[Any] = [
            role_prompt,
            trim_history,
            dynamic_model_selector,
            safe_tool_wrapper,
            output_guard,
        ]

        if enable_hitl:
            interrupt_on = {tool: True for tool in settings.hitl_high_risk_tools}
            middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

        return create_agent(
            model=self.basic_model,
            tools=tools,
            middleware=middleware,
            context_schema=AgentRuntimeContext,
            checkpointer=self.persistence.checkpointer,
            store=self.persistence.store,
            name=f"{role}_agent",
        )

    def _build_orchestration_graph(self) -> Any:
        """构建 LangGraph 条件边编排流程。"""
        workflow = StateGraph(OrchestrationState)

        workflow.add_node("analyze", self._node_analyze)
        workflow.add_node("knowledge", self._node_knowledge)
        workflow.add_node("action", self._node_action)
        workflow.add_node("escalation", self._node_escalation)
        workflow.add_node("respond", self._node_respond)

        workflow.set_entry_point("analyze")

        workflow.add_conditional_edges(
            "analyze",
            self._route_after_analyze,
            {
                "knowledge": "knowledge",
                "action": "action",
                "escalation": "escalation",
                "respond": "respond",
            },
        )

        workflow.add_conditional_edges(
            "knowledge",
            self._route_after_knowledge,
            {
                "action": "action",
                "respond": "respond",
            },
        )

        workflow.add_conditional_edges(
            "action",
            self._route_after_execution,
            {
                "respond": "respond",
                "end": END,
            },
        )

        workflow.add_conditional_edges(
            "escalation",
            self._route_after_execution,
            {
                "respond": "respond",
                "end": END,
            },
        )

        workflow.add_edge("respond", END)
        return workflow.compile()

    def _infer_intent(self, message: str) -> str:
        lowered = message.lower().strip()
        if not lowered:
            return "other"
        if any(token in lowered for token in self.ESCALATION_HINTS):
            return "complaint"
        if any(token in lowered for token in self.REQUEST_HINTS):
            return "request"
        if any(token in lowered for token in self.QUESTION_HINTS):
            return "question"
        if any(token in lowered for token in ("hello", "hi", "你好", "感谢", "谢谢", "thanks")):
            return "greeting"
        return "other"

    def _infer_risk(self, message: str, sentiment: Optional[SentimentResult]) -> str:
        lowered = message.lower()
        if any(token in lowered for token in self.ESCALATION_HINTS):
            return "high"
        if sentiment and sentiment.frustration_score >= 0.75:
            return "high"
        if sentiment and sentiment.frustration_score >= 0.45:
            return "medium"
        return "low"

    def _normalize_intent(self, value: Any) -> str:
        valid = {"question", "complaint", "request", "feedback", "greeting", "other"}
        text = str(value or "").strip().lower()
        return text if text in valid else "other"

    def _normalize_risk(self, value: Any) -> str:
        valid = {"low", "medium", "high"}
        text = str(value or "").strip().lower()
        return text if text in valid else "low"

    def _normalize_sentiment_label(self, value: Any) -> str:
        valid = {"positive", "neutral", "negative"}
        text = str(value or "").strip().lower()
        return text if text in valid else "neutral"

    def _as_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on", "y"}
        return False

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            result = float(value)
            return max(0.0, min(1.0, result))
        except Exception:
            return default

    def _extract_json_payload(self, text: str) -> Optional[Dict[str, Any]]:
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

    def _llm_analyze_message(
        self,
        message: str,
        baseline_intent: str,
        baseline_risk: str,
        baseline_sentiment: SentimentResult,
    ) -> Optional[Dict[str, Any]]:
        if not self.llm_enabled or self.basic_model is None:
            return None

        system_prompt = (
            "你是客服路由决策器。"
            "请严格返回 JSON，不要输出任何额外解释。"
            "字段："
            "intent(question|complaint|request|feedback|greeting|other),"
            "risk(low|medium|high),"
            "sentiment_label(positive|neutral|negative),"
            "frustration_score(0到1),"
            "needs_knowledge(boolean),"
            "needs_action(boolean),"
            "needs_escalation(boolean),"
            "reason(字符串)。"
        )
        user_prompt = (
            f"用户消息：{message}\n"
            f"基线意图：{baseline_intent}\n"
            f"基线风险：{baseline_risk}\n"
            f"基线情绪：{baseline_sentiment.label}\n"
            f"基线挫败分：{baseline_sentiment.frustration_score:.2f}"
        )

        try:
            response = self.basic_model.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            payload = self._extract_json_payload(str(response.content))
            if not payload:
                return None
            return {
                "intent": self._normalize_intent(payload.get("intent")),
                "risk": self._normalize_risk(payload.get("risk")),
                "sentiment_label": self._normalize_sentiment_label(payload.get("sentiment_label")),
                "frustration_score": self._safe_float(
                    payload.get("frustration_score"),
                    default=baseline_sentiment.frustration_score,
                ),
                "needs_knowledge": self._as_bool(payload.get("needs_knowledge")),
                "needs_action": self._as_bool(payload.get("needs_action")),
                "needs_escalation": self._as_bool(payload.get("needs_escalation")),
                "reason": str(payload.get("reason", "")).strip(),
            }
        except Exception as error:
            logger.warning(f"LLM analysis failed, fallback to heuristic routing: {error}")
            return None

    def _plan_route(
        self,
        intent: str,
        risk: str,
        needs_knowledge: bool,
        needs_action: bool,
        needs_escalation: bool,
    ) -> Tuple[str, bool]:
        if risk == "high":
            needs_escalation = True
        if needs_escalation:
            return "escalation", False
        if needs_knowledge:
            return "knowledge", needs_action
        if needs_action:
            return "action", False
        if intent in {"question", "complaint"}:
            return "knowledge", False
        if intent == "request":
            return "action", False
        return "supervisor", False

    def _merge_unique(self, left: List[str], right: List[str]) -> List[str]:
        merged = list(left)
        for item in right:
            cleaned = str(item).strip()
            if cleaned and cleaned not in merged:
                merged.append(cleaned)
        return merged

    def _ticket_id_from_text(self, text: str) -> Optional[str]:
        match = re.search(r"TKT-\d{8,14}-\d{3,6}", text, re.IGNORECASE)
        return match.group(0).upper() if match else None

    def _extract_citations(self, text: str) -> List[str]:
        citations = re.findall(r"来源：([^\n]+)", text)
        unique: List[str] = []
        for cite in citations:
            cleaned = cite.strip()
            if cleaned and cleaned not in unique:
                unique.append(cleaned)
        return unique

    def _agent_thread_id(self, thread_id: str, role: str) -> str:
        return f"{settings.langgraph_thread_prefix}:{role}:{thread_id}"

    def _extract_interrupts(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw_interrupts = result.get("__interrupt__", []) if isinstance(result, dict) else []
        payloads: List[Dict[str, Any]] = []
        for item in raw_interrupts:
            payloads.append(
                {
                    "id": getattr(item, "id", None),
                    "value": getattr(item, "value", item),
                }
            )
        return payloads

    def _extract_ai_text(self, result: Dict[str, Any]) -> str:
        messages = result.get("messages", []) if isinstance(result, dict) else []
        for message in reversed(messages):
            if isinstance(message, AIMessage) and message.content:
                return str(message.content)
        return "抱歉，我暂时无法给出有效回复。"

    def _fallback_response(
        self,
        role: str,
        user_id: str,
        message: str,
        sentiment: Optional[SentimentResult],
        retrieval_text: str = "",
        tool_text: str = "",
    ) -> str:
        intro = ""
        if sentiment and sentiment.frustration_score >= 0.45:
            intro = "我理解你现在比较着急，我们先快速定位问题。"

        lowered = message.lower()
        if role == "knowledge":
            try:
                return intro + search_faq.invoke({"query": message, "category": None})
            except Exception as error:
                return intro + f"知识检索失败：{error}"

        if role == "action":
            if "工单" in message and any(token in lowered for token in ("创建", "create", "open", "new")):
                return intro + create_ticket.invoke(
                    {
                        "user_id": user_id,
                        "subject": message[:80],
                        "description": message,
                        "priority": "medium",
                    }
                )
            if "状态" in message or "status" in lowered:
                ticket_id = self._ticket_id_from_text(message)
                if ticket_id:
                    return intro + get_ticket_status.invoke({"ticket_id": ticket_id})
                return intro + get_user_tickets.invoke({"user_id": user_id, "status": None})
            if "账户" in message or "account" in lowered:
                return intro + lookup_account.invoke({"user_id": user_id})
            return intro + "请提供更具体的执行信息（例如工单号、账户ID、操作目标）。"

        if role == "escalation":
            return intro + escalate_to_human.invoke(
                {
                    "user_id": user_id,
                    "reason": "高风险或高情绪场景",
                    "conversation_summary": message[:400],
                }
            )

        if role == "responder":
            response_parts: List[str] = []
            if retrieval_text:
                response_parts.append(retrieval_text)
            if tool_text:
                response_parts.append(tool_text)
            if response_parts:
                return intro + "\n\n".join(response_parts)
            return (
                intro
                + "我可以帮助你处理产品使用、账单账户、技术排障与工单问题。"
                "请告诉我你遇到的具体现象（操作步骤、报错信息、期望结果）。"
            )

        return (
            intro
            + "我可以帮助你处理产品使用、账单账户、技术排障与工单问题。"
            "请告诉我你遇到的具体现象（操作步骤、报错信息、期望结果）。"
        )

    def _record_history(self, user_id: str, role: str, content: str) -> None:
        with self._lock:
            self._history.setdefault(user_id, []).append(
                {
                    "role": role,
                    "content": content,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            max_items = max(settings.max_conversation_history * 4, 40)
            if len(self._history[user_id]) > max_items:
                self._history[user_id] = self._history[user_id][-max_items:]

    def _save_turn_memory(
        self,
        user_id: str,
        thread_id: str,
        intent: str,
        active_agent: str,
        user_message: str,
        assistant_message: str,
        sentiment: Optional[SentimentResult],
    ) -> None:
        if not self.enable_memory:
            return
        payload = {
            "kind": "conversation_turn",
            "thread_id": thread_id,
            "intent": intent,
            "active_agent": active_agent,
            "message": user_message,
            "response": assistant_message,
            "sentiment": sentiment.label if sentiment else "neutral",
            "frustration_score": sentiment.frustration_score if sentiment else 0.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._save_memory_item(user_id=user_id, payload=payload)

        for pattern in (r"我叫([^\s，。,.!?？!]+)", r"my name is ([a-zA-Z]+)"):
            match = re.search(pattern, user_message, re.IGNORECASE)
            if match:
                self._save_memory_item(
                    user_id=user_id,
                    payload={
                        "kind": "user_fact",
                        "fact": f"用户姓名可能是：{match.group(1)}",
                        "thread_id": thread_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )

    def _call_role_agent(
        self,
        role: str,
        user_id: str,
        thread_id: str,
        intent: str,
        risk: str,
        message: str,
    ) -> Dict[str, Any]:
        """调用指定角色 Agent。"""
        if not self.llm_enabled or role not in self.role_agents:
            fallback = self._fallback_response(role, user_id, message, None)
            return {"messages": [AIMessage(content=fallback)]}

        agent = self.role_agents[role]
        config = {"configurable": {"thread_id": self._agent_thread_id(thread_id, role)}}
        context = AgentRuntimeContext(
            user_id=user_id,
            thread_id=thread_id,
            active_agent=role,
            intent=intent,
            risk=risk,
        )
        return agent.invoke(
            {"messages": [HumanMessage(content=message)]},
            config=config,
            context=context,
        )

    def _route_after_analyze(self, state: OrchestrationState) -> str:
        selected = state.get("selected_agent", "supervisor")
        if selected == "escalation":
            return "escalation"
        if selected == "knowledge":
            return "knowledge"
        if selected == "action":
            return "action"
        return "respond"

    def _route_after_knowledge(self, state: OrchestrationState) -> str:
        if state.get("needs_action_after_knowledge"):
            return "action"
        return "respond"

    def _route_after_execution(self, state: OrchestrationState) -> str:
        if state.get("run_status") == "interrupted":
            return "end"
        return "respond"

    def _compose_knowledge_prompt(self, state: OrchestrationState) -> str:
        return (
            f"用户问题：{state.get('current_message', '')}\n"
            "任务：调用知识检索工具，输出关键结论，并附上来源行（格式：来源：xxx）。"
        )

    def _compose_action_prompt(self, state: OrchestrationState) -> str:
        retrieval = state.get("retrieval_text", "")
        retrieval_block = f"\n可用检索证据：\n{retrieval}\n" if retrieval else ""
        return (
            f"用户请求：{state.get('current_message', '')}\n"
            f"{retrieval_block}"
            "任务：判断是否需要调用工单/账户类工具，必要时执行，输出执行结果。"
        )

    def _compose_escalation_prompt(self, state: OrchestrationState) -> str:
        retrieval = state.get("retrieval_text", "")
        retrieval_block = f"\n可用检索证据：\n{retrieval}\n" if retrieval else ""
        return (
            f"用户消息：{state.get('current_message', '')}\n"
            f"{retrieval_block}"
            "任务：按升级策略处理，高风险场景优先人工升级并给出交接摘要。"
        )

    def _compose_responder_prompt(self, state: OrchestrationState) -> str:
        retrieval = state.get("retrieval_text", "")
        tool_text = state.get("tool_text", "")
        ticket_id = state.get("ticket_id")
        escalation_note = "是" if state.get("escalated") else "否"
        return (
            "请基于以下结构化上下文生成最终客服答复：\n"
            f"- 用户问题：{state.get('current_message', '')}\n"
            f"- 识别意图：{state.get('intent', 'other')}\n"
            f"- 风险等级：{state.get('risk', 'low')}\n"
            f"- 情绪标签：{state.get('sentiment_label', 'neutral')}\n"
            f"- 挫败分：{state.get('frustration_score', 0.0):.2f}\n"
            f"- 是否人工升级：{escalation_note}\n"
            f"- 工单号：{ticket_id or '无'}\n"
            f"- 检索证据：\n{retrieval or '无'}\n"
            f"- 工具执行结果：\n{tool_text or '无'}\n"
            "要求：中文输出；先结论后步骤；若有来源请保留“来源：xxx”；"
            "如果证据不足，要明确告知并给出下一步建议。"
        )

    def _node_analyze(self, state: OrchestrationState) -> Dict[str, Any]:
        message = state.get("current_message", "")
        baseline_sentiment = SentimentResult(
            polarity=0.0,
            subjectivity=0.0,
            label=self._normalize_sentiment_label(state.get("sentiment_label")),
            frustration_score=self._safe_float(state.get("frustration_score"), 0.0),
            keywords=[],
        )
        baseline_intent = self._infer_intent(message)
        baseline_risk = self._infer_risk(message, baseline_sentiment)

        needs_knowledge = baseline_intent in {"question", "complaint"}
        needs_action = baseline_intent == "request" or any(
            token in message.lower() for token in self.TICKET_HINTS + self.ACCOUNT_HINTS
        )
        needs_escalation = baseline_risk == "high"

        llm_analysis = self._llm_analyze_message(
            message=message,
            baseline_intent=baseline_intent,
            baseline_risk=baseline_risk,
            baseline_sentiment=baseline_sentiment,
        )

        intent = baseline_intent
        risk = baseline_risk
        sentiment_label = baseline_sentiment.label
        frustration_score = baseline_sentiment.frustration_score
        reason = "heuristic"

        if llm_analysis:
            intent = llm_analysis["intent"]
            risk = llm_analysis["risk"]
            sentiment_label = llm_analysis["sentiment_label"]
            frustration_score = llm_analysis["frustration_score"]
            needs_knowledge = llm_analysis["needs_knowledge"] or needs_knowledge
            needs_action = llm_analysis["needs_action"] or needs_action
            needs_escalation = llm_analysis["needs_escalation"] or (risk == "high")
            reason = llm_analysis.get("reason", "") or "llm"
        elif intent == "request" and "怎么" in message:
            needs_knowledge = True

        selected_agent, needs_action_after_knowledge = self._plan_route(
            intent=intent,
            risk=risk,
            needs_knowledge=needs_knowledge,
            needs_action=needs_action,
            needs_escalation=needs_escalation,
        )

        user_id = state.get("user_id", "unknown_user")
        thread_id = state.get("thread_id", "unknown_thread")
        self._save_memory_item(
            user_id=user_id,
            payload={
                "kind": "routing_decision",
                "thread_id": thread_id,
                "message": message,
                "intent": intent,
                "risk": risk,
                "selected_agent": selected_agent,
                "needs_knowledge": needs_knowledge,
                "needs_action": needs_action,
                "needs_action_after_knowledge": needs_action_after_knowledge,
                "needs_escalation": needs_escalation,
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        return {
            "intent": intent,
            "risk": risk,
            "sentiment_label": sentiment_label,
            "frustration_score": frustration_score,
            "selected_agent": selected_agent,
            "active_agent": selected_agent,
            "needs_knowledge": needs_knowledge,
            "needs_action": needs_action,
            "needs_action_after_knowledge": needs_action_after_knowledge,
            "needs_escalation": needs_escalation,
            "run_status": "completed",
            "interrupts": [],
        }

    def _node_knowledge(self, state: OrchestrationState) -> Dict[str, Any]:
        user_id = state.get("user_id", "unknown_user")
        thread_id = state.get("thread_id", "unknown_thread")
        intent = state.get("intent", "question")
        risk = state.get("risk", "low")

        if self.llm_enabled:
            result = self._call_role_agent(
                role="knowledge",
                user_id=user_id,
                thread_id=thread_id,
                intent=intent,
                risk=risk,
                message=self._compose_knowledge_prompt(state),
            )
            retrieval_text = self._extract_ai_text(result)
            interrupts = self._extract_interrupts(result)
            if interrupts:
                return {
                    "active_agent": "knowledge",
                    "run_status": "interrupted",
                    "interrupts": interrupts,
                }
        else:
            retrieval_text = self._fallback_response("knowledge", user_id, state.get("current_message", ""), None)

        citations = self._merge_unique(state.get("citations", []), self._extract_citations(retrieval_text))
        return {
            "active_agent": "knowledge",
            "retrieval_text": retrieval_text,
            "citations": citations,
        }

    def _node_action(self, state: OrchestrationState) -> Dict[str, Any]:
        user_id = state.get("user_id", "unknown_user")
        thread_id = state.get("thread_id", "unknown_thread")
        intent = state.get("intent", "request")
        risk = state.get("risk", "medium")

        if self.llm_enabled:
            result = self._call_role_agent(
                role="action",
                user_id=user_id,
                thread_id=thread_id,
                intent=intent,
                risk=risk,
                message=self._compose_action_prompt(state),
            )
            interrupts = self._extract_interrupts(result)
            if interrupts:
                return {
                    "active_agent": "action",
                    "run_status": "interrupted",
                    "interrupts": interrupts,
                }
            tool_text = self._extract_ai_text(result)
        else:
            tool_text = self._fallback_response("action", user_id, state.get("current_message", ""), None)

        ticket_id = self._ticket_id_from_text(tool_text) or state.get("ticket_id")
        citations = self._merge_unique(state.get("citations", []), self._extract_citations(tool_text))
        return {
            "active_agent": "action",
            "tool_text": tool_text,
            "ticket_id": ticket_id,
            "run_status": "completed",
            "interrupts": [],
            "citations": citations,
        }

    def _node_escalation(self, state: OrchestrationState) -> Dict[str, Any]:
        user_id = state.get("user_id", "unknown_user")
        thread_id = state.get("thread_id", "unknown_thread")
        intent = state.get("intent", "complaint")

        if self.llm_enabled:
            result = self._call_role_agent(
                role="escalation",
                user_id=user_id,
                thread_id=thread_id,
                intent=intent,
                risk="high",
                message=self._compose_escalation_prompt(state),
            )
            interrupts = self._extract_interrupts(result)
            if interrupts:
                return {
                    "active_agent": "escalation",
                    "run_status": "interrupted",
                    "interrupts": interrupts,
                    "escalated": True,
                }
            tool_text = self._extract_ai_text(result)
        else:
            tool_text = self._fallback_response("escalation", user_id, state.get("current_message", ""), None)

        ticket_id = self._ticket_id_from_text(tool_text) or state.get("ticket_id")
        citations = self._merge_unique(state.get("citations", []), self._extract_citations(tool_text))
        return {
            "active_agent": "escalation",
            "tool_text": tool_text,
            "ticket_id": ticket_id,
            "escalated": True,
            "run_status": "completed",
            "interrupts": [],
            "citations": citations,
        }

    def _node_respond(self, state: OrchestrationState) -> Dict[str, Any]:
        if state.get("run_status") == "interrupted":
            return {}

        user_id = state.get("user_id", "unknown_user")
        thread_id = state.get("thread_id", "unknown_thread")
        intent = state.get("intent", "other")
        risk = state.get("risk", "low")

        if self.llm_enabled:
            result = self._call_role_agent(
                role="responder",
                user_id=user_id,
                thread_id=thread_id,
                intent=intent,
                risk=risk,
                message=self._compose_responder_prompt(state),
            )
            final_message = self._extract_ai_text(result)
        else:
            final_message = self._fallback_response(
                role="responder",
                user_id=user_id,
                message=state.get("current_message", ""),
                sentiment=None,
                retrieval_text=state.get("retrieval_text", ""),
                tool_text=state.get("tool_text", ""),
            )

        citations = self._merge_unique(state.get("citations", []), self._extract_citations(final_message))
        ticket_id = state.get("ticket_id") or self._ticket_id_from_text(final_message)
        escalated = bool(
            state.get("escalated")
            or state.get("selected_agent") == "escalation"
            or ("人工" in final_message and ticket_id)
        )

        return {
            "final_message": final_message,
            "ticket_id": ticket_id,
            "escalated": escalated,
            "citations": citations,
            "run_status": "completed",
        }

    def _build_sentiment_result(
        self,
        baseline: SentimentResult,
        state: OrchestrationState,
    ) -> SentimentResult:
        return SentimentResult(
            polarity=baseline.polarity,
            subjectivity=baseline.subjectivity,
            label=self._normalize_sentiment_label(state.get("sentiment_label", baseline.label)),
            frustration_score=self._safe_float(
                state.get("frustration_score"),
                default=baseline.frustration_score,
            ),
            keywords=baseline.keywords,
        )

    def _resolve_sources(self, state: OrchestrationState) -> List[str]:
        sources: List[str] = []
        if state.get("retrieval_text"):
            sources.extend(["Hybrid RAG", "FAQ Knowledge Base"])
        if state.get("tool_text"):
            sources.append("Support Tools")
        if state.get("selected_agent") == "escalation" or state.get("escalated"):
            sources.append("Human Handoff")
        if self.llm_enabled:
            sources.append("Responder Agent")
        else:
            sources.append("Fallback Policy")
        return self._merge_unique([], sources)

    def chat(
        self,
        user_id: str,
        message: str,
        thread_id: Optional[str] = None,
    ) -> SupportResponse:
        """处理用户消息。"""
        thread = thread_id or str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        self._thread_user[thread] = user_id
        self._trace_by_thread[thread] = trace_id

        baseline_sentiment = (
            self.sentiment_analyzer.analyze(message)
            if self.sentiment_analyzer
            else SentimentResult(
                polarity=0.0,
                subjectivity=0.0,
                label="neutral",
                frustration_score=0.0,
                keywords=[],
            )
        )

        self._record_history(user_id, "user", message)

        initial_state: OrchestrationState = {
            "user_id": user_id,
            "thread_id": thread,
            "current_message": message,
            "intent": self._infer_intent(message),
            "risk": self._infer_risk(message, baseline_sentiment),
            "sentiment_label": baseline_sentiment.label,
            "frustration_score": baseline_sentiment.frustration_score,
            "selected_agent": "supervisor",
            "active_agent": "supervisor",
            "needs_knowledge": False,
            "needs_action": False,
            "needs_action_after_knowledge": False,
            "needs_escalation": False,
            "retrieval_text": "",
            "tool_text": "",
            "final_message": "",
            "citations": [],
            "run_status": "completed",
            "interrupts": [],
            "ticket_id": None,
            "escalated": False,
        }

        try:
            final_state = self.orchestration_graph.invoke(initial_state)
        except Exception as error:
            logger.error(f"Graph invocation failed, fallback to template: {error}")
            fallback = self._fallback_response(
                role="supervisor",
                user_id=user_id,
                message=message,
                sentiment=baseline_sentiment,
            )
            self._record_history(user_id, "assistant", fallback)
            return SupportResponse(
                message=fallback,
                intent=initial_state["intent"],
                sentiment=baseline_sentiment,
                sources=["Fallback Policy"],
                escalated=False,
                ticket_created=self._ticket_id_from_text(fallback),
                thread_id=thread,
                run_status="completed",
                interrupts=[],
                citations=self._extract_citations(fallback),
                active_agent=initial_state["selected_agent"],
                trace_id=trace_id,
            )

        if final_state.get("run_status") == "interrupted":
            pending_role = final_state.get("active_agent", final_state.get("selected_agent", "action"))
            self._pending_role[thread] = pending_role
            self._pending_state[thread] = dict(final_state)
            wait_message = "检测到高风险动作，已暂停执行，等待人工审批（approve/edit/reject）。"
            self._record_history(user_id, "assistant", wait_message)
            sentiment = self._build_sentiment_result(baseline_sentiment, final_state)
            return SupportResponse(
                message=wait_message,
                intent=final_state.get("intent", "other"),
                sentiment=sentiment,
                sources=["HITL Middleware"],
                escalated=pending_role == "escalation",
                ticket_created=None,
                thread_id=thread,
                run_status="interrupted",
                interrupts=final_state.get("interrupts", []),
                citations=final_state.get("citations", []),
                active_agent=pending_role,
                trace_id=trace_id,
            )

        response_message = final_state.get("final_message") or final_state.get("tool_text") or final_state.get(
            "retrieval_text"
        ) or "抱歉，我暂时无法给出有效回复。"
        citations = final_state.get("citations", [])
        ticket_id = final_state.get("ticket_id") or self._ticket_id_from_text(response_message)
        active_agent = final_state.get("selected_agent", "supervisor")
        sentiment = self._build_sentiment_result(baseline_sentiment, final_state)
        sources = self._resolve_sources(final_state)

        self._record_history(user_id, "assistant", response_message)
        self._save_turn_memory(
            user_id=user_id,
            thread_id=thread,
            intent=final_state.get("intent", "other"),
            active_agent=active_agent,
            user_message=message,
            assistant_message=response_message,
            sentiment=sentiment,
        )

        return SupportResponse(
            message=response_message,
            intent=final_state.get("intent", "other"),
            sentiment=sentiment,
            sources=sources,
            escalated=bool(final_state.get("escalated")),
            ticket_created=ticket_id,
            thread_id=thread,
            run_status="completed",
            interrupts=[],
            citations=citations,
            active_agent=active_agent,
            trace_id=trace_id,
        )

    def resume(
        self,
        thread_id: str,
        decisions: List[Dict[str, Any]],
    ) -> SupportResponse:
        """恢复 HITL 中断会话。"""
        user_id = self._thread_user.get(thread_id, "unknown_user")
        trace_id = self._trace_by_thread.get(thread_id, str(uuid.uuid4()))
        role = self._pending_role.get(thread_id)
        pending_state = self._pending_state.get(thread_id, {})

        neutral_sentiment = SentimentResult(
            polarity=0.0,
            subjectivity=0.0,
            label="neutral",
            frustration_score=0.0,
            keywords=[],
        )

        if role is None:
            return SupportResponse(
                message="未找到待审批线程，无法恢复。",
                intent="resume",
                sentiment=neutral_sentiment,
                sources=["HITL Middleware"],
                thread_id=thread_id,
                run_status="error",
                active_agent="unknown",
                trace_id=trace_id,
            )

        if not self.llm_enabled or role not in self.role_agents:
            self._pending_role.pop(thread_id, None)
            self._pending_state.pop(thread_id, None)
            return SupportResponse(
                message="当前处于无 LLM 模式，已跳过审批并结束流程。",
                intent="resume",
                sentiment=neutral_sentiment,
                sources=["Fallback Policy"],
                thread_id=thread_id,
                run_status="completed",
                active_agent=role,
                trace_id=trace_id,
            )

        agent = self.role_agents[role]
        config = {"configurable": {"thread_id": self._agent_thread_id(thread_id, role)}}
        context = AgentRuntimeContext(
            user_id=user_id,
            thread_id=thread_id,
            active_agent=role,
            intent=pending_state.get("intent", "resume"),
            risk=pending_state.get("risk", "medium"),
        )

        try:
            result = agent.invoke(
                Command(resume={"decisions": decisions}),
                config=config,
                context=context,
            )
        except Exception as error:
            logger.error(f"Resume failed: {error}")
            return SupportResponse(
                message=f"恢复会话失败：{error}",
                intent="resume",
                sentiment=neutral_sentiment,
                sources=["HITL Middleware"],
                thread_id=thread_id,
                run_status="error",
                active_agent=role,
                trace_id=trace_id,
            )

        interrupts = self._extract_interrupts(result)
        if interrupts:
            return SupportResponse(
                message="仍有待审批动作，请继续提交决策。",
                intent="resume",
                sentiment=neutral_sentiment,
                sources=["HITL Middleware"],
                thread_id=thread_id,
                run_status="interrupted",
                interrupts=interrupts,
                active_agent=role,
                trace_id=trace_id,
            )

        resumed_text = self._extract_ai_text(result)
        merged_state: OrchestrationState = dict(pending_state)
        merged_state["active_agent"] = role
        merged_state["run_status"] = "completed"
        merged_state["interrupts"] = []
        merged_state["citations"] = self._merge_unique(
            merged_state.get("citations", []),
            self._extract_citations(resumed_text),
        )

        if role in {"action", "escalation"}:
            merged_state["tool_text"] = resumed_text
            merged_state["ticket_id"] = merged_state.get("ticket_id") or self._ticket_id_from_text(resumed_text)
        if role == "escalation":
            merged_state["escalated"] = True

        responder_updates = self._node_respond(merged_state)
        merged_state.update(responder_updates)
        response_message = merged_state.get("final_message", resumed_text)

        self._pending_role.pop(thread_id, None)
        self._pending_state.pop(thread_id, None)
        self._record_history(user_id, "assistant", response_message)
        self._save_turn_memory(
            user_id=user_id,
            thread_id=thread_id,
            intent=merged_state.get("intent", "resume"),
            active_agent=merged_state.get("selected_agent", role),
            user_message="[resume]",
            assistant_message=response_message,
            sentiment=neutral_sentiment,
        )

        return SupportResponse(
            message=response_message,
            intent=merged_state.get("intent", "resume"),
            sentiment=neutral_sentiment,
            sources=self._resolve_sources(merged_state),
            escalated=bool(merged_state.get("escalated")),
            ticket_created=merged_state.get("ticket_id"),
            thread_id=thread_id,
            run_status="completed",
            citations=merged_state.get("citations", []),
            active_agent=merged_state.get("selected_agent", role),
            trace_id=trace_id,
        )

    def stream_chat(
        self,
        user_id: str,
        message: str,
        thread_id: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        """将聊天结果包装为可流式事件。"""
        response = self.chat(user_id=user_id, message=message, thread_id=thread_id)
        yield {
            "type": "node",
            "thread_id": response.thread_id,
            "active_agent": response.active_agent,
            "trace_id": response.trace_id,
        }
        if response.run_status == "interrupted":
            yield {
                "type": "interrupt",
                "thread_id": response.thread_id,
                "interrupts": response.interrupts,
            }
            yield {"type": "done", "payload": response.to_dict()}
            return

        for token in response.message.split():
            yield {"type": "token", "content": token}
        yield {"type": "done", "payload": response.to_dict()}

    def reindex_knowledge(self, clear_existing: bool = False) -> str:
        """重建知识库索引。"""
        return reindex_knowledge_base.invoke({"clear_existing": clear_existing})

    def reset_conversation(self, user_id: str) -> None:
        """重置本地会话历史。"""
        with self._lock:
            self._history[user_id] = []
            to_remove = [thread for thread, owner in self._thread_user.items() if owner == user_id]
            for thread in to_remove:
                self._pending_role.pop(thread, None)
                self._pending_state.pop(thread, None)
                self._thread_user.pop(thread, None)
                self._trace_by_thread.pop(thread, None)

    def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """读取本地会话历史（用于 API 展示）。"""
        with self._lock:
            history = self._history.get(user_id, [])
            return history[-max(1, limit):]


_support_agent: Optional[SupportAgent] = None
_agent_lock = threading.Lock()


def peek_support_agent() -> Optional[SupportAgent]:
    """仅查看全局实例，不触发初始化。"""
    return _support_agent


def get_support_agent() -> SupportAgent:
    """获取全局 SupportAgent 实例。"""
    global _support_agent
    with _agent_lock:
        if _support_agent is None:
            _support_agent = SupportAgent()
        return _support_agent
